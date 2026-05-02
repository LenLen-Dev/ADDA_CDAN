"""Training entrypoint for PL-EPA-ADDA on image classification UDA.

This script keeps the original ADDA two-stage structure:

1. pretrain a source classifier with source labels;
2. copy it to a target classifier and adversarially align target features to
   frozen source features.

The improvement adds target-side semantic constraints during stage 2:

- confidence-filtered pseudo-label self-training;
- entropy minimization for confident target decision boundaries;
- source class prototype alignment using high-confidence target pseudo labels.
"""
from __future__ import annotations

import argparse
import copy
import os.path as osp
import random
import shutil
import sys
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = PROJECT_ROOT / "examples" / "domain_adaptation" / "image_classification"
for path in (str(PROJECT_ROOT), str(EXAMPLE_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import utils
from pl_epa_adda import ImprovedADDALoss
from tllib.alignment.adda import ImageClassifier
from tllib.alignment.dann import DomainAdversarialLoss
from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.modules.grl import WarmStartGradientReverseLayer
from tllib.utils.analysis import a_distance, collect_feature, tsne
from tllib.utils.data import ForeverDataIterator
from tllib.utils.logger import CompleteLogger
from tllib.utils.meter import AverageMeter, ProgressMeter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_requires_grad(net: nn.Module, requires_grad: bool = False) -> None:
    """Enable or disable gradients for all parameters in a module."""
    for param in net.parameters():
        param.requires_grad = requires_grad


def regularizer_ramp(epoch: int, iteration: int, args: argparse.Namespace) -> float:
    """Linear warm-up for noisy target-side losses."""
    current = epoch + iteration / max(1, args.iters_per_epoch)
    if current < args.regularizer_start_epoch:
        return 0.0
    if args.regularizer_rampup_epochs <= 0:
        return 1.0
    progress = (current - args.regularizer_start_epoch) / args.regularizer_rampup_epochs
    return max(0.0, min(1.0, progress))


def build_classifier(args: argparse.Namespace, num_classes: int) -> ImageClassifier:
    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    pool_layer = nn.Identity() if args.no_pool else None
    return ImageClassifier(
        backbone,
        num_classes,
        bottleneck_dim=args.bottleneck_dim,
        pool_layer=pool_layer,
        finetune=not args.scratch,
    ).to(device)


def main(args: argparse.Namespace) -> None:
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This turns on CUDNN deterministic mode, "
            "which can slow training and may affect checkpoint restart behavior."
        )

    cudnn.benchmark = True

    train_transform = utils.get_train_transform(
        args.train_resizing,
        scale=args.scale,
        ratio=args.ratio,
        random_horizontal_flip=not args.no_hflip,
        random_color_jitter=False,
        resize_size=args.resize_size,
        norm_mean=args.norm_mean,
        norm_std=args.norm_std,
    )
    val_transform = utils.get_val_transform(
        args.val_resizing,
        resize_size=args.resize_size,
        norm_mean=args.norm_mean,
        norm_std=args.norm_std,
    )
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = (
        utils.get_dataset(args.data, args.root, args.source, args.target, train_transform, val_transform)
    )
    train_source_loader = DataLoader(
        train_source_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
    )
    train_target_loader = DataLoader(
        train_target_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    source_classifier = build_classifier(args, num_classes)

    if args.phase == "train":
        if args.pretrain is None:
            print("Pretraining the model on source domain.")
            args.pretrain = logger.get_checkpoint_path("pretrain")
            pretrain_model = build_classifier(args, num_classes)
            pretrain_optimizer = SGD(
                pretrain_model.get_parameters(),
                args.pretrain_lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                nesterov=True,
            )
            pretrain_lr_scheduler = LambdaLR(
                pretrain_optimizer,
                lambda x: args.pretrain_lr * (1.0 + args.lr_gamma * float(x)) ** (-args.lr_decay),
            )
            for epoch in range(args.pretrain_epochs):
                print("pretrain lr:", pretrain_lr_scheduler.get_last_lr())
                utils.empirical_risk_minimization(
                    train_source_iter,
                    pretrain_model,
                    pretrain_optimizer,
                    pretrain_lr_scheduler,
                    epoch,
                    args,
                    device,
                )
                utils.validate(val_loader, pretrain_model, args, device)
            torch.save(pretrain_model.state_dict(), args.pretrain)
            print("Pretraining process is done.")

        checkpoint = torch.load(args.pretrain, map_location="cpu")
        source_classifier.load_state_dict(checkpoint)

    target_classifier = copy.deepcopy(source_classifier)

    if args.phase != "train":
        checkpoint = torch.load(logger.get_checkpoint_path("best"), map_location="cpu")
        target_classifier.load_state_dict(checkpoint)

    set_requires_grad(source_classifier, False)
    # Keep source_classifier.training=True because tllib Classifier returns
    # (logits, features) only in training mode; freeze BN separately.
    source_classifier.train()
    source_classifier.freeze_bn()

    if not args.optimize_head:
        set_requires_grad(target_classifier.head, False)

    domain_discri = DomainDiscriminator(
        in_feature=source_classifier.features_dim,
        hidden_size=args.discriminator_hidden_size,
    ).to(device)
    grl = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=2.0, max_iters=1000, auto_step=True)
    domain_adv = DomainAdversarialLoss(domain_discri, grl=grl).to(device)
    improve_loss = ImprovedADDALoss(
        num_classes=num_classes,
        feature_dim=source_classifier.features_dim,
        pseudo_threshold=args.pseudo_threshold,
        prototype_momentum=args.prototype_momentum,
    ).to(device)

    if args.phase == "analysis":
        feature_extractor = nn.Sequential(
            target_classifier.backbone,
            target_classifier.pool_layer,
            target_classifier.bottleneck,
        ).to(device)
        source_feature = collect_feature(train_source_loader, feature_extractor, device)
        target_feature = collect_feature(train_target_loader, feature_extractor, device)
        tsne_filename = osp.join(logger.visualize_directory, "TSNE.pdf")
        tsne.visualize(source_feature, target_feature, tsne_filename)
        print("Saving t-SNE to", tsne_filename)
        a_dist = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", a_dist)
        return

    if args.phase == "test":
        acc1 = utils.validate(test_loader, target_classifier, args, device)
        print(acc1)
        return

    optimizer = SGD(
        target_classifier.get_parameters(optimize_head=args.optimize_head) + domain_discri.get_parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1.0 + args.lr_gamma * float(x)) ** (-args.lr_decay))

    best_acc1 = 0.0
    for epoch in range(args.epochs):
        print("lr:", lr_scheduler.get_last_lr())
        train(
            train_source_iter,
            train_target_iter,
            source_classifier,
            target_classifier,
            domain_adv,
            improve_loss,
            optimizer,
            lr_scheduler,
            epoch,
            args,
        )

        acc1 = utils.validate(val_loader, target_classifier, args, device)
        torch.save(target_classifier.state_dict(), logger.get_checkpoint_path("latest"))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path("latest"), logger.get_checkpoint_path("best"))
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))
    target_classifier.load_state_dict(torch.load(logger.get_checkpoint_path("best"), map_location="cpu"))
    acc1 = utils.validate(test_loader, target_classifier, args, device)
    print("test_acc1 = {:3.1f}".format(acc1))
    logger.close()


def _as_float(value):
    return value.item() if hasattr(value, "item") else float(value)


def train(
    train_source_iter: ForeverDataIterator,
    train_target_iter: ForeverDataIterator,
    source_model: ImageClassifier,
    target_model: ImageClassifier,
    domain_adv: DomainAdversarialLoss,
    improve_loss: ImprovedADDALoss,
    optimizer: SGD,
    lr_scheduler: LambdaLR,
    epoch: int,
    args: argparse.Namespace,
) -> None:
    batch_time = AverageMeter("Time", ":5.2f")
    data_time = AverageMeter("Data", ":5.2f")
    losses = AverageMeter("Loss", ":6.2f")
    losses_transfer = AverageMeter("Transfer", ":6.2f")
    losses_pseudo = AverageMeter("Pseudo", ":6.2f")
    losses_entropy = AverageMeter("Entropy", ":6.2f")
    losses_proto = AverageMeter("Proto", ":6.2f")
    pseudo_ratios = AverageMeter("PL Ratio", ":5.2f")
    proto_ready = AverageMeter("Proto Ready", ":5.2f")
    domain_accs = AverageMeter("Domain Acc", ":3.1f")
    progress = ProgressMeter(
        args.iters_per_epoch,
        [
            batch_time,
            data_time,
            losses,
            losses_transfer,
            losses_pseudo,
            losses_entropy,
            losses_proto,
            pseudo_ratios,
            proto_ready,
            domain_accs,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # tllib Classifier returns features only when training=True. Source weights
    # are frozen, and BatchNorm layers are put back to eval by freeze_bn().
    source_model.train()
    source_model.freeze_bn()
    target_model.train()
    domain_adv.train()
    improve_loss.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)[:2]
        x_t = next(train_target_iter)[0]

        x_s = x_s.to(device)
        labels_s = labels_s.to(device)
        x_t = x_t.to(device)

        data_time.update(time.time() - end)

        with torch.no_grad():
            _, f_s = source_model(x_s)
        y_t, f_t = target_model(x_t)

        improve_loss.update_source_prototypes(f_s, labels_s)
        transfer_loss = domain_adv(f_s, f_t)
        regularizer_scale = regularizer_ramp(epoch, i, args)
        loss_terms = improve_loss(y_t, f_t)

        pseudo_loss = loss_terms["pseudo"]
        entropy_loss = loss_terms["entropy"]
        prototype_loss = loss_terms["prototype"]
        loss = (
            args.trade_off * transfer_loss
            + regularizer_scale * args.pseudo_label_weight * pseudo_loss
            + regularizer_scale * args.entropy_weight * entropy_loss
            + regularizer_scale * args.prototype_weight * prototype_loss
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        batch_size = x_s.size(0)
        losses.update(loss.item(), batch_size)
        losses_transfer.update(transfer_loss.item(), batch_size)
        losses_pseudo.update(pseudo_loss.item(), batch_size)
        losses_entropy.update(entropy_loss.item(), batch_size)
        losses_proto.update(prototype_loss.item(), batch_size)
        pseudo_ratios.update(improve_loss.last_stats["pseudo_ratio"], batch_size)
        proto_ready.update(improve_loss.last_stats["prototype_ready_ratio"], batch_size)
        domain_accs.update(_as_float(domain_adv.domain_discriminator_accuracy), batch_size)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PL-EPA-ADDA for Unsupervised Domain Adaptation")
    parser.add_argument("root", metavar="DIR", help="root path of dataset")
    parser.add_argument(
        "-d",
        "--data",
        metavar="DATA",
        default="Office31",
        choices=utils.get_dataset_names(),
        help="dataset: " + " | ".join(utils.get_dataset_names()) + " (default: Office31)",
    )
    parser.add_argument("-s", "--source", help="source domain(s)", nargs="+")
    parser.add_argument("-t", "--target", help="target domain(s)", nargs="+")
    parser.add_argument("--train-resizing", type=str, default="default")
    parser.add_argument("--val-resizing", type=str, default="default")
    parser.add_argument("--resize-size", type=int, default=224, help="image size after resizing")
    parser.add_argument("--scale", type=float, nargs="+", default=[0.08, 1.0], metavar="PCT")
    parser.add_argument("--ratio", type=float, nargs="+", default=[3.0 / 4.0, 4.0 / 3.0], metavar="RATIO")
    parser.add_argument("--no-hflip", action="store_true", help="disable random horizontal flip")
    parser.add_argument("--norm-mean", type=float, nargs="+", default=(0.485, 0.456, 0.406))
    parser.add_argument("--norm-std", type=float, nargs="+", default=(0.229, 0.224, 0.225))

    parser.add_argument("-a", "--arch", metavar="ARCH", default="resnet18", choices=utils.get_model_names())
    parser.add_argument("--pretrain", type=str, default=None, help="source pretrain checkpoint")
    parser.add_argument("--bottleneck-dim", default=256, type=int, help="dimension of bottleneck")
    parser.add_argument("--no-pool", action="store_true", help="no pool layer after feature extractor")
    parser.add_argument("--scratch", action="store_true", help="train backbone from scratch")
    parser.add_argument("--optimize-head", action="store_true", help="also update target classifier head during ADDA stage")

    parser.add_argument("-b", "--batch-size", default=32, type=int, metavar="N")
    parser.add_argument("--lr", "--learning-rate", default=0.001, type=float, metavar="LR", dest="lr")
    parser.add_argument("--pretrain-lr", default=0.001, type=float)
    parser.add_argument("--lr-gamma", default=0.0003, type=float)
    parser.add_argument("--lr-decay", default=0.75, type=float)
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M")
    parser.add_argument("--wd", "--weight-decay", default=1e-3, type=float, metavar="W", dest="weight_decay")
    parser.add_argument("-j", "--workers", default=2, type=int, metavar="N")
    parser.add_argument("--epochs", default=20, type=int, metavar="N")
    parser.add_argument("--pretrain-epochs", default=3, type=int, metavar="N")
    parser.add_argument("-i", "--iters-per-epoch", default=1000, type=int)
    parser.add_argument("-p", "--print-freq", default=100, type=int, metavar="N")
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--per-class-eval", action="store_true")
    parser.add_argument("--log", type=str, default="logs/pl_epa_adda")
    parser.add_argument("--phase", type=str, default="train", choices=["train", "test", "analysis"])

    parser.add_argument("--discriminator-hidden-size", default=1024, type=int)
    parser.add_argument("--trade-off", default=1.0, type=float, help="domain adversarial loss weight")
    parser.add_argument("--pseudo-threshold", default=0.95, type=float, help="pseudo-label confidence threshold")
    parser.add_argument("--pseudo-label-weight", default=0.1, type=float)
    parser.add_argument("--entropy-weight", default=0.01, type=float)
    parser.add_argument("--prototype-weight", default=0.1, type=float)
    parser.add_argument("--prototype-momentum", default=0.9, type=float)
    parser.add_argument("--regularizer-start-epoch", default=1.0, type=float)
    parser.add_argument("--regularizer-rampup-epochs", default=5.0, type=float)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
