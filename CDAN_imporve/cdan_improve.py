"""Training entrypoint for TLC-CDAN on image classification domain adaptation.

The script mirrors the original examples/domain_adaptation/image_classification/cdan.py
but replaces CDAN's outer-product / fixed-random conditional map with a learnable
compact conditional map and a temperature-controlled class condition vector.
"""
import argparse
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
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import utils
from lc_cdan import TemperatureConditionalDomainAdversarialLoss
from tllib.alignment.cdan import ImageClassifier
from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.utils.analysis import a_distance, collect_feature, tsne
from tllib.utils.data import ForeverDataIterator
from tllib.utils.logger import CompleteLogger
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.metric import accuracy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    train_transform = utils.get_train_transform(args.train_resizing, scale=args.scale, ratio=args.ratio,
                                                random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source, args.target, train_transform, val_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                 pool_layer=pool_layer, finetune=not args.scratch).to(device)
    classifier_feature_dim = classifier.features_dim

    domain_discri = DomainDiscriminator(args.condition_dim, hidden_size=args.discriminator_hidden_size).to(device)
    domain_adv = TemperatureConditionalDomainAdversarialLoss(
        domain_discri,
        features_dim=classifier_feature_dim,
        num_classes=num_classes,
        condition_dim=args.condition_dim,
        entropy_conditioning=args.entropy,
        condition_dropout=args.condition_dropout,
        max_temperature=args.max_temperature,
        min_temperature=args.min_temperature,
        temperature_schedule=args.temperature_schedule,
    ).to(device)

    all_parameters = classifier.get_parameters() + domain_discri.get_parameters() + domain_adv.get_parameters()
    optimizer = SGD(all_parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    if args.phase == 'analysis':
        feature_extractor = nn.Sequential(classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
        source_feature = collect_feature(train_source_loader, feature_extractor, device)
        target_feature = collect_feature(train_target_loader, feature_extractor, device)
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.pdf')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        acc1 = utils.validate(test_loader, classifier, args, device)
        print(acc1)
        return

    best_acc1 = 0.
    total_steps = args.epochs * args.iters_per_epoch
    for epoch in range(args.epochs):
        print("lr:", lr_scheduler.get_last_lr()[0])
        train(train_source_iter, train_target_iter, classifier, domain_adv, optimizer,
              lr_scheduler, epoch, total_steps, args)

        acc1 = utils.validate(val_loader, classifier, args, device)

        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best'), map_location='cpu'))
    acc1 = utils.validate(test_loader, classifier, args, device)
    print("test_acc1 = {:3.1f}".format(acc1))

    logger.close()


def _as_float(value):
    return value.item() if hasattr(value, 'item') else float(value)


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator, model: ImageClassifier,
          domain_adv: TemperatureConditionalDomainAdversarialLoss, optimizer: SGD,
          lr_scheduler: LambdaLR, epoch: int, total_steps: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    trans_losses = AverageMeter('Trans Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    domain_accs = AverageMeter('Domain Acc', ':3.1f')
    temp_meter = AverageMeter('Temp', ':3.2f')
    progress_meter = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, trans_losses, cls_accs, domain_accs, temp_meter],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    domain_adv.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)[:2]
        x_t, = next(train_target_iter)[:1]

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)

        data_time.update(time.time() - end)

        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x)
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)

        global_step = epoch * args.iters_per_epoch + i
        train_progress = global_step / max(1, total_steps - 1)

        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = domain_adv(y_s, f_s, y_t, f_t, progress=train_progress)
        domain_acc = domain_adv.domain_discriminator_accuracy
        loss = cls_loss + transfer_loss * args.trade_off

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(_as_float(cls_acc), x_s.size(0))
        domain_accs.update(_as_float(domain_acc), x_s.size(0))
        trans_losses.update(transfer_loss.item(), x_s.size(0))
        temp_meter.update(domain_adv.current_temperature, x_s.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress_meter.display(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TLC-CDAN for Unsupervised Domain Adaptation')
    parser.add_argument('root', metavar='DIR', help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) + ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224, help='the image size after resizing')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--no-hflip', action='store_true', help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+', default=(0.485, 0.456, 0.406),
                        help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+', default=(0.229, 0.224, 0.225),
                        help='normalization std')

    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=utils.get_model_names(),
                        help='backbone architecture: ' + ' | '.join(utils.get_model_names()) + ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=256, type=int, help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true', help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')

    parser.add_argument('--condition-dim', default=1024, type=int,
                        help='Dimension of learnable compact conditional representation')
    parser.add_argument('--condition-dropout', default=0.1, type=float,
                        help='Dropout probability in learnable conditional map')
    parser.add_argument('--discriminator-hidden-size', default=1024, type=int,
                        help='Hidden size of the domain discriminator')
    parser.add_argument('--max-temperature', default=2.0, type=float,
                        help='Initial / constant softmax temperature for conditional vector')
    parser.add_argument('--min-temperature', default=1.0, type=float,
                        help='Final softmax temperature when using linear schedule')
    parser.add_argument('--temperature-schedule', default='linear', choices=['linear', 'constant'],
                        help='Temperature schedule for class-condition vector')
    parser.add_argument('--entropy', default=False, action='store_true',
                        help='use CDAN entropy-aware instance weighting, not target entropy minimization')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')

    parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR',
                        help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float, metavar='W',
                        help='weight decay (default: 1e-3)', dest='weight_decay')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int, metavar='N',
                        help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument('--log', type=str, default='tlc_cdan',
                        help='Where to save logs, checkpoints and debugging images.')
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model. When phase is 'analysis', only analysis the model.")
    main(parser.parse_args())
