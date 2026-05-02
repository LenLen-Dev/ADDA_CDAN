"""Core losses for PL-EPA-ADDA.

PL-EPA-ADDA extends ADDA with three target-side regularizers:

1. confidence-filtered pseudo-label self-training;
2. target entropy minimization;
3. source-class prototype alignment for confident target features.

The module is intentionally independent from the training script so it can be
unit-tested with random tensors and reused by other entrypoints.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PseudoLabelStats:
    """Diagnostics produced by pseudo-label based losses."""

    confidence: torch.Tensor
    pseudo_labels: torch.Tensor
    mask: torch.Tensor

    @property
    def selected_ratio(self) -> torch.Tensor:
        return self.mask.float().mean()


def entropy_minimization_loss(logits: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Minimize prediction entropy on unlabeled target samples.

    Args:
        logits: Raw classifier outputs with shape ``(N, C)``.
        reduction: ``"mean"``, ``"sum"`` or ``"none"``.
    """
    probabilities = F.softmax(logits, dim=1)
    entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-6))).sum(dim=1)
    if reduction == "mean":
        return entropy.mean()
    if reduction == "sum":
        return entropy.sum()
    if reduction == "none":
        return entropy
    raise ValueError(f"Unsupported reduction: {reduction}")


class ConfidencePseudoLabelLoss(nn.Module):
    """Cross-entropy on target samples whose maximum probability is reliable.

    Pseudo labels are generated from a detached copy of the current prediction,
    so the target model learns from high-confidence decisions without backpropagating
    through the label-generation path.
    """

    def __init__(self, threshold: float = 0.95):
        super().__init__()
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.threshold = threshold

    def forward(self, logits: torch.Tensor) -> Tuple[torch.Tensor, PseudoLabelStats]:
        probabilities = F.softmax(logits.detach(), dim=1)
        confidence, pseudo_labels = probabilities.max(dim=1)
        mask = confidence.ge(self.threshold)

        per_sample_loss = F.cross_entropy(logits, pseudo_labels, reduction="none")
        if mask.any():
            loss = per_sample_loss[mask].mean()
        else:
            loss = logits.sum() * 0.0

        return loss, PseudoLabelStats(confidence=confidence, pseudo_labels=pseudo_labels, mask=mask)


class SourcePrototypeAlignmentLoss(nn.Module):
    """Align confident target features to source-domain class prototypes.

    Source prototypes are updated online from labeled source mini-batches with an
    exponential moving average. Target features are assigned to their confident
    pseudo-label class and penalized by cosine distance to the corresponding
    source prototype.
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        momentum: float = 0.9,
        normalize: bool = True,
        min_confidence: float = 0.95,
    ):
        super().__init__()
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.normalize = normalize
        self.min_confidence = min_confidence

        self.register_buffer("prototypes", torch.zeros(num_classes, feature_dim))
        self.register_buffer("initialized", torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def update_source(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """Update source prototypes from labeled source features."""
        features = features.detach()
        labels = labels.detach().long()
        if self.normalize:
            features = F.normalize(features, dim=1)

        for class_idx in labels.unique():
            class_id = int(class_idx.item())
            if class_id < 0 or class_id >= self.num_classes:
                continue
            class_features = features[labels == class_idx]
            if class_features.numel() == 0:
                continue
            batch_proto = class_features.mean(dim=0)
            if self.normalize:
                batch_proto = F.normalize(batch_proto, dim=0)

            if self.initialized[class_id]:
                updated = self.momentum * self.prototypes[class_id] + (1.0 - self.momentum) * batch_proto
            else:
                updated = batch_proto
                self.initialized[class_id] = True
            if self.normalize:
                updated = F.normalize(updated, dim=0)
            self.prototypes[class_id].copy_(updated)

    def forward(
        self,
        target_features: torch.Tensor,
        pseudo_labels: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return cosine prototype alignment loss for confident target samples."""
        if mask is None:
            if confidence is None:
                mask = torch.ones_like(pseudo_labels, dtype=torch.bool)
            else:
                mask = confidence.ge(self.min_confidence)
        else:
            mask = mask.bool()

        proto_ready = self.initialized[pseudo_labels.long()]
        mask = mask & proto_ready
        if not mask.any():
            return target_features.sum() * 0.0

        selected_features = target_features[mask]
        selected_labels = pseudo_labels[mask].long()
        selected_prototypes = self.prototypes[selected_labels].detach()

        if self.normalize:
            selected_features = F.normalize(selected_features, dim=1)
            selected_prototypes = F.normalize(selected_prototypes, dim=1)

        cosine_similarity = (selected_features * selected_prototypes).sum(dim=1)
        return (1.0 - cosine_similarity).mean()

    def ready_ratio(self) -> torch.Tensor:
        return self.initialized.float().mean()


class ImprovedADDALoss(nn.Module):
    """Bundle pseudo-label, entropy, and prototype losses for ADDA training."""

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        pseudo_threshold: float = 0.95,
        prototype_momentum: float = 0.9,
    ):
        super().__init__()
        self.pseudo_label = ConfidencePseudoLabelLoss(threshold=pseudo_threshold)
        self.prototype = SourcePrototypeAlignmentLoss(
            num_classes=num_classes,
            feature_dim=feature_dim,
            momentum=prototype_momentum,
            min_confidence=pseudo_threshold,
        )
        self.last_stats: Dict[str, float] = {}

    @torch.no_grad()
    def update_source_prototypes(self, source_features: torch.Tensor, source_labels: torch.Tensor) -> None:
        self.prototype.update_source(source_features, source_labels)

    def forward(self, target_logits: torch.Tensor, target_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        pseudo_loss, stats = self.pseudo_label(target_logits)
        entropy_loss = entropy_minimization_loss(target_logits)
        prototype_loss = self.prototype(
            target_features,
            stats.pseudo_labels,
            confidence=stats.confidence,
            mask=stats.mask,
        )

        self.last_stats = {
            "pseudo_ratio": float(stats.selected_ratio.detach().cpu()),
            "prototype_ready_ratio": float(self.prototype.ready_ratio().detach().cpu()),
            "mean_confidence": float(stats.confidence.mean().detach().cpu()),
        }
        return {
            "pseudo": pseudo_loss,
            "entropy": entropy_loss,
            "prototype": prototype_loss,
        }
