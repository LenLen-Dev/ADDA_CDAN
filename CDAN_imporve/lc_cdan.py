"""Improved CDAN components: learnable compact conditional mapping.

This module implements TLC-CDAN (Temperature-guided Learnable Compact CDAN),
a CDAN variant that replaces the original high-dimensional outer product or
fixed random multilinear map with a learnable low-dimensional interaction map.
"""
from typing import Dict, List, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tllib.modules.entropy import entropy
from tllib.modules.grl import WarmStartGradientReverseLayer
from tllib.utils.metric import accuracy, binary_accuracy

__all__ = [
    "LearnableCompactConditionalMap",
    "TemperatureConditionalDomainAdversarialLoss",
]


class LearnableCompactConditionalMap(nn.Module):
    r"""Learn a compact conditional representation for CDAN.

    Original CDAN uses the outer product :math:`g \otimes f`, whose dimension is
    :math:`C \times F`, or a fixed random multilinear map. This module learns two
    projections and fuses them by element-wise multiplication:

    .. math::
        z_f = \phi_f(f),\quad z_g = \phi_g(g),\quad
        T_\theta(f,g) = \operatorname{Norm}(z_f \odot z_g / \sqrt{d}).

    Args:
        features_dim: Dimension of image features ``f``.
        num_classes: Number of classes, i.e. dimension of class probability ``g``.
        output_dim: Dimension of compact conditional representation.
        dropout_p: Dropout probability used after each projection branch.
        use_layer_norm: Whether to normalize fused representation with LayerNorm.

    Shape:
        - f: ``(N, features_dim)``
        - g: ``(N, num_classes)``
        - output: ``(N, output_dim)``
    """

    def __init__(
        self,
        features_dim: int,
        num_classes: int,
        output_dim: int = 1024,
        dropout_p: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        if features_dim <= 0:
            raise ValueError("features_dim must be positive")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if not 0.0 <= dropout_p < 1.0:
            raise ValueError("dropout_p must be in [0, 1)")

        dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        self.feature_proj = nn.Sequential(
            nn.Linear(features_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            dropout,
        )
        self.class_proj = nn.Sequential(
            nn.Linear(num_classes, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity(),
        )
        self.norm = nn.LayerNorm(output_dim) if use_layer_norm else nn.Identity()
        self.output_dim = output_dim

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if f.dim() != 2 or g.dim() != 2:
            raise ValueError("f and g must both be 2-D tensors")
        z_f = self.feature_proj(f)
        z_g = self.class_proj(g)
        h = torch.mul(z_f, z_g) / math.sqrt(float(self.output_dim))
        return self.norm(h)

    def get_parameters(self) -> List[Dict]:
        return [{"params": self.parameters(), "lr": 1.0}]


class TemperatureConditionalDomainAdversarialLoss(nn.Module):
    r"""Conditional domain adversarial loss with learnable compact mapping.

    This is a drop-in CDAN-style adversarial loss. It keeps the source-domain
    classification objective outside this module and computes only the transfer
    loss:

    .. math::
        \mathcal{L}_{adv} = BCE(D(T_\theta(f_s, g_s^T)), 1)
        + BCE(D(T_\theta(f_t, g_t^T)), 0)

    where :math:`g^T = softmax(logits / T)`. The temperature can be constant or
    linearly annealed from ``max_temperature`` to ``min_temperature`` by passing
    training ``progress`` in ``[0, 1]`` to :meth:`forward`.

    Notes:
        ``g`` is detached after softmax, matching the original project CDAN
        implementation. Therefore, the adversarial loss optimizes the feature
        representation and condition map without directly forcing classifier
        logits through the conditional vector branch.
    """

    def __init__(
        self,
        domain_discriminator: nn.Module,
        features_dim: int,
        num_classes: int,
        condition_dim: int = 1024,
        entropy_conditioning: bool = False,
        condition_dropout: float = 0.1,
        max_temperature: float = 2.0,
        min_temperature: float = 1.0,
        temperature_schedule: str = "linear",
        reduction: str = "mean",
        sigmoid: bool = True,
        grl: Optional[nn.Module] = None,
    ):
        super().__init__()
        if max_temperature <= 0 or min_temperature <= 0:
            raise ValueError("temperatures must be positive")
        if min_temperature > max_temperature:
            raise ValueError("min_temperature must not exceed max_temperature")
        if temperature_schedule not in {"linear", "constant"}:
            raise ValueError("temperature_schedule must be 'linear' or 'constant'")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of: none, mean, sum")

        self.domain_discriminator = domain_discriminator
        self.map = LearnableCompactConditionalMap(
            features_dim=features_dim,
            num_classes=num_classes,
            output_dim=condition_dim,
            dropout_p=condition_dropout,
        )
        self.grl = grl if grl is not None else WarmStartGradientReverseLayer(
            alpha=1.0, lo=0.0, hi=1.0, max_iters=1000, auto_step=True
        )
        self.entropy_conditioning = entropy_conditioning
        self.max_temperature = float(max_temperature)
        self.min_temperature = float(min_temperature)
        self.temperature_schedule = temperature_schedule
        self.reduction = reduction
        self.sigmoid = sigmoid
        self.domain_discriminator_accuracy = None
        self.current_temperature = self.max_temperature

    def _temperature(self, progress: Optional[float] = None) -> float:
        if self.temperature_schedule == "constant" or progress is None:
            return self.max_temperature
        progress = min(1.0, max(0.0, float(progress)))
        return self.max_temperature - (self.max_temperature - self.min_temperature) * progress

    def forward(
        self,
        g_s: torch.Tensor,
        f_s: torch.Tensor,
        g_t: torch.Tensor,
        f_t: torch.Tensor,
        progress: Optional[float] = None,
    ) -> torch.Tensor:
        f = torch.cat((f_s, f_t), dim=0)
        g = torch.cat((g_s, g_t), dim=0)
        self.current_temperature = self._temperature(progress)
        g = F.softmax(g / self.current_temperature, dim=1).detach()

        h = self.grl(self.map(f, g))
        d = self.domain_discriminator(h)

        if self.sigmoid:
            d_label = torch.cat((
                torch.ones((g_s.size(0), 1), device=g_s.device),
                torch.zeros((g_t.size(0), 1), device=g_t.device),
            ))
            self.domain_discriminator_accuracy = binary_accuracy(d, d_label)

            if self.entropy_conditioning:
                weight = 1.0 + torch.exp(-entropy(g))
                weight = weight / torch.sum(weight) * f.size(0)
                return F.binary_cross_entropy(d, d_label, weight.view_as(d), reduction=self.reduction)
            return F.binary_cross_entropy(d, d_label, reduction=self.reduction)

        d_label = torch.cat((
            torch.ones((g_s.size(0),), device=g_s.device),
            torch.zeros((g_t.size(0),), device=g_t.device),
        )).long()
        self.domain_discriminator_accuracy = accuracy(d, d_label)
        if self.entropy_conditioning:
            raise NotImplementedError("entropy_conditioning is only implemented for sigmoid=True")
        return F.cross_entropy(d, d_label, reduction=self.reduction)

    def get_parameters(self) -> List[Dict]:
        """Return trainable parameters introduced by the improved condition map."""
        return self.map.get_parameters()
