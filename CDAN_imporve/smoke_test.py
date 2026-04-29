"""Random-tensor smoke test for TLC-CDAN components."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from lc_cdan import LearnableCompactConditionalMap, TemperatureConditionalDomainAdversarialLoss
from tllib.modules.domain_discriminator import DomainDiscriminator


def main():
    torch.manual_seed(0)
    batch_size = 8
    features_dim = 32
    num_classes = 5
    condition_dim = 16

    f_s = torch.randn(batch_size, features_dim, requires_grad=True)
    f_t = torch.randn(batch_size, features_dim, requires_grad=True)
    g_s = torch.randn(batch_size, num_classes, requires_grad=True)
    g_t = torch.randn(batch_size, num_classes, requires_grad=True)

    condition_map = LearnableCompactConditionalMap(features_dim, num_classes, condition_dim)
    mapped = condition_map(f_s, torch.softmax(g_s, dim=1))
    assert mapped.shape == (batch_size, condition_dim), mapped.shape

    discriminator = DomainDiscriminator(condition_dim, hidden_size=32)
    loss_fn = TemperatureConditionalDomainAdversarialLoss(
        discriminator,
        features_dim=features_dim,
        num_classes=num_classes,
        condition_dim=condition_dim,
        max_temperature=2.0,
        min_temperature=1.0,
        temperature_schedule="linear",
    )
    loss = loss_fn(g_s, f_s, g_t, f_t, progress=0.5)
    loss.backward()

    assert torch.isfinite(loss).item(), loss
    assert f_s.grad is not None and torch.isfinite(f_s.grad).all().item()
    assert f_t.grad is not None and torch.isfinite(f_t.grad).all().item()
    assert any(p.grad is not None for p in loss_fn.map.parameters())
    assert any(p.grad is not None for p in discriminator.parameters())
    print("TLC-CDAN smoke test passed.")
    print("loss = {:.6f}, temperature = {:.3f}, domain_acc = {}".format(
        loss.item(), loss_fn.current_temperature, loss_fn.domain_discriminator_accuracy
    ))


if __name__ == "__main__":
    main()
