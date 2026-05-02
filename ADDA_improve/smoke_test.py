"""Random-tensor smoke test for PL-EPA-ADDA losses."""
import torch

from pl_epa_adda import ImprovedADDALoss


def main():
    torch.manual_seed(7)
    num_classes = 5
    feature_dim = 16
    loss_fn = ImprovedADDALoss(num_classes, feature_dim, pseudo_threshold=0.2)

    source_features = torch.randn(32, feature_dim)
    source_labels = torch.randint(0, num_classes, (32,))
    target_features = torch.randn(32, feature_dim, requires_grad=True)
    target_logits = torch.randn(32, num_classes, requires_grad=True)

    loss_fn.update_source_prototypes(source_features, source_labels)
    losses = loss_fn(target_logits, target_features)
    total_loss = losses["pseudo"] + 0.01 * losses["entropy"] + 0.1 * losses["prototype"]
    total_loss.backward()

    assert torch.isfinite(total_loss).item(), "total loss is not finite"
    assert target_logits.grad is not None, "target logits did not receive gradients"
    assert target_features.grad is not None, "target features did not receive gradients"
    assert loss_fn.prototype.initialized.any().item(), "source prototypes were not initialized"

    print("smoke test passed")
    print({name: round(value.item(), 6) for name, value in losses.items()})
    print(loss_fn.last_stats)


if __name__ == "__main__":
    main()
