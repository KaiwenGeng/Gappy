import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedMSELoss(nn.Module):
    """
    Mean‑squared‑error with optional per‑sample weights.

    Forward signature:
        loss = crit(input, target, weight=None)

    • input  : (N, 1) or (N,)
    • target : (N,)      – ground‑truth values
    • weight : (N,) or None
    """
    def __init__(self, reduction: str = "mean", eps: float = 1e-8):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.reduction = reduction
        self.eps = eps                # tiny constant to avoid /0

    def forward(self,
                input:  torch.Tensor,
                target: torch.Tensor,
                weight: torch.Tensor | None = None) -> torch.Tensor:

        # If no weight is supplied, behave exactly like nn.MSELoss
        if weight is None:
            return F.mse_loss(input, target, reduction=self.reduction)

        # Flatten everything → (N,)
        input  = input.view(-1)
        target = target.view(-1)
        weight = weight.view(-1).to(input.dtype)

        loss_per_sample = weight * (input - target) ** 2

        if self.reduction == "none":
            return loss_per_sample

        if self.reduction == "sum":
            return loss_per_sample.sum()

        # self.reduction == "mean"
        return loss_per_sample.sum() / (weight.sum() + self.eps)


torch.manual_seed(42)
N   = 8
ŷ   = torch.randn(N, 1)
y    = torch.randn(N)
crit = WeightedMSELoss()           # default reduction='mean'

# 1) All weights equal  → identical to nn.MSELoss
w = torch.full((N,), 7.3)
ref = nn.MSELoss()(ŷ.squeeze(), y)
got = crit(ŷ, y, w)
print(ref, got)
assert torch.allclose(ref, got), "Should match nn.MSELoss when weights equal"

# 2) Different weights  → matches hand calculation
w = torch.tensor([1., 0.5, 2., 3., 1., 4., 1.2, 0.7])
hand = (w * (ŷ.squeeze() - y) ** 2).sum() / w.sum()
got  = crit(ŷ, y, w)
print(hand, got)
assert torch.allclose(hand, got), "Weighted mean incorrect"

# 3) Other reductions
