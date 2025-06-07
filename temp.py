import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedSmoothL1Loss(nn.Module):
    """
    Smooth‑L1 (Huber) loss with optional per‑sample weights.

    Forward signature
    -----------------
        loss = crit(input, target, weight=None)

    Arguments
    ----------
    input   : (N, 1) or (N,)
    target  : (N,)
    weight  : (N,) or None          – non‑negative weights
    beta    : float  – threshold between L1 and L2 regions
    reduction : "mean" | "sum" | "none"
    """
    def __init__(self, beta: float = 1.0, reduction: str = "mean", eps: float = 1e-8):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.beta = float(beta)
        self.reduction = reduction
        self.eps = eps        # avoids /0 when weight.sum()==0

    def forward(
        self,
        input:  torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None
    ) -> torch.Tensor:

        # No weighting requested → behave like nn.SmoothL1Loss
        if weight is None:
            return F.smooth_l1_loss(
                input, target, beta=self.beta, reduction=self.reduction
            )

        # Flatten to (N,)
        input  = input.view(-1)
        target = target.view(-1)
        weight = weight.view(-1).to(input.dtype)

        # Per‑element Smooth‑L1 (Huber) error
        diff = torch.abs(input - target)
        per_elem = torch.where(
            diff < self.beta,
            0.5 * diff**2 / self.beta,
            diff - 0.5 * self.beta,
        )

        # Apply weights (still 1‑to‑1 with samples here)
        loss_per_sample = weight * per_elem

        if self.reduction == "none":
            return loss_per_sample          # (N,)

        if self.reduction == "sum":
            return loss_per_sample.sum()

        # self.reduction == "mean"
        return loss_per_sample.sum() / (weight.sum() + self.eps)


torch.manual_seed(123)
N  = 10
ŷ  = torch.randn(N, 1)
y   = torch.randn(N)

# 1) All‑equal weights → identical to nn.SmoothL1Loss
w_equal = torch.full((N,), 2.5)
crit    = WeightedSmoothL1Loss(beta=1.0)           # mean reduction
ref     = nn.SmoothL1Loss(beta=1.0)(ŷ.squeeze(), y)
got     = crit(ŷ, y, w_equal)
print(ref, got)
assert torch.allclose(ref, got), "Should match nn.SmoothL1Loss when weights equal"

# 2) Different weights → matches hand calculation
w = torch.linspace(0.5, 3.0, N)
hand = (w * F.smooth_l1_loss(ŷ.squeeze(), y, beta=1.0, reduction='none')).sum() / w.sum()
got  = crit(ŷ, y, w)
print(hand, got)
assert torch.allclose(hand, got), "Weighted mean incorrect"



def weighted_pearson_loss(pred, target, weight):
    pred   = pred.flatten().to(torch.float32)
    target = target.flatten().to(pred.dtype)
    w      = weight.flatten().to(pred)

    w_sum = w.sum()
    if w_sum == 0:
        return pred.new_tensor(0.)  # or raise

    # weighted means
    μ_p = (w * pred).sum()   / w_sum
    μ_t = (w * target).sum() / w_sum

    # centred
    p_c = pred   - μ_p
    t_c = target - μ_t

    # weighted cov and var
    cov = (w * p_c * t_c).sum()
    var_p = (w * p_c**2).sum() + 1e-12
    var_t = (w * t_c**2).sum() + 1e-12

    ic = cov / torch.sqrt(var_p * var_t)
    return -ic 