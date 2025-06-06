import torch
from torch import nn

def weighted_mse_loss(output: torch.Tensor,
                      target: torch.Tensor,
                      weight: torch.Tensor,
                      reduction: str = "mean") -> torch.Tensor:
    """
    Weighted mean–squared‑error loss.
    
    Args
    ----
    output : shape (N, 1) or (N,)
    target : shape (N,)          —  true values
    weight : shape (N,)          —  non‑negative per‑sample weights
    reduction : "mean" | "sum" | "none"
    
    Returns
    -------
    Tensor — scalar unless reduction="none".
    """
    # flatten everything to (N,)
    output = output.view(-1)
    target = target.view(-1)
    weight = weight.view(-1).to(output.dtype)

    loss_per_sample = weight * (output - target) ** 2

    if reduction == "none":
        return loss_per_sample
    if reduction == "sum":
        return loss_per_sample.sum()
    if reduction == "mean":
        return loss_per_sample.sum() / weight.sum()
    raise ValueError("reduction must be 'mean', 'sum', or 'none'")


torch.manual_seed(0)
N = 6
ŷ = torch.randn(N, 1)
y  = torch.randn(N)
w  = torch.full((N,), 3.14)          # every weight is the same constant

mse_ref   = nn.MSELoss()(ŷ.squeeze(), y)
mse_weighted = weighted_mse_loss(ŷ, y, w)

print("Reference MSELoss :", mse_ref.item())
print("Weighted MSELoss :", mse_weighted.item())
assert torch.allclose(mse_ref, mse_weighted)