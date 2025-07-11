import torch
import torchsort  # pip install torchsort

def rank_gauss_soft(x, dim=0, eps=1e-6):
    # Differentiable ranks
    ranks = torchsort.soft_rank(x, dim=dim, regularization_strength=tau)

    N = x.size(dim)
    U = (ranks + 0.5) / (N + 1.0)
    Z = torch.sqrt(torch.tensor(2.0, device=x.device)) \
        * torch.special.erfinv(2 * U - 1)
    return torch.clamp(Z, -3, 3)
