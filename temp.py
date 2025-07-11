def rank_gauss(x, dim=0, eps=1e-6):
    """
    Map values to an (approximate) N(0,1) using ranks.
    x   : tensor (N, T, F) or similar
    dim : axis along which to make each slice Gaussian
    """
    N = x.size(dim)

    # Get ranks (0 … N-1) for each slice along dim
    ranks = x.argsort(dim=dim).argsort(dim=dim).float()

    # Convert to open‐interval (0,1): U = (rank + 0.5)/(N + 1)
    U = (ranks + 0.5) / (N + 1.0)

    # Map uniform U→Z via Φ⁻¹.  Φ⁻¹(u) = √2 ⋅ erfinv(2u−1)
    Z = torch.sqrt(torch.tensor(2.0, device=x.device)) \
        * torch.special.erfinv(2 * U - 1)

    # (Optional) clip extremely large values for numerical stability
    Z = torch.clamp(Z, -5, 5)

    return Z
