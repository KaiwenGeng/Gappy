def cross_sectional_norm(x, eps=1e-6):
    """
    x: Tensor with shape (N_stocks, T, F).
    Returns: z-scored tensor with the same shape.
    """
    mean = x.mean(dim=0, keepdim=True)              # shape (1, T, F)
    std  = x.std(dim=0, unbiased=False, keepdim=True)
    return (x - mean) / (std + eps)
