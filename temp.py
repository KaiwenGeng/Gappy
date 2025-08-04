class LatentNorm(nn.Module):
    """
    Learn μ, σ from a *light* network that *does not share weights*
    with the main encoder.  Statistics flow into the normaliser by
    stop-gradient (no back-prop through them).
    """
    def __init__(self, d_model, d_hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),               # pool over time
            nn.Flatten(),                          # [B, d_model]
            nn.Linear(d_model, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, 2 * d_model)       # μ and log σ
        )

    def forward(self, v_raw):
        with torch.no_grad():                      # <--- key line
            stats = self.mlp(v_raw.transpose(1,2)) # [B, 2·d_model]
        mu, log_sigma = stats.chunk(2, dim=-1)
        sigma = torch.exp(log_sigma).clamp(1e-3, 1e1)
        v = (v_raw - mu.unsqueeze(1)) / sigma.unsqueeze(1)
        return v, mu, sigma
