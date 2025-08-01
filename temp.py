class VolFiLMLinear(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(2 * d_model, 2 * d_model)   # no hidden layer
        self.eps  = 1e-6

    def forward(self, x, mu, var):
        log_sigma = 0.5 * torch.log(var + self.eps)
        h = torch.cat([mu, log_sigma], dim=-1)            # [B, 2D]
        gamma, beta = self.proj(h).chunk(2, dim=-1)       # [B, D] each
        return gamma.unsqueeze(1) * x + beta.unsqueeze(1)
