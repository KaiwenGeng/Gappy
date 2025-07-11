class GaussianProject(nn.Module):
    """Rough analogue to tanh, but targets std-normal marginal *if* h~Logistic(0,1)."""
    def forward(self, h):
        u = torch.sigmoid(h)                # (0, 1)
        z = torch.sqrt(torch.tensor(2.0)) * torch.erfinv(2*u - 1)
        return z
