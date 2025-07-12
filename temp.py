class FiLMHead(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.gamma = nn.Linear(d_in, d_in)
        self.beta  = nn.Linear(d_in, d_in)
        self.pred  = nn.Linear(d_in, 1)

    def forward(self, h):
        h = self.gamma(h) * h + self.beta(h)
        return self.pred(h)
