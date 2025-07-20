class VariableSelection(nn.Module):
    def __init__(self, num_feat, d_model):
        super().__init__()
        self.context = nn.GRU(num_feat, d_model, batch_first=True)
        self.selector = nn.Sequential(
            nn.Linear(d_model, num_feat),
            nn.Softmax(dim=-1)
        )
        self.proj = nn.Linear(num_feat, d_model)

    def forward(self, x):          # x: (B, T, 6)
        _, h = self.context(x)     # h: (1, B, d)
        alpha = self.selector(h.squeeze(0))        # (B, 6)
        x_sel = (x * alpha.unsqueeze(1)).sum(-1)   # (B, T)
        emb = self.proj(x_sel)                     # (B, T, d)
        return emb, alpha   
