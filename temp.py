meta = torch.cat([
        obs_rate.squeeze(1),    # [B,N]
        mean.squeeze(1),        # [B,N]
        var.squeeze(1)          # [B,N]
    ], dim=-1)                  # [B,3N]
meta = torch.cat([meta, market_weight.unsqueeze(-1)], dim=-1)  # [B,3N+1]




class MetaGate(nn.Module):
    def __init__(self, meta_dim: int, dmodel: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(meta_dim, 4 * dmodel),
            nn.ReLU(),
            nn.Linear(4 * dmodel, dmodel)   # → [B, dmodel]
        )

    def forward(self, meta, pred_lin, pred_nonlin):
        # meta:        [B, meta_dim]
        # pred_lin:    [B, T, dmodel]
        # pred_nonlin: [B, T, dmodel]
        alpha = torch.sigmoid(self.net(meta)).unsqueeze(1)  # [B,1,dmodel]
        return alpha * pred_lin + (1 - alpha) * pred_nonlin 


meta_dim = 3 * n_features + 1 


mixed_hidden  = self.meta_gate(meta, pred_lin, pred_nonlin) 
