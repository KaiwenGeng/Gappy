class VarSelection(nn.Module):
    """
    Variable‑selection layer à la Temporal Fusion Transformer.
    * Each raw feature x[..., i] is first turned into an embedding e_i.
    * A context network produces per‑timestep gating weights α_t.
    * Weighted sum of transformed features feeds downstream model.
    Returns both the fused embedding and α for interpretability.
    """

    def __init__(self, num_feat: int, d_model: int, d_emb: int = 16):
        super().__init__()
        # 1) Feature‑wise nonlinear projections
        self.feat_proj = nn.ModuleList(
            [nn.Linear(1, d_emb) for _ in range(num_feat)]
        )

        # 2) Context network (can be GRU/LSTM/CNN/...); here a 1‑layer GRU
        self.context = nn.GRU(
            input_size=num_feat, hidden_size=d_model, batch_first=True
        )

        # 3) Gating head → α_t ∈ ℝ^{B,T,num_feat}
        self.selector = nn.Linear(d_model, num_feat)

        # 4) Final projection from fused embedding to model dimension
        self.out_proj = nn.Linear(d_emb, d_model)

    def forward(self, x):             # x: (B,T,num_feat)
        B, T, F = x.shape

        # a) Per‑feature embeddings  (B,T,F,d_emb)  → stack to (B,T,F,d_emb)
        e_list = []
        for i, proj in enumerate(self.feat_proj):
            e_list.append(
                proj(x[..., i : i + 1])
            )  # keep last dim=1 for linear
        e = torch.stack(e_list, dim=2)  # (B,T,F,d_emb)

        # b) Context hidden state per timestep   h_t : (B,T,d_model)
        h_seq, _ = self.context(x)      # feed raw features, not embeddings

        # c) Gating weights
        alpha = F.softmax(self.selector(h_seq), dim=-1)  # (B,T,F)

        # d) Weighted fusion across features
        z = (alpha.unsqueeze(-1) * e).sum(dim=2)         # (B,T,d_emb)

        # e) Project to model dimension
        emb = self.out_proj(z)                           # (B,T,d_model)
        return emb, alpha
