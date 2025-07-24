class VariableSelection(nn.Module):
    def __init__(self, in_dim: int, hidden_mult: int = 2):
        super().__init__()
        hidden = in_dim * hidden_mult

        # feature branch – sees only x
        self.x_net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, in_dim)
        )

        # mask branch – sees only the (0/1) mask
        # A much smaller network is enough because the signal is binary
        self.m_net = nn.Sequential(
            nn.Linear(in_dim, in_dim, bias=False),  # learnable per‑feature bias
        )

    def forward(self, x, mask):
        """
        x, mask: shape (B, T, D); mask == 1 for observed values
        """
        # logits from each branch
        logits_x = self.x_net(x)              # (B, T, D)
        logits_m = self.m_net(mask.float())   # (B, T, D)

        logits = logits_x + logits_m          # sum, no concat
        logits = logits.masked_fill(mask == 0, torch.finfo(x.dtype).min)

        # masked soft‑max, safe when every feature is missing
        w = torch.softmax(logits, dim=-1) * mask
        w = w / (w.sum(-1, keepdim=True) + 1e-8)

        return x * w   




class MoEHead(nn.Module):
    """
    Mixture‑of‑Experts projection head.

    Args
    ----
    d_model      : hidden size of the backbone
    c_out        : output dimension
    num_experts  : total experts (E)
    k_active     : experts selected per token (k ≤ E)
    dropout      : dropout prob applied inside each expert
    """

    def __init__(self,
                 d_model: int,
                 c_out:    int,
                 num_experts: int = 3,
                 k_active:    int = 1,
                 dropout:     float = 0.0):
        super().__init__()
        assert 1 <= k_active <= num_experts, "k_active must be in [1, num_experts]"

        self.num_experts = num_experts
        self.k_active    = k_active
        self.c_out       = c_out

        h = max(1, round(d_model / 4))          # bottleneck width

        # ---------- Experts --------------------------------------------------
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, h, bias=True),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(h, c_out, bias=False)
            )
            for _ in range(num_experts)
        ])

        # ---------- Router gate ---------------------------------------------
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    # -----------------------------------------------------------------------
    @torch.no_grad()
    def _topk_mask(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Return a *hard* (0/1) mask with the top‑k experts per token.
        Shape: [..., num_experts]  with exactly k ones along the last dim.
        """
        topk_idx = logits.topk(self.k_active, dim=-1).indices   # [..., k]
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(-1, topk_idx, True)
        return mask

    def _compute_gates(self, x: torch.Tensor) -> torch.Tensor:
        """
        Straight‑through top‑k gating.
        Returns non‑negative gates that sum to 1 along the expert dim.
        """
        logits = self.gate(x)                                   # [..., E]
        if self.k_active == self.num_experts:                   # full softmax
            return logits.softmax(dim=-1)

        mask = self._topk_mask(logits)                          # hard one‑hot
        soft = logits.softmax(dim=-1)
        gates = mask.float() + (soft - soft.detach())           # straight‑through
        gates = gates / gates.sum(dim=-1, keepdim=True)         # normalise
        return gates

    # -----------------------------------------------------------------------
    def forward(self, x):
        """
        x  : [B, L, D]
        out: [B, L, C]
        """
        B, L, D = x.shape
        gates   = self._compute_gates(x)         # [B, L, E]

        x_flat  = x.reshape(-1, D)               # [N, D]  (N = B·L)
        g_flat  = gates.reshape(-1, self.num_experts)  # [N, E]

        out_flat = torch.zeros(
            x_flat.size(0), self.c_out, dtype=x.dtype, device=x.device
        )

        # ----- run only the selected experts --------------------------------
        for e_id, expert in enumerate(self.experts):
            sel = g_flat[:, e_id] > 0            # bool mask for expert e_id
            if sel.any():
                y = expert(x_flat[sel])                          # [n_sel, C]
                y = y * g_flat[sel, e_id].unsqueeze(-1)          # gate scale
                out_flat[sel] += y                               # scatter‑add

        out = out_flat.view(B, L, self.c_out)    # [B, L, C]

        return out
