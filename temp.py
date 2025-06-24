class ExoMixerGLU(nn.Module):
    def __init__(self, n_exo, gate_bias=0.0, p_drop = 0.25):
        super().__init__()
        hidden = n_exo * 2
        self.pre = nn.Linear(n_exo, hidden * 2)   # value | gate
        nn.init.xavier_uniform_(self.pre.weight)
        # set the biases: first half (value) = 0, second half (gate) = gate_bias
        with torch.no_grad():
            self.pre.bias[:hidden].zero_()
            self.pre.bias[hidden:].fill_(gate_bias)

        self.glu  = nn.GLU(dim=-1)
        self.dropout = nn.Dropout(p_drop)
        self.post = nn.Linear(hidden, 1)

    def forward(self, exo_pred):                  # [bs,T,n_exo]
        x = self.pre(exo_pred)
        x = self.glu(x)                           # gate starts near 0
        x = self.dropout(x) 
        return self.post(x).squeeze(-1)     
