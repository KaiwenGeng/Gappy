mlp = nn.Sequential(
    nn.Linear(config.dmodel, hidden_dim),  # input → hidden
    nn.GELU(),                             # non-linearity
    nn.Dropout(drop_p),                    # dropout
    nn.Linear(hidden_dim, config.c_out)    # hidden → output
)
