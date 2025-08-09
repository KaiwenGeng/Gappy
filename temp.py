optimizer = optimizer_cls([
        # everything except dean → gets default lr (set at the end)
        {'params': [p for n, p in model.named_parameters() if not n.startswith('dean.')]},

        # dean sublayers → custom LRs
        {'params': model.dean.mean_layer.parameters(),    'lr': lr * model.dean.mean_lr},
        {'params': model.dean.scaling_layer.parameters(), 'lr': lr * model.dean.scale_lr},
        {'params': model.dean.gating_layer.parameters(),  'lr': lr * model.dean.gate_lr},
    ], lr=lr)  # default LR for groups without their own
    return optimizer
