import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedSmoothL1Loss(nn.Module):
    """
    Smooth‑L1 (Huber) loss with optional per‑sample weights.

    Forward signature
    -----------------
        loss = crit(input, target, weight=None)

    Arguments
    ----------
    input   : (N, 1) or (N,)
    target  : (N,)
    weight  : (N,) or None          – non‑negative weights
    beta    : float  – threshold between L1 and L2 regions
    reduction : "mean" | "sum" | "none"
    """
    def __init__(self, beta: float = 1.0, reduction: str = "mean", eps: float = 1e-8):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.beta = float(beta)
        self.reduction = reduction
        self.eps = eps        # avoids /0 when weight.sum()==0

    def forward(
        self,
        input:  torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None
    ) -> torch.Tensor:

        # No weighting requested → behave like nn.SmoothL1Loss
        if weight is None:
            return F.smooth_l1_loss(
                input, target, beta=self.beta, reduction=self.reduction
            )

        # Flatten to (N,)
        input  = input.view(-1)
        target = target.view(-1)
        weight = weight.view(-1).to(input.dtype)

        # Per‑element Smooth‑L1 (Huber) error
        diff = torch.abs(input - target)
        per_elem = torch.where(
            diff < self.beta,
            0.5 * diff**2 / self.beta,
            diff - 0.5 * self.beta,
        )

        # Apply weights (still 1‑to‑1 with samples here)
        loss_per_sample = weight * per_elem

        if self.reduction == "none":
            return loss_per_sample          # (N,)

        if self.reduction == "sum":
            return loss_per_sample.sum()

        # self.reduction == "mean"
        return loss_per_sample.sum() / (weight.sum() + self.eps)


torch.manual_seed(123)
N  = 10
ŷ  = torch.randn(N, 1)
y   = torch.randn(N)

# 1) All‑equal weights → identical to nn.SmoothL1Loss
w_equal = torch.full((N,), 2.5)
crit    = WeightedSmoothL1Loss(beta=1.0)           # mean reduction
ref     = nn.SmoothL1Loss(beta=1.0)(ŷ.squeeze(), y)
got     = crit(ŷ, y, w_equal)
print(ref, got)
assert torch.allclose(ref, got), "Should match nn.SmoothL1Loss when weights equal"

# 2) Different weights → matches hand calculation
w = torch.linspace(0.5, 3.0, N)
hand = (w * F.smooth_l1_loss(ŷ.squeeze(), y, beta=1.0, reduction='none')).sum() / w.sum()
got  = crit(ŷ, y, w)
print(hand, got)
assert torch.allclose(hand, got), "Weighted mean incorrect"



def weighted_pearson_loss(pred, target, weight):
    pred   = pred.flatten().to(torch.float32)
    target = target.flatten().to(pred.dtype)
    w      = weight.flatten().to(pred)

    w_sum = w.sum()
    if w_sum == 0:
        return pred.new_tensor(0.)  # or raise

    # weighted means
    μ_p = (w * pred).sum()   / w_sum
    μ_t = (w * target).sum() / w_sum

    # centred
    p_c = pred   - μ_p
    t_c = target - μ_t

    # weighted cov and var
    cov = (w * p_c * t_c).sum()
    var_p = (w * p_c**2).sum() + 1e-12
    var_t = (w * t_c**2).sum() + 1e-12

    ic = cov / torch.sqrt(var_p * var_t)
    return -ic 





import copy, time, torch, optuna, numpy as np
from torch import nn, optim
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import KFold            # or TimeSeriesSplit if chronological order matters
from exp_long_term_forecast import Exp_Long_Term_Forecast   # your wrapper class

# ---------------------------------------------------------------------
def tune_timexer(
        base_args,                         # argparse.Namespace or SimpleNamespace with *default* settings
        train_loader,                      # your existing DataLoader (train split only)
        n_trials       = 50,               # Optuna trial budget
        n_splits       = 5,                # K‑folds
        epochs_per_cv  = 5,                # quick inner‑loop training for each trial
        device         = ("cuda" if torch.cuda.is_available() else "cpu"),
        direction      = "minimize",       # minimize val MSE
        seed           = 42
    ):
    """
    Hyper‑parameter search for TimeXer using Optuna + K‑fold CV.
    Returns (best_params: dict, study: optuna.study.Study).

    The function touches *only* the data inside `train_loader`.
    Test data remain unseen until the very end of your pipeline.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ----------------------------------------------------------
    # Helper: build & train one model on (train_idx, val_idx)
    # ----------------------------------------------------------
    def _run_fold(args, fold_id, train_idx, val_idx):
        # Build fresh dataloaders from the original dataset
        full_dataset = train_loader.dataset
        tr_loader = DataLoader(Subset(full_dataset, train_idx),
                               batch_size=args.batch_size,  # keep your original setting
                               shuffle=True, num_workers=train_loader.num_workers)
        val_loader = DataLoader(Subset(full_dataset, val_idx),
                                batch_size=args.batch_size,
                                shuffle=False, num_workers=train_loader.num_workers)

        # ---------------------------
        exp = Exp_Long_Term_Forecast(args)             # wrapper builds model/optim/loss
        model = exp.model.to(device)
        optimizer = optim.AdamW(model.parameters(),
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay)
        criterion = nn.MSELoss()

        # quick training loop
        model.train()
        for ep in range(epochs_per_cv):
            for xb, yb in tr_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(criterion(model(xb), yb).item())
        return float(np.mean(val_losses))

    # ----------------------------------------------------------
    # Optuna objective
    # ----------------------------------------------------------
    def objective(trial: optuna.trial.Trial):

        # ---- sample hyper‑parameters ----
        sampled = {
            # architecture
            "d_model"   : trial.suggest_categorical("d_model",  [32, 64, 128, 256, 512]),
            "d_ffn"     : trial.suggest_categorical("d_ffn",    [64, 128, 256, 512, 1024]),
            "n_heads"   : trial.suggest_categorical("n_heads",  [2, 4, 8]),
            "e_layers"  : trial.suggest_int(         "e_layers", 1, 6),
            "dropout"   : trial.suggest_float(       "dropout",  0.0, 0.5),
            "patch_len" : trial.suggest_categorical("patch_len", [8, 16, 24, 32, 48]),

            # optimiser
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True),
            "weight_decay" : trial.suggest_float("weight_decay",  1e-6, 1e-2, log=True),
        }

        # ---- populate a fresh args for this trial ----
        args = copy.deepcopy(base_args)
        for k, v in sampled.items():
            setattr(args, k, v)

        # (Optuna pruning at trial level)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_losses = []
        for fold_id, (tr_idx, val_idx) in enumerate(kf.split(range(len(train_loader.dataset)))):
            fold_loss = _run_fold(args, fold_id, tr_idx, val_idx)
            fold_losses.append(fold_loss)

            # report intermediate result so Optuna can prune early
            trial.report(fold_loss, step=fold_id)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return float(np.mean(fold_losses))

    # ----------------------------------------------------------
    # Launch the study
    # ----------------------------------------------------------
    sampler  = optuna.samplers.TPESampler(seed=seed)
    pruner   = optuna.pruners.MedianPruner(n_warmup_steps=1)
    study    = optuna.create_study(direction=direction,
                                   sampler=sampler,
                                   pruner=pruner)

    start = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"Tuning finished in {(time.time()-start)/60:.1f} min")
    print("Best value  :", study.best_value)
    print("Best params :", study.best_params)

    return study.best_params, study
# ---------------------------------------------------------------------

#######################################################################
# Example usage in your main script (after loading data)
#######################################################################
# base_args = get_args()              # however you currently build your Namespace
# train_loader, test_loader = ...
# best_params, study = tune_timexer(base_args, train_loader, n_trials=80, n_splits=5)

# # Now train a *final* model on the entire training set with the best params,
# # then evaluate on test_loader exactly once.
# for k, v in best_params.items():
#     setattr(base_args, k, v)
# final_exp = Exp_Long_Term_Forecast(base_args)
# final_exp.train(train_loader)       # your usual training routine
# final_metric = final_exp.evaluate(test_loader)
# print("Test MSE:", final_metric)




torch.manual_seed(42)
N   = 8
ŷ   = torch.randn(N, 1)
y    = torch.randn(N)
crit = WeightedMSELoss()           # default reduction='mean'

# 1) All weights equal  → identical to nn.MSELoss
w = torch.full((N,), 7.3)
ref = nn.MSELoss()(ŷ.squeeze(), y)
got = crit(ŷ, y, w)
print(ref, got)
assert torch.allclose(ref, got), "Should match nn.MSELoss when weights equal"

# 2) Different weights  → matches hand calculation
w = torch.tensor([1., 0.5, 2., 3., 1., 4., 1.2, 0.7])
hand = (w * (ŷ.squeeze() - y) ** 2).sum() / w.sum()
got  = crit(ŷ, y, w)
print(hand, got)
assert torch.allclose(hand, got), "Weighted mean incorrect"





class ExoMixerGLU(nn.Module):
    def __init__(self, n_exo=6, hidden=16):
        super().__init__()
        self.pre  = nn.Linear(n_exo, hidden * 2)  # value | gate
        self.glu  = nn.GLU(dim=-1)
        self.post = nn.Linear(hidden, 1)         # back to scalar

    def forward(self, exo_pred):                 # [bs, T, 6]
        x = self.pre(exo_pred)                   # [bs, T, 32]
        x = self.glu(x)                          # [bs, T, 16]
        return self.post(x).squeeze(-1)          # [bs, T]

core = dec_out[:, :, -1]            # CI forecast of the target
exo  = dec_out[:, :, :-1]           # 6 exogenous series
pred = core + self.exo_mixer(exo)   # blended result
return pred







class ExoMixerGLU(nn.Module):
    def __init__(self, n_exo=6, hidden=16, gate_bias=-4.0):
        super().__init__()
        self.pre = nn.Linear(n_exo, hidden * 2)   # value | gate
        nn.init.xavier_uniform_(self.pre.weight)
        # set the biases: first half (value) = 0, second half (gate) = gate_bias
        with torch.no_grad():
            self.pre.bias[:hidden].zero_()
            self.pre.bias[hidden:].fill_(gate_bias)

        self.glu  = nn.GLU(dim=-1)
        self.post = nn.Linear(hidden, 1)

    def forward(self, exo_pred):                  # [bs,T,n_exo]
        x = self.pre(exo_pred)
        x = self.glu(x)                           # gate starts near 0
        return self.post(x).squeeze(-1)


mu  = core.mean(dim=1, keepdim=True)                        # [B, 1]
std = core.std (dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)

# ── 2.  Broadcast and normalise the exogenous slice  ───────────────────
exo = (exo - mu.unsqueeze(-1)) / std.unsqueeze(-1)          # [B, T, k]



from sklearn.model_selection import TimeSeriesSplit

X = train_year_df[feature_cols]
y = train_year_df[target_col]

# e.g. 5 folds over time
tscv = TimeSeriesSplit(n_splits=5)


import xgboost as xgb
import pandas as pd

# prepare a DataFrame to hold cumulative importances
importances = pd.DataFrame(index=feature_cols)
importances['gain'] = 0.0

for train_idx, val_idx in tscv.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=10,
        verbose=False
    )

    # get raw gain importances
    booster = model.get_booster()
    fold_gain = booster.get_score(importance_type='gain')
    # map from 'f0', 'f1'… back to your feature names
    fold_gain = {
        feature_cols[int(k[1:])]: v
        for k, v in fold_gain.items()
    }
    # add to cumulative sum (missing features get 0)
    for feat, g in fold_gain.items():
        importances.at[feat, 'gain'] += g

# average across folds
importances['gain'] /= tscv.get_n_splits()
# sort descending
importances = importances.sort_values('gain', ascending=False)

top_k = 50
selected_feats = importances.head(top_k).index.tolist()

thresh = 0.01
selected_feats = importances[importances['gain'] > thresh].index.tolist()
