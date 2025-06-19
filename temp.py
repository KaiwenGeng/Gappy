 conda install cudatoolkit==11.8 -c nvidia
 pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
!pip install causal_conv1d==1.4.0
!pip install mamba_ssm==2.2.2


pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# 4 ) install your project-specific libs
pip install causal_conv1d==1.4.0 mamba_ssm==2.2.2


!pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118 \
             --extra-index-url https://download.pytorch.org/whl/cu118
!pip install causal_conv1d==1.4.0 mamba_ssm==2.2.2



# Linux example
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run   # deselect driver if you already have one
echo 'export CUDA_HOME=/usr/local/cuda-11.8' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH'     >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc        # reload shell
nvcc --version          # should now print release 11.8

# then reinstall
pip install --no-cache-dir causal_conv1d==1.4.0 mamba_ssm==2.2.2



# 1. Detect what PyTorch can actually see *inside this process*
# ------------------------------------------------------------------
visible_gpu_count = torch.cuda.device_count()        # after Ray masking
has_cuda          = visible_gpu_count > 0

# ------------------------------------------------------------------
# 2. Choose the primary device
#    • If the user asked for GPU but none are visible, fall back.
#    • Inside Ray, "cuda" (== first visible) is the safest choice.
# ------------------------------------------------------------------
if args.use_gpu and has_cuda:
    # Ray may have remapped GPU indices, so ignore args.gpu
    args.device = torch.device("cuda")               # always the first
    print(f"Using GPU 0 in this worker "
          f"(physical id(s): {os.getenv('CUDA_VISIBLE_DEVICES')})")
else:
    # Apple metal first, then CPU
    if torch.backends.mps.is_available():
        args.device = torch.device("mps")
        print("Using Apple MPS")
    else:
        args.device = torch.device("cpu")
        print("Using CPU")

# ------------------------------------------------------------------
# 3. Multi-GPU (DataParallel / DDP) – build a local list
#    Ray exposes only the GPUs it gave us, so enumerate again.
# ------------------------------------------------------------------
if args.use_gpu and args.use_multi_gpu and has_cuda:
    # Build a list of *local* ids: 0,1,2,… up to visible_gpu_count-1
    args.device_ids = list(range(visible_gpu_count))
    # First one becomes the primary
    args.gpu = args.device_ids[0]
    print(f"Multi-GPU mode → local ids {args.device_ids}")
else:
    args.device_ids = [0] if has_cuda else []



def _acquire_device(self):
    """
    Decide where this process should run.
    Works both on a plain machine and inside a Ray worker.
    """
    # --------------------------- CUDA ---------------------------
    if self.args.use_gpu and self.args.gpu_type == "cuda" and torch.cuda.is_available():
        visible_gpu_count = torch.cuda.device_count()    # after Ray masking
        # Pick the first visible GPU; Ray guarantees that is id 0.
        device = torch.device("cuda")
        msg = f"Use GPU 0 (physical id(s): {os.getenv('CUDA_VISIBLE_DEVICES')})"

        # If the user asked for multi-GPU, expose all local ids.
        if self.args.use_multi_gpu and visible_gpu_count > 1:
            self.args.device_ids = list(range(visible_gpu_count))
            msg += f" | multi-GPU mode → local ids {self.args.device_ids}"
        else:
            self.args.device_ids = [0]

        print(msg)

    # --------------------------- Apple M-series ---------------------------
    elif self.args.use_gpu and self.args.gpu_type == "mps" and hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Use GPU: mps")
        else:
            device = torch.device("cpu")
            print("MPS unavailable – falling back to CPU")

    # --------------------------- CPU fallback ---------------------------
    else:
        device = torch.device("cpu")
        print("Use CPU")

    return device



import torch, numpy as np, shap

def shap_feature_importance(
    model,
    train_loader,
    test_loader,
    *,
    device: torch.device | str = "cpu",
    n_background: int = 128,
    n_explain:    int = 64,
    feature_names: list[str] | None = None,
):
    """
    Global per-feature SHAP importance for finance sequence models.

    Each loader must yield dicts with key "features" shaped (N_d, 121, 6).
    """

    device = torch.device(device)
    model.eval().to(device)

    # ---------- 1. grab background sequences --------------------------------
    need = n_background
    bg_list = []
    for batch in train_loader:                   # shuffled, good random mix
        x = batch["features"].to(device)
        take = min(need, x.shape[0])
        bg_list.append(x[:take])
        need -= take
        if need == 0:
            break
    background = torch.cat(bg_list)              # (n_background, 121, 6)

    # ---------- 2. grab sequences to explain --------------------------------
    need = n_explain
    ex_list = []
    for batch in test_loader:                    # not shuffled – fine
        x = batch["features"].to(device)
        take = min(need, x.shape[0])
        ex_list.append(x[:take])
        need -= take
        if need == 0:
            break
    explain_x = torch.cat(ex_list)               # (n_explain, 121, 6)

    # ---------- 3. choose explainer -----------------------------------------
    try:
        explainer = shap.DeepExplainer(model, background)
    except Exception:
        explainer = shap.GradientExplainer(model, background)

    shap_vals = explainer.shap_values(explain_x)  # returns np.array (n,121,6)
    if isinstance(shap_vals, list):               # just in case
        shap_vals = shap_vals[0]

    shap_vals = torch.as_tensor(shap_vals, device=device)

    # ---------- 4. aggregate over samples & time -----------------------------
    importance = shap_vals.abs().mean(dim=(0, 1)).cpu().numpy()   # (6,)

    if feature_names is not None:
        order = importance.argsort()[::-1]
        print("Top-10 drivers:")
        for i in order:
            print(f"{feature_names[i]:<20s}  {importance[i]:.4g}")

    return importance
