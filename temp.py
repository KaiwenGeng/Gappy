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


import torch
from captum.attr import IntegratedGradients
from typing import Callable, Iterable, Tuple, Union

def integrated_gradients_heatmap(
    model: torch.nn.Module,
    data_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    *,
    device: Union[str, torch.device] = "cuda",
    n_steps: int = 64,
    baseline_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """
    Compute an (seq_len × n_feat) attribution map with Integrated Gradients.

    Parameters
    ----------
    model : nn.Module
        Already-trained network. Forward(x) must return shape [batch] or [batch, 1].
    data_loader : iterable
        Yields batches X (or (X, y)) where X has shape [B, seq_len, n_feat].
    device : str | torch.device, default "cuda"
        Where the computation runs.
    n_steps : int, default 64
        Number of IG interpolation steps (higher = smoother, slower).
    baseline_fn : callable(X) → baseline, optional
        Generates a reference tensor of identical shape to X.
        If None, a zero tensor is used.

    Returns
    -------
    torch.Tensor
        Heat-map of absolute attributions, shape [seq_len, n_feat],
        averaged over all samples in `data_loader`.
    """
    model = model.to(device).eval()
    ig = IntegratedGradients(model)

    heat_accum = None        # will become [seq_len, n_feat]
    sample_count = 0

    for batch in data_loader:
        # handle loaders that return (X, y) or just X
        X = batch[0] if isinstance(batch, (list, tuple)) else batch
        X = X.to(device)

        B, seq_len, n_feat = X.shape
        if baseline_fn is None:
            baseline = torch.zeros_like(X)
        else:
            baseline = baseline_fn(X).to(device)

        # Integrated Gradients
        attr = ig.attribute(
            inputs=X,
            baselines=baseline,
            n_steps=n_steps,
            target=None,          # whatever forward() returns
        ).abs()                  # keep magnitude only

        # accumulate
        attr_sum = attr.sum(dim=0)           # [seq_len, n_feat]
        heat_accum = attr_sum if heat_accum is None else heat_accum + attr_sum
        sample_count += B

    heat_mean = heat_accum / sample_count     # average over all samples
    return heat_mean.cpu()                    # move to CPU for convenience


def plot_ig_heatmap(heat: torch.Tensor,
                    feature_names=None,
                    title="Integrated Gradients importance"):
    """Visualise a (time-lag × feature) tensor as a heat-map."""
    heat_np = heat.numpy()               # move to CPU & np

    seq_len, n_features = heat_np.shape
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]

    # y-ticks: 0 = most recent bar (easier to read for traders)
    ytick = list(range(seq_len))[::-1]   # flip so 0 is at bottom
    ylabels = [f"{-i}" for i in ytick]   # “-5” means 5 bars back

    plt.figure(figsize=(1.2 * n_features, 0.25 * seq_len + 2))
    ax = sns.heatmap(
        heat_np[::-1],                   # flip rows for recency
        cmap="rocket_r",                 # dark = unimportant, bright = important
        xticklabels=feature_names,
        yticklabels=ylabels,
        cbar_kws={"label": "|IG| (avg over samples)"}
    )
    ax.set_xlabel("Feature")
    ax.set_ylabel("Bars back (0 = today)")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()



    I'm using captum attr to do feature importance analysis. When I run it on my trained lstm, everything looks find, the most recent days give higher value while dates that are back in days gives less. For instance, i observe that at lookback 1 most features are relatively important,  while as we go back in time until end of my lookback window, like 121, is basically very low. However, when I run it my trained mamba, I noticed the exact behavior but one big difference: at the exact tail of my lookback window, ie day 121, the importance is huge, almost the same as the most recent days. This is Not even on day 120, 119, as their importance are also very low which is expected.  The importance is just high at the exact tail of my lookback... Is this saying my mamba model is wrong? If not, what's the explanation for that? My entire pipeline is the same other than the seq2seq part..