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
