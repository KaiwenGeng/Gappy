[
    "bash -lc \"pip install --upgrade pip && "
    "pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 "
    "torchaudio==2.3.1+cu118 --index-url https://download.pytorch.org/whl/cu118 && "
    "pip install causal_conv1d==1.4.0 mamba_ssm==2.2.2 --no-build-isolation\""
]