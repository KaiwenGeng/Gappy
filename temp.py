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