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
