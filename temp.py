[kg15317@prdobhfml01 ~]$ pip install --trusted-host github.com https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.0cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
Defaulting to user installation because normal site-packages is not writeable
Collecting mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse
  Downloading https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.0cxx11abiFALSE-cp311-cp311-linux_x86_64.whl (343.4 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 343.4/343.4 MB 323.5 MB/s eta 0:00:00
Requirement already satisfied: torch in /home/kg15317/.local/lib/python3.11/site-packages (from mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (2.3.1+cu118)
Requirement already satisfied: packaging in /mnt/netapp_hfalgo/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages (from mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (23.2)
Requirement already satisfied: ninja in /home/kg15317/.local/lib/python3.11/site-packages (from mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (1.11.1.4)
Requirement already satisfied: einops in /home/kg15317/.local/lib/python3.11/site-packages (from mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (0.8.0)
Requirement already satisfied: triton in /home/kg15317/.local/lib/python3.11/site-packages (from mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (2.3.1)
Requirement already satisfied: transformers in /mnt/netapp_hfalgo/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages (from mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (4.46.2)
Requirement already satisfied: filelock in /mnt/netapp_hfalgo/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages (from torch->mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (3.18.0)
Requirement already satisfied: typing-extensions>=4.8.0 in /mnt/netapp_hfalgo/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages (from torch->mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (4.12.2)
Requirement already satisfied: sympy in /mnt/netapp_hfalgo/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages (from torch->mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (1.12)
Requirement already satisfied: networkx in /mnt/netapp_hfalgo/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages (from torch->mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (3.5)
Requirement already satisfied: jinja2 in /mnt/netapp_hfalgo/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages (from torch->mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (3.1.6)
Requirement already satisfied: fsspec in /mnt/netapp_hfalgo/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages (from torch->mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (2025.3.0)
Requirement already satisfied: nvidia-cuda-nvrtc-cu11==11.8.89 in /home/kg15317/.local/lib/python3.11/site-packages (from torch->mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (11.8.89)
Requirement already satisfied: nvidia-cuda-runtime-cu11==11.8.89 in /home/kg15317/.local/lib/python3.11/site-packages (from torch->mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (11.8.89)
Requirement already satisfied: nvidia-cuda-cupti-cu11==11.8.87 in /home/kg15317/.local/lib/python3.11/site-packages (from torch->mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (11.8.87)
Requirement already satisfied: nvidia-cudnn-cu11==8.7.0.84 in /home/kg15317/.local/lib/python3.11/site-packages (from torch->mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (8.7.0.84)
Requirement already satisfied: nvidia-cublas-cu11==11.11.3.6 in /home/kg15317/.local/lib/python3.11/site-packages (from torch->mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (11.11.3.6)
Requirement already satisfied: nvidia-cufft-cu11==10.9.0.58 in /mnt/netapp_hfalgo/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages (from torch->mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (10.9.0.58)
Requirement already satisfied: nvidia-curand-cu11==10.3.0.86 in /home/kg15317/.local/lib/python3.11/site-packages (from torch->mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (10.3.0.86)
Requirement already satisfied: nvidia-cusolver-cu11==11.4.1.48 in /home/kg15317/.local/lib/python3.11/site-packages (from torch->mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (11.4.1.48)
Requirement already satisfied: nvidia-cusparse-cu11==11.7.5.86 in /home/kg15317/.local/lib/python3.11/site-packages (from torch->mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (11.7.5.86)
Requirement already satisfied: nvidia-nccl-cu11==2.20.5 in /home/kg15317/.local/lib/python3.11/site-packages (from torch->mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (2.20.5)
Requirement already satisfied: nvidia-nvtx-cu11==11.8.86 in /home/kg15317/.local/lib/python3.11/site-packages (from torch->mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (11.8.86)
Requirement already satisfied: MarkupSafe>=2.0 in /mnt/netapp_hfalgo/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages (from jinja2->torch->mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (3.0.2)
Requirement already satisfied: mpmath>=0.19 in /mnt/netapp_hfalgo/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages (from sympy->torch->mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (1.3.0)
Requirement already satisfied: huggingface-hub<1.0,>=0.23.2 in /mnt/netapp_hfalgo/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages (from transformers->mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (0.33.0)
Requirement already satisfied: numpy>=1.17 in /home/kg15317/.local/lib/python3.11/site-packages (from transformers->mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (1.23.5)
Requirement already satisfied: pyyaml>=5.1 in /mnt/netapp_hfalgo/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages (from transformers->mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (6.0.2)
Requirement already satisfied: regex!=2019.12.17 in /mnt/netapp_hfalgo/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages (from transformers->mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (2024.11.6)
Requirement already satisfied: requests in /mnt/netapp_hfalgo/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages (from transformers->mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (2.32.4)
Requirement already satisfied: safetensors>=0.4.1 in /mnt/netapp_hfalgo/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages (from transformers->mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (0.4.5)
Requirement already satisfied: tokenizers<0.21,>=0.20 in /mnt/netapp_hfalgo/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages (from transformers->mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (0.20.3)
Requirement already satisfied: tqdm>=4.27 in /home/kg15317/.local/lib/python3.11/site-packages (from transformers->mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (4.64.1)
Requirement already satisfied: hf-xet<2.0.0,>=1.1.2 in /mnt/netapp_hfalgo/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages (from huggingface-hub<1.0,>=0.23.2->transformers->mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (1.1.4)
Requirement already satisfied: charset_normalizer<4,>=2 in /mnt/netapp_hfalgo/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages (from requests->transformers->mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (3.4.2)
Requirement already satisfied: idna<4,>=2.5 in /mnt/netapp_hfalgo/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages (from requests->transformers->mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (3.7)
Requirement already satisfied: urllib3<3,>=1.21.1 in /mnt/netapp_hfalgo/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages (from requests->transformers->mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (1.26.20)
Requirement already satisfied: certifi>=2017.4.17 in /mnt/netapp_hfalgo/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages (from requests->transformers->mamba-ssm==2.2.2+cu118torch2.0cxx11abifalse) (2025.6.15)
Installing collected packages: mamba-ssm
Successfully installed mamba-ssm-2.2.2



pip install --trusted-host github.com https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.0cxx11abiFALSE-cp311-cp311-linux_x86_64.whl




[kg15317@prdobhfml01 ~]$ python temp.py 
Traceback (most recent call last):
  File "/data01/home/kg15317/temp.py", line 2, in <module>
    from mamba_ssm import Mamba
  File "/home/kg15317/.local/lib/python3.11/site-packages/mamba_ssm/__init__.py", line 3, in <module>
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
  File "/home/kg15317/.local/lib/python3.11/site-packages/mamba_ssm/ops/selective_scan_interface.py", line 16, in <module>
    import selective_scan_cuda
ImportError: /home/kg15317/.local/lib/python3.11/site-packages/selective_scan_cuda.cpython-311-x86_64-linux-gnu.so: undefined symbol: _ZN2at4_ops10zeros_like4callERKNS_6TensorEN3c108optionalINS5_10ScalarTypeEEENS6_INS5_6LayoutEEENS6_INS5_6DeviceEEENS6_IbEENS6_INS5_12MemoryFormatEEE
