start_lines = [
    "bash -lc \"\
pip install --upgrade pip setuptools wheel setuptools-scm && \
pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118 \
            --index-url https://download.pytorch.org/whl/cu118 && \
pip install causal_conv1d==1.4.0 --use-pep517 --no-binary :all: && \
pip install mamba_ssm==2.2.2 --use-pep517 --no-build-isolation\
\""
]