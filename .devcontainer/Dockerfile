# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# ============ Environment ============
ARG TZ
ENV TZ="$TZ"
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV CONDA_DIR=/opt/conda
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV DEVCONTAINER=true
ENV SHELL=/bin/bash
ENV EDITOR=nano
ENV VISUAL=nano

# ============ System packages ============
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    build-essential \
    python3-pip \
    ca-certificates \
    less \
    git \
    procps \
    sudo \
    man-db \
    unzip \
    gnupg2 \
    jq \
    nano \
    vim \
    locales \
    && locale-gen en_US.UTF-8

# ============ Node.js + gh CLI ============
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update && apt-get install -y nodejs gh

# ============ Non-root user ============
ARG USERNAME=sandboxuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ============ Working directory ============
RUN mkdir -p /workspace /home/$USERNAME/.cache \
    && chown -R $USERNAME:$USERNAME /home/$USERNAME/.cache
WORKDIR /workspace

# ============ git-delta ============
ARG GIT_DELTA_VERSION
RUN ARCH=$(dpkg --print-architecture) && \
    wget "https://github.com/dandavison/delta/releases/download/${GIT_DELTA_VERSION}/git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
    dpkg -i "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
    rm "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb"

# ============ Miniconda ============
RUN ARCH=$(uname -m) && \
    wget --quiet "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${ARCH}.sh" -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    chown -R $USERNAME:$USERNAME $CONDA_DIR

# ============ Switch to the non-root user ============
USER $USERNAME

# Conda init, as sandboxuser
RUN $CONDA_DIR/bin/conda init bash \
    && $CONDA_DIR/bin/conda config --set auto_activate_base false

# ============ Training stack (PyTorch 2.4 + gsplat 1.5.3) ============

USER root

# PyTorch 2.4 (CUDA 12.4)
RUN python3 -m pip install --no-cache-dir \
    torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

# gsplat 1.5.3, prebuilt for PyTorch 2.4 / CUDA 12.4. The wheel index hosts
# gsplat alone, so pip cannot resolve its dependencies there: the wheel goes in
# with --no-deps and jaxtyping, rich and ninja come from PyPI.
RUN python3 -m pip download --no-cache-dir --no-deps gsplat==1.5.3 -d /tmp/gsplat \
        --index-url https://docs.gsplat.studio/whl/pt24cu124 && \
    python3 -m pip install --no-cache-dir --no-deps /tmp/gsplat/gsplat-*.whl && \
    python3 -m pip install --no-cache-dir jaxtyping rich ninja && \
    rm -rf /tmp/gsplat

# Python dependencies. The headless OpenCV build: the image has no libGL.
RUN python3 -m pip install --no-cache-dir \
    numpy opencv-python-headless Pillow pyyaml tqdm

# Verify the installation
RUN python3 -c "import torch, gsplat; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}, gsplat {gsplat.__version__}')"

# Back to the non-root user
USER $USERNAME
