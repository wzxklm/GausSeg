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

# ============ Node.js + gh CLI + hf CLI ============
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update && apt-get install -y nodejs gh \
    && python3 -m pip install --no-cache-dir 'huggingface_hub[cli]'

# ============ Non-root user ============
ARG USERNAME=sandboxuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ============ Working directory ============
RUN mkdir -p /workspace /home/$USERNAME/.cache/huggingface \
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

# ============ Reconstruction stack (COLMAP 4.0.3 + OpenMVS v2.4.0) ============

USER root

# Build dependencies for COLMAP 4.0.3 and OpenMVS v2.4.0, plus the OpenCV runtime.
# Note: apt's libgoogle-glog-dev / libceres-dev are too old, so both are built
# from source below and deliberately not installed here.
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists \
    apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    cmake \
    ninja-build \
    gcc-10 \
    g++-10 \
    # --- shared by COLMAP and OpenMVS ---
    libeigen3-dev \
    libglew-dev \
    # --- COLMAP 4.0.3 ---
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libsuitesparse-dev \
    libmetis-dev \
    libgflags-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    libflann-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libopenimageio-dev \
    openimageio-tools \
    # --- OpenMVS v2.4.0 ---
    libboost-iostreams-dev \
    libboost-serialization-dev \
    libopencv-dev \
    libglfw3-dev \
    libcgal-dev \
    # --- OpenCV Python runtime ---
    libgl1-mesa-glx \
    libglib2.0-0 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Build glog 0.7.1: apt's 0.4.0 has a CHECK_OP_LOG macro that ignores custom
# handlers, so COLMAP's THROW_CHECK cannot raise and pycolmap aborts on init.
RUN git clone --depth 1 --branch v0.7.1 https://github.com/google/glog.git /tmp/build/glog && \
    cd /tmp/build/glog && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF && \
    make -j$(nproc) && make install && \
    cd / && rm -rf /tmp/build/glog

# cuDSS (NVIDIA Direct Sparse Solver), required by Ceres for GPU sparse BA.
# The nvidia/cuda:12.4.0 base image already has the NVIDIA CUDA apt repo.
# The static package is mandatory: the dev package's cudss-static-targets.cmake
# is included unconditionally by its config, so a missing libcudss_static.a makes
# every downstream find_package(cudss) a FATAL_ERROR. Its non-standard install
# path is outside the default ld cache and has to be registered by hand.
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists \
    apt-get update && apt-get install -y --no-install-recommends \
    libcudss0-cuda-12 libcudss0-dev-cuda-12 libcudss0-static-cuda-12 && \
    echo "/usr/lib/x86_64-linux-gnu/libcudss/12" > /etc/ld.so.conf.d/cudss.conf && \
    ldconfig

# Ceres master@806af05 (2.3.0-dev, untagged upstream; 2.2.0 has no cuDSS support).
# cudss_INCLUDE_DIR must be passed explicitly: the deb puts headers under
# /usr/include/libcudss/12, while cudss-config.cmake's relative search only looks
# at the /usr/include root and misses them.
RUN git clone https://github.com/ceres-solver/ceres-solver.git /tmp/build/ceres && \
    cd /tmp/build/ceres && \
    git checkout 806af056feb22a1236baca247b4f4b6e1ea911e5 && \
    git submodule update --init --depth 1 third_party/abseil-cpp && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF \
             -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
             -Dcudss_DIR=/usr/lib/x86_64-linux-gnu/libcudss/12/cmake/cudss \
             -Dcudss_INCLUDE_DIR=/usr/include/libcudss/12 && \
    make -j$(nproc) && make install && \
    cd / && rm -rf /tmp/build/ceres && ldconfig

# COLMAP 4.0.3, with ONNX LightGlue support.
# The cudss_* arguments repeat here because Ceres exports cuDSS as a transitive
# dependency, so COLMAP's cmake has to be able to find it too.
RUN git clone --depth 1 --branch 4.0.3 https://github.com/colmap/colmap.git /tmp/build/colmap && \
    cd /tmp/build/colmap && mkdir build && cd build && \
    cmake .. -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
        -DCMAKE_CUDA_ARCHITECTURES=89 \
        -DCUDA_ENABLED=ON \
        -DONNX_ENABLED=ON \
        -DFETCH_ONNX=ON \
        -Dcudss_DIR=/usr/lib/x86_64-linux-gnu/libcudss/12/cmake/cudss \
        -Dcudss_INCLUDE_DIR=/usr/include/libcudss/12 && \
    ninja -j$(nproc) && ninja install && \
    ldconfig

# pycolmap, built straight from COLMAP 4.0.3's own bindings via cmake.
# Needs pybind11 3.0.1 and links against the COLMAP libraries installed above.
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir 'pybind11[global]==3.0.1' && \
    cd /tmp/build/colmap/python && mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
        -DCMAKE_CUDA_ARCHITECTURES=89 \
        -DSKBUILD_PROJECT_NAME=pycolmap \
        -DSKBUILD_PROJECT_VERSION=4.0.3 \
        -DGENERATE_STUBS=OFF \
        -Donnxruntime_LIBRARY_DIR_HINTS=/usr/local/lib \
        -Dpybind11_DIR=$(pybind11-config --cmakedir) \
        -Dcudss_DIR=/usr/lib/x86_64-linux-gnu/libcudss/12/cmake/cudss \
        -Dcudss_INCLUDE_DIR=/usr/include/libcudss/12 && \
    make -j$(nproc) && \
    SITE=$(python3 -c "import site; print(site.getsitepackages()[0])") && \
    cp -r /tmp/build/colmap/python/pycolmap "$SITE/pycolmap" && \
    cp _core.cpython-*.so "$SITE/pycolmap/" && \
    cd / && rm -rf /tmp/build/colmap

# nanoflann 1.6.3: apt's 1.4.2 lacks SearchParameters, which OpenMVS requires.
RUN git clone --depth 1 --branch v1.6.3 https://github.com/jlblancoc/nanoflann.git /tmp/build/nanoflann && \
    cd /tmp/build/nanoflann && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DNANOFLANN_BUILD_EXAMPLES=OFF -DNANOFLANN_BUILD_TESTS=OFF && \
    make install && \
    cd / && rm -rf /tmp/build/nanoflann

# OpenMVS v2.4.0.
# Header-only dependencies, cloned for the build and discarded with /tmp/build
# because the apt versions are too old:
#   - CGAL 6.0.1 (apt's 5.4 has no AABB_traits_3.h)
#   - VCGLib (the mesh library OpenMVS builds on)
# Patch: libjxl is disabled. The pipeline needs no .jxl support, Ubuntu 22.04
#        ships no libjxl-dev, and OpenCV 4.5 has no IMWRITE_JPEGXL_QUALITY.
RUN git clone --depth 1 --branch v6.0.1 https://github.com/CGAL/cgal.git /tmp/build/cgal && \
    git clone --depth 1 https://github.com/cnr-isti-vclab/vcglib.git /tmp/build/vcglib && \
    git clone --recursive --branch v2.4.0 https://github.com/cdcseacave/openMVS.git /tmp/build/openMVS && \
    cd /tmp/build/openMVS && \
    sed -i 's/pkg_check_modules(${PREFIX} REQUIRED IMPORTED_TARGET/pkg_check_modules(${PREFIX} IMPORTED_TARGET/' \
        libs/IO/CMakeLists.txt && \
    sed -i 's/cv::IMWRITE_JPEGXL_QUALITY/cv::IMWRITE_JPEG_QUALITY/' \
        libs/Common/Types.inl && \
    mkdir _build && cd _build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
        -DCMAKE_CUDA_ARCHITECTURES=89 \
        -DOpenMVS_USE_CUDA=ON \
        -DVCG_ROOT=/tmp/build/vcglib \
        -DCGAL_DIR=/tmp/build/cgal && \
    make -j$(nproc) && make install && \
    ldconfig && \
    cd / && rm -rf /tmp/build

# PyTorch 2.4 (CUDA 12.4)
RUN python3 -m pip install --no-cache-dir \
    torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Put PyTorch's bundled cuDNN 9 on the system library path: COLMAP's ONNX
# LightGlue / ALIKED CUDA providers look for libcudnn.so.9.
RUN echo "/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib" \
    > /etc/ld.so.conf.d/torch-cudnn.conf && ldconfig

# Python dependencies.
# safetensors is a transitive dependency megaloc's hubconf.py imports at runtime
# without declaring; without it the first weight download raises ModuleNotFoundError.
RUN python3 -m pip install --no-cache-dir \
    numpy opencv-python plyfile Pillow matplotlib scipy \
    scikit-image imageio open3d pyyaml safetensors tqdm \
    git+https://github.com/facebookresearch/segment-anything.git

# pointops (vendored CUDA extension)
COPY pointops /tmp/pointops
RUN cd /tmp/pointops && \
    rm -rf build pointops.egg-info dist && \
    CXX=g++-10 python3 -m pip install --no-build-isolation . && \
    cd / && rm -rf /tmp/pointops

# hloc (SuperPoint + LightGlue), installed from source as it is not on PyPI.
# The sed drops its pycolmap>=3.13.0 pin: this image's pycolmap 4.0.3 was built
# with cmake and copied into site-packages above, leaving no pip metadata behind,
# so the pin would pull a PyPI wheel over the custom build and lose it.
# third_party is copied because hloc/extractors/superpoint.py reaches the
# SuperGluePretrainedNetwork submodule via sys.path.append("../../third_party");
# pip installs only the Python package, so without the copy the SuperPoint
# extractor raises ModuleNotFoundError when it loads.
RUN git clone --recursive https://github.com/cvg/Hierarchical-Localization /tmp/build/hloc && \
    sed -i '/^pycolmap/d' /tmp/build/hloc/requirements.txt && \
    python3 -m pip install --no-cache-dir /tmp/build/hloc && \
    cp -r /tmp/build/hloc/third_party /usr/local/lib/python3.10/dist-packages/third_party && \
    rm -rf /tmp/build/hloc

# Put the OpenMVS CLI tools on PATH; they install into a /usr/local/bin/OpenMVS/ subdirectory.
ENV PATH="/usr/local/bin/OpenMVS:$PATH"

# Verify the installation
RUN python3 -c "import pycolmap; assert pycolmap.__file__ == '/usr/local/lib/python3.10/dist-packages/pycolmap/__init__.py', pycolmap.__file__; print(f'pycolmap {pycolmap.__version__}')" && \
    ls /usr/local/bin/OpenMVS/InterfaceCOLMAP && echo "InterfaceCOLMAP OK" && \
    ls /usr/local/bin/OpenMVS/DensifyPointCloud && echo "DensifyPointCloud OK" && \
    ls /usr/local/bin/OpenMVS/ReconstructMesh && echo "ReconstructMesh OK" && \
    python3 -c "import pointops; print('pointops OK')" && \
    python3 -c "from hloc import extract_features, match_features, pairs_from_retrieval; from hloc.triangulation import import_features, import_matches, estimation_and_geometric_verification; from hloc.extractors.superpoint import SuperPoint; from hloc.matchers.lightglue import LightGlue; print('hloc OK')" && \
    python3 -c "import huggingface_hub, safetensors; print(f'huggingface_hub {huggingface_hub.__version__}, safetensors {safetensors.__version__}')" && \
    python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"

# Back to the non-root user
USER $USERNAME
