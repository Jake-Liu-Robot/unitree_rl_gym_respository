# ============================================================
# Dockerfile for unitree_rl_gym
# Base: CUDA 12.1 + Ubuntu 22.04
#
# IMPORTANT - Before building, you must manually download
# Isaac Gym from https://developer.nvidia.com/isaac-gym
# and place the extracted folder (named "isaacgym") in the
# same directory as this Dockerfile.
# ============================================================

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# ── System deps ─────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates \
    libgl1-mesa-glx libglib2.0-0 \
    libgomp1 \
    libegl1 libgles2 \
    vulkan-tools libvulkan1 \
    python3-dev build-essential \
    libglfw3 libglfw3-dev libglew-dev patchelf ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ── Miniconda ────────────────────────────────────────────────
ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -u -p ${CONDA_DIR} && \
    rm /tmp/miniconda.sh
ENV PATH="${CONDA_DIR}/bin:${PATH}"

# ── Accept Anaconda Terms of Service (required for non-interactive builds) ──
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# ── Conda env with Python 3.8 ────────────────────────────────
RUN conda create -y -n unitree-rl python=3.8 && \
    conda clean -afy

# All subsequent commands run inside the conda env
SHELL ["conda", "run", "-n", "unitree-rl", "/bin/bash", "-c"]

# ── PyTorch 2.3.1 + CUDA 12.1 ────────────────────────────────
RUN conda install -y \
    pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    pytorch-cuda=12.1 \
    -c pytorch -c nvidia && \
    conda clean -afy

# ── Isaac Gym (must be pre-downloaded into build context) ─────


COPY IsaacGym_Preview_4_Package /opt/isaacgym
RUN cd /opt/isaacgym/isaacgym/python && pip install -e . --no-cache-dir

# ── MuJoCo 2.3.7 ─────────────────────────────────────────────
# Install mujoco pip package (includes binaries, no manual download needed)
RUN pip install mujoco==2.3.7 --no-cache-dir

# Install mujoco-py for legacy compatibility (optional but common)
RUN pip install mujoco-py==2.1.2.14 --no-cache-dir

# Install gymnasium with MuJoCo extras
RUN pip install gymnasium[mujoco]==0.29.1 --no-cache-dir

# ── rsl_rl (copied from local) ───────────────────────────────
COPY rsl_rl /opt/rsl_rl
RUN cd /opt/rsl_rl && pip install -e . --no-cache-dir

# ── unitree_rl_gym (copied from local) ───────────────────────
COPY unitree_rl_gym /opt/unitree_rl_gym
RUN cd /opt/unitree_rl_gym && pip install -e . --no-cache-dir

# ── unitree_sdk2py (copied from local) ───────────────────────
COPY unitree_sdk2_python /opt/unitree_sdk2_python
RUN cd /opt/unitree_sdk2_python && pip install -e . --no-cache-dir

# ── Python requirements ──────────────────────────────────────
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── Runtime env vars ─────────────────────────────────────────
ENV PYTHONPATH="/opt/isaacgym/isaacgym/python:/opt/unitree_rl_gym:/opt/rsl_rl:${PYTHONPATH}"
# Prevent Isaac Gym from trying to open a display by default
ENV ISAACGYM_PATH=/opt/isaacgym

# Make conda env activate automatically
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate unitree-rl" >> ~/.bashrc
ENV PATH="${CONDA_DIR}/envs/unitree-rl/bin:${PATH}"

# Ensure conda is initialized and env is activated for non-interactive shells too
ENV BASH_ENV=/root/.bashrc
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate unitree-rl" > /etc/bash.bashrc.d/conda.sh 2>/dev/null || \
    echo "source /opt/conda/etc/profile.d/conda.sh && conda activate unitree-rl" >> /etc/bash.bashrc

RUN ln -s /opt/conda/envs/unitree-rl/lib/libpython3.8.so.1.0 /usr/lib/libpython3.8.so.1.0 && ldconfig

WORKDIR /workspace
CMD ["/bin/bash"]