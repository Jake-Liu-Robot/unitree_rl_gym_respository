#!/usr/bin/env bash
# =============================================================
# unitree_rl_gym  –  Docker Build & Run Reference
# =============================================================

# ──────────────────────────────────────────────────────────────
# STEP 0  Pre-requisites
# ──────────────────────────────────────────────────────────────
# 1. Install NVIDIA Container Toolkit (if not already done):
#
#    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
#        sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
#    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
#        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
#        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
#    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
#    sudo systemctl restart docker
#
# 2. Download Isaac Gym from https://developer.nvidia.com/isaac-gym
#    Extract it so the folder structure looks like:
#
#    build-context/
#    ├── Dockerfile
#    ├── requirements.txt
#    ├── docker_commands.sh   ← this file
#    └── isaacgym/            ← extracted Isaac Gym (required!)
#        ├── python/
#        ├── docs/
#        └── ...

# ──────────────────────────────────────────────────────────────
# STEP 1  Build the image
# ──────────────────────────────────────────────────────────────
IMAGE_NAME="rl_final"
IMAGE_TAG="latest"

# ──────────────────────────────────────────────────────────────
# STEP 2  Run (interactive shell, GPU enabled)
# ──────────────────────────────────────────────────────────────
# Option A – headless training (no display)
# docker run -it --rm \
#     --gpus all \
#     --ipc=host \
#     --ulimit memlock=-1 \
#     --ulimit stack=67108864 \
#     --name unitree-rl \
#     -v "$(dirname $(dirname $(pwd)))":/workspace \
#     "${IMAGE_NAME}:${IMAGE_TAG}"

# ──────────────────────────────────────────────────────────────
# Option B – with GUI / display (Isaac Gym viewer)
# Run this on the HOST first to allow X11 forwarding:
# xhost +local:docker

# docker run -it --rm \
#     --gpus all \
#     --ipc=host \
#     --ulimit memlock=-1 \
#     --ulimit stack=67108864 \
#     --name unitree-rl \
#     --env DISPLAY="${DISPLAY}" \
#     --env QT_X11_NO_MITSHM=1 \
#     --volume /tmp/.X11-unix:/tmp/.X11-unix \
#     -v "$(dirname $(dirname $(pwd)))":/workspace \
#     "${IMAGE_NAME}:${IMAGE_TAG}"

xhost +local:docker


docker run -it --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --name unitree-rl \
    --env DISPLAY="${DISPLAY}" \
    --env QT_X11_NO_MITSHM=1 \
    --env NVIDIA_DRIVER_CAPABILITIES=all \
    --runtime=nvidia \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    -v /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d \
    -v "$(dirname $(dirname $(pwd)))":/workspace \
    -v "$(dirname $(dirname $(pwd)))/unitree_rl_gym_respository/unitree_rl_gym":/opt/unitree_rl_gym \
    -v "$(dirname $(dirname $(pwd)))/unitree_rl_gym_respository/rsl_rl":/opt/rsl_rl \
    "${IMAGE_NAME}:${IMAGE_TAG}"

# ──────────────────────────────────────────────────────────────
# STEP 3  Verify GPU inside the container
# ──────────────────────────────────────────────────────────────
# Run from inside the container:
#   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# ──────────────────────────────────────────────────────────────
# STEP 4  Verify Isaac Gym (requires Option B / display)
# ──────────────────────────────────────────────────────────────
# Run from inside the container:
#   cd /opt/isaacgym/python/examples
#   python 1080_balls_of_solitude.py

# ──────────────────────────────────────────────────────────────
# STEP 5  Start training (example – Go2 walking)
# ──────────────────────────────────────────────────────────────
# Run from inside the container:
#   cd /opt/unitree_rl_gym
#   python legged_gym/scripts/train.py --task=go2

# ──────────────────────────────────────────────────────────────
# Useful Docker management commands
# ──────────────────────────────────────────────────────────────
# Stop a running container:    docker stop unitree-rl
# Remove the image:            docker rmi rl_final:latest
# Exec into running container: docker exec -it unitree-rl bash
