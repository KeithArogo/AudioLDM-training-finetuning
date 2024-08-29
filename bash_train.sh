#!/bin/bash

# Define environment names, paths, and cache directories
CONDA_ENV_PATH="/mnt/fast/nobackup/scratch4weeks/ko00537/conda_envs/audioldm_train"
CONDA_ENV_NAME="audioldm_train"
PYTHON_VERSION="3.10"
PROJECT_DIR="/mnt/fast/nobackup/scratch4weeks/ko00537/AudioLDM-training-finetuning"

# Activate the Conda environment
source activate ${CONDA_ENV_PATH}

# Navigate to the project directory
cd ${PROJECT_DIR}

# Run the main script using the new paths
poetry run python3 audioldm_train/train/latent_diffusion.py -c audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original.yaml --reload_from_ckpt data/checkpoints/audioldm-s-full


