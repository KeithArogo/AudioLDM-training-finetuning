#!/bin/bash

# Define environment names, paths, and cache directories
CONDA_ENV_PATH="/mnt/fast/nobackup/scratch4weeks/ko00537/conda_envs/audioldm_train"
CONDA_ENV_NAME="audioldm_train"
PYTHON_VERSION="3.10"
PROJECT_DIR="/mnt/fast/nobackup/scratch4weeks/ko00537/AudioLDM-training-finetuning"

# Log the start time
echo "Script started at: $(date)"

# Activate the Conda environment
echo "Activating Conda environment: ${CONDA_ENV_PATH}"
source activate ${CONDA_ENV_PATH}

# Verify the activation
echo "Conda environment activated. Current Python version: $(python --version)"

# Navigate to the project directory
echo "Navigating to project directory: ${PROJECT_DIR}"
cd ${PROJECT_DIR}

# Log the start of the main script execution
echo "Starting the main script using Poetry"

# Run the main script using Poetry
#poetry run python3 audioldm_train/eval.py --log_path all
echo "VAE Eval - Best Params"
poetry run python3 audioldm_train/eval.py --log_path log/latent_diffusion/Best_Hyperparameter_Experiments

# Log the completion of the main script
echo "Main script completed at: $(date)"