#!/bin/bash

# Define environment names, paths, and cache directories
CONDA_ENV_PATH="/mnt/fast/nobackup/scratch4weeks/ko00537/conda_envs/audioldm_train"
CONDA_ENV_NAME="audioldm_train"
PYTHON_VERSION="3.10"
PROJECT_DIR="/mnt/fast/nobackup/scratch4weeks/ko00537/AudioLDM-training-finetuning"

# Array of YAML files
YAML_FILES=(
  "audioldm_train/config/2023_11_13_vae_autoencoder/Best_Hyperparams/best_hyperparams_snr_fix.yaml",
  #"audioldm_train/config/2023_11_13_vae_autoencoder/Best_Hyperparams/best_hyperparams.yaml"
)

# Activate the Conda environment
source activate ${CONDA_ENV_PATH}

# Navigate to the project directory
cd ${PROJECT_DIR}

# Ensure the necessary packages are installed (install only once, not during each run)
#poetry add python_speech_features scipy

# Train the VAE sequentially for each YAML file
for YAML_FILE in "${YAML_FILES[@]}"; do
    echo "Training with ${YAML_FILE}"
    poetry run python3 audioldm_train/train/autoencoder_eval.py -c ${YAML_FILE}
done
