#!/bin/bash

# Define environment names, paths, and cache directories
CONDA_ENV_PATH="/mnt/fast/nobackup/scratch4weeks/ko00537/conda_envs/audioldm_train"
CONDA_ENV_NAME="audioldm_train"
PYTHON_VERSION="3.10"
PROJECT_DIR="/mnt/fast/nobackup/scratch4weeks/ko00537/AudioLDM-training-finetuning"

# Array of YAML files
YAML_FILES=(
    "audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original.yaml"
    "audioldm_train/config/2023_08_23_reproduce_audioldm/Best_Hyperparameter_Experiments/best_hyperparameter_experiments.yaml"
    "audioldm_train/config/2023_08_23_reproduce_audioldm/Best_Hyperparameter_Experiments/best_hyperparameter_experiments_newvae.yaml"
    "audioldm_train/config/2023_08_23_reproduce_audioldm/Best_Hyperparameter_Experiments/best_hyperparameter_experiments_newvae2.yaml"
)

# Array of corresponding checkpoint paths
CHECKPOINT_PATHS=(
    "log/latent_diffusion/2023_08_23_reproduce_audioldm/audioldm_original/checkpoints/checkpoint-fad-133.00-global_step=164999.ckpt"
    "log/latent_diffusion/Best_Hyperparameter_Experiments/best_hyperparameter_experiments/checkpoints/checkpoint-fad-133.00-global_step=374999.ckpt"
    "log/latent_diffusion/Best_Hyperparameter_Experiments/best_hyperparameter_experiments_newvae/checkpoints/checkpoint-fad-133.00-global_step=54999.ckpt"
    "log/latent_diffusion/Best_Hyperparameter_Experiments/best_hyperparameter_experiments_newvae2/checkpoints/checkpoint-fad-133.00-global_step=64999.ckpt"
)

# Activate the Conda environment
source activate ${CONDA_ENV_PATH}

# Navigate to the project directory
cd ${PROJECT_DIR}

# Perform inference sequentially for each YAML file with its associated checkpoint
for index in "${!YAML_FILES[@]}"; do
    YAML_FILE=${YAML_FILES[$index]}
    CHECKPOINT_PATH=${CHECKPOINT_PATHS[$index]}
    
    echo "Inference with ${YAML_FILE} using checkpoint ${CHECKPOINT_PATH}"
    poetry run python3 audioldm_train/infer.py --config_yaml ${YAML_FILE} --list_inference tests/captionlist/inference_test.lst --reload_from_ckpt ${CHECKPOINT_PATH}
done
