#!/bin/bash

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
source $HOME/miniconda/bin/activate

# Create and activate conda environment
conda create -n audioldm_train python=3.10 -y
source activate audioldm_train

# Install poetry
pip install poetry

# Install dependencies using poetry
poetry install

# Install additional Python packages (if any)
pip install pyyaml

# Run the main script
python3 audioldm_train/train/latent_diffusion.py -c audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original.yaml --reload_from_ckpt data/checkpoints/audioldm-s-full
