[![arXiv](https://img.shields.io/badge/arXiv-2301.12503-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2301.12503) [![arXiv](https://img.shields.io/badge/arXiv-2308.05734-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2308.05734)

# 🔊 AudioLDM training, finetuning, inference and evaluation

- [Prepare Python running environment](#prepare-python-running-environment)
  * [Download checkpoints and dataset](#download-checkpoints-and-dataset)
- [Play around with the code](#play-around-with-the-code)
  * [Train the AudioLDM model](#train-the-audioldm-model)
  * [Finetuning of the pretrained model](#finetuning-of-the-pretrained-model)
  * [Evaluate the model output](#evaluate-the-model-output)
  * [Inference with the pretrained model](#inference-with-the-pretrained-model)
  * [Train the model using your own dataset](#train-the-model-using-your-own-dataset)
- [Cite this work](#cite-this-work)

# Prepare Python running environment

```shell 
# Create conda environment
conda create -n audioldm_train python=3.10
conda activate audioldm_train
# Clone the repo
git clone https://github.com/KeithArogo/AudioLDM-training-finetuning; cd AudioLDM-training-finetuning
# Install running environment
pip install poetry
poetry install
```

## Download checkpoints and dataset
1. Download checkpoints from Google Drive: [link](https://drive.google.com/file/d/1T6EnuAHIc8ioeZ9kB1OZ_WGgwXAVGOZS/view?usp=drive_link). The checkpoints including pretrained VAE, AudioMAE, CLAP, 16kHz HiFiGAN, and 48kHz HiFiGAN.
2. Uncompress the checkpoint tar file and place the content into **data/checkpoints/**
3. Download the preprocessed dataset from Google Drive: [link](https://drive.google.com/file/d/1-4X5l5Q8CP6Jcv8Dpu7e0MaU0VDLmgxN/view?usp=sharing)
4. Similarly, uncompress the dataset tar file and place the content into **data/dataset**

To double check if dataset or checkpoints are ready, run the following command:
```shell
python3 tests/validate_dataset_checkpoint.py
```
If the structure is not correct or partly missing. You will see the error message.

# Play around with the code

## Train the AudioLDM model
```python
# Train the AudioLDM (latent diffusion part)
python3 audioldm_train/train/latent_diffusion.py -c audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original.yaml

# Train the VAE
python3 audioldm_train/train/autoencoder.py -c audioldm_train/config/2023_11_13_vae_autoencoder/16k_64.yaml
```

The program will perform generation on the evaluation set every 5 epochs of training. After obtaining the audio generation folders (named val_<training-steps>), you can proceed to the next step for model evaluation.

## Finetuning of the pretrained model

You can finetune with two pretrained checkpoint, first download the one that you like (e.g., using wget):
1. Medium size AudioLDM: https://zenodo.org/records/7884686/files/audioldm-m-full.ckpt
2. Small size AudioLDM: https://zenodo.org/records/7884686/files/audioldm-s-full

Place the checkpoint in the *data/checkpoints* folder

Then perform finetuning with one of the following commands:
```shell
# Medium size AudioLDM
python3 audioldm_train/train/latent_diffusion.py -c audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml --reload_from_ckpt data/checkpoints/audioldm-m-full.ckpt

# Small size AudioLDM
python3 audioldm_train/train/latent_diffusion.py -c audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original.yaml --reload_from_ckpt data/checkpoints/audioldm-s-full
```
You can specify your own dataset following the same format as the provided AudioCaps dataset.

Note that the pretrained AudioLDM checkpoints are under CC-by-NC 4.0 license, which is not allowed for commerial use.

## Experiment Directory and File Setup

This script (`create_experiment_dirs_and_files.py`) automates the creation of directories and sub-files for organizing experiments related to the AudioLDM project. The script is designed to help structure and manage multiple experiment configurations efficiently.

### Experiment Structure

The script uses a dictionary to define the structure, where each key represents an experiment name, and each value indicates the number of sub-files to be created within that experiment’s folder.

Example configuration:
```python
experiments = {
    "7.experiments_warmupsteps": 3,
    "17.experiments_use_spatial_transformer": 3,
    "18.experiments_transformer_depth": 3,
    "11.experiments_out_channels": 3,
    ...
}
```
### How It Works

1. The script creates a set of folders and corresponding sub-files based on predefined experiment names and the number of files specified for each experiment.
2. Root Directory: All experiment folders and files will be created under the following root directory
3. Script Behavior: Directory Creation: The script checks if each experiment directory already exists. If it doesn’t, the script creates it.
4. File Creation: For each experiment, the specified number of sub-files is created inside its corresponding directory. The files are named sequentially as experiment_X.sub where X is the file number.
 
Example Output:

After running the script, the directory structure will look like this:
```python
Z:\\AudioLDM-training-finetuning\\
│
├── 7.experiments_warmupsteps\\
│   ├── experiment_1.sub
│   ├── experiment_2.sub
│   └── experiment_3.sub
│
├── 17.experiments_use_spatial_transformer\\
│   ├── experiment_1.sub
│   ├── experiment_2.sub
│   └── experiment_3.sub
│
...
```
Usage:
To use this script, simply run it in your Python environment. It will create the necessary directories and files based on the experiments dictionary configuration. This setup helps keep your experimental data organized and easy to manage.'

## YAML File Generator for LDM Parameter Experiments

This Python script (`experimental_yaml_setup.py`) automates the creation of directories and YAML configuration files for conducting experiments with various parameters related to Latent Diffusion Models (LDM). The script is designed to help efficiently organize and manage multiple experiment configurations by generating the necessary directories and files based on predefined parameters and their proposed values.

## How It Works

The script iterates through a set of parameters, each associated with a list of proposed values. For each parameter, a directory is created under a base directory. Within each parameter's directory, a YAML file is generated for each proposed value, containing predefined content that can be customized for specific experiments.

### Base Directory

All experiment directories and YAML files will be created under a specified base directory:

### Parameter Configuration

The script uses a dictionary to define the parameters and their corresponding values. For example:

```python
parameters = {
    'Learning_Rate': ['1.0e-4', '5.0e-5', '1.0e-5', '5.0e-6', '1.0e-6'],
    'Batch_Size': ['2', '4', '8', '16'],
    ...
}
```
Script Behavior
1. Directory Creation: For each parameter, a directory is created within the specified base directory.
2. YAML File Creation: For each value associated with a parameter, a YAML file is created within the parameter's directory. Each file is populated with a predefined YAML content template.

Example Output
After running the script, the directory structure will look like this:
```python
LDM_parameter_experiments/
├── Learning_Rate/
│   ├── Learning_Rate_1.yaml
│   ├── Learning_Rate_2.yaml
│   ├── Learning_Rate_3.yaml
│   ├── Learning_Rate_4.yaml
│   └── Learning_Rate_5.yaml
├── Batch_Size/
│   ├── Batch_Size_1.yaml
│   ├── Batch_Size_2.yaml
│   ├── Batch_Size_3.yaml
│   └── Batch_Size_4.yaml
...
```
Usage:
To use this script, simply run it in your Python environment. It will create the necessary directories and YAML files based on the parameters dictionary configuration. This setup helps keep your experiment configurations organized and easy to manage.

Running the Script:
```python
python create_ldm_experiment_yamls.py
```
Customization:
You can customize the content of the generated YAML files by modifying the yaml_content variable within the script. This allows you to tailor the configurations to meet the specific needs of your experiments.

## Evaluate the model output
Automatically evaluation based on each of the folder with generated audio
```python

# Evaluate all existing generated folder
python3 audioldm_train/eval.py --log_path all

# Evaluate only a specific experiment folder
python3 audioldm_train/eval.py --log_path <path-to-the-experiment-folder>
```
The evaluation result will be saved in a json file at the same level of the audio folder.

## Evaluate the VAE
```python

# Evaluate specific VAE configuration
python3 run python3 audioldm_train/train/autoencoder_eval.py -c ${YAML_FILE}
```

## Inference with the pretrained model
Use the following syntax:

```shell
python3 audioldm_train/infer.py --config_yaml ${YAML_FILE}  --list_inference <the-filelist-you-want-to-generate> --reload_from_ckpt ${CHECKPOINT_PATH}
```

The generated audio will be named with the caption by default. If you like to specify the filename to use, please checkout the format of *tests/captionlist/inference_test_with_filename.lst*.

This repo only support inference with the model you trained by yourself. If you want to use the pretrained model directly, please use these two repos: [AudioLDM](https://github.com/haoheliu/AudioLDM) and [AudioLDM2](https://github.com/haoheliu/AudioLDM2).

## Train the model using your own dataset
Super easy, simply follow these steps:

1. Prepare the metadata with the same format as the provided dataset. 
2. Register in the metadata of your dataset in **data/dataset/metadata/dataset_root.json**
3. Use your dataset in the YAML file.

You do not need to resample or pre-segment the audiofile. The dataloader will do most of the jobs.

# Cite this work
If you found this tool useful, please consider citing

```bibtex
@article{audioldm2-2024taslp,
  author={Liu, Haohe and Yuan, Yi and Liu, Xubo and Mei, Xinhao and Kong, Qiuqiang and Tian, Qiao and Wang, Yuping and Wang, Wenwu and Wang, Yuxuan and Plumbley, Mark D.},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={AudioLDM 2: Learning Holistic Audio Generation With Self-Supervised Pretraining}, 
  year={2024},
  volume={32},
  pages={2871-2883},
  doi={10.1109/TASLP.2024.3399607}
}

@article{liu2023audioldm,
  title={{AudioLDM}: Text-to-Audio Generation with Latent Diffusion Models},
  author={Liu, Haohe and Chen, Zehua and Yuan, Yi and Mei, Xinhao and Liu, Xubo and Mandic, Danilo and Wang, Wenwu and Plumbley, Mark D},
  journal={Proceedings of the International Conference on Machine Learning},
  year={2023}
  pages={21450-21474}
}
```

# Acknowledgement
We greatly appreciate the open-soucing of the following code bases. Open source code base is the real-world infinite stone 💎!
- https://github.com/CompVis/stable-diffusion
- https://github.com/LAION-AI/CLAP
- https://github.com/jik876/hifi-gan

> This research was partly supported by the British Broadcasting Corporation Research and Development, Engineering and Physical Sciences Research Council (EPSRC) Grant EP/T019751/1 "AI for Sound", and a PhD scholarship from the Centre for Vision, Speech and Signal Processing (CVSSP), Faculty of Engineering and Physical Science (FEPS), University of Surrey. For the purpose of open access, the authors have applied a Creative Commons Attribution (CC BY) license to any Author Accepted Manuscript version arising. We would like to thank Tang Li, Ke Chen, Yusong Wu, Zehua Chen and Jinhua Liang for their support and discussions.

