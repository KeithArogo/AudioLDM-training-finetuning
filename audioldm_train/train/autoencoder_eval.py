import sys
import os
import wandb
import argparse
import yaml
import torch
from pytorch_lightning.strategies.ddp import DDPStrategy
from audioldm_train.utilities.data.dataset import AudioDataset
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from audioldm_train.modules.latent_encoder.autoencoder import AutoencoderKL

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith("."):
            yield f

def main(configs, exp_group_name, exp_name):
    if "precision" in configs.keys():
        torch.set_float32_matmul_precision(configs["precision"])
    batch_size = config_yaml["model"]["params"]["batchsize"]
    log_path = config_yaml["log_directory"]

    if "dataloader_add_ons" in configs["data"].keys():
        dataloader_add_ons = configs["data"]["dataloader_add_ons"]
    else:
        dataloader_add_ons = []

    # Load the validation dataset
    val_dataset = AudioDataset(config_yaml, split="val", add_ons=dataloader_add_ons)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=8, shuffle=False
    )

    # Define the model
    model = AutoencoderKL(
        ddconfig=config_yaml["model"]["params"]["ddconfig"],
        lossconfig=config_yaml["model"]["params"]["lossconfig"],
        embed_dim=config_yaml["model"]["params"]["embed_dim"],
        image_key=config_yaml["model"]["params"]["image_key"],
        base_learning_rate=config_yaml["model"]["base_learning_rate"],
        subband=config_yaml["model"]["params"]["subband"],
        sampling_rate=config_yaml["preprocessing"]["audio"]["sampling_rate"],
    )

    # Ensure log_dir is set for the model
    model.set_log_dir(log_path, exp_group_name, exp_name)

    # Checkpoint path handling
    checkpoint_path = os.path.join(log_path, exp_group_name, exp_name, "checkpoints")

    # Verify if the checkpoint exists
    if len(os.listdir(checkpoint_path)) > 0:
        print(f"Loading checkpoint from path: {checkpoint_path}")
        # Assuming the latest checkpoint is used
        checkpoint_file = max(listdir_nohidden(checkpoint_path), key=lambda x: os.path.getctime(os.path.join(checkpoint_path, x)))
        resume_from_checkpoint = os.path.join(checkpoint_path, checkpoint_file)
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
    else:
        raise FileNotFoundError("No checkpoints found in the directory. Please ensure checkpoints are available.")

    devices = torch.cuda.device_count()

    wandb_logger = WandbLogger(
        save_dir=os.path.join(log_path, exp_group_name, exp_name),
        project=config_yaml["project"],
        config=config_yaml,
        name=f"{exp_group_name}/{exp_name}",
    )

    # Trainer setup for evaluation
    trainer = Trainer(
        accelerator="gpu",
        devices=devices,
        logger=wandb_logger,
        strategy=DDPStrategy(find_unused_parameters=True),
    )

    # EVALUATION
    print("Running evaluation on validation set...")
    trainer.test(model, val_loader, ckpt_path=resume_from_checkpoint)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--autoencoder_config", type=str, required=True, help="path to autoencoder config .yaml"
    )
    args = parser.parse_args()

    config_yaml = args.autoencoder_config
    exp_name = os.path.basename(config_yaml.split(".")[0])
    exp_group_name = os.path.basename(os.path.dirname(config_yaml))

    config_yaml = os.path.join(config_yaml)
    config_yaml = yaml.load(open(config_yaml, "r"), Loader=yaml.FullLoader)

    main(config_yaml, exp_group_name, exp_name)
