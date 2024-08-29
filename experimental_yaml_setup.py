import os

# Define the base directory for the experiments
base_experiment_dir = 'LDM_parameter_experiments'

# Parameters and their corresponding proposed values
parameters = {
    'Learning_Rate': ['1.0e-4', '5.0e-5', '1.0e-5', '5.0e-6', '1.0e-6'],
    'Learning_Rate_AE': ['8.0e-06', '8.0e-07', '8.0e-05', '4.0e-04', '2.0e-02'],
    'Batch_Size': ['2', '4', '8', '16'],
    'Batch_Size_AE': ['4', '8', '16', '32'],
    'Latent_Embed_Dim': ['8', '16', '32', '64'],
    'Latent_T_Size': ['256', '512'],
    'Latent_F_Size': ['16', '8', '32', '64'],
    'Max_Steps': ['800000', '100000', '200000', '400000'],
    'Warmup_Steps': ['2000', '3000', '1000', '500'],
    'KL_Weight': ['1000.0', '500.0', '1500', '2000.0'],
    'Disc_Weight': ['0.5', '0.3', '0.7', '0.9'],
    'In_Channels': ['8', '16', '32', '64'],
    'Out_Channels': ['8', '16', '32', '64'],
    'Model_Channels': ['128', '256', '512'],
    'Attention_Resolutions': ['[8, 4, 2]', '[4, 2, 1]', '[16, 8, 4]', '[32, 16, 8]'],
    'Num_Res_Blocks': ['2', '4', '8', '16'],
    'Channel_Mult': ['[1, 2, 3, 5]', '[1, 2, 4, 6]'],
    'Num_Head_Channels': ['32', '8', '16', '64'],
    'Use_Spatial_Transformer': ['true', 'false'],
    'Transformer_Depth': ['1', '2', '3', '4']
}

# The content to be added to each YAML file
yaml_content = '''metadata_root: "./data/dataset/metadata/dataset_root.json"
log_directory: "./log/latent_diffusion"
project: "audioldm"
precision: "high"

variables:
  sampling_rate: &sampling_rate 16000 
  mel_bins: &mel_bins 64
  latent_embed_dim: &latent_embed_dim 8
  latent_t_size: &latent_t_size 256 # TODO might need to change
  latent_f_size: &latent_f_size 16
  in_channels: &unet_in_channels 8
  optimize_ddpm_parameter: &optimize_ddpm_parameter true
  optimize_gpt: &optimize_gpt true
  warmup_steps: &warmup_steps 2000

data: 
  train: ["musicQA"]
  val: "musicQA"
  test: "musicQA"
  class_label_indices: "audioset_eval_subset"
  dataloader_add_ons: [] 

step:
  validation_every_n_epochs: 5
  save_checkpoint_every_n_steps: 5000
  # limit_val_batches: 2
  max_steps: 800000
  save_top_k: 1

preprocessing:
  audio:
    sampling_rate: *sampling_rate
    max_wav_value: 32768.0
    duration: 10.24
  stft:
    filter_length: 1024
    hop_length: 160
    win_length: 1024
  mel:
    n_mel_channels: *mel_bins
    mel_fmin: 0
    mel_fmax: 8000 

augmentation:
  mixup: 0.0

model:
  target: audioldm_train.modules.latent_diffusion.ddpm.LatentDiffusion
  params: 
    # Autoencoder
    first_stage_config:
      base_learning_rate: 8.0e-06
      target: audioldm_train.modules.latent_encoder.autoencoder.AutoencoderKL
      params: 
        reload_from_ckpt: "data/checkpoints/vae_mel_16k_64bins.ckpt"
        sampling_rate: *sampling_rate
        batchsize: 4
        monitor: val/rec_loss
        image_key: fbank
        subband: 1
        embed_dim: *latent_embed_dim
        time_shuffle: 1
        lossconfig:
          target: audioldm_train.losses.LPIPSWithDiscriminator
          params:
            disc_start: 50001
            kl_weight: 1000.0
            disc_weight: 0.5
            disc_in_channels: 1
        ddconfig: 
          double_z: true
          mel_bins: *mel_bins # The frequency bins of mel spectrogram
          z_channels: 8
          resolution: 256
          downsample_time: false
          in_channels: 1
          out_ch: 1
          ch: 128 
          ch_mult:
          - 1
          - 2
          - 4
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
    
    # Other parameters
    base_learning_rate: 1.0e-4
    warmup_steps: *warmup_steps
    optimize_ddpm_parameter: *optimize_ddpm_parameter
    sampling_rate: *sampling_rate
    batchsize: 2
    linear_start: 0.0015
    linear_end: 0.0195
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    unconditional_prob_cfg: 0.1
    parameterization: eps # [eps, x0, v]
    first_stage_key: fbank
    latent_t_size: *latent_t_size # TODO might need to change
    latent_f_size: *latent_f_size
    channels: *latent_embed_dim # TODO might need to change
    monitor: val/loss_simple_ema
    scale_by_std: true
    unet_config:
      target: audioldm_train.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 64 
        extra_film_condition_dim: 512 # If you use film as extra condition, set this parameter. For example if you have two conditioning vectors each have dimension 512, then this number would be 1024
        # context_dim: 
        # - 768
        in_channels: *unet_in_channels # The input channel of the UNet model
        out_channels: *latent_embed_dim # TODO might need to change
        model_channels: 128 # TODO might need to change
        attention_resolutions:
        - 8
        - 4
        - 2
        num_res_blocks: 2
        channel_mult: 
        - 1
        - 2
        - 3
        - 5
        num_head_channels: 32
        use_spatial_transformer: true
        transformer_depth: 1
        extra_sa_layer: false
    
    cond_stage_config:
      film_clap_cond1:
        cond_stage_key: text
        conditioning_key: film
        target: audioldm_train.conditional_models.CLAPAudioEmbeddingClassifierFreev2
        params:
          pretrained_path: data/checkpoints/clap_htsat_tiny.pt
          sampling_rate: 16000
          embed_mode: text # or text
          amodel: HTSAT-tiny

    evaluation_params:
      unconditional_guidance_scale: 3.5
      ddim_sampling_steps: 200
      n_candidates_per_samples: 3
'''

# Create the base directory for the experiments
os.makedirs(base_experiment_dir, exist_ok=True)

# Loop through each parameter and create the corresponding directories and YAML files
for parameter, values in parameters.items():
    # Create a directory for the parameter
    param_dir = os.path.join(base_experiment_dir, parameter)
    os.makedirs(param_dir, exist_ok=True)
    
    # Create YAML files for each proposed value
    for idx, value in enumerate(values, 1):
        yaml_file = f'{parameter}_{idx}.yaml'
        file_path = os.path.join(param_dir, yaml_file)
        with open(file_path, 'w') as f:
            f.write(yaml_content)  # Write the provided content to each file

print(f"Directories and YAML files created and populated successfully under '{base_experiment_dir}'.")
