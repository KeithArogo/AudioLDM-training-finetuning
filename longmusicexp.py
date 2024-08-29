import os

# Not enough GPU memory to conduct most experiments

# Define the base directory for the experiments
base_experiment_dir = 'Longermusic_experiments'

# Only the latent_t_size and audio_duration parameters will vary
latent_t_size_audio_duration_pairs = [
    {'latent_t_size': '512', 'audio_duration': '90'},
    {'latent_t_size': '512', 'audio_duration': '100'},
    {'latent_t_size': '512', 'audio_duration': '110'},
    {'latent_t_size': '512', 'audio_duration': '120'},
    {'latent_t_size': '1024', 'audio_duration': '45'},
    {'latent_t_size': '1024', 'audio_duration': '50'},
    {'latent_t_size': '1024', 'audio_duration': '55'},
    {'latent_t_size': '1024', 'audio_duration': '60'},
    {'latent_t_size': '2048', 'audio_duration': '30'},
    {'latent_t_size': '2048', 'audio_duration': '35'},
    {'latent_t_size': '2048', 'audio_duration': '40'},
    {'latent_t_size': '4096', 'audio_duration': '15'},
    {'latent_t_size': '4096', 'audio_duration': '17.5'},
    {'latent_t_size': '4096', 'audio_duration': '20'},
]

# Default values
default_values = {
    'Learning_Rate': '5.0e-6',
    'Disc_Start': '50001',
    'KL_Weight': '1500.0',
    'Disc_Weight': '0.3',
    'Embed_Dim': '8',
    'Double_Z': 'true',
    'Mel_Bins': '64',
    'Z_Channels': '8',
    'Resolution': '256',
    'Downsample_Time': 'false',
    'In_Channels': '8',
    'Out_Channels': '1',
    'Channels': '128',
    'Channel_Mult': '[1, 2, 4]',
    'Num_Res_Blocks': '2',
    'Attn_Resolutions': '[]',
    'Dropout': '0.0',
    'Batch_Size': '4',
    'latent_t_size': '256',  # Default latent_t_size
    'audio_duration': '10.24'  # Default duration (in seconds)
}

# The base YAML content with placeholders for parameters
yaml_template = '''metadata_root: "./data/dataset/metadata/dataset_root.json"
log_directory: "./log/latent_diffusion"
project: "audioldm"
precision: "high"

variables:
  sampling_rate: &sampling_rate 16000 
  mel_bins: &mel_bins {Mel_Bins}
  latent_embed_dim: &latent_embed_dim {Embed_Dim}
  latent_t_size: &latent_t_size {latent_t_size}
  latent_f_size: &latent_f_size 16
  in_channels: &unet_in_channels {In_Channels}
  optimize_ddpm_parameter: &optimize_ddpm_parameter true
  optimize_gpt: &optimize_gpt true
  warmup_steps: &warmup_steps 4000

data: 
  train: ["musicQA"]
  val: "musicQA"
  test: "musicQA"
  class_label_indices: "audioset_eval_subset"
  dataloader_add_ons: [] 

step:
  validation_every_n_epochs: 5
  save_checkpoint_every_n_steps: 5000
  max_steps: 200000
  save_top_k: 1

preprocessing:
  audio:
    sampling_rate: *sampling_rate
    max_wav_value: 32768.0
    duration: {audio_duration}
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
    first_stage_config:
      base_learning_rate: 2.0e-02
      target: audioldm_train.modules.latent_encoder.autoencoder.AutoencoderKL
      params: 
        reload_from_ckpt: "data/checkpoints/vae_mel_16k_64bins.ckpt"
        sampling_rate: *sampling_rate
        batchsize: {Batch_Size}
        monitor: val/rec_loss
        image_key: fbank
        subband: 1
        embed_dim: *latent_embed_dim
        time_shuffle: 1
        lossconfig:
          target: audioldm_train.losses.LPIPSWithDiscriminator
          params:
            disc_start: {Disc_Start}
            kl_weight: {KL_Weight}
            disc_weight: {Disc_Weight}
            disc_in_channels: 1
        ddconfig: 
          double_z: {Double_Z}
          mel_bins: *mel_bins
          z_channels: {Z_Channels}
          resolution: {Resolution}
          downsample_time: {Downsample_Time}
          in_channels: 1
          out_ch: {Out_Channels}
          ch: {Channels}
          ch_mult: {Channel_Mult}
          num_res_blocks: {Num_Res_Blocks}
          attn_resolutions: {Attn_Resolutions}
          dropout: {Dropout}

    base_learning_rate: {Learning_Rate}
    warmup_steps: *warmup_steps
    optimize_ddpm_parameter: *optimize_ddpm_parameter
    sampling_rate: *sampling_rate
    batchsize: {Batch_Size}
    linear_start: 0.0015
    linear_end: 0.0195
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    unconditional_prob_cfg: 0.1
    parameterization: eps
    first_stage_key: fbank
    latent_t_size: *latent_t_size
    latent_f_size: *latent_f_size
    channels: *latent_embed_dim
    monitor: val/loss_simple_ema
    scale_by_std: true
    unet_config:
      target: audioldm_train.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 64
        extra_film_condition_dim: 512
        in_channels: *unet_in_channels
        out_channels: *latent_embed_dim
        model_channels: 128
        attention_resolutions:
        - 4
        - 2
        - 1
        num_res_blocks: 4
        channel_mult:
        - 1
        - 2
        - 3
        - 5
        num_head_channels: 8
        use_spatial_transformer: true
        transformer_depth: 2
        extra_sa_layer: false
    
    cond_stage_config:
      film_clap_cond1:
        cond_stage_key: text
        conditioning_key: film
        target: audioldm_train.conditional_models.CLAPAudioEmbeddingClassifierFreev2
        params:
          pretrained_path: data/checkpoints/clap_htsat_tiny.pt
          sampling_rate: 16000
          embed_mode: text
          amodel: HTSAT-tiny

    evaluation_params:
      unconditional_guidance_scale: 3.5
      ddim_sampling_steps: 200
      n_candidates_per_samples: 3
'''

# Create the base directory for the experiments
os.makedirs(base_experiment_dir, exist_ok=True)

# Loop through latent_t_size and audio_duration pairs
for idx, pair in enumerate(latent_t_size_audio_duration_pairs, 1):
    yaml_file = f'latent_t_size_audio_duration_{idx}.yaml'
    param_dir = os.path.join(base_experiment_dir, 'latent_t_size_audio_duration')
    os.makedirs(param_dir, exist_ok=True)
    file_path = os.path.join(param_dir, yaml_file)
    
    # Create a copy of the default values
    yaml_values = default_values.copy()
    
    # Override the latent_t_size and audio_duration with the pair values
    yaml_values['latent_t_size'] = pair['latent_t_size']
    yaml_values['audio_duration'] = pair['audio_duration']
    
    # Replace the placeholders in the YAML content with the specific parameter values
    specific_yaml_content = yaml_template.format(**yaml_values)
    
    # Write the customized content to each file
    with open(file_path, 'w') as f:
        f.write(specific_yaml_content)

print(f"Directories and YAML files created and populated successfully under '{base_experiment_dir}'.")
