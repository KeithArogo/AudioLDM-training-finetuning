import os

# Define the experiments and their respective file counts
experiments = {
    "7.experiments_warmupsteps": 3,
    "17.experiments_use_spatial_transformer": 3,
    "18.experiments_transformer_depth": 3,
    "11.experiments_out_channels": 3,
    "14.experiments_num_res_blocks": 3,
    "16.experiments_num_head_channels": 3,
    "12.experiments_model_channels": 1,
    "6.experiments_maxsteps": 3,
    "5.experiments_latentfsize": 3,
    "3.experiments_latentembed": 3,
    "4.experiments_latenettsize": 1,
    "8.experiments_klweight": 3,
    "10.experiments_in_channels": 3,
    "9.experiments_discweight": 3,
    "15.experiments_channel_mult": 1,
    "13.experiments_attention_resolutions": 3
}

# Define the root directory where the folders and files will be created
root_dir = r"Z:\AudioLDM-training-finetuning"

# Create each folder and sub-files
for exp_name, num_files in experiments.items():
    # Create the folder path
    folder_path = os.path.join(root_dir, exp_name)
    
    # Create the folder if it does not exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created directory: {folder_path}")
    else:
        print(f"Directory already exists: {folder_path}")
    
    # Create the sub-files with experiment number
    for i in range(1, num_files + 1):
        file_name = f"experiment_{i}.sub"
        file_path = os.path.join(folder_path, file_name)
        
        # Create an empty file
        with open(file_path, 'w') as file:
            file.write('')  # You can add initial content here if needed
        print(f"Created file: {file_path}")
