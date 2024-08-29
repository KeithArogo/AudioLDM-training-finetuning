import os
import yaml
import torch
import logging
from audioldm_eval import EvaluationHelper

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SAMPLE_RATE = 16000
device = torch.device(f"cuda:{0}")
evaluator = EvaluationHelper(SAMPLE_RATE, device)

def locate_yaml_file(path):
    """Locate the first YAML file in the specified directory."""
    for file in os.listdir(path):
        if file.endswith(".yaml"):
            return os.path.join(path, file)
    return None

def is_evaluated(path):
    """Check if the experiment directory has been evaluated."""
    candidates = [f for f in os.listdir(os.path.dirname(path)) if f.endswith(".json")]
    folder_name = os.path.basename(path)
    return any(folder_name in candidate for candidate in candidates)

def locate_validation_output(path):
    """Locate validation output folders that have not been evaluated."""
    folders = []
    for file in os.listdir(path):
        dirname = os.path.join(path, file)
        if "val_" in file and os.path.isdir(dirname) and not is_evaluated(dirname):
            folders.append(dirname)
    return folders

def evaluate_exp_performance(exp_name):
    """Evaluate performance for a given experiment."""
    abs_path_exp = os.path.join(latent_diffusion_model_log_path, exp_name)
    config_yaml_path = locate_yaml_file(abs_path_exp)

    if config_yaml_path is None:
        logging.warning(f"{exp_name} does not contain a yaml configuration file")
        return

    folders_todo = locate_validation_output(abs_path_exp)

    for folder in folders_todo:
        logging.info(f"Evaluating folder: {folder}")

        test_dataset = "musicQA"
        test_audio_data_folder = os.path.join(test_audio_path, test_dataset)

        try:
            evaluator.main(folder, test_audio_data_folder)
        except Exception as e:
            logging.error(f"Error evaluating folder {folder}: {e}")

def eval(exps):
    """Evaluate a list of experiments."""
    for exp in exps:
        try:
            evaluate_exp_performance(exp)
        except Exception as e:
            logging.error(f"Error processing experiment {exp}: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AudioLDM model evaluation")
    parser.add_argument("-l", "--log_path", type=str, help="the log path", required=True)
    parser.add_argument("-e", "--exp_name", type=str, help="the experiment name", default=None)

    args = parser.parse_args()

    test_audio_path = "log/testset_data"
    latent_diffusion_model_log_path = args.log_path

    if latent_diffusion_model_log_path != "all":
        exp_name = args.exp_name
        if exp_name is None:
            exps = os.listdir(latent_diffusion_model_log_path)
            eval(exps)
        else:
            eval([exp_name])
    else:
        todo_list = [os.path.abspath("log/latent_diffusion")]
        for todo in todo_list:
            for latent_diffusion_model_log_path in os.listdir(todo):
                latent_diffusion_model_log_path = os.path.join(todo, latent_diffusion_model_log_path)
                if os.path.isdir(latent_diffusion_model_log_path):
                    logging.info(f"Processing directory: {latent_diffusion_model_log_path}")
                    exps = os.listdir(latent_diffusion_model_log_path)
                    eval(exps)
