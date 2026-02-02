"""
Feintuning ab best_model.pt
- lädt bestes Modell
- trainiert mit kleiner LR weiter
"""

import os
import torch
from train import train

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.dirname(script_dir)

    config_dict = dict()
    config_dict['seed'] = 42
    config_dict['testset_ratio'] = 0.1
    config_dict['validset_ratio'] = 0.1
    config_dict['results_path'] = os.path.join(code_dir, "results_nirwana")
    config_dict['data_path'] = os.path.join(code_dir, "data", "dataset")
    config_dict['resume_from'] = os.path.join(config_dict['results_path'], "best_model.pt")
    config_dict['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Feintuning-Settings
    config_dict['learningrate'] = 1e-5
    config_dict['weight_decay'] = 5e-7
    config_dict['n_updates'] = 30000
    config_dict['batchsize'] = 12
    config_dict['early_stopping_patience'] = 10
    config_dict['use_wandb'] = False
    config_dict['gradient_clip_value'] = 0.2
    config_dict['use_tta'] = True
    config_dict['accumulation_steps'] = 2
    config_dict['warmup_steps'] = 1000

    config_dict['print_train_stats_at'] = 10
    config_dict['print_stats_at'] = 50
    config_dict['plot_at'] = 50
    config_dict['validate_at'] = 50

    network_config = {
        'n_in_channels': 4
    }
    config_dict['network_config'] = network_config

    train(**config_dict)
