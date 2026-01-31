"""
    Author: Your Name
    HTL-Grieskirchen 5. Jahrgang, Schuljahr 2025/26
    main.py
"""

import os
import torch
from utils import create_predictions


from train import train


if __name__ == '__main__':
    # Berechne Basispfad basierend auf Ort dieses Skripts
    script_dir = os.path.dirname(os.path.abspath(__file__))  # src Ordner
    code_dir = os.path.dirname(script_dir)  # code Ordner
    
    config_dict = dict()

    config_dict['seed'] = 42
    config_dict['testset_ratio'] = 0.1
    config_dict['validset_ratio'] = 0.1
    config_dict['results_path'] = os.path.join(code_dir, "results_improved")
    config_dict['data_path'] = os.path.join(code_dir, "data", "dataset")
    config_dict['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    config_dict['learningrate'] = 3e-4  # Optimiert für OneCycleLR
    config_dict['weight_decay'] = 1e-5  # Reduziert für Pre-trained Model
    config_dict['n_updates'] = 50000  # Weniger Updates bei besserer Architektur
    config_dict['batchsize'] = 16  # Größere Batches für bessere Stabilität
    config_dict['early_stopping_patience'] = 15
    config_dict['use_wandb'] = False

    config_dict['print_train_stats_at'] = 10
    config_dict['print_stats_at'] = 50  # Öfter validieren
    config_dict['plot_at'] = 50
    config_dict['validate_at'] = 50

    network_config = {
        'n_in_channels': 4
    }
    
    config_dict['network_config'] = network_config

    train(**config_dict)
    
    challenge_dir = os.path.dirname(code_dir)  # Image_Inpainting_Challenge Ordner
    testset_path = os.path.join(challenge_dir, "code", "data", "challenge_testset.npz")
    state_dict_path = os.path.join(config_dict['results_path'], "best_model.pt")
    save_path = os.path.join(config_dict['results_path'], "testset", "my_submission_name.npz")
    plot_path = os.path.join(config_dict['results_path'], "testset", "plots")

    # Comment out, if predictions are required
    create_predictions(config_dict['network_config'], state_dict_path, testset_path, None, save_path, plot_path, plot_at=20)
