"""
Erstellt Predictions vom gespeicherten best_model.pt
Kann jederzeit nach Training-Abbruch ausgeführt werden
"""

import os
import torch
from architecture import MyModel
from utils import create_predictions

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.dirname(script_dir)
    
    results_path = os.path.join(code_dir, "results_nirwana")
    
    network_config = {
        'n_in_channels': 4
    }
    
    challenge_dir = os.path.dirname(code_dir)
    testset_path = os.path.join(challenge_dir, "code", "data", "challenge_testset.npz")
    state_dict_path = os.path.join(results_path, "best_model.pt")
    save_path = os.path.join(results_path, "testset", "my_submission_name.npz")
    plot_path = os.path.join(results_path, "testset", "plots")

    # Prüfe ob Modell existiert
    if os.path.exists(state_dict_path):
        print("=" * 50)
        print("✅ BEST MODEL LOADED")
        print("=" * 50)
        print(f"Model path: {state_dict_path}")
        print(f"File size:  {os.path.getsize(state_dict_path) / 1e6:.1f} MB")
        print()
        print("⚠️ Check your training output for:")
        print("   - Validation RMSE (val_RMSE)")
        print("   - Best epoch")
        print("=" * 50)
        print()

    print(f"Loading model from: {state_dict_path}")
    create_predictions(network_config, state_dict_path, testset_path, None, save_path, plot_path, 
                      plot_at=20, use_tta=True)
    print(f"Predictions saved to: {save_path}")
