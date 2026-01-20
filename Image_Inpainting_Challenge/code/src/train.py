"""
    Author: Your Name
    HTL-Grieskirchen 5. Jahrgang, Schuljahr 2025/26
    train.py
"""

import datasets
from architecture import MyModel
from utils import plot, evaluate_model

import torch
import numpy as np
import os

from torch.utils.data import DataLoader
from torch.utils.data import Subset

import wandb


def train(seed, testset_ratio, validset_ratio, data_path, results_path, early_stopping_patience, device, learningrate,
          weight_decay, n_updates, use_wandb, print_train_stats_at, print_stats_at, plot_at, validate_at, batchsize,
          network_config: dict):
    
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    # Benchmark an für Speed
    torch.backends.cudnn.benchmark = True

    if device is None:
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    if isinstance(device, str):
        device = torch.device(device)

    if use_wandb:
        wandb.login()
        wandb.init(project="image_inpainting", config={
            "learning_rate": learningrate,
            "weight_decay": weight_decay,
            "n_updates": n_updates,
            "batch_size": batchsize,
            "validation_ratio": validset_ratio,
            "testset_ratio": testset_ratio,
            "early_stopping_patience": early_stopping_patience,
        })

    plotpath = os.path.join(results_path, "plots")
    os.makedirs(plotpath, exist_ok=True)

    image_dataset = datasets.ImageDataset(datafolder=data_path)

    n_total = len(image_dataset)
    n_test = int(n_total * testset_ratio)
    n_valid = int(n_total * validset_ratio)
    n_train = n_total - n_test - n_valid
    indices = np.random.permutation(n_total)
    
    dataset_train = Subset(image_dataset, indices=indices[0:n_train])
    dataset_valid = Subset(image_dataset, indices=indices[n_train:n_train + n_valid])
    dataset_test = Subset(image_dataset, indices=indices[n_train + n_valid:n_total])

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batchsize,
                                  num_workers=2, shuffle=True, pin_memory=True)
    dataloader_valid = DataLoader(dataset=dataset_valid, batch_size=1,
                                  num_workers=0, shuffle=False)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=1,
                                 num_workers=0, shuffle=False)

    network = MyModel(**network_config)
    network.to(device)
    network.train()

    mse_loss = torch.nn.MSELoss()
    
    # BACK TO BASICS: Adam statt AdamW, da dies bei dir besser lief.
    optimizer = torch.optim.Adam(network.parameters(), lr=learningrate, weight_decay=weight_decay)
    
    # Scheduler: ReduceLR ist oft einfacher zu handhaben als OneCycle
    # verbose=True entfernt wegen Fehler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    if use_wandb:
        wandb.watch(network, mse_loss, log="all", log_freq=10)

    i = 0
    counter = 0
    best_validation_loss = np.inf
    loss_list = []

    saved_model_path = os.path.join(results_path, "best_model.pt")

    print(f"Started training on device {device}")

    while i < n_updates:

        for input, target in dataloader_train:

            input, target = input.to(device), target.to(device)

            if (i + 1) % print_train_stats_at == 0:
                print(f'Update Step {i + 1} of {n_updates}: Current loss: {loss_list[-1] if loss_list else 0:.5f}')

            optimizer.zero_grad()

            output = network(input)

            loss = mse_loss(output, target)

            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

            if use_wandb and (i+1) % print_stats_at == 0:
                wandb.log({"training/loss_per_batch": loss.item()}, step=i)

            if (i + 1) % plot_at == 0:
                print(f"Plotting images, current update {i + 1}")
                plot(input.cpu().numpy(), target.detach().cpu().numpy(), output.detach().cpu().numpy(), plotpath, i)

            if (i + 1) % validate_at == 0:
                print(f"Evaluation of the model:")
                val_loss, val_rmse = evaluate_model(network, dataloader_valid, mse_loss, device)
                print(f"val_loss: {val_loss:.6f}, val_RMSE: {val_rmse:.4f}")
                
                # Scheduler Step
                scheduler.step(val_loss)
                
                # Current LR ausgeben
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Current LR: {current_lr}")

                if use_wandb:
                    wandb.log({"validation/loss": val_loss, "validation/RMSE": val_rmse}, step=i)

                if val_loss < best_validation_loss:
                    best_validation_loss = val_loss
                    torch.save(network.state_dict(), saved_model_path)
                    print(f"Saved new best model with val_loss: {best_validation_loss:.6f}")
                    counter = 0
                else:
                    counter += 1

            if counter >= early_stopping_patience:
                print("Stopped training because of early stopping")
                i = n_updates
                break

            i += 1
            if i >= n_updates:
                print("Finished training because maximum number of updates reached")
                break

    print("Evaluating the self-defined testset")
    network.load_state_dict(torch.load(saved_model_path))
    testset_loss, testset_rmse = evaluate_model(network=network, dataloader=dataloader_test, loss_fn=mse_loss,
                                                device=device)

    print(f'testset_loss of model: {testset_loss}, RMSE = {testset_rmse}')

    if use_wandb:
        wandb.summary["testset/loss"] = testset_loss
        wandb.summary["testset/RMSE"] = testset_rmse
        wandb.finish()