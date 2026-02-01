"""
    Advanced Training mit Perceptual Loss, SSIM, EMA, Mixed Precision
    HTL-Grieskirchen 5. Jahrgang, Schuljahr 2025/26
    train.py
"""

import datasets
from architecture import MyModel
from utils import plot, evaluate_model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import os
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch_ema import ExponentialMovingAverage

from torch.utils.data import DataLoader
from torch.utils.data import Subset


class PerceptualLoss(nn.Module):
    """Perceptual Loss basierend auf VGG16 Features"""
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights='DEFAULT').features[:16]  # Bis relu3_3
        self.vgg = vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # ImageNet Normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    
    def normalize(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)
    
    def forward(self, pred, target):
        pred_norm = self.normalize(pred)
        target_norm = self.normalize(target)
        pred_features = self.vgg(pred_norm)
        target_features = self.vgg(target_norm)
        return F.mse_loss(pred_features, target_features)


class SSIMLoss(nn.Module):
    """SSIM Loss für bessere perceptuelle Qualität"""
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.channel = 3
        self.window = self.create_window(window_size, self.channel)
    
    def gaussian(self, window_size, sigma=1.5):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
    
    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def ssim(self, img1, img2):
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
        
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=self.channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, self.window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, self.window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, self.window, padding=self.window_size//2, groups=self.channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()
    
    def forward(self, pred, target):
        return 1 - self.ssim(pred, target)


class CombinedLoss(nn.Module):
    """MSE-only Loss für direktes Optimieren der MSE-Metrik"""
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred, target):
        # Direkt auf MSE optimieren
        mse = self.mse_loss(pred, target)
        return mse

import wandb


class SAM(torch.optim.Optimizer):
    """Sharpness Aware Minimization"""
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert isinstance(base_optimizer, type)
        self.base_optimizer = base_optimizer
        self.rho = rho
        super(SAM, self).__init__(params, dict(rho=rho, **kwargs))

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["e_w"] = p.grad * scale
                p.add_(self.state[p]["e_w"])
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def _grad_norm(self):
        norm = torch.norm(
            torch.stack([
                ((p.grad ** 2).sum()).sqrt()
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ])
        )
        return norm


def train(seed, testset_ratio, validset_ratio, data_path, results_path, early_stopping_patience, device, learningrate,
          weight_decay, n_updates, use_wandb, print_train_stats_at, print_stats_at, plot_at, validate_at, batchsize,
          network_config: dict, gradient_clip_value=1.0, use_tta=False, accumulation_steps=1, warmup_steps=0):
    
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
                                  num_workers=0, shuffle=True, pin_memory=True)
    dataloader_valid = DataLoader(dataset=dataset_valid, batch_size=1,
                                  num_workers=0, shuffle=False)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=1,
                                 num_workers=0, shuffle=False)

    network = MyModel(**network_config)
    network.to(device)
    network.train()

    # Multi-Scale Loss Function (L1 + MSE + Perceptual + SSIM)
    loss_fn = CombinedLoss()
    loss_fn.to(device)
    
    # Optimizer (AdamW mit aggressiven Settings)
    optimizer = torch.optim.AdamW(
        network.parameters(), 
        lr=learningrate, 
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # EMA (Exponential Moving Average) für bessere Modelle
    ema_model = ExponentialMovingAverage(network.parameters(), decay=0.999)
    
    # Mixed Precision Training
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Cosine Annealing mit Warm Restarts für bessere Konvergenz
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10000,  # Restart alle 10k Updates
        T_mult=2,   # Verdopple Periode nach jedem Restart
        eta_min=1e-6  # Minimale LR
    )

    if use_wandb:
        wandb.watch(network, loss_fn, log="all", log_freq=10)

    i = 0
    counter = 0
    best_validation_loss = np.inf
    loss_list = []
    accumulation_counter = 0

    # Speichere Basis LR für Warmup
    base_lr = learningrate
    warmup_factor = 1.0 if warmup_steps == 0 else 0.0

    saved_model_path = os.path.join(results_path, "best_model.pt")

    print(f"Started training on device {device}")
    print(f"Using Gated Convolutions + Transformer Blocks + CBAM + Mixed Precision + EMA + Cosine Annealing")
    print(f"Model Parameters: {sum(p.numel() for p in network.parameters()):,}")

    total_batches = len(dataloader_train)

    while i < n_updates:

        for batch_idx, (input, target) in enumerate(tqdm(dataloader_train, desc=f"Update {i}/{n_updates}")):

            input, target = input.to(device), target.to(device)

            if (i + 1) % print_train_stats_at == 0:
                print(f'Update Step {i + 1} of {n_updates}: Current loss: {loss_list[-1] if loss_list else 0:.5f}')

            # Mixed Precision Training mit Gradient Accumulation
            device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            with autocast(device_type=device_type):
                output = network(input)
                output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0)
                output = torch.clamp(output, 0.0, 1.0)
                loss = loss_fn(output, target) / accumulation_steps

            if not torch.isfinite(loss):
                print("Non-finite loss detected. Skipping step.")
                optimizer.zero_grad(set_to_none=True)
                accumulation_counter = 0
                scaler.update()
                continue

            scaler.scale(loss).backward()
            accumulation_counter += 1
            
            # Optimizer Step nach Accumulation
            if accumulation_counter == accumulation_steps:
                scaler.unscale_(optimizer)
                
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=gradient_clip_value)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Update EMA
                ema_model.update()
                
                # Warmup Phase
                if i < warmup_steps:
                    warmup_factor = i / warmup_steps
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = base_lr * warmup_factor
                
                # Cosine Annealing LR Scheduler Step
                if i >= warmup_steps:
                    scheduler.step(i - warmup_steps + (counter / len(dataloader_train)))
                
                accumulation_counter = 0

            loss_list.append((loss.item() * accumulation_steps))  # Speichere den tatsächlichen Loss

            if use_wandb and (i+1) % print_stats_at == 0:
                wandb.log({"training/loss_per_batch": loss.item()}, step=i)

            if (i + 1) % validate_at == 0:
                print(f"Plotting images, current update {i + 1}")
                plot(input.cpu().numpy(), target.detach().cpu().numpy(), output.detach().cpu().numpy(), plotpath, i)

            # Validierung nur am Ende der Epoch (wenn Balken 100% ist)
            if (batch_idx + 1) == total_batches:
                print(f"Evaluation of the model:")
                val_loss, val_rmse = evaluate_model(network, dataloader_valid, loss_fn, device)
                print(f"val_loss: {val_loss:.6f}, val_RMSE: {val_rmse:.4f}")
                
                # Current LR ausgeben
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Current LR: {current_lr:.2e}")

                if use_wandb:
                    wandb.log({"validation/loss": val_loss, "validation/RMSE": val_rmse}, step=i)

                if val_loss < best_validation_loss:
                    best_validation_loss = val_loss
                    # Speichere EMA Model
                    ema_model.store(network.parameters())
                    ema_model.copy_to(network.parameters())
                    torch.save(network.state_dict(), saved_model_path)
                    ema_model.restore(network.parameters())
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
    testset_loss, testset_rmse = evaluate_model(network=network, dataloader=dataloader_test, loss_fn=loss_fn,
                                                device=device)

    print(f'testset_loss of model: {testset_loss}, RMSE = {testset_rmse}')

    if use_wandb:
        wandb.summary["testset/loss"] = testset_loss
        wandb.summary["testset/RMSE"] = testset_rmse
        wandb.finish()