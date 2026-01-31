"""
    Datasets mit Advanced Augmentation
    HTL-Grieskirchen 5. Jahrgang, Schuljahr 2025/26
    datasets.py
"""

import torch
import numpy as np
import os
import glob
from PIL import Image
import torchvision.transforms as T
import random

IMAGE_DIMENSION = 100


def get_augmentation_pipeline():
    """Leichte Augmentation für bessere MSE-Konvergenz"""
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),  # Konvertiert zu [0, 1] Tensor
    ])

def create_arrays_from_image(image_tensor: torch.Tensor, offset: tuple, spacing: tuple) -> tuple[np.ndarray, np.ndarray]:
    """
    Erstellt den Input und die Maske.
    WICHTIG: Raster (blau) = 1, Hintergrund (grau) = 0.
    """
    image_array = image_tensor.numpy()
    c, h, w = image_array.shape
    
    # Maske initialisieren: 0 = Pixel fehlt
    known_array = np.zeros((1, h, w), dtype=np.float32)
    
    offset_y, offset_x = offset
    spacing_y, spacing_x = spacing
    
    # Raster setzen: 1 = Pixel bekannt
    known_array[0, offset_y::spacing_y, offset_x::spacing_x] = 1.0
    
    # Bild maskieren (alles außer Raster wird 0)
    input_array = image_array * known_array
    
    return input_array, known_array


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, datafolder: str):
        super().__init__()
        self.datafolder = datafolder
        
        # Suche DIREKT im Ordner (nicht rekursiv in Unterordnern)
        search_pattern = os.path.join(datafolder, "*.jpg")
        self.imagefiles = sorted(glob.glob(search_pattern))
        
        if len(self.imagefiles) == 0:
             search_pattern_png = os.path.join(datafolder, "*.png")
             self.imagefiles = sorted(glob.glob(search_pattern_png))
        
        print(f"Found {len(self.imagefiles)} images in {datafolder}")
        
        self.augmentation = get_augmentation_pipeline()

    def __len__(self):
        return len(self.imagefiles)

    def __getitem__(self, idx: int):
        img_path = self.imagefiles[idx]
        
        try:
            image_pil = Image.open(img_path).convert("RGB")
            image_pil = image_pil.resize((IMAGE_DIMENSION, IMAGE_DIMENSION), Image.BILINEAR)
            
            # Augmentations - torchvision transforms erwarten PIL Image und geben Tensor zurück
            target_tensor = self.augmentation(image_pil)
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))
        
        # Zufälliges Gitter
        offset = (random.randint(0, 8), random.randint(0, 8))
        spacing = (random.randint(2, 6), random.randint(2, 6))
        
        masked_image_np, mask_np = create_arrays_from_image(target_tensor, offset, spacing)
        
        masked_image_tensor = torch.from_numpy(masked_image_np)
        mask_tensor = torch.from_numpy(mask_np)
        
        # 4 Channel Input: Maskiertes Bild + Maske
        input_tensor = torch.cat((masked_image_tensor, mask_tensor), dim=0)
        
        return input_tensor, target_tensor