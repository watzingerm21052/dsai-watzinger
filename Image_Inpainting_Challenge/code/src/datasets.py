"""
    Author: Dein Name
    HTL-Grieskirchen 5. Jahrgang, Schuljahr 2025/26
    datasets.py
"""

import torch
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
import random

IMAGE_DIMENSION = 100

def create_arrays_from_image(image_array: np.ndarray, offset: tuple, spacing: tuple) -> tuple[np.ndarray, np.ndarray]:
    """
    Erstellt den Input und die Maske basierend auf Offset und Spacing.
    Interpretation laut Abbildung 1:
    - Die blauen Quadrate im Diagramm sind durch Offset/Spacing definiert -> Das sind die BEKANNTEN Pixel.
    - Der graue Hintergrund ist "auf 0 gesetzt" -> Das sind die FEHLENDEN Pixel.
    Wir behalten also nur ein Gitter und müssen den Rest rekonstruieren.
    """
    
    # Sicherstellen, dass wir mit numpy arrays arbeiten
    if isinstance(image_array, torch.Tensor):
        image_array = image_array.numpy()
        
    c, h, w = image_array.shape
    
    # Maske initialisieren: 0 = Pixel fehlt (Hintergrund), 1 = Pixel bekannt (Raster)
    known_array = np.zeros((1, h, w), dtype=np.float32)
    
    offset_y, offset_x = offset
    spacing_y, spacing_x = spacing
    
    # Raster setzen: An diesen Stellen ist die Information vorhanden (1)
    # Slicing: start:stop:step
    known_array[0, offset_y::spacing_y, offset_x::spacing_x] = 1.0
    
    # Das Eingabebild wird maskiert (Pixelwerte an fehlenden Stellen auf 0 setzen)
    input_array = image_array * known_array
    
    return input_array, known_array


class ImageDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading images from a folder
    """

    def __init__(self, datafolder):
        super().__init__()
        self.datafolder = datafolder
        # Suche nach Bilddateien im Ordner
        self.image_files = sorted([
            os.path.join(datafolder, f) for f in os.listdir(datafolder) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        
        # Transformationen
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_DIMENSION, IMAGE_DIMENSION)),
            # Stärkere Augmentation für bessere Generalisierung bei wenigen Datenpunkten
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(), # [0, 1] Range
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            # Fallback falls ein Bild korrupt ist
            return self.__getitem__((idx + 1) % len(self))
        
        target_tensor = self.transform(image)
        
        # Zufällige Parameter für Offset (0-8) und Spacing (2-6) wählen
        # Spacing 2-6 bedeutet wir behalten nur jeden 2. bis 6. Pixel!
        # Das Bild besteht also zu 75% bis 97% aus schwarzen Löchern.
        offset_y = random.randint(0, 8)
        offset_x = random.randint(0, 8)
        spacing_y = random.randint(2, 6)
        spacing_x = random.randint(2, 6)
        
        offset = (offset_y, offset_x)
        spacing = (spacing_y, spacing_x)
        
        masked_image_np, mask_np = create_arrays_from_image(target_tensor, offset, spacing)
        
        masked_image_tensor = torch.from_numpy(masked_image_np)
        mask_tensor = torch.from_numpy(mask_np)
        
        # Input: 4 Channels (RGB + Maske)
        input_tensor = torch.cat((masked_image_tensor, mask_tensor), dim=0)
        
        return input_tensor, target_tensor