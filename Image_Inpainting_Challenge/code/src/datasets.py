"""
    Author: Dein Name
    HTL-Grieskirchen 5. Jahrgang, Schuljahr 2025/26
    datasets.py
"""

import torch
import numpy as np
import os
import glob
from PIL import Image
import torchvision.transforms.functional as TF
import random

IMAGE_DIMENSION = 100

def resize(img: Image.Image) -> Image.Image:
    """Resized das Bild auf die Zielgröße (100x100)."""
    return img.resize((IMAGE_DIMENSION, IMAGE_DIMENSION), Image.BILINEAR)

def preprocess(img: Image.Image) -> torch.Tensor:
    """
    Konvertiert das PIL Image in einen Tensor und normalisiert auf [0, 1].
    Hier können auch Augmentations eingefügt werden.
    """
    # Random Flip für Data Augmentation (macht das Modell robuster)
    if random.random() > 0.5:
        img = TF.hflip(img)
    if random.random() > 0.5:
        img = TF.vflip(img)
        
    return TF.to_tensor(img)

def create_arrays_from_image(image_tensor: torch.Tensor, offset: tuple, spacing: tuple) -> tuple[np.ndarray, np.ndarray]:
    """
    Erstellt den sparse Input und die Maske basierend auf Offset und Spacing.
    
    Logik laut PDF (Abbildung 1):
    - Das Raster (definiert durch offset/spacing) sind die BEKANNTEN Pixel (Blau).
    - Der Rest wird auf 0 gesetzt (Grau/Fehlt).
    """
    # Tensor zu Numpy konvertieren für die Masken-Erstellung
    image_array = image_tensor.numpy()
    c, h, w = image_array.shape
    
    # Maske initialisieren: 0 = Pixel fehlt (Hintergrund), 1 = Pixel bekannt (Raster)
    # Wir starten mit einem schwarzen Blatt.
    known_array = np.zeros((1, h, w), dtype=np.float32)
    
    offset_y, offset_x = offset
    spacing_y, spacing_x = spacing
    
    # Raster setzen: Wir markieren die Stellen, die wir "sehen" dürfen, mit 1.
    # Slicing: start:stop:step
    known_array[0, offset_y::spacing_y, offset_x::spacing_x] = 1.0
    
    # Bild maskieren: Wir behalten nur die Pixel an den Raster-Positionen.
    # Der Rest wird mathematisch auf 0 gezwungen.
    input_array = image_array * known_array
    
    return input_array, known_array


class ImageDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading images from a folder using glob
    """

    def __init__(self, datafolder: str):
        super().__init__()
        self.datafolder = datafolder
        
        # Rekursive Suche nach JPGs (Struktur vom Lehrer)
        # glob.glob gibt eine Liste von Pfaden zurück, die dem Muster entsprechen
        search_pattern = os.path.join(datafolder, "**", "*.jpg")
        self.imagefiles = sorted(glob.glob(search_pattern, recursive=True))
        
        # Fallback, falls Datensatz PNGs enthält (z.B. CIFAR oder ähnliche)
        if len(self.imagefiles) == 0:
             search_pattern_png = os.path.join(datafolder, "**", "*.png")
             self.imagefiles = sorted(glob.glob(search_pattern_png, recursive=True))

    def __len__(self):
        return len(self.imagefiles)

    def __getitem__(self, idx: int):
        img_path = self.imagefiles[idx]
        
        try:
            # Bild laden und sicherstellen, dass es RGB ist (3 Channel)
            image = Image.open(img_path).convert("RGB")
            
            # 1. Resize
            image = resize(image)
            
            # 2. Preprocess (Augmentation + ToTensor)
            target_tensor = preprocess(image)
            
        except Exception as e:
            print(f"Fehler beim Laden von {img_path}: {e}")
            # Bei Fehler einfach das nächste Bild nehmen
            return self.__getitem__((idx + 1) % len(self))
        
        # Zufälliges Gitter laut Angabe
        # offset 0-8, spacing 2-6
        offset = (random.randint(0, 8), random.randint(0, 8))
        spacing = (random.randint(2, 6), random.randint(2, 6))
        
        masked_image_np, mask_np = create_arrays_from_image(target_tensor, offset, spacing)
        
        masked_image_tensor = torch.from_numpy(masked_image_np)
        mask_tensor = torch.from_numpy(mask_np)
        
        # 4 Channel Input: Maskiertes Bild + Maske
        input_tensor = torch.cat((masked_image_tensor, mask_tensor), dim=0)
        
        return input_tensor, target_tensor