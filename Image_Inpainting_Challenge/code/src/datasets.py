"""
    Author: Matthias Watzinger
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
    Erstellt das maskierte Eingabebild und die Maske selbst basierend auf Offset und Spacing.
    
    Args:
        image_array: Das Originalbild (C, H, W) oder (H, W, C). Wir erwarten hier (C, H, W) nach ToTensor().
        offset: Tupel (offset_y, offset_x)
        spacing: Tupel (spacing_y, spacing_x)
    Returns:
        image_array: Das maskierte Bild (Pixel auf 0 gesetzt wo Maske 0 ist).
        known_array: Die Maske (1 wo Pixel bekannt sind, 0 wo sie fehlen).
    """
    
    # Sicherstellen, dass wir mit numpy arrays arbeiten
    if isinstance(image_array, torch.Tensor):
        image_array = image_array.numpy()
        
    c, h, w = image_array.shape
    
    # Maske initialisieren: 1 = Pixel bekannt, 0 = Pixel fehlt
    known_array = np.ones((1, h, w), dtype=np.float32)
    
    offset_y, offset_x = offset
    spacing_y, spacing_x = spacing
    
    # Raster anwenden: Setze 0 an den Stellen, die durch Offset und Spacing definiert sind
    # Slicing Notation: start:stop:step
    # Wir setzen die Stellen auf 0, wo das Gitter "trifft".
    # Die Aufgabenstellung sagt: "Bild, mit grauen Pixeln auf 0 gesetzt".
    # Das bedeutet, an den Rasterpunkten fehlen die Informationen.
    
    # Achtung: Numpy Slicing erlaubt direkte Zuweisung auf das Gitter
    known_array[0, offset_y::spacing_y, offset_x::spacing_x] = 0.0
    
    # Das Eingabebild wird maskiert (Pixelwerte an fehlenden Stellen auf 0 setzen)
    masked_image_array = image_array * known_array
    
    return masked_image_array, known_array


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
        
        # Transformationen: Resize und Tensor-Konvertierung
        # Data Augmentation (zufälliges Flippen) hilft dem Modell, besser zu generalisieren
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_DIMENSION, IMAGE_DIMENSION)),
            transforms.RandomHorizontalFlip(p=0.5), # Augmentation
            transforms.RandomVerticalFlip(p=0.5),   # Augmentation
            transforms.ToTensor(), # Konvertiert Bild zu (C, H, W) und Wertebereich [0, 1]
        ])
        
        # Validierung/Test Transformation (ohne Random Flip) falls nötig, 
        # aber hier für Einfachheit im Training inkludiert.
        # Für striktere Trennung könnte man separate Transforms übergeben.

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # Bild laden und in RGB konvertieren (um Fehler bei RGBA/Graustufen zu vermeiden)
        image = Image.open(img_path).convert("RGB")
        
        # Transformationen anwenden (gibt Tensor (3, 100, 100) zurück)
        target_tensor = self.transform(image)
        
        # Zufällige Parameter für Offset (0-8) und Spacing (2-6) wählen
        # Aufgabenstellung: "ganzzahlig zwischen 0 und 8" ->randint(0, 8) ist inklusiv
        offset_y = random.randint(0, 8)
        offset_x = random.randint(0, 8)
        spacing_y = random.randint(2, 6)
        spacing_x = random.randint(2, 6)
        
        offset = (offset_y, offset_x)
        spacing = (spacing_y, spacing_x)
        
        # Maskierung anwenden
        # image_array ist hier der target_tensor als numpy array
        masked_image_np, mask_np = create_arrays_from_image(target_tensor, offset, spacing)
        
        # Zurück zu Tensor konvertieren
        masked_image_tensor = torch.from_numpy(masked_image_np)
        mask_tensor = torch.from_numpy(mask_np)
        
        # Input für das Netz: Concatenation von Maskiertem Bild (3 Channel) und Maske (1 Channel)
        # Ergebnis Shape: (4, 100, 100)
        input_tensor = torch.cat((masked_image_tensor, mask_tensor), dim=0)
        
        return input_tensor, target_tensor