# Advanced Image Inpainting Model - Improvements

## 🚀 State-of-the-Art Implementationen

### Architecture Improvements

#### 1. **Residual Dense Blocks (RDB)**
- Kombiniert ResNet + DenseNet Ideen
- Dense Connections zwischen Schichten ermöglichen effizienteres Lernen
- Weighted Residuals (0.2) für stabile Gradienten

#### 2. **Convolutional Block Attention Module (CBAM)**
- **Channel Attention**: Wichtige Features priorisieren
- **Spatial Attention**: Räumliche Fokussierung auf relevante Regionen
- Kombiniert für optimale Feature Auswahl

#### 3. **Partial Convolution**
- Speziell für Inpainting Tasks entwickelt
- Berücksichtigt nur bekannte Pixel (Maske-aware)
- Verhindert Artefakte an Rändern unbekannter Regionen

#### 4. **Skip Connections mit Dekoder**
- Encoder-Decoder Architektur mit U-Net inspiriertem Design
- Preserviert Details von allen Skalen
- Bottleneck mit Extra-RDBs für tiefere Features

---

## 🎯 Training Optimierungen

### 1. **Sharpness Aware Minimization (SAM)**
- State-of-the-art Optimizer
- Findet flacheren Minima für bessere Generalisierung
- Zwei-Schritt Update: Gradient ascent + descent
- **Resultat**: Bessere Generalisierung, robustere Modelle

### 2. **Exponential Moving Average (EMA)**
- Speichert exponentiellen Durchschnitt der Modell-Gewichte
- Bessere Stabilität während Training
- Oft bessere Testset Performance
- Erhöht Robustheit gegen Overfitting

### 3. **Mixed Precision Training**
- Verwendet FP16 für schnellere Berechnung
- Behält FP32 für wichtige Operationen
- **Vorteile**: 2-3x schneller, weniger GPU Memory, keine Genauigkeitsverluste
- `GradScaler` verhindert Gradient Underflow

### 4. **Advanced Learning Rate Scheduling**
- **Warmup Phase** (1000 Steps): Sanfter Start
- **Linear Decay**: Graduelles Senken der LR über Zeit
- **ReduceLROnPlateau**: Zusätzlicher Decay bei Plateau
- Kombination für optimale Konvergenz

---

## 📊 Daten & Augmentation

### Albumentations Pipeline
- **Geometric**: Rotation, HFlip, VFlip
- **Color**: Brightness/Contrast, Gaussian Noise, Blur
- **Advanced**: ColorJitter, Downscaling
- **Normalization**: ImageNet Standard (mean, std)
- **Resultat**: Robusteres Modell, bessere Generalisierung

---

## 🛠️ Loss Function

### Kombinierte Loss-Funktion
```
Loss = 0.7 * HuberLoss + 0.3 * MSELoss
```

**Warum diese Kombination?**
- **HuberLoss**: Robust gegen Outliers, smooth Gradienten
- **MSELoss**: Präzise bei kleinen Fehlern
- **70/30 Split**: HuberLoss dominiert, MSE für Feinheiten

---

## 📈 Hyperparameter Settings

| Parameter | Wert | Begründung |
|-----------|------|-----------|
| Learning Rate | 5e-5 | SAM erlaubt kleinere LR für feineres Lernen |
| Batch Size | 12 | Klein für bessere SAM Gradienten |
| Weight Decay | 1e-4 | Moderate Regularisierung |
| Optimizer | SAM + AdamW | Best-of-both-worlds |
| n_updates | 150000 | Länger trainieren für Konvergenz |
| Warmup Steps | 1000 | Sanfter Start |

---

## 💾 Model Architecture Summary

```
Input (4 channels: RGB + Mask)
    ↓
Initial Conv (64 channels)
    ↓
Encoder Stage 1: PartialConv → RDB → CBAM → MaxPool (128 ch)
    ↓
Encoder Stage 2: PartialConv → RDB → CBAM → MaxPool (256 ch)
    ↓
Encoder Stage 3: PartialConv → RDB → CBAM → MaxPool (512 ch)
    ↓
Bottleneck: 2x RDB + CBAM (512 ch)
    ↓
Decoder Stage 3: Upsample + RDB + CBAM (256 ch)
    ↓
Decoder Stage 2: Upsample + RDB + CBAM (128 ch)
    ↓
Decoder Stage 1: Upsample + RDB + CBAM (64 ch)
    ↓
Output Conv: 32ch → 3 channels (RGB)
    ↓
Sigmoid: [0, 1] range
```

---

## 📦 Requirements

```
torch>=2.0.0
torchvision>=0.15.0
albumentations>=1.3.0
torch-ema>=0.3
numpy>=1.24.0
Pillow>=9.0.0
matplotlib>=3.7.0
opencv-python>=4.7.0
tqdm>=4.65.0
wandb>=0.14.0
kornia>=0.7.0
```

---

## 🎓 Verwendete State-of-the-Art Techniken

1. ✅ **SAM Optimizer** - Moderner Optimizer für bessere Generalisierung
2. ✅ **EMA** - Robusteres Training
3. ✅ **Mixed Precision** - Schnelleres Training
4. ✅ **Partial Convolution** - Inpainting-spezifisch
5. ✅ **Residual Dense Blocks** - Effizientere Features
6. ✅ **Channel + Spatial Attention** - Intelligente Feature-Auswahl
7. ✅ **Albumentations** - Advanced Augmentation
8. ✅ **Warmup + Decay Scheduling** - Optimale Konvergenz
9. ✅ **Kombinierte Loss-Funktion** - Robustheit gegen Outliers
10. ✅ **Instance Normalization** - Besser für Inpainting

---

## 🎉 Erwartete Verbesserungen

- **MSE/RMSE**: 30-50% Reduktion durch SAM + EMA + bessere Architecture
- **Konvergenzgeschwindigkeit**: ~2-3x schneller durch Mixed Precision
- **Generalisierung**: Deutlich besser durch Partial Conv + Attention
- **Stabilität**: Robuster durch kombinierte Loss-Funktion

