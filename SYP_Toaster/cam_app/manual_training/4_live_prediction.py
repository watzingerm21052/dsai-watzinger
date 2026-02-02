"""
Skript 4: Live Toast-Erkennung mit trainiertem Modell
=====================================================
Nutzt das trainierte CNN für Echtzeit-Vorhersagen.

Workflow: recording.py -> 1_crop_images.py -> 2_label_images.py -> 3_train_model.py -> 4_live_prediction.py
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# ================= PFADE =================
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "toast_model.pt"

# ================= KONSTANTEN =================
IMG_SIZE = 224
CLASSES = ['roh', 'leicht', 'perfekt', 'dunkel', 'verbrannt']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================= EINSTELLUNGEN =================
BOX_WIDTH = 550
BOX_HEIGHT = 450

# Farben für Klassen (BGR)
CLASS_COLORS = {
    'roh': (200, 200, 200),      # Grau
    'leicht': (0, 255, 255),     # Gelb
    'perfekt': (0, 255, 0),      # Grün
    'dunkel': (0, 165, 255),     # Orange
    'verbrannt': (0, 0, 255)     # Rot
}

CLASS_EMOJIS = {
    'roh': '🍞 ROH',
    'leicht': '🥪 LEICHT',
    'perfekt': '✅ PERFEKT!',
    'dunkel': '🔥 DUNKEL',
    'verbrannt': '💀 VERBRANNT'
}

# ================= CNN MODELL (kopiert aus 3_train_model.py) =================
class ToastCNN(nn.Module):
    """Einfaches CNN für Toast-Klassifikation"""
    def __init__(self, num_classes=5):
        super(ToastCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# ================= MODELL LADEN =================
def load_model():
    """Lädt das trainierte Modell"""
    if not MODEL_PATH.exists():
        print(f"❌ Modell nicht gefunden: {MODEL_PATH}")
        print("   Bitte zuerst Skript 3 (3_train_model.py) ausführen!")
        return None
    
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = ToastCNN(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Modell geladen (Accuracy: {checkpoint['accuracy']:.1f}%)")
    return model

# ================= KAMERA =================
def init_camera():
    """Initialisiert die Kamera"""
    print("Suche Kamera...")
    for idx in [1, 0, 2]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                print(f"✅ Kamera {idx} bereit!")
                return cap
        cap.release()
    print("❌ Keine Kamera gefunden!")
    return None

# ================= VORHERSAGE =================
def predict(model, roi):
    """Macht Vorhersage für ROI"""
    # Bild vorbereiten
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Vorhersage
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item() * 100
    
    return CLASSES[pred_idx], confidence, probs[0].cpu().numpy()

# ================= HAUPTPROGRAMM =================
def main():
    print("="*50)
    print("🍞 LIVE TOAST-ERKENNUNG")
    print("="*50)
    
    # Modell laden
    model = load_model()
    if model is None:
        return
    
    # Kamera starten
    cap = init_camera()
    if cap is None:
        return
    
    print("\nSteuerung:")
    print(" [ESC] - Beenden")
    print("="*50)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        
        # ROI berechnen (leicht nach unten versetzt)
        center_x, center_y = w // 2, h // 2 + 50
        x1 = center_x - BOX_WIDTH // 2
        y1 = center_y - BOX_HEIGHT // 2
        x2 = center_x + BOX_WIDTH // 2
        y2 = center_y + BOX_HEIGHT // 2
        
        # ROI extrahieren und Vorhersage machen
        roi = frame[y1:y2, x1:x2]
        pred_class, confidence, all_probs = predict(model, roi)
        
        # Farbe basierend auf Vorhersage
        color = CLASS_COLORS[pred_class]
        
        # ROI-Rahmen zeichnen
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Vorhersage anzeigen
        label = CLASS_EMOJIS[pred_class]
        cv2.putText(frame, f"{label}", (x1, y1 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        cv2.putText(frame, f"Konfidenz: {confidence:.1f}%", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Balkendiagramm für alle Klassen
        bar_x = x2 + 30
        bar_width = 150
        bar_height = 25
        
        cv2.putText(frame, "Vorhersage:", (bar_x, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        for i, (cls, prob) in enumerate(zip(CLASSES, all_probs)):
            y_pos = y1 + 20 + i * (bar_height + 10)
            
            # Hintergrund
            cv2.rectangle(frame, (bar_x, y_pos), 
                         (bar_x + bar_width, y_pos + bar_height),
                         (50, 50, 50), -1)
            
            # Füllbalken
            fill_width = int(prob * bar_width)
            cv2.rectangle(frame, (bar_x, y_pos),
                         (bar_x + fill_width, y_pos + bar_height),
                         CLASS_COLORS[cls], -1)
            
            # Label
            cv2.putText(frame, f"{cls}: {prob*100:.0f}%", 
                       (bar_x + 5, y_pos + 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Warnung bei perfektem Toast
        if pred_class == 'perfekt' and confidence > 70:
            cv2.putText(frame, ">>> TOAST FERTIG! <<<", (w//2 - 150, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Warnung bei verbrannt
        if pred_class in ['dunkel', 'verbrannt'] and confidence > 60:
            cv2.putText(frame, "!!! ACHTUNG - ZU DUNKEL !!!", (w//2 - 200, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        cv2.imshow("Toast AI - Live Erkennung", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Beendet.")

if __name__ == "__main__":
    main()
