import cv2
import pickle
import time
import os
from pathlib import Path

# ================= EINSTELLUNGEN =================
# Muss exakt zum Hauptprogramm passen!
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "auto_training_data"
OUTPUT_FILE = OUTPUT_DIR / "training_data.pkl"
IMAGE_SIZE = (224, 224) 

# Die Klassen (Tasten 0-4)
CLASSES = {
    0: "ROH (Weissbrot)",
    1: "LEICHT (Hell)",
    2: "PERFEKT (Gold)",
    3: "DUNKEL (Braun)",
    4: "VERBRANNT (Schwarz)"
}

def manual_data_collection():
    # Ordner erstellen, falls nicht existiert
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)

    # Kamera Setup (sucht automatisch nach der richtigen Cam)
    cap = None
    # Wir testen Port 1 (extern), dann 0 (intern), dann 2
    for i in [1, 0, 2]:
        # DSHOW ist unter Windows schneller/stabiler
        temp = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if temp.isOpened():
            ret, _ = temp.read()
            if ret:
                cap = temp
                # Auflösung hochsetzen für gute Vorschau
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) # Autofokus aus (wichtig bei Toaster!)
                print(f"✅ Kamera {i} gefunden und verbunden.")
                break
        temp.release()
    
    if not cap:
        print("❌ FEHLER: Keine Kamera gefunden!")
        return

    print("\n" + "="*60)
    print("🎓 MANUELLER LEHRER-MODUS")
    print("="*60)
    print("ANLEITUNG:")
    print("1. Leertaste drücken zum Starten der Aufnahme")
    print("2. Toast beobachten")
    print("3. Tasten 0-4 drücken, um den aktuellen Zustand zu setzen")
    print("-" * 30)
    for k, v in CLASSES.items():
        print(f" Taste [{k}] -> {v}")
    print("-" * 30)
    print(" [S]     -> Speichern und Beenden")
    print(" [Q]     -> Abbrechen ohne Speichern")
    print("="*60)

    samples = []
    current_label = 0 # Startet standardmäßig mit "Roh"
    recording = False
    start_time = 0
    last_save = 0
    
    # Alte Daten laden (damit man mehrere Toasts nacheinander aufnehmen kann)
    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE, 'rb') as f:
                samples = pickle.load(f)
            print(f"ℹ️ {len(samples)} existierende Bilder geladen. Neue werden angehängt.")
        except Exception as e:
            print(f"⚠️ Warnung: Konnte alte Daten nicht laden ({e}). Fange neu an.")
            samples = []

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Vorschau-Bild kopieren (damit Text nicht im Trainingsbild landet)
        display = frame.copy()
        
        # --- AUFNAHME LOGIK ---
        if recording:
            now = time.time()
            # Wir speichern 2 Bilder pro Sekunde (alle 0.5s)
            if now - last_save > 0.5: 
                # WICHTIG: Nur Grayscale speichern (robuster gegen Belichtung!)
                small_frame = cv2.resize(frame, IMAGE_SIZE)
                small_frame_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                elapsed = now - start_time
                
                # Format: (Grayscale, Label, Timestamp)
                samples.append((small_frame_gray, current_label, elapsed))
                
                last_save = now
                print(f"📸 Gelernt: {CLASSES[current_label]} (S/W) (Total: {len(samples)})")
            
            # Roter Punkt blinkend
            if int(now * 2) % 2 == 0:
                cv2.circle(display, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(display, "REC", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(display, "PAUSE (Leertaste zum Starten)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # --- GUI ANZEIGE ---
        # Schwarzer Balken unten für Text
        h, w = display.shape[:2]
        cv2.rectangle(display, (0, h-80), (w, h), (0, 0, 0), -1)
        
        # Aktuelles Label anzeigen
        label_text = f"LERNEN: {CLASSES[current_label]} (Taste {current_label})"
        cv2.putText(display, label_text, (20, h-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Hilfe-Text
        info_text = f"Samples: {len(samples)} | [0]=Roh [1]=Leicht [2]=Perfekt [3]=Dunkel [4]=Verbrannt"
        cv2.putText(display, info_text, (20, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Grayscale Version für parallele Anzeige
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_display = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)  # Zurück zu 3 Kanälen für Kompatibilität
        
        # Schwarzer Balken für S/W Version
        cv2.rectangle(gray_display, (0, h-80), (w, h), (0, 0, 0), -1)
        cv2.putText(gray_display, label_text, (20, h-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(gray_display, info_text, (20, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        if recording:
            now = time.time()
            if int(now * 2) % 2 == 0:
                cv2.circle(gray_display, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(gray_display, "REC", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(gray_display, "PAUSE (Leertaste zum Starten)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Lehrer Modus - FARBE", display)
        cv2.imshow("Lehrer Modus - S/W", gray_display)

        # --- TASTENSTEUERUNG ---
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '): # Leertaste: Start/Stop
            recording = not recording
            if recording:
                start_time = time.time() # Reset Timer bei neuem Start
                print("▶️ Aufnahme gestartet!")
            else:
                print("⏸️ Pausiert.")
        
        # Label Tasten 0-4
        elif key in [ord('0'), ord('1'), ord('2'), ord('3'), ord('4')]:
            current_label = int(chr(key))
            print(f"👉 Label geändert auf: {CLASSES[current_label]}")
            
        elif key == ord('s'): # Speichern
            if len(samples) == 0:
                print("⚠️ Keine Daten zum Speichern!")
                continue
                
            print(f"\n💾 Speichere {len(samples)} Datensätze in:")
            print(f"   {OUTPUT_FILE}")
            
            # Ordner sicherstellen
            if not OUTPUT_DIR.exists(): OUTPUT_DIR.mkdir(parents=True)
            
            with open(OUTPUT_FILE, 'wb') as f:
                pickle.dump(samples, f)
            
            print("✅ GESPEICHERT! Du kannst jetzt 'pid_toaster_control.py' starten und trainieren.")
            break
            
        elif key == ord('q'): # Quit ohne Speichern
            print("❌ Abbruch ohne Speichern.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    manual_data_collection()