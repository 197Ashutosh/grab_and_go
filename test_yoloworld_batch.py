import os
from pathlib import Path
from ultralytics import YOLOWorld

m = YOLOWorld(r'c:\Users\197as\Documents\Document\7th SEM\Final Year Project\ai_vision_engine\yolov8s-world.pt')
m.set_classes(['chips packet', 'lays', 'potato chips', 'biscuit packet', 'cold drink', 'bottle', 'chocolate', 'cell phone', 'instant noodles', 'snack packet'])

raw_dir = Path(r'c:\Users\197as\Documents\Document\7th SEM\Final Year Project\ai_vision_engine\custom_dataset\raw_images')
detected_count = 0
for img_p in list(raw_dir.glob('*.jpg'))[:20]:
    res = m.predict(str(img_p), conf=0.10, verbose=False)[0]
    if len(res.boxes) > 0:
        detected_count += 1
        for b in res.boxes:
            print(f"{img_p.name}: {m.names[int(b.cls[0])]} ({float(b.conf[0]):.2f})")

print(f"Total images with detections: {detected_count}/20")
