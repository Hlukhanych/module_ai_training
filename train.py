"""
YOLO Fine-tuning Script
Simple script for fine-tuning YOLOv8
"""

from ultralytics import YOLO

# Load model (will be automatically downloaded from the internet)
model = YOLO('yolov8n.pt')

# Three lines - and done! Training on custom data
results = model.train(
    data='data/dataset.yaml',  # Path to your dataset
    epochs=10,                 # Number of epochs
    imgsz=640,                 # Image size
    device='mps',              # GPU id 0 or -1 for CPU or 'mps' for Apple Silicon
    batch=8,                   # Batch size
    patience=20,               # Early stopping
)

print("✨ Done!")
