# 🍽️ YOLO Food Detection

Object detection system for food items using YOLOv8 with advanced image preprocessing.

## 📦 Dataset

**Source:** [Roboflow - Food Detection Dataset](https://universe.roboflow.com/ahmad-nabil/food-detection-for-yolo-training/dataset/2)

Contains labeled images of various dishes split into train/valid/test sets.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python train.py

# Run prediction with preprocessing
python predict.py
```

## 📁 Project Structure

```
├── data/                    # Dataset (train/valid/test)
├── train.py                 # Model training
├── predict.py               # Inference with preprocessing
└── runs/                    # Training results and weights
```

## 🖼️ Image Preprocessing

The project includes advanced preprocessing to improve detection:

1. **Noise Reduction** - Removes image noise
2. **Gaussian Blur** - Smooths the image
3. **CLAHE** - Enhances contrast
4. **Sharpening** - Improves edge definition

## 📊 Usage

### Training
```bash
python train.py
```
Model saved to `runs/detect/train/weights/best.pt`

### Prediction
```bash
python predict.py
```
Results:
- `preprocessed_image.jpg` - Processed input
- `prediction_result.jpg` - Detection results

## 🛠️ Configuration

Edit `train.py` to adjust training parameters:
```python
epochs=10           # Training epochs
batch=8            # Batch size
device='mps'       # 'mps' for Mac, 0 for GPU, 'cpu' for CPU
```

Edit `predict.py` to adjust preprocessing:
```python
h=10               # Denoising strength
clipLimit=2.0      # CLAHE contrast
```

## 📈 Metrics

- **mAP50** - Accuracy at IoU threshold 0.5
- **mAP50-95** - Average accuracy across IoU thresholds
- **Precision** - Correct detections ratio
- **Recall** - Found objects ratio

## 🐛 Troubleshooting

**Out of memory:** Reduce batch size in `train.py`
**Missing dependencies:** Run `pip install -r requirements.txt`
**No GPU:** Model automatically uses CPU (slower but works)

## 📄 License

Dataset: CC BY 4.0

---

Built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
