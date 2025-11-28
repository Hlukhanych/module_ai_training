"""
Inference script with advanced image preprocessing
Includes noise reduction, Gaussian blur, and contrast enhancement
"""

from ultralytics import YOLO
import cv2
import numpy as np

def preprocess_image(image_path):
    """
    Apply image preprocessing to improve detection quality

    Steps:
    1. Load image
    2. Denoise (remove noise)
    3. Gaussian blur (smooth image)
    4. Contrast enhancement (CLAHE)
    5. Sharpening (optional)

    Args:
        image_path: Path to input image

    Returns:
        Preprocessed image
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    print("🔄 Preprocessing image...")
    print("   Step 1: Denoising (removing noise)...")
    # Apply Non-local Means Denoising to remove noise
    # h=10: Filter strength (higher = more denoising)
    denoised = cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10,
                                                templateWindowSize=7, searchWindowSize=21)

    print("   Step 2: Applying Gaussian blur...")
    # Apply Gaussian blur to smooth the image
    # (5, 5): Kernel size, 0: sigma (auto-calculated)
    blurred = cv2.GaussianBlur(denoised, (5, 5), 0)

    print("   Step 3: Enhancing contrast (CLAHE)...")
    # Convert to LAB color space for better contrast enhancement
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Merge channels and convert back to BGR
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    print("   Step 4: Sharpening image...")
    # Create sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    # Save preprocessed image for comparison
    cv2.imwrite('preprocessed_image.jpg', sharpened)
    print("   ✅ Preprocessed image saved as 'preprocessed_image.jpg'")

    return sharpened

# Load model
model = YOLO('runs/detect/train/weights/best.pt')

# Path to image
image_path = 'image.jpg'

# Preprocess image
preprocessed_img = preprocess_image(image_path)

# Run inference on preprocessed image
print("\n🚀 Running detection on preprocessed image...")
results = model.predict(preprocessed_img, conf=0.25, save=True)

# Display results
for result in results:
    print(f"\n✅ Objects found: {len(result.boxes)}")

    # Print information for each detected object
    for i, box in enumerate(result.boxes):
        cls_name = result.names[int(box.cls)]
        confidence = float(box.conf)
        coords = box.xyxy[0].cpu().numpy()
        print(f"   [{i+1}] {cls_name}")
        print(f"       Confidence: {confidence:.2%}")
        print(f"       Coordinates: [{coords[0]:.0f}, {coords[1]:.0f}, {coords[2]:.0f}, {coords[3]:.0f}]")

    # Save result with bounding boxes
    if len(result.boxes) > 0:
        result_image = result.plot()
        cv2.imwrite('prediction_result.jpg', result_image)
        print(f"\n💾 Result saved to 'prediction_result.jpg'")
    else:
        print("\n❌ No objects detected")
