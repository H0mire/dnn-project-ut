# Ultrasound Organ Classification

This project implements a neural network-based classification system for ultrasound images, specifically designed to classify bladder and kidney images with limited training data (50 images per class).

## Approach

### Feature Extraction Strategy

Given the constraint of only 50 images per class and no pretrained models allowed, I implemented a dual approach:

1. **Handcrafted Features** (287 features total):
   - **Texture Features (60 features)**: Gray Level Co-occurrence Matrix (GLCM) properties including contrast, dissimilarity, homogeneity, energy, and correlation at multiple distances and angles
   - **Intensity Features (23 features)**: Statistical measures (mean, std, median, percentiles) and histogram features
   - **Shape Features (5 features)**: Region properties including area, eccentricity, solidity, and extent

2. **Shallow CNN Architecture**:
   - Designed specifically for small datasets with regularization
   - 3 convolutional blocks with batch normalization and dropout
   - Adaptive pooling to handle varying input sizes
   - Feature extraction capability for hybrid approaches

### Data Augmentation

To address the limited data:
- Random rotation (±10 degrees)
- Random horizontal flipping
- Random affine transformations
- Color jittering for brightness and contrast

### Model Architecture

The `ShallowCNN` consists of:
- Conv Block 1: 16 filters (5x5 kernel) → BatchNorm → ReLU → MaxPool
- Conv Block 2: 32 filters (3x3 kernel) → BatchNorm → ReLU → MaxPool
- Conv Block 3: 64 filters (3x3 kernel) → BatchNorm → ReLU → AdaptiveAvgPool
- Classifier: Dropout(0.5) → FC(1024→128) → ReLU → Dropout(0.5) → FC(128→2)

### Training Strategy

- Train/Validation split: 80/20 with stratification
- Batch size: 8 (suitable for small dataset)
- Optimizer: Adam with weight decay (1e-4)
- Learning rate scheduler: ReduceLROnPlateau
- Epochs: 100 with early stopping potential

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

```bash
python ultrasound_classifier.py
```

This will:
1. Load the ultrasound images from `./datasets/img/`
2. Split data into train/validation sets
3. Train the CNN model with augmentation
4. Evaluate on the test set
5. Save the trained model as `ultrasound_classifier.pth`
6. Generate visualizations:
   - `training_history.png`: Loss and accuracy curves
   - `confusion_matrix.png`: Classification performance

### Using the Trained Model

```python
from ultrasound_classifier import ShallowCNN
import torch
import torchvision.transforms as transforms
from PIL import Image

# Load model
model = ShallowCNN(num_classes=2)
model.load_state_dict(torch.load('ultrasound_classifier.pth'))
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Predict
image = Image.open('path_to_ultrasound_image.png').convert('RGB')
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(image_tensor)
    _, predicted = torch.max(output, 1)
    
class_names = ['bladder', 'kidney']
print(f"Predicted: {class_names[predicted.item()]}")
```

### Extracting Features

```python
from ultrasound_classifier import HandcraftedFeatureExtractor
import cv2

extractor = HandcraftedFeatureExtractor()
image = cv2.imread('path_to_image.png')
image = cv2.resize(image, (128, 128))
image = image.astype(np.float32) / 255.0

features = extractor.extract_all_features(image)
print(f"Extracted {len(features)} features")
```

## Results

The model achieves binary classification between bladder and kidney ultrasound images. Performance metrics including accuracy, precision, recall, and F1-score are displayed after training, along with a confusion matrix visualization.

## Future Improvements

1. **Hybrid Model**: Combine CNN features with handcrafted features using the `HybridClassifier` class
2. **Multi-class Extension**: Extend to classify all 6 organs (bowel, bladder, gallbladder, kidney, spleen, liver)
3. **Ensemble Methods**: Combine multiple models for better performance
4. **Cross-validation**: Implement k-fold cross-validation for more robust evaluation

## Project Structure

```
.
├── datasets/
│   └── img/
│       ├── train/
│       │   ├── bladder/ (50 images)
│       │   └── kidney/ (50 images)
│       └── test/
│           ├── bladder/ (10 images)
│           └── kidney/ (10 images)
├── ultrasound_classifier.py
├── requirements.txt
└── README.md
```
