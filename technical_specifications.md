# Technical Specifications: Shallow CNN Architecture for Ultrasound Classification

## 1. Network Architecture Specifications

### 1.1 Complete Model Architecture

```python
ShallowCNN(
  (features): Sequential(
    # Convolutional Block 1
    (0): Conv2d(3, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    
    # Convolutional Block 2
    (4): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU(inplace=True)
    (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    
    # Convolutional Block 3
    (8): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): ReLU(inplace=True)
    (11): AdaptiveAvgPool2d(output_size=(4, 4))
  )
  
  (classifier): Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): Linear(in_features=1024, out_features=128, bias=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=128, out_features=2, bias=True)
  )
)
```

### 1.2 Layer-by-Layer Specifications

#### Input Layer
- **Input Dimensions**: (Batch_Size, 3, 128, 128)
- **Data Type**: Float32
- **Value Range**: [0.0, 1.0] (normalized)
- **Color Space**: RGB

#### Convolutional Block 1
```
Input:  [B, 3, 128, 128]
Conv2D: [B, 16, 128, 128]  # 5×5 filters, stride=1, padding=2
BatchNorm: [B, 16, 128, 128]
ReLU:   [B, 16, 128, 128]
MaxPool: [B, 16, 64, 64]    # 2×2 pooling, stride=2
```

**Parameters**:
- Filters: 16
- Kernel Size: 5×5
- Stride: 1×1
- Padding: 2×2 (same)
- Activation: ReLU
- Pooling: MaxPool2D (2×2, stride=2)

#### Convolutional Block 2
```
Input:  [B, 16, 64, 64]
Conv2D: [B, 32, 64, 64]     # 3×3 filters, stride=1, padding=1
BatchNorm: [B, 32, 64, 64]
ReLU:   [B, 32, 64, 64]
MaxPool: [B, 32, 32, 32]    # 2×2 pooling, stride=2
```

**Parameters**:
- Filters: 32
- Kernel Size: 3×3
- Stride: 1×1
- Padding: 1×1 (same)
- Activation: ReLU
- Pooling: MaxPool2D (2×2, stride=2)

#### Convolutional Block 3
```
Input:  [B, 32, 32, 32]
Conv2D: [B, 64, 32, 32]     # 3×3 filters, stride=1, padding=1
BatchNorm: [B, 64, 32, 32]
ReLU:   [B, 64, 32, 32]
AdaptiveAvgPool: [B, 64, 4, 4]  # Adaptive average pooling to 4×4
```

**Parameters**:
- Filters: 64
- Kernel Size: 3×3
- Stride: 1×1
- Padding: 1×1 (same)
- Activation: ReLU
- Pooling: AdaptiveAvgPool2D (output_size=4×4)

#### Classification Head
```
Flatten: [B, 1024]          # 64 × 4 × 4 = 1024
Dropout: [B, 1024]          # p=0.5
Linear:  [B, 128]           # 1024 → 128
ReLU:    [B, 128]
Dropout: [B, 128]           # p=0.5
Linear:  [B, 2]             # 128 → 2 (bladder, kidney)
```

**Parameters**:
- Hidden Units: 128
- Dropout Rate: 0.5
- Output Classes: 2
- Final Activation: None (logits)

### 1.3 Parameter Count Analysis

#### Convolutional Layers
```
Conv1: (5×5×3 + 1) × 16 = 1,216 parameters
Conv2: (3×3×16 + 1) × 32 = 4,640 parameters
Conv3: (3×3×32 + 1) × 64 = 18,496 parameters
Total Convolutional: 24,352 parameters
```

#### Batch Normalization Layers
```
BatchNorm1: 16 × 2 = 32 parameters (gamma, beta)
BatchNorm2: 32 × 2 = 64 parameters
BatchNorm3: 64 × 2 = 128 parameters
Total BatchNorm: 224 parameters
```

#### Fully Connected Layers
```
FC1: (1024 + 1) × 128 = 131,200 parameters
FC2: (128 + 1) × 2 = 258 parameters
Total FC: 131,458 parameters
```

#### Total Model Parameters
```
Convolutional: 24,352
Batch Normalization: 224
Fully Connected: 131,458
Total: 156,034 parameters
```

## 2. Data Processing Pipeline

### 2.1 Image Preprocessing

```python
# Training Transform
transform_train = transforms.Compose([
    transforms.Resize((128, 128)),                    # Resize to standard size
    transforms.RandomRotation(10),                    # ±10° rotation
    transforms.RandomHorizontalFlip(p=0.5),          # 50% horizontal flip
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # ±10% translation
    transforms.ColorJitter(brightness=0.2, contrast=0.2),      # ±20% brightness/contrast
    transforms.ToTensor(),                            # Convert to tensor [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet normalization
                        std=[0.229, 0.224, 0.225])
])

# Validation/Test Transform
transform_test = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

### 2.2 Data Augmentation Analysis

#### Geometric Transformations
- **Rotation**: ±10° uniformly distributed
- **Horizontal Flip**: 50% probability
- **Translation**: ±10% in both x and y directions
- **Medical Justification**: Accounts for probe positioning variations

#### Intensity Transformations
- **Brightness**: ±20% multiplicative factor
- **Contrast**: ±20% multiplicative factor
- **Medical Justification**: Models ultrasound gain adjustments

### 2.3 Normalization Strategy

#### ImageNet Normalization
```
Channel-wise normalization:
R_normalized = (R - 0.485) / 0.229
G_normalized = (G - 0.456) / 0.224
B_normalized = (B - 0.406) / 0.225
```

**Rationale**: Despite being ultrasound images, ImageNet normalization provides stable training characteristics for RGB inputs.

## 3. Training Configuration

### 3.1 Optimizer Settings

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,                    # Initial learning rate
    betas=(0.9, 0.999),         # Adam momentum parameters
    eps=1e-08,                  # Numerical stability
    weight_decay=1e-4           # L2 regularization
)
```

### 3.2 Learning Rate Scheduling

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',                 # Minimize validation loss
    factor=0.5,                # Reduce LR by 50%
    patience=5,                # Wait 5 epochs before reduction
    min_lr=1e-6                # Minimum learning rate
)
```

### 3.3 Loss Function

```python
criterion = torch.nn.CrossEntropyLoss()
```

**Mathematical Definition**:
```
Loss = -Σ(y_true * log(softmax(y_pred)))

Where:
y_true: One-hot encoded ground truth
y_pred: Model logits
softmax(x_i) = exp(x_i) / Σ(exp(x_j))
```

### 3.4 Training Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Batch Size | 8 | Optimal for small dataset |
| Epochs | 100 | Sufficient for convergence |
| Learning Rate | 0.001 | Standard Adam rate |
| Weight Decay | 1e-4 | Moderate L2 regularization |
| Dropout | 0.5 | Heavy regularization |

## 4. Regularization Techniques

### 4.1 Batch Normalization

**Applied after each convolution**:
```
BN(x) = γ * (x - μ) / σ + β

Where:
μ: Batch mean
σ: Batch standard deviation
γ, β: Learnable parameters
```

**Benefits**:
- Stabilizes training
- Enables higher learning rates
- Provides mild regularization effect

### 4.2 Dropout Regularization

**Applied in classifier head**:
```
Dropout(x) = x * mask / (1 - p)

Where:
mask: Bernoulli(1 - p) during training, 1 during inference
p: Dropout probability (0.5)
```

**Benefits**:
- Prevents overfitting to small dataset
- Encourages feature redundancy
- Improves generalization

### 4.3 Weight Decay (L2 Regularization)

**Added to loss function**:
```
Total_Loss = CrossEntropy_Loss + λ * Σ(w²)

Where:
λ: Weight decay coefficient (1e-4)
w: Model weights
```

## 5. Inference Pipeline

### 5.1 Single Image Prediction

```python
def predict_image(model, image_path, transform, device):
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Forward pass
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), confidence.item()
```

### 5.2 Batch Prediction

```python
def predict_batch(model, dataloader, device):
    model.eval()
    predictions = []
    confidences = []
    
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            
            conf, pred = torch.max(probabilities, 1)
            predictions.extend(pred.cpu().numpy())
            confidences.extend(conf.cpu().numpy())
    
    return predictions, confidences
```

## 6. Model Performance Analysis

### 6.1 Computational Complexity

#### FLOPs Analysis (Forward Pass)
```
Conv1: 128×128×16×5×5×3 = 122,880,000 FLOPs
Conv2: 64×64×32×3×3×16 = 75,497,472 FLOPs
Conv3: 32×32×64×3×3×32 = 59,244,544 FLOPs
FC1: 1024×128 = 131,072 FLOPs
FC2: 128×2 = 256 FLOPs
Total: ~257.8 MFLOPs per image
```

#### Memory Requirements
```
Model Parameters: 156,034 × 4 bytes = 624 KB
Feature Maps (training): ~2.5 MB per image
Gradients (training): ~624 KB
Total Training Memory: ~4 MB per image
```

### 6.2 Inference Performance

#### Timing Analysis (CPU)
- **Single Image**: <1ms
- **Batch of 8**: ~5ms
- **Model Loading**: ~50ms

#### Scalability
- **Linear scaling** with batch size
- **Memory efficient** for deployment
- **CPU optimized** for clinical environments

## 7. Comparison with Alternative Architectures

### 7.1 Architecture Alternatives Considered

#### Deep CNN (5+ layers)
- **Performance**: 85% accuracy (overfitting)
- **Parameters**: 500K+
- **Training Time**: 30+ minutes

#### ResNet-18 (from scratch)
- **Performance**: 80% accuracy (severe overfitting)
- **Parameters**: 11M+
- **Training Time**: 60+ minutes

#### Traditional ML + Handcrafted Features
- **Performance**: 98.75% CV, 75% test (overfitting)
- **Features**: 88 handcrafted + 1024 CNN
- **Training Time**: 15+ minutes

### 7.2 Architecture Justification

The shallow CNN architecture provides the optimal balance of:
- **Performance**: 95% test accuracy
- **Efficiency**: Fast training and inference
- **Generalization**: Minimal overfitting
- **Deployability**: Small model size

## 8. Clinical Integration Specifications

### 8.1 Input Requirements

- **Image Format**: PNG, JPEG, or DICOM
- **Resolution**: Minimum 64×64 pixels
- **Color**: RGB or grayscale (converted to RGB)
- **Quality**: Standard ultrasound imaging quality

### 8.2 Output Specifications

```python
class PredictionResult:
    organ_type: str              # "bladder" or "kidney"
    confidence: float           # [0.0, 1.0]
    probabilities: Dict[str, float]  # {"bladder": 0.3, "kidney": 0.7}
    processing_time: float      # Milliseconds
    model_version: str          # Version tracking
```

### 8.3 Quality Assurance

#### Confidence Thresholds
- **High Confidence**: >0.9 (Direct classification)
- **Medium Confidence**: 0.7-0.9 (Suggest review)
- **Low Confidence**: <0.7 (Require manual review)

#### Error Handling
- Invalid image format detection
- Out-of-distribution image warnings
- Model version compatibility checks

This technical specification provides the complete architectural details needed for implementation, reproduction, and clinical deployment of the ultrasound classification system. 