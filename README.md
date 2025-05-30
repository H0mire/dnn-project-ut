# dnn-project-ut

Des geilste Projekt ever
# Ultrasound Organ Classification

## Overview
This project implements a lightweight convolutional neural network (MiniVGGNet) 
for classifying ultrasound images into six classes: bladder, bowel, gallbladder, kidney, liver, and spleen. 
It is designed for small datasets with no use of pretrained models.

---

## Model Architecture
- 2 convolutional blocks:
  - Each block: 2 Conv layers + BatchNorm + ReLU
  - MaxPooling after each block
- Adaptive Average Pooling
- Fully connected classifier:
  - FC(1024 → 128) + ReLU + Dropout
  - FC(128 → 6 classes)

---

## Dataset
- Ultrasound images organized into 6 organ classes
- Training set: 50 images per class
- Test set: 10 images per class

---

## Training Details
- Optimizer: Adam
- Loss Function: CrossEntropyLoss
- Batch size: 8
- Epochs: 30
- Data Augmentation:
  - Random rotations
  - Horizontal flips
  - Affine transformations
- Framework: PyTorch

---

## Results
- **Training Accuracy**: ~97%
- **Validation Accuracy**: up to 100%
- **Test Accuracy**: 93.3%
- **Macro F1-Score**: 0.93

### Confusion Matrix


### Training Curves



