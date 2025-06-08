import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops_table, label
from skimage.filters import sobel
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


class UltrasoundDataset(Dataset):
    """Custom dataset for ultrasound images with augmentation"""
    
    def __init__(self, image_paths, labels, transform=None, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.augment = augment
        
        # Define augmentation transforms
        if augment:
            self.augmentation = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ])
        else:
            self.augmentation = None
            
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        # Apply augmentation if training
        if self.augmentation:
            image = self.augmentation(image)
            
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        return image, label


class ShallowCNN(nn.Module):
    """Shallow CNN designed for small datasets"""
    
    def __init__(self, num_classes=2):
        super(ShallowCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def get_features(self, x):
        """Extract features from the last conv layer"""
        x = self.features(x)
        return x.view(x.size(0), -1)


def load_data(data_dir, classes=['bladder', 'kidney']):
    """Load ultrasound images and labels"""
    images_train = []
    labels_train = []
    images_test = []
    labels_test = []
    
    # Load training data
    for i, class_name in enumerate(classes):
        train_dir = os.path.join(data_dir, 'train', class_name)
        test_dir = os.path.join(data_dir, 'test', class_name)
        
        # Training images
        for img_name in os.listdir(train_dir):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(train_dir, img_name)
                images_train.append(img_path)
                labels_train.append(i)
                
        # Test images
        for img_name in os.listdir(test_dir):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(test_dir, img_name)
                images_test.append(img_path)
                labels_test.append(i)
    
    return (np.array(images_train), np.array(labels_train), 
            np.array(images_test), np.array(labels_test))


def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001):
    """Train the CNN model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        scheduler.step(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, '
                  f'Val Accuracy: {val_accuracy:.2f}%')
    
    return model, train_losses, val_losses, val_accuracies


def evaluate_model(model, test_loader, class_names):
    """Evaluate the model on test data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return all_predictions, all_labels


def plot_training_history(train_losses, val_losses, val_accuracies):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(val_accuracies, label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()


def main():

    class_names = ['bladder', 'kidney', 'liver', 'gallbladder', 'spleen', 'bowel']
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load data
    print("Loading data...")
    data_dir = './datasets/img'
    X_train_paths, y_train, X_test_paths, y_test = load_data(data_dir, classes=class_names)
    
    # Create validation set from training data
    X_train_paths, X_val_paths, y_train, y_val = train_test_split(
        X_train_paths, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Training samples: {len(X_train_paths)}")
    print(f"Validation samples: {len(X_val_paths)}")
    print(f"Test samples: {len(X_test_paths)}")
    
    # Define transforms
    transform_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = UltrasoundDataset(X_train_paths, y_train, transform_train, augment=True)
    val_dataset = UltrasoundDataset(X_val_paths, y_val, transform_test, augment=False)
    test_dataset = UltrasoundDataset(X_test_paths, y_test, transform_test, augment=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Initialize model
    print("\nInitializing model...")
    model = ShallowCNN(num_classes=len(class_names))
    
    # Train model
    print("\nTraining model...")
    model, train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, num_epochs=100, lr=0.001
    )
    
    # Plot training history
    plot_training_history(train_losses, val_losses, val_accuracies)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    class_names = class_names
    evaluate_model(model, test_loader, class_names)
    
    # Save model
    torch.save(model.state_dict(), 'ultrasound_classifier.pth')
    print("\nModel saved as 'ultrasound_classifier.pth'")
    
    # Extract and save features for visualization
    print("\nExtracting features for analysis...")
    feature_extractor = HandcraftedFeatureExtractor()
    
    # Sample feature extraction
    sample_image_path = X_train_paths[0]
    sample_image = cv2.imread(sample_image_path)
    sample_image = cv2.resize(sample_image, (128, 128))
    sample_image = sample_image.astype(np.float32) / 255.0
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main() 