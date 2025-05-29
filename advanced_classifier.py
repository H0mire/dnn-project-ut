import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from ultrasound_classifier import (
    HandcraftedFeatureExtractor, ShallowCNN, load_data,
    UltrasoundDataset, train_model, evaluate_model
)
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import cv2
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib


class AdvancedHybridClassifier:
    """Advanced classifier combining multiple feature extraction methods"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cnn = ShallowCNN(num_classes=2).to(self.device)
        self.feature_extractor = HandcraftedFeatureExtractor()
        self.scaler = StandardScaler()
        self.ensemble_classifiers = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', probability=True, random_state=42)
        }
        
    def extract_cnn_features(self, image_paths, transform):
        """Extract CNN features from images"""
        features = []
        self.cnn.eval()
        
        with torch.no_grad():
            for path in image_paths:
                from PIL import Image
                img = Image.open(path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(self.device)
                feature = self.cnn.get_features(img_tensor)
                features.append(feature.cpu().numpy().flatten())
                
        return np.array(features)
    
    def extract_handcrafted_features(self, image_paths):
        """Extract handcrafted features from images"""
        features = []
        
        for path in image_paths:
            img = cv2.imread(path)
            img = cv2.resize(img, (128, 128))
            img = img.astype(np.float32) / 255.0
            feature = self.feature_extractor.extract_all_features(img)
            features.append(feature)
            
        return np.array(features)
    
    def train_hybrid_model(self, X_train_paths, y_train, X_val_paths, y_val):
        """Train the hybrid model with both CNN and handcrafted features"""
        
        # Define transform
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # First, train the CNN
        print("Training CNN backbone...")
        train_dataset = UltrasoundDataset(X_train_paths, y_train, transform, augment=True)
        val_dataset = UltrasoundDataset(X_val_paths, y_val, transform, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        self.cnn, _, _, _ = train_model(self.cnn, train_loader, val_loader, num_epochs=50)
        
        # Extract features
        print("\nExtracting hybrid features...")
        cnn_features_train = self.extract_cnn_features(X_train_paths, transform)
        handcrafted_features_train = self.extract_handcrafted_features(X_train_paths)
        
        cnn_features_val = self.extract_cnn_features(X_val_paths, transform)
        handcrafted_features_val = self.extract_handcrafted_features(X_val_paths)
        
        # Combine features
        X_train_combined = np.hstack([cnn_features_train, handcrafted_features_train])
        X_val_combined = np.hstack([cnn_features_val, handcrafted_features_val])
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_combined)
        X_val_scaled = self.scaler.transform(X_val_combined)
        
        # Train ensemble classifiers
        print("\nTraining ensemble classifiers...")
        results = {}
        
        for name, clf in self.ensemble_classifiers.items():
            print(f"Training {name}...")
            clf.fit(X_train_scaled, y_train)
            
            # Validation accuracy
            val_pred = clf.predict(X_val_scaled)
            val_acc = accuracy_score(y_val, val_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
            
            results[name] = {
                'model': clf,
                'val_accuracy': val_acc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"{name} - Validation Accuracy: {val_acc:.4f}, "
                  f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return results
    
    def predict_ensemble(self, X_test_paths):
        """Make predictions using ensemble voting"""
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Extract features
        cnn_features = self.extract_cnn_features(X_test_paths, transform)
        handcrafted_features = self.extract_handcrafted_features(X_test_paths)
        X_combined = np.hstack([cnn_features, handcrafted_features])
        X_scaled = self.scaler.transform(X_combined)
        
        # Get predictions from each classifier
        predictions = []
        for name, clf in self.ensemble_classifiers.items():
            pred = clf.predict(X_scaled)
            predictions.append(pred)
        
        # Majority voting
        predictions = np.array(predictions)
        ensemble_pred = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=0, arr=predictions
        )
        
        return ensemble_pred
    
    def save_model(self, filepath='hybrid_model.pkl'):
        """Save the trained hybrid model"""
        model_data = {
            'cnn_state_dict': self.cnn.state_dict(),
            'scaler': self.scaler,
            'ensemble_classifiers': self.ensemble_classifiers,
            'feature_extractor': self.feature_extractor
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")


def analyze_feature_importance(classifier, X_train_paths, y_train):
    """Analyze the importance of different feature types"""
    
    # Extract features separately
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    cnn_features = classifier.extract_cnn_features(X_train_paths, transform)
    handcrafted_features = classifier.extract_handcrafted_features(X_train_paths)
    
    # Train classifiers on different feature sets
    from sklearn.ensemble import RandomForestClassifier
    
    rf_cnn = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_handcrafted = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_combined = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Scale features
    scaler_cnn = StandardScaler()
    scaler_hand = StandardScaler()
    scaler_comb = StandardScaler()
    
    X_cnn_scaled = scaler_cnn.fit_transform(cnn_features)
    X_hand_scaled = scaler_hand.fit_transform(handcrafted_features)
    X_combined = np.hstack([cnn_features, handcrafted_features])
    X_comb_scaled = scaler_comb.fit_transform(X_combined)
    
    # Cross-validation scores
    cv_cnn = cross_val_score(rf_cnn, X_cnn_scaled, y_train, cv=5)
    cv_hand = cross_val_score(rf_handcrafted, X_hand_scaled, y_train, cv=5)
    cv_combined = cross_val_score(rf_combined, X_comb_scaled, y_train, cv=5)
    
    print("\nFeature Analysis Results:")
    print(f"CNN Features Only: {cv_cnn.mean():.4f} (+/- {cv_cnn.std():.4f})")
    print(f"Handcrafted Features Only: {cv_hand.mean():.4f} (+/- {cv_hand.std():.4f})")
    print(f"Combined Features: {cv_combined.mean():.4f} (+/- {cv_combined.std():.4f})")
    
    # Feature importance for combined model
    rf_combined.fit(X_comb_scaled, y_train)
    importances = rf_combined.feature_importances_
    
    cnn_importance = importances[:1024].mean()
    hand_importance = importances[1024:].mean()
    
    print(f"\nAverage Feature Importance:")
    print(f"CNN Features: {cnn_importance:.4f}")
    print(f"Handcrafted Features: {hand_importance:.4f}")


def main():
    """Main function to demonstrate advanced classification"""
    
    # Load data
    print("Loading data...")
    data_dir = './datasets/img'
    X_train_paths, y_train, X_test_paths, y_test = load_data(data_dir)
    
    # Split training data for validation
    from sklearn.model_selection import train_test_split
    X_train_paths, X_val_paths, y_train, y_val = train_test_split(
        X_train_paths, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Initialize advanced classifier
    print("\nInitializing Advanced Hybrid Classifier...")
    classifier = AdvancedHybridClassifier()
    
    # Train hybrid model
    results = classifier.train_hybrid_model(X_train_paths, y_train, X_val_paths, y_val)
    
    # Test the ensemble
    print("\nEvaluating on test set...")
    ensemble_pred = classifier.predict_ensemble(X_test_paths)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    
    print(f"\nEnsemble Test Accuracy: {ensemble_acc:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, ensemble_pred, 
                              target_names=['bladder', 'kidney']))
    
    # Analyze feature importance
    analyze_feature_importance(classifier, X_train_paths, y_train)
    
    # Save the model
    classifier.save_model()
    
    print("\nAdvanced classification complete!")


if __name__ == "__main__":
    main() 