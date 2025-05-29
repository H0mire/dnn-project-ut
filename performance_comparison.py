import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from ultrasound_classifier import ShallowCNN, load_data, UltrasoundDataset
from advanced_classifier import AdvancedHybridClassifier
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import seaborn as sns


def evaluate_normal_cnn():
    """Evaluate the normal CNN classifier"""
    print("=== NORMAL CNN CLASSIFIER ===")
    
    # Load data
    data_dir = './datasets/img'
    X_train_paths, y_train, X_test_paths, y_test = load_data(data_dir)
    
    # Load trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ShallowCNN(num_classes=2)
    model.load_state_dict(torch.load('ultrasound_classifier.pth', map_location=device))
    model.eval()
    model.to(device)
    
    # Create test dataset
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = UltrasoundDataset(X_test_paths, y_test, transform, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Make predictions
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=['bladder', 'kidney']))
    
    return all_labels, all_predictions, accuracy


def evaluate_hybrid_classifier():
    """Evaluate the hybrid classifier if available"""
    print("\n=== HYBRID CLASSIFIER ===")
    
    # Load data
    data_dir = './datasets/img'
    X_train_paths, y_train, X_test_paths, y_test = load_data(data_dir)
    
    # Split training data for validation
    X_train_paths, X_val_paths, y_train, y_val = train_test_split(
        X_train_paths, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Initialize and train hybrid classifier
    classifier = AdvancedHybridClassifier()
    
    # Train the model (this will take some time)
    print("Training hybrid model...")
    results = classifier.train_hybrid_model(X_train_paths, y_train, X_val_paths, y_val)
    
    # Test the ensemble
    ensemble_pred = classifier.predict_ensemble(X_test_paths)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    
    print(f"Test Accuracy: {ensemble_acc:.4f} ({ensemble_acc*100:.1f}%)")
    
    print("\nClassification Report:")
    print(classification_report(y_test, ensemble_pred, target_names=['bladder', 'kidney']))
    
    return y_test, ensemble_pred, ensemble_acc


def plot_comparison(cnn_labels, cnn_preds, hybrid_labels, hybrid_preds, cnn_acc, hybrid_acc):
    """Plot comparison between models"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Confusion matrices
    cm_cnn = confusion_matrix(cnn_labels, cnn_preds)
    cm_hybrid = confusion_matrix(hybrid_labels, hybrid_preds)
    
    # CNN confusion matrix
    sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['bladder', 'kidney'], yticklabels=['bladder', 'kidney'],
                ax=axes[0])
    axes[0].set_title(f'CNN Classifier\nAccuracy: {cnn_acc:.1%}')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # Hybrid confusion matrix
    sns.heatmap(cm_hybrid, annot=True, fmt='d', cmap='Reds',
                xticklabels=['bladder', 'kidney'], yticklabels=['bladder', 'kidney'],
                ax=axes[1])
    axes[1].set_title(f'Hybrid Classifier\nAccuracy: {hybrid_acc:.1%}')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    # Accuracy comparison
    models = ['CNN', 'Hybrid']
    accuracies = [cnn_acc, hybrid_acc]
    colors = ['blue', 'red']
    
    bars = axes[2].bar(models, accuracies, color=colors, alpha=0.7)
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Model Comparison')
    axes[2].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.1%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison plot saved as 'model_comparison.png'")


def analyze_why_hybrid_performs_worse():
    """Analyze potential reasons for hybrid classifier underperformance"""
    
    print("\n=== ANALYSIS: Why Hybrid Classifier Performs Worse ===")
    
    potential_reasons = [
        "1. **Overfitting in Ensemble**: The ensemble might be overfitting to the validation set",
        "2. **Feature Redundancy**: CNN features might already capture the handcrafted features",
        "3. **Ensemble Voting Issues**: Simple majority voting might not be optimal",
        "4. **Training Data Size**: Very small dataset makes ensemble approaches less effective",
        "5. **Feature Scaling**: Different feature scales might affect ensemble performance",
        "6. **Cross-validation vs Test**: CV performance doesn't always translate to test performance"
    ]
    
    for reason in potential_reasons:
        print(reason)
    
    print("\n=== RECOMMENDATIONS ===")
    recommendations = [
        "• Use the **Normal CNN Classifier** for this specific task",
        "• The CNN alone is sufficient and less prone to overfitting",
        "• Handcrafted features might be better as auxiliary information for interpretation",
        "• For larger datasets, hybrid approaches typically perform better"
    ]
    
    for rec in recommendations:
        print(rec)


def main():
    """Main comparison function"""
    print("ULTRASOUND CLASSIFIER PERFORMANCE COMPARISON")
    print("=" * 50)
    
    # Evaluate Normal CNN
    cnn_labels, cnn_preds, cnn_acc = evaluate_normal_cnn()
    
    # Evaluate Hybrid Classifier (this will retrain, so it might take time)
    print("\nNote: Hybrid classifier will retrain, this may take a few minutes...")
    hybrid_labels, hybrid_preds, hybrid_acc = evaluate_hybrid_classifier()
    
    # Plot comparison
    plot_comparison(cnn_labels, cnn_preds, hybrid_labels, hybrid_preds, cnn_acc, hybrid_acc)
    
    # Summary
    print(f"\n{'='*50}")
    print("FINAL COMPARISON SUMMARY")
    print(f"{'='*50}")
    print(f"Normal CNN Classifier:    {cnn_acc:.1%}")
    print(f"Hybrid Classifier:        {hybrid_acc:.1%}")
    print(f"Difference:              {(cnn_acc - hybrid_acc)*100:+.1f} percentage points")
    
    if cnn_acc > hybrid_acc:
        print(f"\n🏆 **WINNER: Normal CNN Classifier**")
        print(f"   The CNN classifier outperforms the hybrid approach by {(cnn_acc - hybrid_acc)*100:.1f} percentage points")
    else:
        print(f"\n🏆 **WINNER: Hybrid Classifier**")
        print(f"   The hybrid classifier outperforms the CNN approach by {(hybrid_acc - cnn_acc)*100:.1f} percentage points")
    
    # Analysis
    analyze_why_hybrid_performs_worse()


if __name__ == "__main__":
    main() 