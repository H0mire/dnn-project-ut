import matplotlib.pyplot as plt
import numpy as np

def create_comparison_plot():
    """Create a visual comparison of both approaches"""
    
    # Data from previous runs
    models = ['Normal CNN', 'Advanced Hybrid']
    accuracies = [0.95, 0.75]
    
    # Detailed metrics
    bladder_precision = [1.00, 1.00]
    bladder_recall = [0.90, 0.50]
    kidney_precision = [0.91, 0.67]
    kidney_recall = [1.00, 1.00]
    f1_scores = [0.95, 0.73]
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Normal CNN vs Advanced Hybrid Classifier Comparison', fontsize=16, fontweight='bold')
    
    # Colors
    colors = ['#2E86AB', '#A23B72']  # Blue for CNN, Red for Hybrid
    
    # 1. Overall Accuracy
    bars1 = axes[0,0].bar(models, accuracies, color=colors, alpha=0.8)
    axes[0,0].set_title('Overall Test Accuracy', fontweight='bold')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].set_ylim(0, 1)
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                      f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Bladder Detection Performance
    x = np.arange(len(models))
    width = 0.35
    axes[0,1].bar(x - width/2, bladder_precision, width, label='Precision', color=colors[0], alpha=0.6)
    axes[0,1].bar(x + width/2, bladder_recall, width, label='Recall', color=colors[1], alpha=0.6)
    axes[0,1].set_title('Bladder Detection', fontweight='bold')
    axes[0,1].set_ylabel('Score')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(models)
    axes[0,1].legend()
    axes[0,1].set_ylim(0, 1.1)
    
    # 3. Kidney Detection Performance
    axes[0,2].bar(x - width/2, kidney_precision, width, label='Precision', color=colors[0], alpha=0.6)
    axes[0,2].bar(x + width/2, kidney_recall, width, label='Recall', color=colors[1], alpha=0.6)
    axes[0,2].set_title('Kidney Detection', fontweight='bold')
    axes[0,2].set_ylabel('Score')
    axes[0,2].set_xticks(x)
    axes[0,2].set_xticklabels(models)
    axes[0,2].legend()
    axes[0,2].set_ylim(0, 1.1)
    
    # 4. F1-Scores
    bars4 = axes[1,0].bar(models, f1_scores, color=colors, alpha=0.8)
    axes[1,0].set_title('F1-Score', fontweight='bold')
    axes[1,0].set_ylabel('F1-Score')
    axes[1,0].set_ylim(0, 1)
    for bar, f1 in zip(bars4, f1_scores):
        height = bar.get_height()
        axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                      f'{f1:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 5. Model Complexity Comparison
    complexity_scores = [3, 8]  # Relative complexity (1-10 scale)
    bars5 = axes[1,1].bar(models, complexity_scores, color=colors, alpha=0.8)
    axes[1,1].set_title('Model Complexity', fontweight='bold')
    axes[1,1].set_ylabel('Complexity Score (1-10)')
    axes[1,1].set_ylim(0, 10)
    for bar, comp in zip(bars5, complexity_scores):
        height = bar.get_height()
        axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.2,
                      f'{comp}', ha='center', va='bottom', fontweight='bold')
    
    # 6. Summary Metrics
    axes[1,2].axis('off')
    summary_text = f"""
    🏆 WINNER: Normal CNN
    
    Key Advantages:
    • 20% higher accuracy
    • Better bladder recall (+40%)
    • Better kidney precision (+24%)
    • Lower complexity
    • More robust to small data
    
    Final Recommendation:
    Use Normal CNN Classifier
    """
    axes[1,2].text(0.1, 0.9, summary_text, transform=axes[1,2].transAxes, 
                   fontsize=12, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('detailed_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Detailed comparison plot saved as 'detailed_comparison.png'")

def print_summary():
    """Print a clear summary of the comparison"""
    
    print("=" * 60)
    print("🔍 ULTRASOUND CLASSIFIER PERFORMANCE COMPARISON")
    print("=" * 60)
    
    print("\n📊 RESULTS SUMMARY:")
    print("-" * 40)
    print(f"{'Metric':<25} {'Normal CNN':<12} {'Hybrid':<12} {'Winner'}")
    print("-" * 40)
    print(f"{'Test Accuracy':<25} {'95.0%':<12} {'75.0%':<12} {'CNN 🏆'}")
    print(f"{'Bladder Precision':<25} {'100.0%':<12} {'100.0%':<12} {'Tie'}")
    print(f"{'Bladder Recall':<25} {'90.0%':<12} {'50.0%':<12} {'CNN 🏆'}")
    print(f"{'Kidney Precision':<25} {'91.0%':<12} {'67.0%':<12} {'CNN 🏆'}")
    print(f"{'Kidney Recall':<25} {'100.0%':<12} {'100.0%':<12} {'Tie'}")
    print(f"{'F1-Score':<25} {'0.95':<12} {'0.73':<12} {'CNN 🏆'}")
    print(f"{'Model Complexity':<25} {'Low':<12} {'High':<12} {'CNN 🏆'}")
    
    print("\n🎯 KEY FINDINGS:")
    print("• Normal CNN outperforms by 20 percentage points")
    print("• Much better at detecting bladders (90% vs 50% recall)")
    print("• More precise at identifying kidneys (91% vs 67%)")
    print("• Simpler architecture, less prone to overfitting")
    
    print("\n💡 WHY CNN WINS:")
    print("1. Perfect match for small dataset (50 images/class)")
    print("2. Appropriate regularization prevents overfitting")
    print("3. Handcrafted features may be redundant with CNN features")
    print("4. Ensemble complexity not justified by performance gain")
    
    print("\n🏆 FINAL RECOMMENDATION:")
    print("Use the Normal CNN Classifier for this ultrasound task!")
    print("=" * 60)

if __name__ == "__main__":
    print_summary()
    create_comparison_plot() 