"""
Utility Helper Functions
Common utility functions for the spam classifier project
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np

def print_results(accuracy, model_name="Model"):
    """Print formatted results"""
    print(f"\n{'='*50}")
    print(f"📊 {model_name} Results")
    print(f"{'='*50}")
    print(f"🎯 Accuracy: {accuracy:.2%}")
    print(f"📈 Performance: {'Excellent' if accuracy > 0.9 else 'Good' if accuracy > 0.8 else 'Fair'}")
    print(f"{'='*50}")

def create_visualizations(df, save_path="results.png"):
    """Create and save visualizations"""
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Text length distribution
    axes[0, 0].hist(df[df['label'] == 0]['length'], alpha=0.7, label='Legitimate', bins=10)
    axes[0, 0].hist(df[df['label'] == 1]['length'], alpha=0.7, label='Spam', bins=10)
    axes[0, 0].set_title('Text Length Distribution')
    axes[0, 0].set_xlabel('Length')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    # Plot 2: Word count distribution
    axes[0, 1].hist(df[df['label'] == 0]['word_count'], alpha=0.7, label='Legitimate', bins=10)
    axes[0, 1].hist(df[df['label'] == 1]['word_count'], alpha=0.7, label='Spam', bins=10)
    axes[0, 1].set_title('Word Count Distribution')
    axes[0, 1].set_xlabel('Word Count')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # Plot 3: Feature importance
    features = ['has_exclamation', 'has_urgent', 'has_all_caps']
    feature_counts = df[features].sum()
    axes[1, 0].bar(feature_counts.index, feature_counts.values)
    axes[1, 0].set_title('Feature Distribution')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Label distribution
    label_counts = df['label'].value_counts()
    axes[1, 1].pie(label_counts.values, labels=['Legitimate', 'Spam'], autopct='%1.1f%%')
    axes[1, 1].set_title('Label Distribution')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Visualizations saved to {save_path}")

def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Spam'],
                yticklabels=['Legitimate', 'Spam'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📈 Confusion matrix saved to {save_path}")

def save_model_results(model, accuracy, save_path="model_results.txt"):
    """Save model results to file"""
    with open(save_path, 'w') as f:
        f.write("Simple AI Project - Model Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Model Type: {type(model).__name__}\n")
        f.write(f"Accuracy: {accuracy:.2%}\n")
        f.write(f"Performance: {'Excellent' if accuracy > 0.9 else 'Good' if accuracy > 0.8 else 'Fair'}\n")
        f.write("\nModel Parameters:\n")
        f.write(str(model.get_params()))
    
    print(f"💾 Model results saved to {save_path}")

def display_sample_predictions(model, texts, labels, num_samples=5):
    """Display sample predictions"""
    print("\n🔮 Sample Predictions:")
    print("-" * 50)
    
    for i in range(min(num_samples, len(texts))):
        pred = model.predict([texts[i]])[0]
        true_label = labels[i]
        status = "✅" if pred == true_label else "❌"
        
        print(f"{status} Text: {texts[i][:50]}...")
        print(f"   True: {'Spam' if true_label == 1 else 'Legitimate'}")
        print(f"   Pred: {'Spam' if pred == 1 else 'Legitimate'}")
        print()

def format_prediction_result(prediction, probability=None):
    """Format prediction result for display"""
    result = {
        'prediction': 'SPAM' if prediction == 1 else 'LEGITIMATE',
        'confidence': probability[0][prediction] if probability is not None else None
    }
    
    if result['confidence'] is not None:
        result['confidence_percentage'] = f"{result['confidence']*100:.1f}%"
    
    return result

def print_project_info():
    """Print project information"""
    print("=" * 60)
    print("📧 Simple AI Project - Email Spam Classifier")
    print("=" * 60)
    print("This project demonstrates:")
    print("  • Data preprocessing and feature extraction")
    print("  • Machine learning model training")
    print("  • Model evaluation and visualization")
    print("  • Real-world application deployment")
    print("=" * 60)

def save_model(model, filepath):
    """Save a trained model to file"""
    import pickle
    from pathlib import Path
    
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"   Model saved to: {filepath}")

def load_model(filepath):
    """Load a trained model from file"""
    import pickle
    import os
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    return model 