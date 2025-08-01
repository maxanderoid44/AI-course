#!/usr/bin/env python3
"""
Simple AI Project - Email Spam Classifier
Main application entry point
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models import SpamClassifier
from src.data import DataPreprocessor
from src.utils import (
    print_results, 
    create_visualizations, 
    save_model_results,
    format_prediction_result,
    print_project_info,
    save_model,
    load_model
)

def load_data():
    """Load sample data for demonstration"""
    # Create sample data for demonstration
    data = {
        'text': [
            'Free money now! Click here!',
            'Meeting tomorrow at 3 PM',
            'You won a prize! Claim now!',
            'Project update: Q4 results',
            'Limited time offer! Buy now!',
            'Team lunch this Friday',
            'Urgent: Your account is suspended',
            'Weekly report attached',
            'Exclusive deal for you!',
            'Code review meeting notes'
        ],
        'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = legitimate
    }
    return pd.DataFrame(data)

def preprocess_data(df):
    """Simple text preprocessing"""
    # Convert text to lowercase
    df['text_processed'] = df['text'].str.lower()
    
    # Create simple features
    df['length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    df['has_exclamation'] = df['text'].str.contains('!').astype(int)
    df['has_urgent'] = df['text'].str.contains('urgent|free|prize|offer|deal').astype(int)
    
    return df

def train_model(X_train, y_train):
    """Train the spam classifier"""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Spam']))
    
    return accuracy

def plot_results(df):
    """Create visualization of results"""
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Text length distribution
    plt.subplot(1, 3, 1)
    df.groupby('label')['length'].hist(alpha=0.7, bins=10)
    plt.title('Text Length by Label')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.legend(['Legitimate', 'Spam'])
    
    # Plot 2: Word count distribution
    plt.subplot(1, 3, 2)
    df.groupby('label')['word_count'].hist(alpha=0.7, bins=10)
    plt.title('Word Count by Label')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.legend(['Legitimate', 'Spam'])
    
    # Plot 3: Feature importance
    plt.subplot(1, 3, 3)
    features = ['length', 'word_count', 'has_exclamation', 'has_urgent']
    feature_counts = df[features].sum()
    feature_counts.plot(kind='bar')
    plt.title('Feature Distribution')
    plt.xlabel('Features')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('results.png')
    plt.show()

def main():
    """Main function"""
    print_project_info()
    
    print("🚀 Starting Simple AI Project - Email Spam Classifier")
    print("=" * 50)
    
    # Load data
    print("📊 Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} samples")
    
    # Preprocess data
    print("🔧 Preprocessing data...")
    df = preprocess_data(df)
    
    # Prepare features
    features = ['length', 'word_count', 'has_exclamation', 'has_urgent']
    X = df[features]
    y = df['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train model
    print("🤖 Training model...")
    model = train_model(X_train, y_train)
    
    # Evaluate model
    print("📈 Evaluating model...")
    accuracy = evaluate_model(model, X_test, y_test)
    
    # Create visualizations
    print("📊 Creating visualizations...")
    plot_results(df)
    
    # Save model
    print("💾 Saving model...")
    save_model(model, "models/spam_classifier.pkl")
    
    # Test predictions
    print("\n🔮 Testing predictions...")
    sample_emails = [
        "Get rich quick! Click here for amazing offers!",
        "Meeting tomorrow at 3 PM in conference room A",
        "URGENT: Your account has been suspended!",
        "Please review the attached quarterly report"
    ]
    
    # Use our SpamClassifier for predictions
    classifier = SpamClassifier()
    classifier.train(df['text'].tolist(), df['label'].tolist())
    
    predictions = classifier.predict(sample_emails)
    probabilities = classifier.predict_proba(sample_emails)
    
    for email, pred, prob in zip(sample_emails, predictions, probabilities):
        result = format_prediction_result(pred, [prob])
        status = "🚨 SPAM" if pred == 1 else "✅ LEGITIMATE"
        print(f"{status}: '{email[:50]}...' (Confidence: {result['confidence_percentage']})")
    
    # Save classifier model
    print("\n💾 Saving classifier model...")
    classifier.save_model("models/spam_classifier_full.pkl")
    
    print("\n✅ Project completed successfully!")
    print(f"📁 Results saved to 'results.png'")
    print(f"🎯 Final accuracy: {accuracy:.2%}")
    print(f"📦 Models saved to 'models/' directory")

if __name__ == "__main__":
    main() 