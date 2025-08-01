#!/usr/bin/env python3
"""
Train Model Script
Independent script to train the spam classifier model
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models import SpamClassifier
from src.data import DataPreprocessor
from src.utils import print_results, create_visualizations, save_model_results

def main():
    """Main training function"""
    print("🚀 Starting Model Training Script")
    print("=" * 50)
    
    try:
        # Initialize data preprocessor
        print("📊 Loading and preprocessing data...")
        preprocessor = DataPreprocessor()
        data = preprocessor.load_sample_data()
        
        print(f"✅ Data loaded successfully!")
        print(f"   Total samples: {len(data)}")
        print(f"   Spam samples: {len(data[data['label'] == 1])}")
        print(f"   Legitimate samples: {len(data[data['label'] == 0])}")
        
        # Get text data for training
        texts, labels = preprocessor.get_text_data()
        
        # Initialize and train classifier
        print("\n🤖 Training the spam classifier...")
        classifier = SpamClassifier()
        classifier.train(texts, labels)
        
        # Evaluate model
        print("\n📈 Evaluating model performance...")
        accuracy = classifier.evaluate(texts, labels)
        
        # Print results
        print_results(accuracy, "Spam Classifier")
        
        # Create visualizations
        print("\n📊 Creating visualizations...")
        processed_data = preprocessor.create_features(data.copy())
        create_visualizations(processed_data, "training_results.png")
        
        # Save model results
        print("\n💾 Saving model results...")
        save_model_results(classifier.model, accuracy, "model_results.txt")
        
        # Save the trained classifier
        print("\n💾 Saving trained classifier...")
        classifier.save_model("models/spam_classifier_trained.pkl")
        
        # Make sample predictions
        print("\n🔮 Making sample predictions...")
        sample_emails = [
            "Get rich quick! Click here for amazing offers!",
            "Meeting tomorrow at 3 PM in conference room A",
            "URGENT: Your account has been suspended!",
            "Please review the attached quarterly report",
            "Limited time offer! 90% off everything!"
        ]
        
        predictions = classifier.predict(sample_emails)
        for email, pred in zip(sample_emails, predictions):
            status = "SPAM" if pred == 1 else "LEGITIMATE"
            print(f"   '{email[:50]}...' -> {status}")
        
        print("\n🎉 Training completed successfully!")
        print(f"📁 Results saved to:")
        print(f"   - training_results.png")
        print(f"   - model_results.txt")
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 