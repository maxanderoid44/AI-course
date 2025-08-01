#!/usr/bin/env python3
"""
Load and Use Saved Model
Demonstrates how to load a saved model and use it for predictions
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models import SpamClassifier
from src.utils import format_prediction_result

def main():
    """Load saved model and make predictions"""
    print("🔍 Loading Saved Model Demo")
    print("=" * 50)
    
    try:
        # Try to load the saved model
        print("📦 Loading saved model...")
        classifier = SpamClassifier.load_model("models/spam_classifier_trained.pkl")
        print("✅ Model loaded successfully!")
        
        # Test predictions with new emails
        test_emails = [
            "Free money now! Click here for amazing offers!",
            "Meeting tomorrow at 3 PM in conference room A",
            "URGENT: Your account has been suspended!",
            "Please review the attached quarterly report",
            "Limited time offer! 90% off everything!",
            "Team lunch this Friday",
            "You've won $1,000,000! Claim your prize now!",
            "Weekly report attached"
        ]
        
        print(f"\n🔮 Making predictions on {len(test_emails)} test emails...")
        print("-" * 60)
        
        predictions = classifier.predict(test_emails)
        probabilities = classifier.predict_proba(test_emails)
        
        for i, (email, pred, prob) in enumerate(zip(test_emails, predictions, probabilities), 1):
            result = format_prediction_result(pred, [prob])
            status = "🚨 SPAM" if pred == 1 else "✅ LEGITIMATE"
            print(f"{i:2d}. {status}: '{email[:40]}...' (Confidence: {result['confidence_percentage']})")
        
        print("\n🎉 Model loading and prediction demo completed successfully!")
        
    except FileNotFoundError:
        print("❌ No saved model found. Please run the training script first:")
        print("   uv run python src/models/train_model.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 