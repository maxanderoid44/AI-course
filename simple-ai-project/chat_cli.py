#!/usr/bin/env python3
"""
Simple AI Project - Email Spam Classifier CLI Chat
Command-line interface for real-time email spam classification
"""

import sys
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models import SpamClassifier
from src.utils import format_prediction_result, print_project_info

def load_or_train_model():
    """Load saved model or train a new one"""
    try:
        # Try to load existing model
        print("📦 Loading pre-trained model...")
        classifier = SpamClassifier.load_model("models/spam_classifier_trained.pkl")
        print("✅ Pre-trained model loaded successfully!")
        return classifier, True
    except FileNotFoundError:
        # Train new model if not available
        print("🔄 No pre-trained model found. Training new model...")
        
        from src.data import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        data = preprocessor.load_sample_data()
        texts, labels = preprocessor.get_text_data()
        
        classifier = SpamClassifier()
        classifier.train(texts, labels)
        
        # Save the model for future use
        classifier.save_model("models/spam_classifier_trained.pkl")
        
        print("✅ Model trained and saved successfully!")
        return classifier, False

def classify_email(classifier, email_text):
    """Classify a single email"""
    try:
        # Make prediction
        prediction = classifier.predict([email_text])[0]
        probabilities = classifier.predict_proba([email_text])[0]
        
        # Format result
        result = format_prediction_result(prediction, [probabilities])
        
        return {
            'prediction': result['prediction'],
            'confidence': result['confidence_percentage'],
            'probabilities': probabilities,
            'is_spam': prediction == 1
        }
    except Exception as e:
        print(f"❌ Error classifying email: {str(e)}")
        return None

def print_chat_header():
    """Print chat interface header"""
    print("\n" + "="*60)
    print("📧 Email Spam Classifier Chat Interface")
    print("="*60)
    print("Type 'quit' or 'exit' to end the chat")
    print("Type 'help' for sample emails")
    print("Type 'stats' to see chat statistics")
    print("="*60)

def print_help():
    """Print help information and sample emails"""
    print("\n💡 Sample emails to try:")
    print("-" * 40)
    
    sample_emails = [
        "Meeting tomorrow at 3 PM in conference room A",
        "Get rich quick! Click here for amazing offers!",
        "Please review the attached quarterly report",
        "URGENT: Your account has been suspended!",
        "Thank you for your application, we will contact you soon",
        "Limited time offer! 90% off everything!"
    ]
    
    for i, email in enumerate(sample_emails, 1):
        print(f"{i}. {email}")
    
    print("\n💡 Tips:")
    print("- Spam indicators: URGENT, FREE, PRIZE, !!!, limited time")
    print("- Legitimate indicators: meeting, report, application, professional language")
    print("-" * 40)

def main():
    """Main CLI chat function"""
    print_project_info()
    
    # Load or train model
    classifier, model_loaded = load_or_train_model()
    
    # Chat statistics
    total_emails = 0
    spam_count = 0
    legitimate_count = 0
    
    # Print chat header
    print_chat_header()
    
    while True:
        try:
            # Get user input
            user_input = input("\n💬 Enter email text: ").strip()
            
            # Check for commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thanks for using Email Spam Classifier Chat!")
                break
            elif user_input.lower() == 'help':
                print_help()
                continue
            elif user_input.lower() == 'stats':
                print(f"\n📊 Chat Statistics:")
                print(f"   Total emails: {total_emails}")
                print(f"   Spam detected: {spam_count}")
                print(f"   Legitimate: {legitimate_count}")
                if total_emails > 0:
                    spam_rate = (spam_count / total_emails) * 100
                    print(f"   Spam rate: {spam_rate:.1f}%")
                continue
            elif not user_input:
                print("❌ Please enter some text to classify.")
                continue
            
            # Classify the email
            print("🔍 Analyzing email...")
            result = classify_email(classifier, user_input)
            
            if result:
                total_emails += 1
                
                # Update statistics
                if result['is_spam']:
                    spam_count += 1
                    icon = "🚨"
                    status = "SPAM"
                else:
                    legitimate_count += 1
                    icon = "✅"
                    status = "LEGITIMATE"
                
                # Print result
                print(f"\n{icon} Classification: {status}")
                print(f"   Confidence: {result['confidence']}")
                print(f"   Legitimate Probability: {result['probabilities'][0]:.1%}")
                print(f"   Spam Probability: {result['probabilities'][1]:.1%}")
                
                # Add some delay for better UX
                time.sleep(0.5)
            
        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Thanks for using Email Spam Classifier!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    # Print final statistics
    if total_emails > 0:
        print(f"\n📊 Final Statistics:")
        print(f"   Total emails classified: {total_emails}")
        print(f"   Spam detected: {spam_count}")
        print(f"   Legitimate: {legitimate_count}")
        spam_rate = (spam_count / total_emails) * 100
        print(f"   Overall spam rate: {spam_rate:.1f}%")

if __name__ == "__main__":
    main() 