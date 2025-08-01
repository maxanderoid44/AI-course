"""
Spam Classifier Module
Simple implementation of email spam classification
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

class SpamClassifier:
    """Simple spam classifier using Random Forest"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
    
    def extract_features(self, texts):
        """Extract simple features from text"""
        features = pd.DataFrame()
        
        # Text length
        features['length'] = [len(text) for text in texts]
        
        # Word count
        features['word_count'] = [len(text.split()) for text in texts]
        
        # Has exclamation
        features['has_exclamation'] = [1 if '!' in text else 0 for text in texts]
        
        # Has urgent words
        urgent_words = ['urgent', 'free', 'prize', 'offer', 'deal', 'limited', 'exclusive']
        features['has_urgent'] = [
            1 if any(word in text.lower() for word in urgent_words) else 0 
            for text in texts
        ]
        
        return features
    
    def train(self, texts, labels):
        """Train the spam classifier"""
        print("Extracting features...")
        features = self.extract_features(texts)
        
        print("Training model...")
        self.model.fit(features, labels)
        self.is_trained = True
        
        print("Model trained successfully!")
    
    def predict(self, texts):
        """Predict spam/legitimate for given texts"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        features = self.extract_features(texts)
        predictions = self.model.predict(features)
        return predictions
    
    def evaluate(self, texts, labels):
        """Evaluate model performance"""
        predictions = self.predict(texts)
        accuracy = accuracy_score(labels, predictions)
        
        print(f"Accuracy: {accuracy:.2%}")
        print("\nClassification Report:")
        print(classification_report(labels, predictions, target_names=['Legitimate', 'Spam']))
        
        return accuracy
    
    def save_model(self, filepath):
        """Save the trained model to file"""
        import pickle
        from pathlib import Path
        
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"Model saved to: {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """Load a trained model from file"""
        import pickle
        import os
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        return model
    
    def predict_proba(self, texts):
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        features = self.extract_features(texts)
        probabilities = self.model.predict_proba(features)
        return probabilities 