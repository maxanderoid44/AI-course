"""
Email spam classifier model implementation
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time


class EmailClassifier:
    """Email spam classifier using multiple ML algorithms"""
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.training_time = None
        self.feature_importance = None
        
    def _create_model(self):
        """Create the specified model"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'logistic_regression':
            return LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        elif self.model_type == 'svm':
            return SVC(
                kernel='rbf',
                random_state=42,
                probability=True
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train):
        """Train the classifier"""
        print(f"Training {self.model_type} model...")
        
        # Create model
        self.model = self._create_model()
        
        # Record training time
        start_time = time.time()
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Calculate training time
        self.training_time = time.time() - start_time
        
        # Get feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        
        print(f"✅ Model trained in {self.training_time:.2f} seconds")
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Print detailed results
        print(f"\n📊 Model Performance:")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Training Time: {self.training_time:.2f} seconds")
        
        # Classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Spam']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n🔢 Confusion Matrix:")
        print(f"   True Negatives (Legitimate): {cm[0,0]}")
        print(f"   False Positives: {cm[0,1]}")
        print(f"   False Negatives: {cm[1,0]}")
        print(f"   True Positives (Spam): {cm[1,1]}")
        
        return accuracy
    
    def get_feature_importance(self, feature_names=None, top_n=10):
        """Get top feature importance scores"""
        if self.feature_importance is None:
            print("Feature importance not available for this model type")
            return None
        
        # Get indices of top features
        top_indices = np.argsort(self.feature_importance)[::-1][:top_n]
        
        # Create feature importance dictionary
        importance_dict = {}
        for idx in top_indices:
            if feature_names is not None and idx < len(feature_names):
                feature_name = feature_names[idx]
            else:
                feature_name = f"Feature_{idx}"
            
            importance_dict[feature_name] = self.feature_importance[idx]
        
        return importance_dict
    
    def save_model(self, filepath):
        """Save the trained model"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_model(cls, filepath):
        """Load a trained model"""
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f) 