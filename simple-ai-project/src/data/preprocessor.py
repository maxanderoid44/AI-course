"""
Data Preprocessor Module
Handles data loading and preprocessing for the spam classifier
"""

import pandas as pd
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    """Handles data preprocessing for spam classification"""
    
    def __init__(self):
        self.data = None
    
    def load_sample_data(self):
        """Load sample data for demonstration"""
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
                'Code review meeting notes',
                'Get rich quick! Amazing offers!',
                'Please review the attached documents',
                'URGENT: Action required immediately!',
                'Monthly team meeting agenda',
                'Special discount just for you!',
                'Project deadline reminder',
                'Congratulations! You are selected!',
                'Budget approval needed',
                'Limited time only! Don\'t miss out!',
                'Quarterly review meeting'
            ],
            'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        }
        
        self.data = pd.DataFrame(data)
        return self.data
    
    def preprocess_text(self, text_series):
        """Basic text preprocessing"""
        # Convert to lowercase
        processed = text_series.str.lower()
        
        # Remove extra whitespace
        processed = processed.str.strip()
        
        return processed
    
    def create_features(self, df):
        """Create features from text data"""
        # Text length
        df['length'] = df['text'].str.len()
        
        # Word count
        df['word_count'] = df['text'].str.split().str.len()
        
        # Has exclamation
        df['has_exclamation'] = df['text'].str.contains('!').astype(int)
        
        # Has urgent words
        urgent_pattern = r'urgent|free|prize|offer|deal|limited|exclusive|rich|quick'
        df['has_urgent'] = df['text'].str.contains(urgent_pattern, case=False).astype(int)
        
        # Has all caps words
        df['has_all_caps'] = df['text'].str.contains(r'\b[A-Z]{3,}\b').astype(int)
        
        return df
    
    def split_data(self, test_size=0.3, random_state=42):
        """Split data into train and test sets"""
        if self.data is None:
            raise ValueError("Data must be loaded before splitting")
        
        # Create features
        df = self.create_features(self.data.copy())
        
        # Prepare features and target
        features = ['length', 'word_count', 'has_exclamation', 'has_urgent', 'has_all_caps']
        X = df[features]
        y = df['label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def get_text_data(self):
        """Get text data for training"""
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        return self.data['text'].tolist(), self.data['label'].tolist() 