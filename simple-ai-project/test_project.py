#!/usr/bin/env python3
"""
Test Project Script
Verify that all components of the simple AI project work correctly
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported"""
    print("🧪 Testing module imports...")
    
    try:
        from src.models import SpamClassifier
        print("✅ SpamClassifier imported successfully")
        
        from src.data import DataPreprocessor
        print("✅ DataPreprocessor imported successfully")
        
        from src.utils import print_results, create_visualizations
        print("✅ Utility functions imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_loading():
    """Test data loading functionality"""
    print("\n📊 Testing data loading...")
    
    try:
        from src.data import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        data = preprocessor.load_sample_data()
        
        print(f"✅ Data loaded successfully")
        print(f"   Total samples: {len(data)}")
        print(f"   Columns: {list(data.columns)}")
        
        return True
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False

def test_model_creation():
    """Test model creation and basic functionality"""
    print("\n🤖 Testing model creation...")
    
    try:
        from src.models import SpamClassifier
        
        classifier = SpamClassifier()
        print("✅ SpamClassifier created successfully")
        
        # Test feature extraction
        test_texts = ["Hello world", "Free money now!"]
        features = classifier.extract_features(test_texts)
        
        print(f"✅ Feature extraction works")
        print(f"   Features shape: {features.shape}")
        print(f"   Feature columns: {list(features.columns)}")
        
        return True
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False

def test_basic_training():
    """Test basic training functionality"""
    print("\n🎯 Testing basic training...")
    
    try:
        from src.models import SpamClassifier
        from src.data import DataPreprocessor
        
        # Load data
        preprocessor = DataPreprocessor()
        data = preprocessor.load_sample_data()
        texts, labels = preprocessor.get_text_data()
        
        # Train model
        classifier = SpamClassifier()
        classifier.train(texts, labels)
        
        print("✅ Model training completed successfully")
        
        # Test prediction
        test_texts = ["Free money now!", "Meeting tomorrow"]
        predictions = classifier.predict(test_texts)
        
        print(f"✅ Predictions work")
        print(f"   Predictions: {predictions}")
        
        return True
    except Exception as e:
        print(f"❌ Training error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Simple AI Project - Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_loading,
        test_model_creation,
        test_basic_training
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Project is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 