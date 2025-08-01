#!/usr/bin/env python3
"""
Simple AI Project - Email Spam Classifier Chat App
Interactive Streamlit application for real-time email spam classification
"""

import streamlit as st
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
        classifier = SpamClassifier.load_model("models/spam_classifier_trained.pkl")
        st.success("✅ Pre-trained model loaded successfully!")
        return classifier, True
    except FileNotFoundError:
        # Train new model if not available
        st.info("🔄 No pre-trained model found. Training new model...")
        
        from src.data import DataPreprocessor
        
        with st.spinner("Training model..."):
            preprocessor = DataPreprocessor()
            data = preprocessor.load_sample_data()
            texts, labels = preprocessor.get_text_data()
            
            classifier = SpamClassifier()
            classifier.train(texts, labels)
            
            # Save the model for future use
            classifier.save_model("models/spam_classifier_trained.pkl")
        
        st.success("✅ Model trained and saved successfully!")
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
        st.error(f"Error classifying email: {str(e)}")
        return None

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Email Spam Classifier Chat",
        page_icon="📧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin: 0.8rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 2px solid;
    }
    .spam-message {
        background-color: #fff5f5;
        border-color: #feb2b2;
        color: #c53030;
    }
    .legitimate-message {
        background-color: #f0fff4;
        border-color: #9ae6b4;
        color: #22543d;
    }
    .user-message {
        background-color: #f7fafc;
        border-color: #90cdf4;
        color: #2a4365;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
    .stChatMessage {
        background-color: transparent !important;
    }
    .stChatMessage [data-testid="chatMessage"] {
        background-color: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">📧 Email Spam Classifier Chat</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("🤖 About the Model")
        st.markdown("""
        This AI model classifies emails as **spam** or **legitimate** using:
        
        **Features:**
        - Text length
        - Word count
        - Presence of exclamation marks
        - Urgent/spam keywords
        
        **Algorithm:** Random Forest Classifier
        
        **Accuracy:** ~95%
        """)
        
        st.header("📊 Model Status")
        
        # Load or train model
        classifier, model_loaded = load_or_train_model()
        
        if model_loaded:
            st.success("✅ Model Ready")
        else:
            st.info("🔄 Model Trained")
        
        st.header("💡 Tips")
        st.markdown("""
        **Spam indicators:**
        - Urgent language ("URGENT", "ACT NOW")
        - Promises of money/prizes
        - Excessive exclamation marks
        - Limited time offers
        
        **Legitimate indicators:**
        - Professional language
        - Specific meeting times
        - Business context
        - Normal punctuation
        """)
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        st.markdown("Enter email text below to classify it as spam or legitimate:")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Enter email text here..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Classify the email
            with st.chat_message("assistant"):
                with st.spinner("Analyzing email..."):
                    result = classify_email(classifier, prompt)
                    
                    if result:
                        # Determine message style
                        if result['is_spam']:
                            message_style = "spam-message"
                            icon = "🚨"
                            status = "SPAM"
                        else:
                            message_style = "legitimate-message"
                            icon = "✅"
                            status = "LEGITIMATE"
                        
                        # Create response message with better styling
                        response = f"""
                        <div class="chat-message {message_style}">
                            <h3 style="margin: 0 0 1rem 0; font-size: 1.2rem;">{icon} Classification: {status}</h3>
                            <div style="margin-bottom: 0.5rem;">
                                <strong>Confidence:</strong> {result['confidence']}
                            </div>
                            <div style="margin-bottom: 0.5rem;">
                                <strong>Legitimate Probability:</strong> {result['probabilities'][0]:.1%}
                            </div>
                            <div style="margin-bottom: 0;">
                                <strong>Spam Probability:</strong> {result['probabilities'][1]:.1%}
                            </div>
                        </div>
                        """
                        
                        st.markdown(response, unsafe_allow_html=True)
                        
                        # Add assistant message to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": f"Classification: {status} (Confidence: {result['confidence']})"
                        })
    
    with col2:
        st.header("📈 Statistics")
        
        # Calculate statistics
        total_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
        spam_count = 0
        legitimate_count = 0
        
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "assistant" and "Classification:" in message["content"]:
                if "SPAM" in message["content"]:
                    spam_count += 1
                elif "LEGITIMATE" in message["content"]:
                    legitimate_count += 1
        
        # Display metrics
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.metric("Total Emails", total_messages)
            st.metric("Spam Detected", spam_count)
        
        with col_b:
            st.metric("Legitimate", legitimate_count)
            if total_messages > 0:
                spam_rate = (spam_count / total_messages) * 100
                st.metric("Spam Rate", f"{spam_rate:.1f}%")
        
        # Sample emails for quick testing
        st.header("🧪 Sample Emails")
        st.markdown("Try these examples:")
        
        sample_emails = [
            "Meeting tomorrow at 3 PM in conference room A",
            "Get rich quick! Click here for amazing offers!",
            "Please review the attached quarterly report",
            "URGENT: Your account has been suspended!",
            "Thank you for your application, we will contact you soon",
            "Limited time offer! 90% off everything!"
        ]
        
        for i, email in enumerate(sample_emails):
            if st.button(f"Sample {i+1}", key=f"sample_{i}"):
                # Simulate chat input
                st.session_state.messages.append({"role": "user", "content": email})
                st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        Built with ❤️ using Streamlit, Scikit-learn, and UV | 
        <a href="https://github.com/yourusername/simple-ai-project" target="_blank">GitHub</a>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 