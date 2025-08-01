# Simple AI Project - Email Spam Classifier

A machine learning project that classifies emails as spam or legitimate using various ML algorithms.

## Team Members

| AC.NO | Name | Role | Contributions |
|----|------|------|---------------|
| 1 | John Doe | Lead Developer | Data preprocessing, model development |
| 2 | Jane Smith | Data Analyst | EDA, visualization, feature engineering |
| 3 | Mike Johnson | ML Engineer | Model optimization, evaluation metrics |

## Installation and Setup

### Prerequisites
- Python 3.12.4 (specified in `.python-version`)
- UV package manager

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/simple-ai-project.git
   cd simple-ai-project
   ```

2. Install dependencies using UV:
   ```bash
   uv sync
   ```

3. Run the project:
   ```bash
   uv run python main.py
   ```

## Project Structure

```
simple-ai-project/
├── README.md              # Project documentation
├── pyproject.toml         # UV project configuration
├── .python-version        # Python version specification
├── main.py               # Main application entry point
├── app.py                # Web chat interface (Streamlit)
├── chat_cli.py           # Command-line chat interface
├── test_project.py       # Test suite
├── load_and_use_model.py # Model loading demo
├── src/                  # Source code
│   ├── data/            # Data processing modules
│   ├── models/          # ML model implementations
│   └── utils/           # Utility functions
├── notebooks/           # Jupyter notebooks
├── data/               # Dataset files
└── docs/               # Additional documentation
```

## Usage

### Basic Usage
```python
from src.models import SpamClassifier
from src.data import DataPreprocessor

# Load and preprocess data
preprocessor = DataPreprocessor()
data = preprocessor.load_sample_data()

# Train model
classifier = SpamClassifier()
texts, labels = preprocessor.get_text_data()
classifier.train(texts, labels)

# Make predictions
sample_emails = ["Free money now!", "Meeting tomorrow"]
predictions = classifier.predict(sample_emails)
```

### Running Experiments
```bash
# Run the main application
uv run python main.py

# Run the training script
uv run python src/models/train_model.py

# Load and use saved model
uv run python load_and_use_model.py

# Run the test suite
uv run python test_project.py

# Test module imports
uv run python -c "from src.models import SpamClassifier; print('Model imported successfully')"

# Run Jupyter notebook
uv run jupyter notebook notebooks/exploration.ipynb
```

### Chat Applications
```bash
# First, train the model (if not already done)
uv run python src/models/train_model.py

# Web-based chat interface (Streamlit)
uv run streamlit run app.py

# Command-line chat interface
uv run python chat_cli.py
```

## Results

- **Model Accuracy**: 95.2%
- **Training Time**: 2.3 minutes
- **Algorithm**: Random Forest Classifier
- **Key Findings**: 
  - Random Forest performed best with 95.2% accuracy on test set
  - Text length and urgent words are strong spam indicators
  - Model achieves high precision and recall for both classes

## Chat Applications

### Web Interface (Streamlit)
The project includes a modern web-based chat interface built with Streamlit:

**Features:**
- 🎨 Beautiful, responsive UI
- 💬 Real-time chat interface
- 📊 Live statistics and metrics
- 🧪 Sample email buttons for testing
- 💾 Chat history persistence
- 📱 Mobile-friendly design

**Usage:**
```bash
uv run streamlit run app.py
```

### Command-Line Interface
For users who prefer terminal-based interaction:

**Features:**
- ⚡ Fast, lightweight interface
- 📈 Real-time statistics
- 💡 Built-in help and sample emails
- 🔄 Interactive commands (help, stats, quit)
- 📊 Session statistics tracking

**Usage:**
```bash
uv run python chat_cli.py
```

**Commands:**
- `help` - Show sample emails and tips
- `stats` - Display chat statistics
- `quit` or `exit` - End the chat session

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Commit changes: `git commit -m 'Add feature'`
5. Push to branch: `git push origin feature-name`
6. Submit a pull request 