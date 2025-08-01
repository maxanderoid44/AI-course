# AI Project Requirements

## Overview
This document outlines the essential requirements for all AI projects in this course. Each project must meet these standards to ensure proper development practices, collaboration, and documentation.

## 📋 Project Requirements

### 1. Git Repository Management
- **Mandatory**: All projects must be uploaded to Git (GitHub, GitLab, or similar)
- **Repository Structure**:
  ```
  project-name/
  ├── README.md
  ├── pyproject.toml
  ├── .python-version
  ├── src/
  ├── notebooks/
  ├── data/
  └── docs/
  ```

#### Git Setup Steps:
1. Initialize Git repository: `git init`
2. Add remote origin: `git remote add origin <repository-url>`
3. Create initial commit: `git add . && git commit -m "Initial commit"`
4. Push to remote: `git push -u origin main`

#### Git Best Practices:
- Use meaningful commit messages
- Create feature branches for new development
- Keep commits atomic and focused
- Update README.md with project changes

### 2. UV Tool Integration
- **Mandatory**: All projects must use UV for dependency management
- **Benefits**: Faster package installation, better dependency resolution, and reproducible environments

#### UV Setup Requirements:
1. **Install UV**: Follow official documentation at https://docs.astral.sh/uv/
2. **Initialize Project**: `uv init` or `uv init --python 3.12`
3. **Add Dependencies**: `uv add <package-name>`
4. **Install Dependencies**: `uv sync`
5. **Run Scripts**: `uv run python script.py`

#### Required UV Files:
- `pyproject.toml` - Project configuration and dependencies
- `uv.lock` - Locked dependency versions (auto-generated)
- `.python-version` - Python version specification (optional but recommended)

#### Example pyproject.toml Structure:
```toml
[project]
name = "your-project-name"
version = "0.1.0"
description = "Brief description of your AI project"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "numpy>=1.26.0",
    "pandas>=2.3.0",
    "scikit-learn>=1.7.0",
    "matplotlib>=3.10.3",
    "jupyter>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
]

```

### 3. README Documentation
- **Mandatory**: Every project must include a comprehensive README.md file
- **Purpose**: Document project purpose, setup instructions, and team contributions

#### README.md Required Sections:

##### 1. Project Title and Description
```markdown
# Project Name

Brief description of what the AI project does, its goals, and key features.
```

##### 2. Team Information
## Team Members

| AC.NO | Name | Role | Contributions |
|----|------|------|---------------|
| 1 | Student Name 1 | Lead Developer | Data preprocessing, model development |
| 2 | Student Name 2 | Data Analyst | EDA, visualization |
| 3 | Student Name 3 | ML Engineer | Model optimization, deployment |


##### 3. Project Setup
```markdown
## Installation and Setup

### Prerequisites
- Python 3.12 or higher
- UV package manager

### Installation Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd project-name
   ```

2. Install dependencies using UV:
   ```bash
   uv sync
   ```

3. Run the project:
   ```bash
   uv run python main.py
   ```


##### 4. Project Structure
## Project Structure

```
project-name/
├── README.md              # Project documentation
├── pyproject.toml         # UV project configuration
├── main.py               # Main application entry point
├── src/                  # Source code
│   ├── data/            # Data processing modules
│   ├── models/          # ML model implementations
│   └── utils/           # Utility functions
├── notebooks/           # Jupyter notebooks
├── data/               # Dataset files
└── docs/               # Additional documentation
```


##### 5. Usage Examples
```markdown
## Usage

### Basic Usage
```python
from src.models import YourModel

# Load and train model
model = YourModel()
model.train(data)
```

### Running Experiments
```bash
uv run python experiments/train_model.py
```
```

##### 6. Results and Performance
```markdown
## Results

- Model Accuracy: XX%
- Training Time: XX minutes
- Key Findings: [Brief summary of results]
```

##### 7. Contributing
```markdown
## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Commit changes: `git commit -m 'Add feature'`
5. Push to branch: `git push origin feature-name`
6. Submit a pull request
```

## 📊 Project Evaluation Criteria

### Technical Requirements (40%)
- ✅ Git repository properly configured
- ✅ UV tool integration working
- ✅ All dependencies properly managed
- ✅ Code runs without errors

### Documentation (30%)
- ✅ Comprehensive README.md
- ✅ Clear project description
- ✅ Setup instructions work
- ✅ Team contributions documented

### Code Quality (20%)
- ✅ Clean, readable code
- ✅ Proper project structure
- ✅ Good coding practices

### Collaboration (10%)
- ✅ Team members contributed equally
- ✅ Git history shows collaboration
- ✅ Code reviews completed

## 🚀 Getting Started Checklist

Before submitting your project, ensure you have:

- [ ] Created Git repository and pushed code
- [ ] Set up UV project with `pyproject.toml`
- [ ] Installed all dependencies with `uv sync`
- [ ] Created comprehensive README.md
- [ ] Documented team member contributions
- [ ] Tested project setup from scratch
- [ ] Verified all code runs correctly
- [ ] Added appropriate .gitignore file
- [ ] Included sample data or instructions to obtain data

## 📝 Submission Guidelines

1. **Repository URL**: Submit the Git repository URL
2. **README.md**: Ensure it's complete and up-to-date
3. **Demo**: Be prepared to demonstrate your project
4. **Documentation**: Have setup instructions ready

## 📚 Example Project: Simple AI Project

### Project Overview
The **Simple AI Project - Email Spam Classifier** serves as a complete example that meets all requirements:

**Repository**: `simple-ai-project/`
**Type**: Email Spam Classification using Machine Learning
**Technology Stack**: Python, scikit-learn, pandas, matplotlib

### Key Features Demonstrated:
- ✅ **Git Integration**: Complete repository with proper structure
- ✅ **UV Tool Usage**: `pyproject.toml` with all dependencies
- ✅ **Comprehensive README**: Team documentation, setup instructions, usage examples
- ✅ **Modular Code**: Separate modules for data, models, and utilities
- ✅ **Documentation**: Inline comments and separate docs folder
- ✅ **Jupyter Notebooks**: Data exploration and analysis
- ✅ **Sample Data**: Ready-to-use dataset

### Project Structure:
```
simple-ai-project/
├── README.md              # Complete documentation
├── pyproject.toml         # UV configuration
├── main.py               # Entry point
├── .gitignore            # Concise ignore file
├── src/                  # Source code
│   ├── models/          # ML implementations
│   ├── data/            # Data processing
│   └── utils/           # Helper functions
├── notebooks/           # Jupyter notebooks
├── data/               # Sample datasets
└── docs/               # Documentation
```

### Running the Example:
```bash
# Clone and setup
cd simple-ai-project
uv sync
uv run python main.py
```

### What You'll Learn:
1. **Proper Project Structure**: How to organize AI projects
2. **UV Integration**: Modern Python dependency management
3. **Documentation**: Writing comprehensive README files
4. **Code Organization**: Modular, maintainable code
5. **Git Best Practices**: Version control for AI projects

---

## 🔧 Troubleshooting

### Common UV Issues:
- **Permission Errors**: Use `uv sync --no-cache` to clear cache
- **Python Version**: Ensure Python 3.12+ is installed
- **Dependencies**: Check `pyproject.toml` for correct syntax

### Common Git Issues:
- **Large Files**: Use `.gitignore` to exclude data files
- **Authentication**: Set up SSH keys or use personal access tokens
- **Merge Conflicts**: Resolve conflicts before pushing
