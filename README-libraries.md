# Setting Up the Python Environment with `uv` and Local Libraries

This project uses a set of pre-downloaded Python libraries and the [`uv`](https://github.com/astral-sh/uv) package manager for fast, offline installation. Follow these steps to set up your environment:

## Prerequisites
- Python 3.8 or newer installed
- [`uv`](https://github.com/astral-sh/uv) installed globally (`pip install uv`)
- PowerShell (recommended for running the script)

## Important
- Ensure that both `installEnvWIthLibrary.ps1` and `libraries.zip` are located in the root directory of your project (the same folder as this README). The script relies on these files being in the project path to work correctly.

## Installation Steps

1. **Download the Libraries Archive**
   - Ensure you have `libraries.zip` in the project root. If not, download it from the repository or ask your instructor.

2. **Run the Setup Script**
   - Open PowerShell in the project directory.
   - Run the following command:
     ```powershell
     .\installEnvWIthLibrary.ps1
     ```
   - The script will:
     - Extract `libraries.zip` if needed
     - Create a virtual environment using `uv`
     - Activate the virtual environment
     - Install all required libraries from the local `libraries` folder (offline, no internet required)

3. **Activate the Virtual Environment (if not already active)**
   - If you open a new terminal, activate with:
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```

## Included Libraries
- opencv-python
- matplotlib
- pandas
- numpy
- scikit-learn
- ipykernel
- tensorflow
- transformers
- torch
- datasets
- keras
- torchvision
- pillow
- flask
- streamlit
- gradio

## Notes
- All installations are performed offline using the local `libraries` folder.
- If you need to add more libraries, place their wheels in the `libraries` folder and add the corresponding `uv pip install ...` command to `installEnvWIthLibrary.ps1`.

## Troubleshooting
- If you see an error about missing `libraries.zip`, ensure it is present in the project root.
- If `uv` is not recognized, install it globally with `pip install uv`.

---
For more details, see the comments in `installEnvWIthLibrary.ps1`.
