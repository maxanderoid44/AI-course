Write-Output "Start unarchive libraries"
if (Test-Path -Path "libraries.zip") {
    Write-Output "libraries.zip found"
    if (!(Test-Path -Path "libraries")) {
        Write-Output "libraries folder not found, extracting archive"
        Expand-Archive -Path libraries.zip -DestinationPath libraries
    } else {
        Write-Output "libraries folder already exists, skipping extraction"
    }
} else {
    Write-Output "libraries.zip not found, please download it from the repository"
    exit
}

if (!(Test-Path -Path ".venv")) {
    Write-Output "Creating virtual environment"
    uv venv
} else {
    Write-Output "Virtual environment already exists, skipping creation"
}

Write-Output "Activate virtual environment"
.\.venv\Scripts\activate.ps1

Write-Output "Install libraries"
uv pip install opencv-python --no-index --find-links libraries  
uv pip install matplotlib --no-index --find-links libraries 
uv pip install pandas --no-index --find-links libraries  
uv pip install numpy --no-index --find-links libraries  
uv pip install scikit-learn --no-index --find-links libraries  
uv pip install ipykernel --no-index --find-links libraries  
uv pip install tensorflow --no-index --find-links libraries 
uv pip install transformers --no-index --find-links libraries  
uv pip install torch --no-index --find-links libraries  
uv pip install datasets --no-index --find-links libraries  
uv pip install keras --no-index --find-links libraries  
uv pip install torchvision --no-index --find-links libraries  
uv pip install pillow --no-index --find-links libraries  
uv pip install flask --no-index --find-links libraries  
uv pip install streamlit --no-index --find-links libraries  
uv pip install gradio --no-index --find-links libraries  

Write-Output "Done"
