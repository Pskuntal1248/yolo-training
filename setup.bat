@echo off
REM Setup script for YOLO training on Windows with NVIDIA GPU

echo 🚀 Setting up YOLO Training Environment...
echo.

REM Check Python installation
echo Checking Python installation...
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found! Please install Python 3.8 or higher.
    pause
    exit /b 1
)

python --version
echo ✅ Python found
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat
echo ✅ Virtual environment created and activated
echo.

REM Install PyTorch with CUDA support
echo Installing PyTorch with CUDA support...
echo This may take a few minutes...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo ✅ PyTorch installed
echo.

REM Install Ultralytics YOLO
echo Installing Ultralytics YOLO...
pip install ultralytics
echo ✅ Ultralytics installed
echo.

REM Verify GPU availability
echo Checking GPU availability...
python -c "import torch; print('✅ CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
echo.

echo ================================================
echo ✅ Setup Complete!
echo ================================================
echo.
echo To start training, run:
echo   python train_simple.py
echo.
echo To activate this environment later, run:
echo   venv\Scripts\activate.bat
echo.
pause
