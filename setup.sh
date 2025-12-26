#!/bin/bash

# Setup script for YOLO training on Linux/Windows with NVIDIA GPU

echo "🚀 Setting up YOLO Training Environment..."
echo ""

# Check Python version
echo "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
    PIP_CMD=pip3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
    PIP_CMD=pip
else
    echo "❌ Python not found! Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Found Python: $($PYTHON_CMD --version)"
echo ""

# Create virtual environment (optional but recommended)
echo "Creating virtual environment..."
$PYTHON_CMD -m venv venv
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null
echo "✅ Virtual environment created and activated"
echo ""

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
echo "This may take a few minutes..."
$PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo "✅ PyTorch installed"
echo ""

# Install Ultralytics YOLO
echo "Installing Ultralytics YOLO..."
$PIP_CMD install ultralytics
echo "✅ Ultralytics installed"
echo ""

# Verify GPU availability
echo "Checking GPU availability..."
$PYTHON_CMD -c "import torch; print('✅ CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
echo ""

echo "================================================"
echo "✅ Setup Complete!"
echo "================================================"
echo ""
echo "To start training, run:"
echo "  python train_simple.py"
echo ""
echo "To activate this environment later, run:"
echo "  source venv/bin/activate    (Linux/Mac)"
echo "  venv\\Scripts\\activate      (Windows)"
echo ""
