# Installation Guide

## Quick Setup

### 1. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Or install individually
pip install highway-env>=1.8.0
pip install gymnasium>=0.29.0
pip install stable-baselines3>=2.1.0
pip install torch>=2.0.0
pip install numpy>=1.24.0
pip install opencv-python>=4.8.0
pip install matplotlib>=3.7.0
pip install tensorboard>=2.13.0
pip install imitation>=0.3.0
pip install scikit-learn>=1.3.0
pip install tqdm>=4.65.0
pip install pillow>=10.0.0
pip install seaborn>=0.12.0
pip install wandb>=0.15.0
```

### 2. Test Installation

```bash
# Test environment creation
python test_environment.py

# Test quick demo
python quick_demo.py --scenario highway --duration 10
```

## Troubleshooting

### Import Error: `cannot import name 'make' from 'highway_env'`

This error occurs with newer versions of `highway_env`. The project includes a compatibility layer to handle this.

**Solution 1: Use the compatibility layer (Recommended)**
The project automatically handles this with the `highway_env_compat.py` module.

**Solution 2: Install specific version**
```bash
pip uninstall highway-env
pip install highway-env==1.8.0
```

**Solution 3: Use gymnasium directly**
```bash
pip install gymnasium[highway-env]
```

### Missing Dependencies

If you get import errors for other packages:

```bash
# Install missing packages
pip install <package_name>

# Or reinstall all requirements
pip install -r requirements.txt --force-reinstall
```

### Environment Issues

If you're using a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Verification

After installation, run these tests:

```bash
# 1. Test imports
python test_environment.py

# 2. Test quick demo
python quick_demo.py --scenario highway --duration 5

# 3. Test live demo (if models exist)
python live_demo.py --scenario highway --duration 10
```

## Common Issues

### 1. PyTorch Installation
If PyTorch fails to install:

```bash
# For CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For CUDA (if you have NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. OpenCV Issues
If OpenCV fails to install:

```bash
# Alternative OpenCV installation
pip install opencv-python-headless
```

### 3. Highway-env Rendering Issues
If you get rendering errors:

```bash
# Install additional dependencies
pip install pygame
pip install pyglet
```

## System Requirements

- Python 3.8+
- 4GB+ RAM
- 2GB+ free disk space
- Optional: NVIDIA GPU for faster training

## Next Steps

After successful installation:

1. **Run Quick Demo**: `python quick_demo.py --interactive`
2. **Train Models**: `python scripts/train_il.py`
3. **Live Simulation**: `python live_demo.py --interactive`
4. **View Results**: `python view_visualizations.py --interactive`