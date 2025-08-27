#!/usr/bin/env python3
"""
Setup script for the Vision-Based Autonomous Driving Agent project.
"""

import os
import sys
import subprocess
from pathlib import Path


def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False
    return True


def create_directories():
    """Create necessary directories."""
    print("Creating project directories...")
    directories = [
        "data",
        "models", 
        "plots",
        "logs",
        "configs",
        "data/expert_demonstrations",
        "data/human_demonstrations",
        "models/cnn_imitation",
        "models/ppo_driving",
        "models/hybrid_agent",
        "plots/training",
        "plots/evaluation",
        "logs/tensorboard",
        "logs/wandb"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  Created: {directory}")
    
    print("✅ Directories created successfully!")


def check_dependencies():
    """Check if all dependencies are available."""
    print("Checking dependencies...")
    
    required_packages = [
        "torch",
        "torchvision", 
        "gymnasium",
        "highway_env",
        "stable_baselines3",
        "opencv-python",
        "matplotlib",
        "seaborn",
        "numpy",
        "tqdm",
        "pyyaml",
        "pillow",
        "scikit-learn",
        "tensorboard",
        "imitation",
        "wandb"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are available!")
    return True


def test_installation():
    """Test the installation by running a simple import test."""
    print("Testing installation...")
    
    try:
        # Test basic imports
        from utils.data_utils import load_config
        from models.cnn_models import DrivingCNN
        from agents.cnn_agent import CNNImitationAgent
        from environments.env_configs import create_environment
        
        print("✅ Basic imports successful!")
        
        # Test configuration loading
        config = load_config("configs/training_config.yaml")
        print("✅ Configuration loading successful!")
        
        # Test model creation
        model = DrivingCNN()
        print("✅ Model creation successful!")
        
        print("✅ Installation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False


def main():
    """Main setup function."""
    print("🚗 Vision-Based Autonomous Driving Agent Setup")
    print("="*50)
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("❌ Python 3.9 or higher is required!")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Create directories
    create_directories()
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Test installation
    if not test_installation():
        return False
    
    print("\n" + "="*50)
    print("🎉 Setup completed successfully!")
    print("="*50)
    print("\nYou can now:")
    print("1. Run the demo: python demo.py")
    print("2. Train imitation learning: python scripts/train_il.py --collect_data")
    print("3. Train reinforcement learning: python scripts/train_rl.py")
    print("4. Evaluate agents: python scripts/evaluate.py --agent_type rl --agent_path ./models/ppo_driving")
    print("\nFor more information, see the README.md file.")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)