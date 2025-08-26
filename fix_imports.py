#!/usr/bin/env python3
"""
Script to fix import issues and test the environment setup.
"""

import subprocess
import sys
import os
from pathlib import Path


def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False


def check_package(package):
    """Check if a package is installed."""
    try:
        __import__(package)
        return True
    except ImportError:
        return False


def fix_highway_env_import():
    """Fix highway_env import issues."""
    print("🔧 Fixing highway_env import issues...")
    
    # Try different approaches
    approaches = [
        ("highway-env==1.8.0", "Installing specific version"),
        ("gymnasium[highway-env]", "Installing gymnasium with highway-env"),
        ("highway-env", "Installing latest highway-env"),
    ]
    
    for package, description in approaches:
        print(f"  {description}...")
        if install_package(package):
            print(f"  ✅ {package} installed successfully")
            return True
        else:
            print(f"  ❌ Failed to install {package}")
    
    return False


def install_missing_dependencies():
    """Install missing dependencies."""
    print("📦 Installing missing dependencies...")
    
    required_packages = [
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "stable-baselines3>=2.1.0",
        "opencv-python>=4.8.0",
        "matplotlib>=3.7.0",
        "tensorboard>=2.13.0",
        "imitation>=0.3.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
        "pillow>=10.0.0",
        "seaborn>=0.12.0",
        "wandb>=0.15.0"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        package_name = package.split('>=')[0].split('==')[0]
        if not check_package(package_name):
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Found {len(missing_packages)} missing packages:")
        for package in missing_packages:
            print(f"  - {package}")
        
        print("\nInstalling missing packages...")
        for package in missing_packages:
            print(f"Installing {package}...")
            if install_package(package):
                print(f"✅ {package} installed")
            else:
                print(f"❌ Failed to install {package}")
    else:
        print("✅ All required packages are installed")


def test_environment():
    """Test if the environment works correctly."""
    print("🧪 Testing environment...")
    
    try:
        # Test basic imports
        import gymnasium as gym
        print("✅ gymnasium imported")
        
        import numpy as np
        print("✅ numpy imported")
        
        # Test highway_env import
        try:
            import highway_env
            print("✅ highway_env imported")
        except ImportError:
            print("⚠️  highway_env not available")
        
        # Test environment creation
        try:
            from environments.env_configs import create_environment
            print("✅ environment configs imported")
            
            # Try to create a simple environment
            env = create_environment("highway-v0")
            print("✅ Environment created successfully")
            
            # Test basic operations
            obs, info = env.reset()
            print("✅ Environment reset successful")
            
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print("✅ Environment step successful")
            
            env.close()
            print("✅ Environment closed successfully")
            
            return True
            
        except Exception as e:
            print(f"❌ Environment creation failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False


def main():
    """Main function."""
    print("🚗 Autonomous Driving Agent - Import Fix")
    print("="*50)
    
    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found. Please run this script from the project root directory.")
        return
    
    # Install missing dependencies
    install_missing_dependencies()
    
    # Fix highway_env import
    fix_highway_env_import()
    
    # Test environment
    if test_environment():
        print("\n🎉 Environment setup successful!")
        print("\nNext steps:")
        print("1. Run quick demo: python quick_demo.py --interactive")
        print("2. Train models: python scripts/train_il.py")
        print("3. Live simulation: python live_demo.py --interactive")
    else:
        print("\n❌ Environment setup failed. Please check the error messages above.")
        print("\nTroubleshooting:")
        print("1. Make sure you have Python 3.8+ installed")
        print("2. Try creating a virtual environment")
        print("3. Check the INSTALL.md file for more details")


if __name__ == "__main__":
    main()