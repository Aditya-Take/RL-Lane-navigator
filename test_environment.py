#!/usr/bin/env python3
"""
Test script to verify environment creation works correctly.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_environment_creation():
    """Test creating different environments."""
    print("🧪 Testing Environment Creation...")
    
    try:
        from environments.env_configs import create_environment
        
        # Test highway environment
        print("Testing highway-v0...")
        env = create_environment("highway-v0")
        print(f"✅ Highway environment created: {type(env)}")
        
        # Test observation and action spaces
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        # Test reset
        obs, info = env.reset()
        print(f"✅ Reset successful, observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✅ Step successful, reward: {reward}")
        
        env.close()
        print("✅ Highway environment test passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        return False

def test_imports():
    """Test all imports work correctly."""
    print("🧪 Testing Imports...")
    
    try:
        import gymnasium as gym
        print("✅ gymnasium imported")
        
        import numpy as np
        print("✅ numpy imported")
        
        import torch
        print("✅ torch imported")
        
        from environments.env_configs import create_environment
        print("✅ environments.env_configs imported")
        
        from environments.wrappers import create_wrapped_environment
        print("✅ environments.wrappers imported")
        
        from agents.cnn_agent import CNNImitationAgent
        print("✅ agents.cnn_agent imported")
        
        from agents.ppo_agent import PPODrivingAgent
        print("✅ agents.ppo_agent imported")
        
        from utils.data_utils import load_config
        print("✅ utils.data_utils imported")
        
        print("✅ All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚗 Autonomous Driving Agent - Environment Test")
    print("="*50)
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        # Test environment creation
        env_ok = test_environment_creation()
        
        if env_ok:
            print("\n🎉 All tests passed! Environment is ready to use.")
        else:
            print("\n❌ Environment creation failed. Please check your installation.")
    else:
        print("\n❌ Import tests failed. Please check your dependencies.")

if __name__ == "__main__":
    main()