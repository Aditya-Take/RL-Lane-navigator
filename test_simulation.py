#!/usr/bin/env python3
"""
Test script to verify simulation capabilities work correctly.
This script runs a quick test of all scenarios to ensure everything is set up properly.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from utils.data_utils import load_config
        print("✅ utils.data_utils imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import utils.data_utils: {e}")
        return False
    
    try:
        from environments.env_configs import create_environment
        print("✅ environments.env_configs imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import environments.env_configs: {e}")
        return False
    
    try:
        from environments.wrappers import create_wrapped_environment
        print("✅ environments.wrappers imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import environments.wrappers: {e}")
        return False
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import OpenCV: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import NumPy: {e}")
        return False
    
    return True

def test_config_loading():
    """Test that configuration can be loaded."""
    print("\nTesting configuration loading...")
    
    try:
        from utils.data_utils import load_config
        config = load_config("configs/training_config.yaml")
        print("✅ Configuration loaded successfully")
        print(f"   Default environment: {config['environment']['name']}")
        return True
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return False

def test_environment_creation():
    """Test that environments can be created."""
    print("\nTesting environment creation...")
    
    try:
        from utils.data_utils import load_config
        from environments.env_configs import create_environment
        
        config = load_config("configs/training_config.yaml")
        
        # Test each scenario
        scenarios = ["highway-v0", "intersection-v0", "roundabout-v0", "parking-v0"]
        
        for scenario in scenarios:
            try:
                env = create_environment(scenario, config=config['environment']['config'])
                print(f"✅ {scenario} environment created successfully")
                env.close()
            except Exception as e:
                print(f"❌ Failed to create {scenario}: {e}")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Failed to test environment creation: {e}")
        return False

def test_quick_simulation():
    """Test the quick simulation script."""
    print("\nTesting quick simulation...")
    
    try:
        from quick_simulation import QuickSimulation
        
        simulator = QuickSimulation()
        print("✅ QuickSimulation class created successfully")
        
        # Test environment creation
        env = simulator.create_environment("highway")
        print("✅ Highway environment created through QuickSimulation")
        
        # Test simple agent creation
        from quick_simulation import SimpleAgent
        agent = SimpleAgent(env.action_space)
        print("✅ SimpleAgent created successfully")
        
        # Test a few steps without rendering
        obs, _ = env.reset()
        for i in range(10):
            action = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        print("✅ Quick simulation test completed successfully")
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Failed to test quick simulation: {e}")
        return False

def test_simulation_viewer():
    """Test the simulation viewer script."""
    print("\nTesting simulation viewer...")
    
    try:
        from simulation_viewer import SimulationViewer
        
        viewer = SimulationViewer()
        print("✅ SimulationViewer class created successfully")
        
        # Test environment creation
        env = viewer.create_environment("highway")
        print("✅ Highway environment created through SimulationViewer")
        
        # Test agent loading (without actual model files)
        try:
            agent = viewer.load_agent("rl")
            print("✅ RL agent loading test completed (using untrained model)")
        except Exception as e:
            print(f"⚠️  RL agent loading test failed (expected without model files): {e}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Failed to test simulation viewer: {e}")
        return False

def run_quick_demo():
    """Run a very quick demo to show everything works."""
    print("\nRunning quick demo...")
    
    try:
        from quick_simulation import QuickSimulation
        
        simulator = QuickSimulation()
        
        # Run a very short demo
        print("Running 50-step highway demo...")
        results = simulator.run_demo(
            scenario="highway",
            render=False,  # No rendering for quick test
            save_video=False,
            max_steps=50
        )
        
        print(f"✅ Demo completed successfully!")
        print(f"   Steps: {results['steps']}")
        print(f"   Total Reward: {results['total_reward']:.2f}")
        print(f"   Success: {results['success']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to run quick demo: {e}")
        return False

def main():
    """Main test function."""
    print("="*60)
    print("SIMULATION CAPABILITIES TEST")
    print("="*60)
    
    tests = [
        ("Import Test", test_imports),
        ("Config Loading", test_config_loading),
        ("Environment Creation", test_environment_creation),
        ("Quick Simulation", test_quick_simulation),
        ("Simulation Viewer", test_simulation_viewer),
        ("Quick Demo", run_quick_demo)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*40}")
        print(f"Running: {test_name}")
        print(f"{'='*40}")
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Simulation capabilities are working correctly.")
        print("\nYou can now run:")
        print("  python quick_simulation.py --scenario highway --render")
        print("  python simulation_viewer.py --interactive")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Check you're in the project root directory")
        print("  3. Ensure all required files are present")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)