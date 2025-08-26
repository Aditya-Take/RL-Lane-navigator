#!/usr/bin/env python3
"""
Quick demonstration of autonomous driving scenarios.
Shows the environment and agent behavior without requiring trained models.
"""

import gymnasium as gym
import numpy as np
import time
import argparse
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environments.env_configs import create_environment
from environments.wrappers import create_wrapped_environment


def simple_heuristic_agent(obs, info):
    """Simple heuristic-based agent for demonstration."""
    # Extract basic information
    if isinstance(obs, dict):
        # For image observations
        if 'image' in obs:
            # Simple lane following heuristic
            # This is a placeholder - in reality you'd process the image
            action = np.array([0.0, 0.5])  # [steering, acceleration]
        else:
            # For state observations
            action = np.array([0.0, 0.5])
    else:
        # For array observations
        action = np.array([0.0, 0.5])
    
    # Add some randomness for demonstration
    action[0] += np.random.normal(0, 0.1)  # Random steering
    action[1] += np.random.normal(0, 0.1)  # Random acceleration
    
    # Clip actions to valid range
    action = np.clip(action, -1, 1)
    
    return action


def run_quick_demo(scenario_name, duration=30, agent_type='heuristic'):
    """Run a quick demonstration of a driving scenario."""
    print(f"🚗 Quick Demo: {scenario_name.upper()}")
    print("="*50)
    
    # Map scenario names to environment names
    scenario_map = {
        'highway': 'highway-v0',
        'intersection': 'intersection-v0', 
        'roundabout': 'roundabout-v0',
        'parking': 'parking-v0'
    }
    
    env_name = scenario_map.get(scenario_name.lower(), 'highway-v0')
    
    # Create environment
    print(f"🌍 Creating {scenario_name} environment...")
    env = create_environment(env_name)
    
    # Apply basic wrappers for better visualization
    wrapper_config = {
        'observation_preprocessing': {
            'enabled': True,
            'target_size': (84, 84),
            'normalize': True
        },
        'frame_stack': {
            'enabled': False
        },
        'domain_randomization': {
            'enabled': False
        },
        'reward_shaping': {
            'enabled': True
        },
        'multi_agent': {
            'enabled': True,
            'vehicles_count': 10
        }
    }
    
    wrapped_env = create_wrapped_environment(env, wrapper_config)
    
    # Run simulation
    obs, info = wrapped_env.reset()
    total_reward = 0
    steps = 0
    start_time = time.time()
    
    print(f"⏱️  Running for {duration} seconds...")
    print("🎮 Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        while time.time() - start_time < duration:
            # Get action
            if agent_type == 'heuristic':
                action = simple_heuristic_agent(obs, info)
            elif agent_type == 'random':
                action = wrapped_env.action_space.sample()
            else:
                action = np.array([0.0, 0.5])  # Default straight driving
            
            # Take step
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            total_reward += reward
            steps += 1
            
            # Display info every 20 steps
            if steps % 20 == 0:
                speed = info.get('speed', 0)
                lane = info.get('lane_index', 0)
                print(f"📍 Step {steps:3d} | Reward: {reward:6.2f} | Speed: {speed:5.1f} | Lane: {lane}")
            
            # Check for termination
            if terminated or truncated:
                print(f"🏁 Episode ended after {steps} steps")
                obs, info = wrapped_env.reset()
                total_reward = 0
                steps = 0
            
            # Render
            wrapped_env.render()
            time.sleep(0.05)  # Control speed
            
    except KeyboardInterrupt:
        print("\n⏹️  Demo stopped by user")
    
    finally:
        wrapped_env.close()
        print(f"✅ Demo completed. Total steps: {steps}")


def run_all_scenarios_quick(duration=20):
    """Run quick demos for all scenarios."""
    scenarios = ['highway', 'intersection', 'roundabout', 'parking']
    
    print("🌍 Quick Demo: All Scenarios")
    print("="*50)
    
    for scenario in scenarios:
        print(f"\n🎯 Scenario: {scenario.upper()}")
        run_quick_demo(scenario, duration, 'heuristic')
        time.sleep(2)  # Pause between scenarios


def interactive_quick_demo():
    """Interactive menu for quick demos."""
    while True:
        print("\n" + "="*50)
        print("🚗 QUICK DEMO MENU")
        print("="*50)
        print("1. Highway Demo")
        print("2. Intersection Demo") 
        print("3. Roundabout Demo")
        print("4. Parking Demo")
        print("5. All Scenarios")
        print("6. Exit")
        print("="*50)
        
        choice = input("Enter your choice (1-6): ").strip()
        
        if choice == '1':
            run_quick_demo('highway', 30, 'heuristic')
        elif choice == '2':
            run_quick_demo('intersection', 30, 'heuristic')
        elif choice == '3':
            run_quick_demo('roundabout', 30, 'heuristic')
        elif choice == '4':
            run_quick_demo('parking', 30, 'heuristic')
        elif choice == '5':
            run_all_scenarios_quick(20)
        elif choice == '6':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter a number between 1-6.")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Quick demonstration of driving scenarios")
    parser.add_argument("--scenario", type=str, choices=['highway', 'intersection', 'roundabout', 'parking'],
                       default='highway', help="Scenario to demo")
    parser.add_argument("--duration", type=int, default=30,
                       help="Demo duration in seconds")
    parser.add_argument("--agent", type=str, choices=['heuristic', 'random'],
                       default='heuristic', help="Agent type")
    parser.add_argument("--all", action="store_true",
                       help="Run all scenarios")
    parser.add_argument("--interactive", action="store_true",
                       help="Run interactive menu")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_quick_demo()
    elif args.all:
        run_all_scenarios_quick(args.duration)
    else:
        run_quick_demo(args.scenario, args.duration, args.agent)


if __name__ == "__main__":
    main()