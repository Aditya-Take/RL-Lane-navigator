#!/usr/bin/env python3
"""
Live demonstration of trained autonomous driving agents in different scenarios.
Shows real-time simulation of the agent driving in highway, intersection, roundabout, and parking environments.
"""

import argparse
import time
import numpy as np
from pathlib import Path
import json
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_utils import load_config
from environments.env_configs import create_environment
from environments.wrappers import create_wrapped_environment
from agents.cnn_agent import CNNImitationAgent
from agents.ppo_agent import PPODrivingAgent, HybridAgent


class LiveDrivingDemo:
    """Live demonstration of autonomous driving agents."""
    
    def __init__(self, config_path="configs/training_config.yaml"):
        """Initialize the live demo."""
        self.config = load_config(config_path)
        self.agents = {}
        self.current_agent = None
        self.current_env = None
        
    def load_agents(self, il_model_path=None, rl_model_path=None):
        """Load trained agents."""
        print("🤖 Loading trained agents...")
        
        # Load IL agent if available
        if il_model_path and Path(il_model_path).exists():
            try:
                self.agents['il'] = CNNImitationAgent(self.config)
                self.agents['il'].load_model(il_model_path)
                print(f"✅ Loaded IL agent from: {il_model_path}")
            except Exception as e:
                print(f"❌ Failed to load IL agent: {e}")
        
        # Load RL agent if available
        if rl_model_path and Path(rl_model_path).exists():
            try:
                # Create a dummy environment for initialization
                dummy_env = create_environment("highway-v0")
                self.agents['rl'] = PPODrivingAgent(dummy_env, self.config)
                self.agents['rl'].load_model(rl_model_path)
                print(f"✅ Loaded RL agent from: {rl_model_path}")
            except Exception as e:
                print(f"❌ Failed to load RL agent: {e}")
        
        # Create hybrid agent if both are available
        if 'il' in self.agents and 'rl' in self.agents:
            self.agents['hybrid'] = HybridAgent(
                self.agents['il'], 
                self.agents['rl'], 
                il_weight=0.3
            )
            print("✅ Created hybrid agent (IL + RL)")
        
        if not self.agents:
            print("⚠️  No trained agents found. Will use random actions for demonstration.")
    
    def create_scenario_environment(self, scenario_name):
        """Create environment for specific scenario."""
        print(f"🌍 Creating {scenario_name} environment...")
        
        # Map scenario names to environment names
        scenario_map = {
            'highway': 'highway-v0',
            'intersection': 'intersection-v0', 
            'roundabout': 'roundabout-v0',
            'parking': 'parking-v0'
        }
        
        env_name = scenario_map.get(scenario_name.lower(), 'highway-v0')
        
        # Create base environment
        env = create_environment(env_name)
        
        # Apply wrappers for better visualization
        wrapper_config = {
            'observation_preprocessing': {
                'enabled': True,
                'target_size': (84, 84),
                'normalize': True
            },
            'frame_stack': {
                'enabled': False  # Disable for live demo
            },
            'domain_randomization': {
                'enabled': False  # Disable for consistent demo
            },
            'reward_shaping': {
                'enabled': True
            },
            'multi_agent': {
                'enabled': True,
                'vehicles_count': 15
            }
        }
        
        self.current_env = create_wrapped_environment(env, wrapper_config)
        print(f"✅ Created {scenario_name} environment")
        
        return self.current_env
    
    def run_live_demo(self, scenario_name, agent_type='hybrid', duration=60, render=True):
        """Run live demonstration of agent driving."""
        print(f"\n🚗 Starting Live Demo: {scenario_name.upper()} - {agent_type.upper()}")
        print("="*60)
        
        # Create environment
        env = self.create_scenario_environment(scenario_name)
        
        # Select agent
        if agent_type in self.agents:
            self.current_agent = self.agents[agent_type]
            print(f"🎯 Using {agent_type.upper()} agent")
        else:
            print("⚠️  Using random actions (no trained agent)")
            self.current_agent = None
        
        # Run simulation
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        start_time = time.time()
        
        print(f"⏱️  Running for {duration} seconds...")
        print("🎮 Controls: Press 'q' to quit, 'r' to reset, 'p' to pause")
        
        try:
            while time.time() - start_time < duration:
                # Get action from agent or random
                if self.current_agent:
                    if agent_type == 'il':
                        action = self.current_agent.predict(obs)
                    elif agent_type == 'rl':
                        action, _ = self.current_agent.predict(obs, deterministic=True)
                    elif agent_type == 'hybrid':
                        action = self.current_agent.predict(obs)
                else:
                    action = env.action_space.sample()
                
                # Take step in environment
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                
                # Display info
                if steps % 30 == 0:  # Every 30 steps
                    speed = info.get('speed', 0)
                    lane = info.get('lane_index', 0)
                    print(f"📍 Step {steps:4d} | Reward: {reward:6.2f} | Speed: {speed:5.1f} | Lane: {lane}")
                
                # Check for termination
                if terminated or truncated:
                    print(f"🏁 Episode ended after {steps} steps")
                    obs, info = env.reset()
                    total_reward = 0
                    steps = 0
                
                # Render if enabled
                if render:
                    env.render()
                    time.sleep(0.05)  # Control simulation speed
                
        except KeyboardInterrupt:
            print("\n⏹️  Demo stopped by user")
        
        finally:
            env.close()
            print(f"✅ Demo completed. Total steps: {steps}")
    
    def run_scenario_comparison(self, scenario_name, duration=30):
        """Compare different agents in the same scenario."""
        print(f"\n🔬 Agent Comparison: {scenario_name.upper()}")
        print("="*60)
        
        available_agents = list(self.agents.keys())
        if not available_agents:
            print("❌ No trained agents available for comparison")
            return
        
        for agent_type in available_agents:
            print(f"\n🎯 Testing {agent_type.upper()} agent...")
            self.run_live_demo(scenario_name, agent_type, duration, render=True)
            time.sleep(2)  # Pause between agents
    
    def run_all_scenarios(self, agent_type='hybrid', duration=30):
        """Run demo across all scenarios."""
        scenarios = ['highway', 'intersection', 'roundabout', 'parking']
        
        print(f"\n🌍 Running All Scenarios with {agent_type.upper()} Agent")
        print("="*60)
        
        for scenario in scenarios:
            print(f"\n🎯 Scenario: {scenario.upper()}")
            self.run_live_demo(scenario, agent_type, duration, render=True)
            time.sleep(3)  # Pause between scenarios
    
    def interactive_demo_menu(self):
        """Interactive menu for live demonstrations."""
        while True:
            print("\n" + "="*60)
            print("🚗 AUTONOMOUS DRIVING - LIVE DEMO MENU")
            print("="*60)
            print("1. Highway Demo")
            print("2. Intersection Demo") 
            print("3. Roundabout Demo")
            print("4. Parking Demo")
            print("5. Agent Comparison (Highway)")
            print("6. All Scenarios Demo")
            print("7. Custom Demo")
            print("8. Exit")
            print("="*60)
            
            choice = input("Enter your choice (1-8): ").strip()
            
            if choice == '1':
                self.run_live_demo('highway', 'hybrid', 60, True)
            elif choice == '2':
                self.run_live_demo('intersection', 'hybrid', 60, True)
            elif choice == '3':
                self.run_live_demo('roundabout', 'hybrid', 60, True)
            elif choice == '4':
                self.run_live_demo('parking', 'hybrid', 60, True)
            elif choice == '5':
                self.run_scenario_comparison('highway', 30)
            elif choice == '6':
                self.run_all_scenarios('hybrid', 30)
            elif choice == '7':
                self.custom_demo()
            elif choice == '8':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter a number between 1-8.")
    
    def custom_demo(self):
        """Custom demo with user-selected parameters."""
        print("\n🎛️  Custom Demo Configuration")
        print("-" * 40)
        
        # Select scenario
        scenarios = ['highway', 'intersection', 'roundabout', 'parking']
        print("Available scenarios:")
        for i, scenario in enumerate(scenarios, 1):
            print(f"  {i}. {scenario}")
        
        scenario_choice = input("Select scenario (1-4): ").strip()
        try:
            scenario = scenarios[int(scenario_choice) - 1]
        except:
            scenario = 'highway'
        
        # Select agent
        available_agents = list(self.agents.keys())
        if available_agents:
            print("\nAvailable agents:")
            for i, agent in enumerate(available_agents, 1):
                print(f"  {i}. {agent}")
            
            agent_choice = input("Select agent: ").strip()
            try:
                agent_type = available_agents[int(agent_choice) - 1]
            except:
                agent_type = available_agents[0]
        else:
            agent_type = 'random'
        
        # Select duration
        duration = input("Duration in seconds (default 60): ").strip()
        try:
            duration = int(duration)
        except:
            duration = 60
        
        # Run custom demo
        self.run_live_demo(scenario, agent_type, duration, True)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Live demonstration of autonomous driving agents")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--il_model", type=str, default="./models/cnn_imitation/model.pth",
                       help="Path to IL model")
    parser.add_argument("--rl_model", type=str, default="./models/ppo_driving/model.zip",
                       help="Path to RL model")
    parser.add_argument("--scenario", type=str, choices=['highway', 'intersection', 'roundabout', 'parking'],
                       help="Specific scenario to demo")
    parser.add_argument("--agent", type=str, choices=['il', 'rl', 'hybrid', 'random'],
                       default='hybrid', help="Agent type to use")
    parser.add_argument("--duration", type=int, default=60,
                       help="Demo duration in seconds")
    parser.add_argument("--interactive", action="store_true",
                       help="Run interactive demo menu")
    parser.add_argument("--no_render", action="store_true",
                       help="Disable rendering (faster)")
    
    args = parser.parse_args()
    
    # Initialize demo
    demo = LiveDrivingDemo(args.config)
    
    # Load agents
    demo.load_agents(args.il_model, args.rl_model)
    
    if args.interactive:
        # Run interactive menu
        demo.interactive_demo_menu()
    elif args.scenario:
        # Run specific scenario
        demo.run_live_demo(args.scenario, args.agent, args.duration, not args.no_render)
    else:
        # Run default demo
        print("🚗 Starting default live demo...")
        demo.run_live_demo('highway', args.agent, args.duration, not args.no_render)


if __name__ == "__main__":
    main()