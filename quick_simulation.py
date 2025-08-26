#!/usr/bin/env python3
"""
Quick Simulation Script for Autonomous Driving Scenarios
This script provides a quick way to see simulations for all scenarios,
even without trained models. It uses simple rule-based or random agents
to demonstrate the environments.

Usage:
    python quick_simulation.py --scenario highway --render
    python quick_simulation.py --scenario all --save_videos
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
import numpy as np
import cv2
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.data_utils import load_config
from environments.env_configs import create_environment
from environments.wrappers import create_wrapped_environment


class SimpleAgent:
    """Simple rule-based agent for demonstration purposes."""
    
    def __init__(self, action_space):
        self.action_space = action_space
        self.step_count = 0
    
    def predict(self, obs, deterministic=True):
        """Predict action based on simple rules."""
        self.step_count += 1
        
        # Simple rule-based behavior
        if hasattr(self.action_space, 'shape'):
            # Continuous action space
            action = np.zeros(self.action_space.shape)
            
            # Basic lane following behavior
            # Accelerate slightly forward
            action[0] = 0.5  # acceleration
            
            # Add some steering variation
            if self.step_count % 50 < 25:
                action[1] = 0.1  # slight right turn
            else:
                action[1] = -0.1  # slight left turn
            
            # Keep speed reasonable
            action[2] = 0.0  # speed control
            
            return action
        else:
            # Discrete action space
            return self.action_space.sample()


class QuickSimulation:
    """Quick simulation runner for demonstration purposes."""
    
    def __init__(self, config_path: str = "configs/training_config.yaml"):
        """Initialize the quick simulation runner."""
        self.config = load_config(config_path)
        self.scenarios = {
            "highway": "highway-v0",
            "intersection": "intersection-v0", 
            "roundabout": "roundabout-v0",
            "parking": "parking-v0"
        }
        self.environments = {}
        
    def create_environment(self, scenario: str) -> Any:
        """Create environment for a specific scenario."""
        if scenario in self.environments:
            return self.environments[scenario]
        
        env_name = self.scenarios[scenario]
        print(f"Creating {scenario} environment...")
        
        # Create base environment
        env = create_environment(env_name, config=self.config['environment']['config'])
        
        # Apply wrappers for simulation
        wrappers_config = {
            'observation_preprocessing': {
                'enabled': True,
                'target_size': (84, 84),
                'normalize': True
            },
            'domain_randomization': {
                'enabled': False
            },
            'reward_shaping': self.config.get('reward_shaping', {}),
            'multi_agent': self.config.get('multi_agent', {})
        }
        
        env = create_wrapped_environment(env, wrappers_config)
        self.environments[scenario] = env
        return env
    
    def run_demo(self, scenario: str, 
                render: bool = True, 
                save_video: bool = False,
                max_steps: int = 500,
                output_dir: str = "./demo_videos") -> Dict[str, Any]:
        """
        Run a demonstration for a specific scenario.
        
        Args:
            scenario: Scenario name
            render: Whether to render the simulation
            save_video: Whether to save simulation video
            max_steps: Maximum steps per episode
            output_dir: Directory to save videos
            
        Returns:
            Demo results
        """
        print(f"\n{'='*60}")
        print(f"Running {scenario.upper()} demonstration")
        print(f"{'='*60}")
        
        # Create environment
        env = self.create_environment(scenario)
        
        # Create simple agent
        agent = SimpleAgent(env.action_space)
        
        # Setup video recording if requested
        video_writer = None
        video_path = None
        if save_video:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            video_path = output_path / f"{scenario}_demo.mp4"
            
            # Get environment render size
            env.render()
            frame = env.render()
            if frame is not None:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(str(video_path), fourcc, 20.0, (width, height))
        
        # Run simulation
        obs, _ = env.reset()
        total_reward = 0
        step_count = 0
        episode_info = {
            'collisions': 0,
            'off_road': 0,
            'lane_violations': 0,
            'speed_violations': 0
        }
        
        print(f"Starting demonstration... (Press 'q' to quit, 'r' to reset)")
        
        while step_count < max_steps:
            # Get agent action
            action = agent.predict(obs)
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # Update episode info
            if info.get('collision', False):
                episode_info['collisions'] += 1
            if info.get('off_road', False):
                episode_info['off_road'] += 1
            if info.get('lane_violation', False):
                episode_info['lane_violations'] += 1
            if info.get('speed_violation', False):
                episode_info['speed_violations'] += 1
            
            # Render environment
            if render:
                frame = env.render()
                if frame is not None and video_writer is not None:
                    # Convert to BGR for OpenCV
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    else:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    video_writer.write(frame_bgr)
                
                # Handle user input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Demonstration stopped by user.")
                    break
                elif key == ord('r'):
                    print("Resetting demonstration...")
                    obs, _ = env.reset()
                    agent.step_count = 0
                    total_reward = 0
                    step_count = 0
                    episode_info = {k: 0 for k in episode_info}
            
            # Check for episode end
            if terminated or truncated:
                print(f"Episode ended after {step_count} steps.")
                break
        
        # Cleanup
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved to: {video_path}")
        
        # Calculate final metrics
        success = not (episode_info['collisions'] > 0 or episode_info['off_road'] > 0)
        avg_reward = total_reward / max(step_count, 1)
        
        results = {
            'scenario': scenario,
            'total_reward': total_reward,
            'avg_reward': avg_reward,
            'steps': step_count,
            'success': success,
            'episode_info': episode_info,
            'video_path': str(video_path) if save_video else None
        }
        
        # Print results
        print(f"\nDemonstration Results:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Steps: {step_count}")
        print(f"  Success: {'Yes' if success else 'No'}")
        print(f"  Collisions: {episode_info['collisions']}")
        print(f"  Off-road: {episode_info['off_road']}")
        print(f"  Lane Violations: {episode_info['lane_violations']}")
        print(f"  Speed Violations: {episode_info['speed_violations']}")
        
        return results
    
    def run_all_demos(self, render: bool = True,
                     save_videos: bool = False,
                     output_dir: str = "./demo_videos") -> Dict[str, Any]:
        """
        Run demonstrations for all scenarios.
        
        Args:
            render: Whether to render demonstrations
            save_videos: Whether to save demonstration videos
            output_dir: Directory to save videos
            
        Returns:
            Results for all scenarios
        """
        print(f"\n{'='*80}")
        print(f"Running demonstrations for ALL scenarios")
        print(f"{'='*80}")
        
        all_results = {}
        
        for scenario in self.scenarios.keys():
            try:
                results = self.run_demo(
                    scenario=scenario,
                    render=render,
                    save_video=save_videos,
                    output_dir=output_dir
                )
                all_results[scenario] = results
                
                # Small delay between scenarios
                time.sleep(1)
                
            except Exception as e:
                print(f"Error running {scenario} demonstration: {e}")
                all_results[scenario] = {'error': str(e)}
        
        # Print summary
        self.print_summary(all_results)
        
        # Save results
        if save_videos:
            results_path = Path(output_dir) / "demo_results.json"
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\nResults saved to: {results_path}")
        
        return all_results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print summary of all demonstration results."""
        print(f"\n{'='*80}")
        print("DEMONSTRATION SUMMARY")
        print(f"{'='*80}")
        
        print(f"{'Scenario':<15} {'Success':<8} {'Reward':<10} {'Steps':<8} {'Collisions':<12}")
        print("-" * 80)
        
        total_success = 0
        total_scenarios = 0
        
        for scenario, result in results.items():
            if 'error' in result:
                print(f"{scenario:<15} {'ERROR':<8} {'N/A':<10} {'N/A':<8} {'N/A':<12}")
                continue
            
            success = "Yes" if result['success'] else "No"
            reward = f"{result['avg_reward']:.2f}"
            steps = str(result['steps'])
            collisions = str(result['episode_info']['collisions'])
            
            print(f"{scenario:<15} {success:<8} {reward:<10} {steps:<8} {collisions:<12}")
            
            if result['success']:
                total_success += 1
            total_scenarios += 1
        
        print("-" * 80)
        if total_scenarios > 0:
            success_rate = (total_success / total_scenarios) * 100
            print(f"Overall Success Rate: {success_rate:.1f}% ({total_success}/{total_scenarios})")
    
    def interactive_mode(self):
        """Run interactive mode for manual scenario selection."""
        print(f"\n{'='*60}")
        print("INTERACTIVE DEMONSTRATION MODE")
        print(f"{'='*60}")
        print("Available scenarios:")
        for i, scenario in enumerate(self.scenarios.keys(), 1):
            print(f"  {i}. {scenario.title()}")
        print("  5. All scenarios")
        print("  0. Exit")
        
        while True:
            try:
                choice = input("\nSelect scenario (0-5): ").strip()
                
                if choice == "0":
                    print("Exiting interactive mode.")
                    break
                elif choice == "5":
                    self.run_all_demos(render=True, save_videos=True)
                elif choice in ["1", "2", "3", "4"]:
                    scenarios = list(self.scenarios.keys())
                    scenario = scenarios[int(choice) - 1]
                    
                    render_choice = input("Render demonstration? (y/n): ").strip().lower()
                    save_choice = input("Save video? (y/n): ").strip().lower()
                    
                    self.run_demo(
                        scenario=scenario,
                        render=render_choice == 'y',
                        save_video=save_choice == 'y'
                    )
                else:
                    print("Invalid choice. Please select 0-5.")
                    
            except KeyboardInterrupt:
                print("\nExiting interactive mode.")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main function for the quick simulation."""
    parser = argparse.ArgumentParser(description="Quick Simulation for Autonomous Driving Scenarios")
    parser.add_argument("--scenario", type=str, 
                       choices=["highway", "intersection", "roundabout", "parking", "all"],
                       default="highway",
                       help="Scenario to demonstrate")
    parser.add_argument("--render", action="store_true",
                       help="Render demonstration")
    parser.add_argument("--save_videos", action="store_true",
                       help="Save demonstration videos")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="./demo_videos",
                       help="Directory to save videos and results")
    parser.add_argument("--max_steps", type=int, default=500,
                       help="Maximum steps per episode")
    
    args = parser.parse_args()
    
    # Create quick simulation runner
    simulator = QuickSimulation(args.config)
    
    try:
        if args.interactive:
            simulator.interactive_mode()
        elif args.scenario == "all":
            simulator.run_all_demos(
                render=args.render,
                save_videos=args.save_videos,
                output_dir=args.output_dir
            )
        else:
            simulator.run_demo(
                scenario=args.scenario,
                render=args.render,
                save_video=args.save_videos,
                max_steps=args.max_steps,
                output_dir=args.output_dir
            )
    
    except KeyboardInterrupt:
        print("\nDemonstration interrupted by user.")
    except Exception as e:
        print(f"Error running demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()