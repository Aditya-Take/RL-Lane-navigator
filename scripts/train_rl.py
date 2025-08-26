#!/usr/bin/env python3
"""
Training script for reinforcement learning agent (PPO).
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.data_utils import load_config
from agents.ppo_agent import PPODrivingAgent
from environments.env_configs import create_environment
from environments.wrappers import create_wrapped_environment


def main():
    """Main training function for reinforcement learning."""
    parser = argparse.ArgumentParser(description="Train reinforcement learning agent")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--env_name", type=str, default=None,
                       help="Environment name (overrides config)")
    parser.add_argument("--save_path", type=str, default="./models/ppo_driving",
                       help="Path to save trained model")
    parser.add_argument("--pretrained_il", type=str, default=None,
                       help="Path to pretrained imitation learning model")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for training (cuda/cpu)")
    parser.add_argument("--timesteps", type=int, default=None,
                       help="Number of timesteps for training (overrides config)")
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Override environment name if provided
    if args.env_name:
        config['environment']['name'] = args.env_name
    
    # Override timesteps if provided
    if args.timesteps:
        config['reinforcement_learning']['training']['total_timesteps'] = args.timesteps
    
    # Create environment
    print(f"Creating environment: {config['environment']['name']}")
    env = create_environment(
        config['environment']['name'],
        config=config['environment']['config']
    )
    
    # Apply wrappers
    wrappers_config = {
        'observation_preprocessing': {
            'enabled': True,
            'target_size': (84, 84),
            'normalize': True
        },
        'domain_randomization': config['domain_randomization'],
        'reward_shaping': config['reward_shaping'],
        'multi_agent': config['multi_agent']
    }
    
    env = create_wrapped_environment(env, wrappers_config)
    
    # Create PPO agent
    print("Initializing PPO agent...")
    agent = PPODrivingAgent(config, env, device=args.device)
    
    # Print model summary
    summary = agent.get_model_summary()
    print("Model Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Train the agent
    print("Starting PPO training...")
    training_history = agent.train(
        save_path=args.save_path,
        pretrained_il_path=args.pretrained_il
    )
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = create_environment(
        config['environment']['name'],
        config=config['environment']['config']
    )
    
    # Apply wrappers for evaluation (without domain randomization)
    eval_wrappers_config = {
        'observation_preprocessing': {
            'enabled': True,
            'target_size': (84, 84),
            'normalize': True
        },
        'domain_randomization': {
            'enabled': False
        },
        'reward_shaping': config['reward_shaping'],
        'multi_agent': config['multi_agent']
    }
    
    eval_env = create_wrapped_environment(eval_env, eval_wrappers_config)
    
    # Evaluate the trained agent
    print("Evaluating trained agent...")
    eval_metrics = agent.evaluate(eval_env, n_episodes=20)
    
    print("Training completed successfully!")
    print(f"Model saved to: {args.save_path}")
    print(f"Evaluation metrics: {eval_metrics}")


if __name__ == "__main__":
    main()