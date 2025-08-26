#!/usr/bin/env python3
"""
Evaluation script for testing trained agents.
"""

import argparse
import os
import sys
from pathlib import Path
import json
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.data_utils import load_config
from agents.cnn_agent import CNNImitationAgent
from agents.ppo_agent import PPODrivingAgent, HybridAgent
from environments.env_configs import create_environment
from environments.wrappers import create_wrapped_environment


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate trained agents")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--agent_type", type=str, choices=["il", "rl", "hybrid"], required=True,
                       help="Type of agent to evaluate")
    parser.add_argument("--agent_path", type=str, required=True,
                       help="Path to trained agent model")
    parser.add_argument("--env_name", type=str, default=None,
                       help="Environment name for evaluation")
    parser.add_argument("--episodes", type=int, default=50,
                       help="Number of episodes for evaluation")
    parser.add_argument("--render", action="store_true",
                       help="Render evaluation episodes")
    parser.add_argument("--save_videos", action="store_true",
                       help="Save evaluation videos")
    parser.add_argument("--output_path", type=str, default="./evaluation_results",
                       help="Path to save evaluation results")
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Override environment name if provided
    if args.env_name:
        config['environment']['name'] = args.env_name
    
    # Create evaluation environment
    print(f"Creating evaluation environment: {config['environment']['name']}")
    env = create_environment(
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
    
    env = create_wrapped_environment(env, eval_wrappers_config)
    
    # Load and evaluate agent
    if args.agent_type == "il":
        agent = CNNImitationAgent(config)
        agent.load_model(args.agent_path)
        results = evaluate_il_agent(agent, env, args.episodes, args.render)
    
    elif args.agent_type == "rl":
        agent = PPODrivingAgent(config, env)
        agent.load_model(args.agent_path)
        results = agent.evaluate(env, n_episodes=args.episodes)
    
    elif args.agent_type == "hybrid":
        # For hybrid agent, we need both IL and RL models
        il_agent = CNNImitationAgent(config)
        il_agent.load_model(args.agent_path + "_il")
        
        rl_agent = PPODrivingAgent(config, env)
        rl_agent.load_model(args.agent_path + "_rl")
        
        hybrid_agent = HybridAgent(il_agent, rl_agent, config)
        results = hybrid_agent.evaluate(env, n_episodes=args.episodes)
    
    # Save evaluation results
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    if args.agent_type == "il":
        print(f"Imitation Learning Agent Results:")
        print(f"  MSE: {results['mse']:.4f}")
        print(f"  MAE: {results['mae']:.4f}")
        print(f"  Correlation: {results['correlation_mean']:.4f}")
    
    elif args.agent_type == "rl":
        print(f"Reinforcement Learning Agent Results:")
        print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Mean Episode Length: {results['mean_length']:.1f}")
        print(f"  Collision Rate: {results['collision_rate']:.2%}")
        print(f"  Off-road Rate: {results['off_road_rate']:.2%}")
        print(f"  Success Rate: {results['success_rate']:.2%}")
    
    elif args.agent_type == "hybrid":
        print(f"Hybrid Agent Results:")
        print(f"  Hybrid Mode:")
        print(f"    Mean Reward: {results['hybrid']['mean_reward']:.2f}")
        print(f"    Success Rate: {results['hybrid']['success_rate']:.2%}")
        print(f"  RL Only Mode:")
        print(f"    Mean Reward: {results['rl_only']['mean_reward']:.2f}")
        print(f"    Success Rate: {results['rl_only']['success_rate']:.2%}")
    
    print(f"\nResults saved to: {output_path}")


def evaluate_il_agent(agent: CNNImitationAgent, env, n_episodes: int, render: bool = False) -> dict:
    """
    Evaluate imitation learning agent.
    
    Args:
        agent: Trained imitation learning agent
        env: Evaluation environment
        n_episodes: Number of episodes to evaluate
        render: Whether to render episodes
        
    Returns:
        Evaluation results
    """
    print(f"Evaluating IL agent over {n_episodes} episodes...")
    
    all_predictions = []
    all_targets = []
    episode_rewards = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_predictions = []
        episode_targets = []
        
        done = False
        while not done:
            # Get expert action (target)
            expert_action = env.action_space.sample()  # This is a placeholder
            
            # Get agent prediction
            prediction = agent.predict(obs)
            
            # Store prediction and target
            episode_predictions.append(prediction)
            episode_targets.append(expert_action)
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(prediction)
            episode_reward += reward
            done = terminated or truncated
            
            if render:
                env.render()
        
        all_predictions.extend(episode_predictions)
        all_targets.extend(episode_targets)
        episode_rewards.append(episode_reward)
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    from utils.data_utils import calculate_metrics
    metrics = calculate_metrics(all_predictions, all_targets)
    
    return {
        'mse': metrics['mse'],
        'mae': metrics['mae'],
        'correlation_mean': metrics['correlation_mean'],
        'correlation_std': metrics['correlation_std'],
        'mean_episode_reward': np.mean(episode_rewards),
        'std_episode_reward': np.std(episode_rewards)
    }


if __name__ == "__main__":
    main()