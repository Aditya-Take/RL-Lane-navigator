#!/usr/bin/env python3
"""
Comprehensive demo script for the Vision-Based Autonomous Driving Agent.
This script demonstrates the complete pipeline from data collection to evaluation.
"""

import os
import sys
import argparse
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.data_utils import load_config
from agents.cnn_agent import CNNImitationAgent
from agents.ppo_agent import PPODrivingAgent, HybridAgent
from environments.env_configs import create_environment
from environments.wrappers import create_wrapped_environment
from training.data_collection import ExpertDataCollector
from utils.visualization import (
    plot_training_history, plot_rl_training_history, 
    plot_evaluation_metrics, create_training_dashboard
)


def demo_data_collection(config_path: str = "configs/training_config.yaml"):
    """Demonstrate expert data collection."""
    print("="*60)
    print("DEMO: Expert Data Collection")
    print("="*60)
    
    # Load configuration
    config = load_config(config_path)
    
    # Create data collector
    collector = ExpertDataCollector(config)
    
    # Collect a small amount of data for demo
    print("Collecting expert demonstrations...")
    stats = collector.collect_expert_data("./data/demo_expert_data", episodes=10)
    
    print(f"Data collection completed!")
    print(f"Successful episodes: {stats['successful_episodes']}/{stats['total_episodes']}")
    print(f"Total frames collected: {stats['total_frames']}")
    
    return "./data/demo_expert_data"


def demo_imitation_learning(data_path: str, config_path: str = "configs/training_config.yaml"):
    """Demonstrate imitation learning training."""
    print("\n" + "="*60)
    print("DEMO: Imitation Learning Training")
    print("="*60)
    
    # Load configuration
    config = load_config(config_path)
    
    # Create imitation learning agent
    print("Initializing imitation learning agent...")
    agent = CNNImitationAgent(config, device="cpu")  # Use CPU for demo
    
    # Print model summary
    summary = agent.get_model_summary()
    print("Model Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Train the agent (with reduced epochs for demo)
    config['imitation_learning']['training']['epochs'] = 5  # Reduced for demo
    agent.epochs = 5
    
    print("Starting imitation learning training...")
    training_history = agent.train(
        train_data_path=data_path,
        save_path="./models/demo_il_model"
    )
    
    # Plot training history
    plot_training_history(training_history, save_path="./plots/demo_il_training.png")
    
    print("Imitation learning training completed!")
    return "./models/demo_il_model"


def demo_reinforcement_learning(il_model_path: str, config_path: str = "configs/training_config.yaml"):
    """Demonstrate reinforcement learning training."""
    print("\n" + "="*60)
    print("DEMO: Reinforcement Learning Training")
    print("="*60)
    
    # Load configuration
    config = load_config(config_path)
    
    # Create environment
    print("Creating training environment...")
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
    config['reinforcement_learning']['training']['total_timesteps'] = 10000  # Reduced for demo
    agent = PPODrivingAgent(config, env, device="cpu")  # Use CPU for demo
    
    # Print model summary
    summary = agent.get_model_summary()
    print("Model Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Train the agent
    print("Starting PPO training...")
    training_history = agent.train(
        save_path="./models/demo_rl_model",
        pretrained_il_path=il_model_path
    )
    
    # Plot training history
    plot_rl_training_history(training_history, save_path="./plots/demo_rl_training.png")
    
    print("Reinforcement learning training completed!")
    return "./models/demo_rl_model"


def demo_evaluation(il_model_path: str, rl_model_path: str, config_path: str = "configs/training_config.yaml"):
    """Demonstrate agent evaluation."""
    print("\n" + "="*60)
    print("DEMO: Agent Evaluation")
    print("="*60)
    
    # Load configuration
    config = load_config(config_path)
    
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
    
    # Load and evaluate IL agent
    print("Evaluating imitation learning agent...")
    il_agent = CNNImitationAgent(config)
    il_agent.load_model(il_model_path)
    il_metrics = il_agent.evaluate("./data/demo_expert_data")
    
    # Load and evaluate RL agent
    print("Evaluating reinforcement learning agent...")
    rl_agent = PPODrivingAgent(config, eval_env)
    rl_agent.load_model(rl_model_path)
    rl_metrics = rl_agent.evaluate(eval_env, n_episodes=5)  # Reduced for demo
    
    # Create hybrid agent
    print("Evaluating hybrid agent...")
    hybrid_agent = HybridAgent(il_agent, rl_agent, config)
    hybrid_metrics = hybrid_agent.evaluate(eval_env, n_episodes=5)  # Reduced for demo
    
    # Plot evaluation metrics
    print("Creating evaluation plots...")
    plot_evaluation_metrics(rl_metrics, save_path="./plots/demo_rl_evaluation.png")
    
    # Create comprehensive dashboard
    create_training_dashboard(
        il_history=il_agent.training_history,
        rl_history=rl_agent.training_history,
        eval_metrics=rl_metrics,
        save_path="./plots/demo_training_dashboard.png"
    )
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print("Imitation Learning Agent:")
    print(f"  MSE: {il_metrics['mse']:.4f}")
    print(f"  MAE: {il_metrics['mae']:.4f}")
    print(f"  Correlation: {il_metrics['correlation_mean']:.4f}")
    
    print("\nReinforcement Learning Agent:")
    print(f"  Mean Reward: {rl_metrics['mean_reward']:.2f} ± {rl_metrics['std_reward']:.2f}")
    print(f"  Success Rate: {rl_metrics['success_rate']:.2%}")
    print(f"  Collision Rate: {rl_metrics['collision_rate']:.2%}")
    
    print("\nHybrid Agent:")
    print(f"  Hybrid Mode - Mean Reward: {hybrid_metrics['hybrid']['mean_reward']:.2f}")
    print(f"  Hybrid Mode - Success Rate: {hybrid_metrics['hybrid']['success_rate']:.2%}")
    print(f"  RL Only Mode - Mean Reward: {hybrid_metrics['rl_only']['mean_reward']:.2f}")
    print(f"  RL Only Mode - Success Rate: {hybrid_metrics['rl_only']['success_rate']:.2%}")
    
    return {
        'il_metrics': il_metrics,
        'rl_metrics': rl_metrics,
        'hybrid_metrics': hybrid_metrics
    }


def demo_live_driving(agent_path: str, agent_type: str = "rl", config_path: str = "configs/training_config.yaml"):
    """Demonstrate live driving with trained agent."""
    print("\n" + "="*60)
    print("DEMO: Live Driving")
    print("="*60)
    
    # Load configuration
    config = load_config(config_path)
    
    # Create environment
    print("Creating driving environment...")
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
        'domain_randomization': {
            'enabled': False  # Disable for live demo
        },
        'reward_shaping': config['reward_shaping'],
        'multi_agent': config['multi_agent']
    }
    
    env = create_wrapped_environment(env, wrappers_config)
    
    # Load agent
    if agent_type == "il":
        agent = CNNImitationAgent(config)
        agent.load_model(agent_path)
    elif agent_type == "rl":
        agent = PPODrivingAgent(config, env)
        agent.load_model(agent_path)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    print(f"Starting live driving with {agent_type.upper()} agent...")
    print("Press 'q' to quit the demo.")
    
    # Run live driving demo
    obs, _ = env.reset()
    total_reward = 0
    step_count = 0
    
    while True:
        # Get agent action
        if agent_type == "il":
            action = agent.predict(obs)
        else:
            action = agent.predict(obs, deterministic=True)
        
        # Take action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        # Render environment
        env.render()
        
        # Check for episode end
        if terminated or truncated:
            print(f"Episode ended. Total reward: {total_reward:.2f}, Steps: {step_count}")
            break
        
        # Check for quit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Demo stopped by user.")
            break
    
    env.close()
    print("Live driving demo completed!")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Autonomous Driving Agent Demo")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--skip_data_collection", action="store_true",
                       help="Skip data collection step")
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip training steps")
    parser.add_argument("--live_demo", action="store_true",
                       help="Run live driving demo")
    parser.add_argument("--agent_type", type=str, choices=["il", "rl", "hybrid"], default="rl",
                       help="Agent type for live demo")
    
    args = parser.parse_args()
    
    # Create necessary directories
    Path("./data").mkdir(exist_ok=True)
    Path("./models").mkdir(exist_ok=True)
    Path("./plots").mkdir(exist_ok=True)
    Path("./logs").mkdir(exist_ok=True)
    
    print("🚗 Vision-Based Autonomous Driving Agent Demo")
    print("="*60)
    print("This demo will showcase the complete pipeline:")
    print("1. Expert data collection")
    print("2. Imitation learning training")
    print("3. Reinforcement learning training")
    print("4. Agent evaluation")
    print("5. Live driving demonstration")
    print("="*60)
    
    data_path = None
    il_model_path = None
    rl_model_path = None
    
    try:
        # Step 1: Data Collection
        if not args.skip_data_collection:
            data_path = demo_data_collection(args.config)
        else:
            data_path = "./data/demo_expert_data"
            print("Skipping data collection step.")
        
        # Step 2: Imitation Learning
        if not args.skip_training:
            il_model_path = demo_imitation_learning(data_path, args.config)
        else:
            il_model_path = "./models/demo_il_model"
            print("Skipping imitation learning training.")
        
        # Step 3: Reinforcement Learning
        if not args.skip_training:
            rl_model_path = demo_reinforcement_learning(il_model_path, args.config)
        else:
            rl_model_path = "./models/demo_rl_model"
            print("Skipping reinforcement learning training.")
        
        # Step 4: Evaluation
        evaluation_results = demo_evaluation(il_model_path, rl_model_path, args.config)
        
        # Step 5: Live Demo
        if args.live_demo:
            demo_live_driving(rl_model_path, args.agent_type, args.config)
        
        print("\n" + "="*60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Generated files:")
        print(f"  - Expert data: {data_path}")
        print(f"  - IL model: {il_model_path}")
        print(f"  - RL model: {rl_model_path}")
        print(f"  - Training plots: ./plots/")
        print(f"  - Logs: ./logs/")
        print("\nYou can now:")
        print("  - Run the full training pipeline with more episodes/timesteps")
        print("  - Experiment with different environments (intersection-v0, roundabout-v0, parking-v0)")
        print("  - Try different agent architectures and hyperparameters")
        print("  - Use the trained models for real-world applications")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Please check the configuration and try again.")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Import cv2 for live demo
    try:
        import cv2
    except ImportError:
        print("Warning: OpenCV not found. Live demo will not be available.")
        cv2 = None
    
    main()