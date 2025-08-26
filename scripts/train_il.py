#!/usr/bin/env python3
"""
Training script for imitation learning agent.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.data_utils import load_config
from agents.cnn_agent import CNNImitationAgent
from training.data_collection import ExpertDataCollector


def main():
    """Main training function for imitation learning."""
    parser = argparse.ArgumentParser(description="Train imitation learning agent")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--data_path", type=str, default="./data/expert_demonstrations",
                       help="Path to expert demonstration data")
    parser.add_argument("--save_path", type=str, default="./models/cnn_imitation",
                       help="Path to save trained model")
    parser.add_argument("--collect_data", action="store_true",
                       help="Collect expert data before training")
    parser.add_argument("--episodes", type=int, default=None,
                       help="Number of episodes for data collection")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for training (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Collect expert data if requested
    if args.collect_data:
        print("Collecting expert demonstrations...")
        collector = ExpertDataCollector(config)
        stats = collector.collect_expert_data(args.data_path, args.episodes)
        print(f"Data collection completed: {stats}")
    
    # Check if data exists
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data path {data_path} does not exist!")
        print("Please run with --collect_data flag to collect expert demonstrations first.")
        return
    
    # Create imitation learning agent
    print("Initializing imitation learning agent...")
    agent = CNNImitationAgent(config, device=args.device)
    
    # Print model summary
    summary = agent.get_model_summary()
    print("Model Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Train the agent
    print("Starting imitation learning training...")
    training_history = agent.train(
        train_data_path=args.data_path,
        save_path=args.save_path
    )
    
    # Evaluate the trained agent
    print("Evaluating trained agent...")
    eval_metrics = agent.evaluate(args.data_path)
    
    print("Training completed successfully!")
    print(f"Model saved to: {args.save_path}")
    print(f"Evaluation metrics: {eval_metrics}")


if __name__ == "__main__":
    main()