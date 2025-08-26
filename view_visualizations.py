#!/usr/bin/env python3
"""
Interactive visualization viewer for the autonomous driving agent.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional
import json


def view_training_plots(plots_dir: str = "./plots"):
    """View all training-related plots."""
    plots_path = Path(plots_dir)
    
    if not plots_path.exists():
        print(f"❌ Plots directory {plots_dir} not found!")
        print("Please run training first to generate plots.")
        return
    
    # Find all plot files
    plot_files = list(plots_path.glob("*.png")) + list(plots_path.glob("*.jpg"))
    
    if not plot_files:
        print("❌ No plot files found!")
        return
    
    print(f"📊 Found {len(plot_files)} visualization files:")
    for i, file in enumerate(plot_files):
        print(f"  {i+1}. {file.name}")
    
    # Display plots
    for i, plot_file in enumerate(plot_files):
        print(f"\n🖼️  Displaying: {plot_file.name}")
        
        # Load and display image
        img = mpimg.imread(plot_file)
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.title(f"Visualization: {plot_file.name}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def view_training_dashboard(plots_dir: str = "./plots"):
    """View the comprehensive training dashboard."""
    dashboard_path = Path(plots_dir) / "demo_training_dashboard.png"
    
    if not dashboard_path.exists():
        print("❌ Training dashboard not found!")
        print("Please run the complete training pipeline first.")
        return
    
    print("📈 Displaying Training Dashboard...")
    img = mpimg.imread(dashboard_path)
    
    plt.figure(figsize=(16, 12))
    plt.imshow(img)
    plt.title("Autonomous Driving Agent - Training Dashboard", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def view_evaluation_results(results_dir: str = "./evaluation_results"):
    """View evaluation results and metrics."""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"❌ Evaluation results directory {results_dir} not found!")
        return
    
    # Load evaluation results
    results_file = results_path / "evaluation_results.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        print("📊 Evaluation Results:")
        print("="*50)
        
        if 'mean_reward' in results:
            print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        if 'success_rate' in results:
            print(f"Success Rate: {results['success_rate']:.2%}")
        if 'collision_rate' in results:
            print(f"Collision Rate: {results['collision_rate']:.2%}")
        if 'off_road_rate' in results:
            print(f"Off-road Rate: {results['off_road_rate']:.2%}")
        
        # Create a summary plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Performance metrics
        if 'mean_reward' in results and 'success_rate' in results:
            perf_metrics = ['Mean Reward', 'Success Rate']
            perf_values = [results['mean_reward'], results['success_rate']]
            
            bars1 = axes[0].bar(perf_metrics, perf_values, color=['blue', 'green'])
            axes[0].set_title('Performance Metrics')
            axes[0].set_ylabel('Value')
            axes[0].grid(True, alpha=0.3)
            
            for bar, value in zip(bars1, perf_values):
                axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom')
        
        # Safety metrics
        if 'collision_rate' in results and 'off_road_rate' in results:
            safety_metrics = ['Collision Rate', 'Off-road Rate']
            safety_values = [results['collision_rate'], results['off_road_rate']]
            
            bars2 = axes[1].bar(safety_metrics, safety_values, color=['red', 'orange'])
            axes[1].set_title('Safety Metrics')
            axes[1].set_ylabel('Rate')
            axes[1].grid(True, alpha=0.3)
            
            for bar, value in zip(bars2, safety_values):
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                            f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()


def view_model_summary(models_dir: str = "./models"):
    """View model summaries and architectures."""
    models_path = Path(models_dir)
    
    if not models_path.exists():
        print(f"❌ Models directory {models_dir} not found!")
        return
    
    print("🤖 Model Summaries:")
    print("="*50)
    
    # Check for different model types
    model_types = ['cnn_imitation', 'ppo_driving', 'hybrid_agent']
    
    for model_type in model_types:
        model_path = models_path / model_type
        if model_path.exists():
            print(f"\n📁 {model_type.upper()} Model:")
            
            # Check for config file
            config_file = model_path / "config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                if 'imitation_learning' in config:
                    il_config = config['imitation_learning']
                    print(f"  - CNN Layers: {il_config['cnn']['conv_layers']}")
                    print(f"  - FC Layers: {il_config['cnn']['fc_layers']}")
                    print(f"  - Learning Rate: {il_config['training']['learning_rate']}")
                
                if 'reinforcement_learning' in config:
                    rl_config = config['reinforcement_learning']
                    print(f"  - PPO Learning Rate: {rl_config['ppo']['learning_rate']}")
                    print(f"  - Total Timesteps: {rl_config['training']['total_timesteps']}")
            
            # Check for training history
            history_file = model_path / "training_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history = json.load(f)
                
                if 'train_loss' in history:
                    print(f"  - Training Epochs: {len(history['train_loss'])}")
                    if history['train_loss']:
                        print(f"  - Final Train Loss: {history['train_loss'][-1]:.4f}")
                        print(f"  - Final Val Loss: {history['val_loss'][-1]:.4f}")


def interactive_visualization_menu():
    """Interactive menu for viewing visualizations."""
    while True:
        print("\n" + "="*60)
        print("🎨 AUTONOMOUS DRIVING AGENT - VISUALIZATION VIEWER")
        print("="*60)
        print("1. View All Training Plots")
        print("2. View Training Dashboard")
        print("3. View Evaluation Results")
        print("4. View Model Summaries")
        print("5. View All Visualizations")
        print("6. Exit")
        print("="*60)
        
        choice = input("Enter your choice (1-6): ").strip()
        
        if choice == '1':
            view_training_plots()
        elif choice == '2':
            view_training_dashboard()
        elif choice == '3':
            view_evaluation_results()
        elif choice == '4':
            view_model_summary()
        elif choice == '5':
            print("🖼️  Loading all visualizations...")
            view_training_plots()
            view_training_dashboard()
            view_evaluation_results()
            view_model_summary()
        elif choice == '6':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter a number between 1-6.")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="View autonomous driving agent visualizations")
    parser.add_argument("--plots_dir", type=str, default="./plots",
                       help="Directory containing plot files")
    parser.add_argument("--results_dir", type=str, default="./evaluation_results",
                       help="Directory containing evaluation results")
    parser.add_argument("--models_dir", type=str, default="./models",
                       help="Directory containing trained models")
    parser.add_argument("--interactive", action="store_true",
                       help="Run interactive visualization menu")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_visualization_menu()
    else:
        # View all visualizations
        print("🎨 Loading all visualizations...")
        view_training_plots(args.plots_dir)
        view_training_dashboard(args.plots_dir)
        view_evaluation_results(args.results_dir)
        view_model_summary(args.models_dir)


if __name__ == "__main__":
    main()