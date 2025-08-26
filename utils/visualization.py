"""
Visualization utilities for training progress, model performance, and driving behavior analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import cv2
from pathlib import Path
import json
import torch


def plot_training_history(history: Dict[str, List[float]], save_path: str = None):
    """
    Plot training history for imitation learning.
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plots
    axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # MSE plots
    train_mse = [metrics['mse'] for metrics in history['train_metrics']]
    val_mse = [metrics['mse'] for metrics in history['val_metrics']]
    axes[0, 1].plot(train_mse, label='Train MSE', color='blue')
    axes[0, 1].plot(val_mse, label='Validation MSE', color='red')
    axes[0, 1].set_title('Training and Validation MSE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # MAE plots
    train_mae = [metrics['mae'] for metrics in history['train_metrics']]
    val_mae = [metrics['mae'] for metrics in history['val_metrics']]
    axes[1, 0].plot(train_mae, label='Train MAE', color='blue')
    axes[1, 0].plot(val_mae, label='Validation MAE', color='red')
    axes[1, 0].set_title('Training and Validation MAE')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Correlation plots
    train_corr = [metrics['correlation_mean'] for metrics in history['train_metrics']]
    val_corr = [metrics['correlation_mean'] for metrics in history['val_metrics']]
    axes[1, 1].plot(train_corr, label='Train Correlation', color='blue')
    axes[1, 1].plot(val_corr, label='Validation Correlation', color='red')
    axes[1, 1].set_title('Training and Validation Correlation')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Correlation')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_rl_training_history(history: Dict[str, List[float]], save_path: str = None):
    """
    Plot training history for reinforcement learning.
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    if history['episode_rewards']:
        axes[0, 0].plot(history['episode_rewards'], alpha=0.6, color='blue')
        # Moving average
        window_size = min(100, len(history['episode_rewards']) // 10)
        if window_size > 1:
            moving_avg = pd.Series(history['episode_rewards']).rolling(window=window_size).mean()
            axes[0, 0].plot(moving_avg, color='red', linewidth=2, label=f'Moving Average (window={window_size})')
            axes[0, 0].legend()
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
    
    # Episode lengths
    if history['episode_lengths']:
        axes[0, 1].plot(history['episode_lengths'], alpha=0.6, color='green')
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Length')
        axes[0, 1].grid(True)
    
    # Evaluation rewards
    if history['eval_rewards']:
        axes[1, 0].plot(history['eval_rewards'], color='orange', marker='o')
        axes[1, 0].set_title('Evaluation Rewards')
        axes[1, 0].set_xlabel('Evaluation')
        axes[1, 0].set_ylabel('Mean Reward')
        axes[1, 0].grid(True)
    
    # Evaluation lengths
    if history['eval_lengths']:
        axes[1, 1].plot(history['eval_lengths'], color='purple', marker='s')
        axes[1, 1].set_title('Evaluation Episode Lengths')
        axes[1, 1].set_xlabel('Evaluation')
        axes[1, 1].set_ylabel('Mean Length')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"RL training history plot saved to {save_path}")
    
    plt.show()


def plot_action_distribution(predictions: np.ndarray, targets: np.ndarray, save_path: str = None):
    """
    Plot action distribution comparison between predictions and targets.
    
    Args:
        predictions: Predicted actions
        targets: Target actions
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Steering distribution
    axes[0, 0].hist(predictions[:, 0], bins=50, alpha=0.7, label='Predictions', color='blue')
    axes[0, 0].hist(targets[:, 0], bins=50, alpha=0.7, label='Targets', color='red')
    axes[0, 0].set_title('Steering Distribution')
    axes[0, 0].set_xlabel('Steering')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Acceleration distribution
    axes[0, 1].hist(predictions[:, 1], bins=50, alpha=0.7, label='Predictions', color='blue')
    axes[0, 1].hist(targets[:, 1], bins=50, alpha=0.7, label='Targets', color='red')
    axes[0, 1].set_title('Acceleration Distribution')
    axes[0, 1].set_xlabel('Acceleration')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Scatter plot: Steering
    axes[1, 0].scatter(targets[:, 0], predictions[:, 0], alpha=0.5, color='blue')
    axes[1, 0].plot([targets[:, 0].min(), targets[:, 0].max()], 
                    [targets[:, 0].min(), targets[:, 0].max()], 'r--', linewidth=2)
    axes[1, 0].set_title('Steering: Predictions vs Targets')
    axes[1, 0].set_xlabel('Target Steering')
    axes[1, 0].set_ylabel('Predicted Steering')
    axes[1, 0].grid(True)
    
    # Scatter plot: Acceleration
    axes[1, 1].scatter(targets[:, 1], predictions[:, 1], alpha=0.5, color='green')
    axes[1, 1].plot([targets[:, 1].min(), targets[:, 1].max()], 
                    [targets[:, 1].min(), targets[:, 1].max()], 'r--', linewidth=2)
    axes[1, 1].set_title('Acceleration: Predictions vs Targets')
    axes[1, 1].set_xlabel('Target Acceleration')
    axes[1, 1].set_ylabel('Predicted Acceleration')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Action distribution plot saved to {save_path}")
    
    plt.show()


def plot_evaluation_metrics(metrics: Dict[str, float], save_path: str = None):
    """
    Plot evaluation metrics.
    
    Args:
        metrics: Evaluation metrics dictionary
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Performance metrics
    if 'mean_reward' in metrics:
        performance_metrics = ['mean_reward', 'success_rate']
        performance_values = [metrics.get(m, 0) for m in performance_metrics]
        
        bars1 = axes[0].bar(performance_metrics, performance_values, color=['blue', 'green'])
        axes[0].set_title('Performance Metrics')
        axes[0].set_ylabel('Value')
        axes[0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, performance_values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    # Safety metrics
    if 'collision_rate' in metrics:
        safety_metrics = ['collision_rate', 'off_road_rate']
        safety_values = [metrics.get(m, 0) for m in safety_metrics]
        
        bars2 = axes[1].bar(safety_metrics, safety_values, color=['red', 'orange'])
        axes[1].set_title('Safety Metrics')
        axes[1].set_ylabel('Rate')
        axes[1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars2, safety_values):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Evaluation metrics plot saved to {save_path}")
    
    plt.show()


def visualize_attention_weights(attention_weights: np.ndarray, original_image: np.ndarray, 
                               save_path: str = None):
    """
    Visualize attention weights overlaid on the original image.
    
    Args:
        attention_weights: Attention weights (H, W)
        original_image: Original image (H, W) or (C, H, W)
        save_path: Path to save the visualization
    """
    # Normalize attention weights
    attention_norm = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min())
    
    # Resize attention weights to match image size
    if len(original_image.shape) == 3:
        h, w = original_image.shape[1], original_image.shape[2]
        attention_resized = cv2.resize(attention_norm, (w, h))
    else:
        h, w = original_image.shape
        attention_resized = cv2.resize(attention_norm, (w, h))
    
    # Create heatmap
    heatmap = cv2.applyColorMap((attention_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Overlay on original image
    if len(original_image.shape) == 3:
        # Convert to BGR for OpenCV
        if original_image.shape[0] == 3:  # CHW format
            img_bgr = np.transpose(original_image, (1, 2, 0))
        else:
            img_bgr = original_image
        img_bgr = (img_bgr * 255).astype(np.uint8)
    else:
        img_bgr = (original_image * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    
    # Blend images
    alpha = 0.7
    overlay = cv2.addWeighted(img_bgr, alpha, heatmap, 1 - alpha, 0)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Attention heatmap
    axes[1].imshow(attention_resized, cmap='jet')
    axes[1].set_title('Attention Weights')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Attention Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention visualization saved to {save_path}")
    
    plt.show()


def plot_trajectory_comparison(il_trajectories: List[np.ndarray], rl_trajectories: List[np.ndarray],
                              hybrid_trajectories: List[np.ndarray], save_path: str = None):
    """
    Plot trajectory comparison between different agents.
    
    Args:
        il_trajectories: Imitation learning trajectories
        rl_trajectories: Reinforcement learning trajectories
        hybrid_trajectories: Hybrid agent trajectories
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot IL trajectories
    for traj in il_trajectories:
        axes[0].plot(traj[:, 0], traj[:, 1], alpha=0.6, color='blue')
    axes[0].set_title('Imitation Learning Trajectories')
    axes[0].set_xlabel('X Position')
    axes[0].set_ylabel('Y Position')
    axes[0].grid(True)
    axes[0].set_aspect('equal')
    
    # Plot RL trajectories
    for traj in rl_trajectories:
        axes[1].plot(traj[:, 0], traj[:, 1], alpha=0.6, color='red')
    axes[1].set_title('Reinforcement Learning Trajectories')
    axes[1].set_xlabel('X Position')
    axes[1].set_ylabel('Y Position')
    axes[1].grid(True)
    axes[1].set_aspect('equal')
    
    # Plot Hybrid trajectories
    for traj in hybrid_trajectories:
        axes[2].plot(traj[:, 0], traj[:, 1], alpha=0.6, color='green')
    axes[2].set_title('Hybrid Agent Trajectories')
    axes[2].set_xlabel('X Position')
    axes[2].set_ylabel('Y Position')
    axes[2].grid(True)
    axes[2].set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Trajectory comparison plot saved to {save_path}")
    
    plt.show()


def create_training_dashboard(il_history: Dict[str, List[float]], rl_history: Dict[str, List[float]],
                             eval_metrics: Dict[str, float], save_path: str = None):
    """
    Create a comprehensive training dashboard.
    
    Args:
        il_history: Imitation learning training history
        rl_history: Reinforcement learning training history
        eval_metrics: Evaluation metrics
        save_path: Path to save the dashboard
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # IL Training Loss
    ax1 = fig.add_subplot(gs[0, 0])
    if il_history and 'train_loss' in il_history:
        ax1.plot(il_history['train_loss'], label='Train Loss', color='blue')
        ax1.plot(il_history['val_loss'], label='Val Loss', color='red')
        ax1.set_title('IL Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
    
    # IL Metrics
    ax2 = fig.add_subplot(gs[0, 1])
    if il_history and 'train_metrics' in il_history:
        train_mse = [m['mse'] for m in il_history['train_metrics']]
        val_mse = [m['mse'] for m in il_history['val_metrics']]
        ax2.plot(train_mse, label='Train MSE', color='blue')
        ax2.plot(val_mse, label='Val MSE', color='red')
        ax2.set_title('IL MSE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MSE')
        ax2.legend()
        ax2.grid(True)
    
    # RL Episode Rewards
    ax3 = fig.add_subplot(gs[0, 2])
    if rl_history and 'episode_rewards' in rl_history:
        ax3.plot(rl_history['episode_rewards'], alpha=0.6, color='green')
        ax3.set_title('RL Episode Rewards')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Reward')
        ax3.grid(True)
    
    # RL Evaluation Rewards
    ax4 = fig.add_subplot(gs[0, 3])
    if rl_history and 'eval_rewards' in rl_history:
        ax4.plot(rl_history['eval_rewards'], color='orange', marker='o')
        ax4.set_title('RL Evaluation Rewards')
        ax4.set_xlabel('Evaluation')
        ax4.set_ylabel('Mean Reward')
        ax4.grid(True)
    
    # Performance Metrics
    ax5 = fig.add_subplot(gs[1, :2])
    if eval_metrics:
        perf_metrics = ['mean_reward', 'success_rate']
        perf_values = [eval_metrics.get(m, 0) for m in perf_metrics]
        bars1 = ax5.bar(perf_metrics, perf_values, color=['blue', 'green'])
        ax5.set_title('Performance Metrics')
        ax5.set_ylabel('Value')
        ax5.grid(True, alpha=0.3)
        
        for bar, value in zip(bars1, perf_values):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
    
    # Safety Metrics
    ax6 = fig.add_subplot(gs[1, 2:])
    if eval_metrics:
        safety_metrics = ['collision_rate', 'off_road_rate']
        safety_values = [eval_metrics.get(m, 0) for m in safety_metrics]
        bars2 = ax6.bar(safety_metrics, safety_values, color=['red', 'orange'])
        ax6.set_title('Safety Metrics')
        ax6.set_ylabel('Rate')
        ax6.grid(True, alpha=0.3)
        
        for bar, value in zip(bars2, safety_values):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom')
    
    # Summary Statistics
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    summary_text = "Training Summary:\n\n"
    if il_history:
        summary_text += f"IL Training Epochs: {len(il_history.get('train_loss', []))}\n"
        if il_history.get('train_loss'):
            summary_text += f"Final IL Train Loss: {il_history['train_loss'][-1]:.4f}\n"
            summary_text += f"Final IL Val Loss: {il_history['val_loss'][-1]:.4f}\n"
    
    if rl_history:
        summary_text += f"RL Training Episodes: {len(rl_history.get('episode_rewards', []))}\n"
        if rl_history.get('episode_rewards'):
            summary_text += f"Final RL Mean Reward: {np.mean(rl_history['episode_rewards'][-100:]):.2f}\n"
    
    if eval_metrics:
        summary_text += f"Evaluation Success Rate: {eval_metrics.get('success_rate', 0):.2%}\n"
        summary_text += f"Evaluation Collision Rate: {eval_metrics.get('collision_rate', 0):.2%}\n"
    
    ax7.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training dashboard saved to {save_path}")
    
    plt.show()


def save_visualization_config():
    """Save visualization configuration."""
    config = {
        'style': 'seaborn-v0_8',
        'figure_size': (12, 8),
        'dpi': 300,
        'save_format': 'png',
        'color_palette': 'viridis'
    }
    
    with open('configs/visualization_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Visualization configuration saved to configs/visualization_config.json")


# Set default style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")