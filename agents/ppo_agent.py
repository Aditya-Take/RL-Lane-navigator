"""
PPO reinforcement learning agent for autonomous driving with vision-based inputs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import gymnasium as gym
from gymnasium import spaces
import os
import json
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from models.cnn_models import DrivingCNN, create_cnn_model
from utils.data_utils import preprocess_frame


class CNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    CNN feature extractor for PPO with vision-based inputs.
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        """
        Initialize CNN feature extractor.
        
        Args:
            observation_space: Observation space
            features_dim: Dimension of extracted features
        """
        super().__init__(observation_space, features_dim)
        
        # Create CNN model
        self.cnn = DrivingCNN(
            input_channels=observation_space.shape[0],
            conv_layers=[32, 64, 128, 256],
            fc_layers=[512, features_dim],
            output_dim=features_dim,
            normalize_output=False  # Don't normalize for feature extraction
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features from observations.
        
        Args:
            observations: Input observations (batch_size, channels, height, width)
            
        Returns:
            Extracted features
        """
        return self.cnn.get_features(observations)


class PPODrivingAgent:
    """
    PPO-based reinforcement learning agent for autonomous driving.
    """
    
    def __init__(self, config: Dict[str, Any], env: gym.Env, device: str = "cuda"):
        """
        Initialize the PPO driving agent.
        
        Args:
            config: Configuration dictionary
            env: Training environment
            device: Device to use for training
        """
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.env = env
        
        # PPO parameters
        ppo_config = config['reinforcement_learning']['ppo']
        self.learning_rate = ppo_config['learning_rate']
        self.n_steps = ppo_config['n_steps']
        self.batch_size = ppo_config['batch_size']
        self.n_epochs = ppo_config['n_epochs']
        self.gamma = ppo_config['gamma']
        self.gae_lambda = ppo_config['gae_lambda']
        self.clip_range = ppo_config['clip_range']
        self.ent_coef = ppo_config['ent_coef']
        self.vf_coef = ppo_config['vf_coef']
        self.max_grad_norm = ppo_config['max_grad_norm']
        
        # Training parameters
        self.total_timesteps = config['reinforcement_learning']['training']['total_timesteps']
        self.eval_freq = config['reinforcement_learning']['training']['eval_freq']
        self.n_eval_episodes = config['reinforcement_learning']['training']['n_eval_episodes']
        self.save_freq = config['reinforcement_learning']['training']['save_freq']
        
        # Create PPO model
        self.model = self._create_ppo_model()
        
        # Training history
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'eval_rewards': [],
            'eval_lengths': []
        }
        
        print(f"PPO Driving Agent initialized on {self.device}")
    
    def _create_ppo_model(self) -> PPO:
        """Create PPO model with custom feature extractor."""
        # Check if observation space is image-based
        if len(self.env.observation_space.shape) == 3:  # (C, H, W)
            # Use CNN feature extractor for vision-based inputs
            policy_kwargs = {
                "features_extractor_class": CNNFeaturesExtractor,
                "features_extractor_kwargs": {
                    "features_dim": 256
                }
            }
        else:
            # Use default MLP for state-based inputs
            policy_kwargs = {
                "net_arch": {
                    "pi": self.config['reinforcement_learning']['policy_network']['mlp']['hidden_layers'],
                    "vf": self.config['reinforcement_learning']['policy_network']['mlp']['hidden_layers']
                }
            }
        
        # Create PPO model
        model = PPO(
            "CnnPolicy" if len(self.env.observation_space.shape) == 3 else "MlpPolicy",
            self.env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log="./logs/ppo_tensorboard/"
        )
        
        return model
    
    def train(self, save_path: str = "./models/ppo_driving", 
              pretrained_il_path: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Train the PPO agent.
        
        Args:
            save_path: Path to save the trained model
            pretrained_il_path: Path to pretrained imitation learning model (optional)
            
        Returns:
            Training history
        """
        print("Starting PPO training...")
        
        # Load pretrained IL weights if provided
        if pretrained_il_path:
            self._load_pretrained_weights(pretrained_il_path)
        
        # Create callbacks
        callbacks = self._create_callbacks(save_path)
        
        # Train the model
        self.model.learn(
            total_timesteps=self.total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        # Save final model
        self.save_model(save_path)
        
        print("PPO training completed!")
        return self.training_history
    
    def _load_pretrained_weights(self, pretrained_path: str) -> None:
        """Load pretrained weights from imitation learning model."""
        print(f"Loading pretrained weights from {pretrained_path}")
        
        # Load IL model
        il_model = torch.load(f"{pretrained_path}/model.pth", map_location=self.device, weights_only=True)
        
        # Extract CNN weights
        cnn_weights = {}
        for key, value in il_model.items():
            if key.startswith('conv_layers') or key.startswith('fc_layers'):
                cnn_weights[key] = value
        
        # Load weights into PPO feature extractor
        if hasattr(self.model.policy, 'features_extractor'):
            feature_extractor_state = self.model.policy.features_extractor.state_dict()
            
            # Update weights that match
            for key in cnn_weights:
                if key in feature_extractor_state:
                    feature_extractor_state[key] = cnn_weights[key]
            
            self.model.policy.features_extractor.load_state_dict(feature_extractor_state)
            print("Pretrained weights loaded successfully")
    
    def _create_callbacks(self, save_path: str) -> List[BaseCallback]:
        """Create training callbacks."""
        callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.save_freq,
            save_path=save_path,
            name_prefix="ppo_model"
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback
        if hasattr(self, 'eval_env'):
            eval_callback = EvalCallback(
                self.eval_env,
                best_model_save_path=save_path,
                log_path=save_path,
                eval_freq=self.eval_freq,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)
        
        # Custom callback for tracking training history
        history_callback = TrainingHistoryCallback(self.training_history)
        callbacks.append(history_callback)
        
        return callbacks
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Predict action for a given observation.
        
        Args:
            observation: Input observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Predicted action
        """
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action
    
    def evaluate(self, eval_env: gym.Env, n_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the trained agent.
        
        Args:
            eval_env: Evaluation environment
            n_episodes: Number of episodes to evaluate
            
        Returns:
            Evaluation metrics
        """
        print(f"Evaluating agent over {n_episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        collision_count = 0
        off_road_count = 0
        
        for episode in range(n_episodes):
            obs, _ = eval_env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
                
                # Track specific events
                if info.get('crashed', False):
                    collision_count += 1
                if info.get('off_road', False):
                    off_road_count += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        # Calculate metrics
        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'collision_rate': collision_count / n_episodes,
            'off_road_rate': off_road_count / n_episodes,
            'success_rate': sum(1 for r in episode_rewards if r > 0) / n_episodes
        }
        
        print(f"Evaluation Results:")
        print(f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"Mean Episode Length: {metrics['mean_length']:.1f}")
        print(f"Collision Rate: {metrics['collision_rate']:.2%}")
        print(f"Off-road Rate: {metrics['off_road_rate']:.2%}")
        print(f"Success Rate: {metrics['success_rate']:.2%}")
        
        return metrics
    
    def save_model(self, save_path: str) -> None:
        """
        Save the trained model.
        
        Args:
            save_path: Path to save the model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save PPO model
        self.model.save(save_path / "ppo_model")
        
        # Save configuration
        with open(save_path / "config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save training history
        with open(save_path / "training_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"PPO model saved to {save_path}")
    
    def load_model(self, load_path: str) -> None:
        """
        Load a trained model.
        
        Args:
            load_path: Path to load the model from
        """
        load_path = Path(load_path)
        
        # Load PPO model
        self.model = PPO.load(load_path / "ppo_model")
        
        # Load configuration
        with open(load_path / "config.json", 'r') as f:
            self.config = json.load(f)
        
        # Load training history if available
        history_path = load_path / "training_history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)
        
        print(f"PPO model loaded from {load_path}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary information."""
        total_params = sum(p.numel() for p in self.model.policy.parameters())
        trainable_params = sum(p.numel() for p in self.model.policy.parameters() if p.requires_grad)
        
        return {
            'model_type': 'PPO Reinforcement Learning',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'policy_type': self.model.policy.__class__.__name__,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'ent_coef': self.ent_coef
        }


class TrainingHistoryCallback(BaseCallback):
    """Callback to track training history."""
    
    def __init__(self, history_dict: Dict[str, List[float]]):
        """
        Initialize training history callback.
        
        Args:
            history_dict: Dictionary to store training history
        """
        super().__init__()
        self.history = history_dict
    
    def _on_step(self) -> bool:
        """Called after each step."""
        # Track episode rewards and lengths
        if self.locals.get('dones') is not None:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    episode_reward = self.locals.get('rewards', [0])[i] if i < len(self.locals.get('rewards', [])) else 0
                    episode_length = self.num_timesteps
                    
                    self.history['episode_rewards'].append(episode_reward)
                    self.history['episode_lengths'].append(episode_length)
        
        return True


class HybridAgent:
    """
    Hybrid agent combining imitation learning and reinforcement learning.
    """
    
    def __init__(self, il_agent: 'CNNImitationAgent', rl_agent: 'PPODrivingAgent',
                 config: Dict[str, Any]):
        """
        Initialize hybrid agent.
        
        Args:
            il_agent: Imitation learning agent
            rl_agent: Reinforcement learning agent
            config: Configuration dictionary
        """
        self.il_agent = il_agent
        self.rl_agent = rl_agent
        self.config = config
        
        # Hybrid parameters
        self.il_weight = config.get('hybrid', {}).get('il_weight', 0.5)
        self.rl_weight = 1.0 - self.il_weight
        
        print(f"Hybrid Agent initialized with IL weight: {self.il_weight}, RL weight: {self.rl_weight}")
    
    def predict(self, observation: np.ndarray, use_hybrid: bool = True) -> np.ndarray:
        """
        Predict action using hybrid approach.
        
        Args:
            observation: Input observation
            use_hybrid: Whether to use hybrid prediction
            
        Returns:
            Predicted action
        """
        if use_hybrid:
            # Get predictions from both agents
            il_action = self.il_agent.predict(observation)
            rl_action = self.rl_agent.predict(observation)
            
            # Combine predictions
            hybrid_action = (self.il_weight * il_action + self.rl_weight * rl_action)
            return hybrid_action
        else:
            # Use only RL agent
            return self.rl_agent.predict(observation)
    
    def evaluate(self, eval_env: gym.Env, n_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the hybrid agent.
        
        Args:
            eval_env: Evaluation environment
            n_episodes: Number of episodes to evaluate
            
        Returns:
            Evaluation metrics
        """
        print(f"Evaluating hybrid agent over {n_episodes} episodes...")
        
        # Evaluate with hybrid prediction
        hybrid_metrics = self._evaluate_mode(eval_env, n_episodes, use_hybrid=True)
        
        # Evaluate with RL-only prediction
        rl_metrics = self._evaluate_mode(eval_env, n_episodes, use_hybrid=False)
        
        # Combine results
        metrics = {
            'hybrid': hybrid_metrics,
            'rl_only': rl_metrics
        }
        
        print("Hybrid Agent Evaluation Results:")
        print(f"Hybrid - Mean Reward: {hybrid_metrics['mean_reward']:.2f}")
        print(f"RL Only - Mean Reward: {rl_metrics['mean_reward']:.2f}")
        
        return metrics
    
    def _evaluate_mode(self, eval_env: gym.Env, n_episodes: int, use_hybrid: bool) -> Dict[str, float]:
        """Evaluate agent in a specific mode."""
        episode_rewards = []
        episode_lengths = []
        collision_count = 0
        
        for episode in range(n_episodes):
            obs, _ = eval_env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action = self.predict(obs, use_hybrid=use_hybrid)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
                
                if info.get('crashed', False):
                    collision_count += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'collision_rate': collision_count / n_episodes,
            'success_rate': sum(1 for r in episode_rewards if r > 0) / n_episodes
        }