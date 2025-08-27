"""
CNN-based imitation learning agent for autonomous driving.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import os
import json
from pathlib import Path

from models.cnn_models import DrivingCNN, create_cnn_model
from utils.data_utils import DrivingDataset, ImageAugmentation, calculate_metrics


class CNNImitationAgent:
    """
    CNN-based imitation learning agent for autonomous driving.
    """
    
    def __init__(self, config: Dict[str, Any], device: str = "cuda"):
        """
        Initialize the CNN imitation agent.
        
        Args:
            config: Configuration dictionary
            device: Device to use for training ("cuda" or "cpu")
        """
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Create model
        self.model = create_cnn_model(config['imitation_learning'], model_type="basic")
        self.model.to(self.device)
        
        # Training parameters
        self.learning_rate = config['imitation_learning']['training']['learning_rate']
        self.batch_size = config['imitation_learning']['training']['batch_size']
        self.epochs = config['imitation_learning']['training']['epochs']
        self.optimizer_name = config['imitation_learning']['training']['optimizer']
        self.loss_function_name = config['imitation_learning']['training']['loss_function']
        
        # Setup optimizer and loss function
        self.optimizer = self._setup_optimizer()
        self.criterion = self._setup_loss_function()
        
        # Data augmentation
        self.augmentation = ImageAugmentation(config['imitation_learning']['augmentation'])
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        print(f"CNN Imitation Agent initialized on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer based on configuration."""
        if self.optimizer_name.lower() == "adam":
            return optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name.lower() == "sgd":
            return optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        elif self.optimizer_name.lower() == "adamw":
            return optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
    
    def _setup_loss_function(self) -> nn.Module:
        """Setup loss function based on configuration."""
        if self.loss_function_name.lower() == "mse":
            return nn.MSELoss()
        elif self.loss_function_name.lower() == "mae":
            return nn.L1Loss()
        elif self.loss_function_name.lower() == "huber":
            return nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss function: {self.loss_function_name}")
    
    def train(self, train_data_path: str, val_data_path: Optional[str] = None,
              save_path: str = "./models/cnn_imitation") -> Dict[str, List[float]]:
        """
        Train the imitation learning agent.
        
        Args:
            train_data_path: Path to training data
            val_data_path: Path to validation data (optional)
            save_path: Path to save the trained model
            
        Returns:
            Training history
        """
        print("Starting imitation learning training...")
        
        # Create datasets
        train_dataset = DrivingDataset(train_data_path, transform=self.augmentation.augment_image)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )
        
        if val_data_path:
            val_dataset = DrivingDataset(val_data_path, transform=None, augment=False)
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                num_workers=4,
                pin_memory=True
            )
        else:
            # Split training data for validation
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=4,
                pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                num_workers=4,
                pin_memory=True
            )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = self.config['imitation_learning']['training']['early_stopping_patience']
        
        for epoch in range(self.epochs):
            # Training phase
            train_loss, train_metrics = self._train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_metrics = self._validate_epoch(val_loader)
            
            # Store history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_metrics'].append(train_metrics)
            self.training_history['val_metrics'].append(val_metrics)
            
            # Print progress
            print(f"Epoch {epoch+1}/{self.epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Train MSE: {train_metrics['mse']:.4f}, Val MSE: {val_metrics['mse']:.4f}")
            print(f"Train MAE: {train_metrics['mae']:.4f}, Val MAE: {val_metrics['mae']:.4f}")
            print("-" * 50)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.save_model(save_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        print("Training completed!")
        return self.training_history
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        for batch_idx, (frames, actions) in enumerate(train_loader):
            frames = frames.to(self.device)
            actions = actions.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(frames)
            loss = self.criterion(predictions, actions)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Store for metrics
            total_loss += loss.item()
            all_predictions.append(predictions.detach().cpu().numpy())
            all_targets.append(actions.detach().cpu().numpy())
        
        # Calculate metrics
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        metrics = calculate_metrics(all_predictions, all_targets)
        
        return total_loss / len(train_loader), metrics
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for frames, actions in val_loader:
                frames = frames.to(self.device)
                actions = actions.to(self.device)
                
                # Forward pass
                predictions = self.model(frames)
                loss = self.criterion(predictions, actions)
                
                # Store for metrics
                total_loss += loss.item()
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(actions.cpu().numpy())
        
        # Calculate metrics
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        metrics = calculate_metrics(all_predictions, all_targets)
        
        return total_loss / len(val_loader), metrics
    
    def predict(self, observation: np.ndarray) -> np.ndarray:
        """
        Predict action for a given observation.
        
        Args:
            observation: Input observation (H, W) or (C, H, W)
            
        Returns:
            Predicted action
        """
        self.model.eval()
        
        # Preprocess observation
        if len(observation.shape) == 2:
            observation = observation[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)
        elif len(observation.shape) == 3:
            observation = observation[np.newaxis, :, :, :]  # (1, C, H, W)
        
        observation = torch.from_numpy(observation).float().to(self.device)
        
        with torch.no_grad():
            prediction = self.model(observation)
        
        return prediction.cpu().numpy()[0]
    
    def predict_batch(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict actions for a batch of observations.
        
        Args:
            observations: Input observations (N, H, W) or (N, C, H, W)
            
        Returns:
            Predicted actions (N, action_dim)
        """
        self.model.eval()
        
        # Preprocess observations
        if len(observations.shape) == 3:
            observations = observations[:, np.newaxis, :, :]  # (N, 1, H, W)
        
        observations = torch.from_numpy(observations).float().to(self.device)
        
        with torch.no_grad():
            predictions = self.model(observations)
        
        return predictions.cpu().numpy()
    
    def save_model(self, save_path: str) -> None:
        """
        Save the trained model.
        
        Args:
            save_path: Path to save the model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save(self.model.state_dict(), save_path / "model.pth")
        
        # Save configuration
        with open(save_path / "config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save training history
        with open(save_path / "training_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str) -> None:
        """
        Load a trained model.
        
        Args:
            load_path: Path to load the model from
        """
        load_path = Path(load_path)
        
        # Load model state
        self.model.load_state_dict(torch.load(load_path / "model.pth", map_location=self.device, weights_only=True))
        
        # Load configuration
        with open(load_path / "config.json", 'r') as f:
            self.config = json.load(f)
        
        # Load training history if available
        history_path = load_path / "training_history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)
        
        print(f"Model loaded from {load_path}")
    
    def evaluate(self, test_data_path: str) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data_path: Path to test data
            
        Returns:
            Evaluation metrics
        """
        print("Evaluating model...")
        
        # Create test dataset
        test_dataset = DrivingDataset(test_data_path, transform=None, augment=False)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        # Evaluate
        test_loss, test_metrics = self._validate_epoch(test_loader)
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test MSE: {test_metrics['mse']:.4f}")
        print(f"Test MAE: {test_metrics['mae']:.4f}")
        print(f"Test Correlation: {test_metrics['correlation_mean']:.4f}")
        
        return test_metrics
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary information."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_type': 'CNN Imitation Learning',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'input_shape': self.model.input_channels,
            'output_shape': self.model.output_dim,
            'optimizer': self.optimizer_name,
            'loss_function': self.loss_function_name,
            'learning_rate': self.learning_rate
        }