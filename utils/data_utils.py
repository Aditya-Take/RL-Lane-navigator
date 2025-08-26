"""
Data utilities for the autonomous driving agent.
Includes configuration loading, data augmentation, and preprocessing functions.
"""

import yaml
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import pickle
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration
    """
    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, indent=2)


class ImageAugmentation:
    """Image augmentation utilities for domain randomization."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize image augmentation.
        
        Args:
            config: Augmentation configuration
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        self.rotation_range = config.get('rotation_range', 5.0)
        self.brightness_range = config.get('brightness_range', 0.1)
        self.noise_std = config.get('noise_std', 0.01)
        self.blur_probability = config.get('blur_probability', 0.1)
    
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply augmentation to an image.
        
        Args:
            image: Input image (H, W, C) or (H, W)
            
        Returns:
            Augmented image
        """
        if not self.enabled:
            return image
        
        # Convert to float if needed
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # Random rotation
        if self.rotation_range > 0:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            image = self._rotate_image(image, angle)
        
        # Random brightness adjustment
        if self.brightness_range > 0:
            brightness_factor = np.random.uniform(
                1 - self.brightness_range, 
                1 + self.brightness_range
            )
            image = image * brightness_factor
            image = np.clip(image, 0, 255)
        
        # Random noise
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, image.shape)
            image = image + noise
            image = np.clip(image, 0, 255)
        
        # Random blur
        if np.random.random() < self.blur_probability:
            kernel_size = np.random.choice([3, 5])
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        return image
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle."""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Perform rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
        return rotated


class DrivingDataset(Dataset):
    """Dataset for driving data (frames and actions)."""
    
    def __init__(self, data_path: str, transform=None, augment: bool = True):
        """
        Initialize driving dataset.
        
        Args:
            data_path: Path to the data directory
            transform: Optional transform to apply
            augment: Whether to apply augmentation
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self.augment = augment
        
        # Load data
        self.frames, self.actions = self._load_data()
        
        print(f"Loaded {len(self.frames)} samples from {data_path}")
    
    def _load_data(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Load frames and actions from data directory."""
        frames = []
        actions = []
        
        # Load frames
        frames_file = self.data_path / "frames.pkl"
        actions_file = self.data_path / "actions.pkl"
        
        if frames_file.exists() and actions_file.exists():
            with open(frames_file, 'rb') as f:
                frames = pickle.load(f)
            with open(actions_file, 'rb') as f:
                actions = pickle.load(f)
        else:
            # Load from individual files
            frame_dir = self.data_path / "frames"
            action_dir = self.data_path / "actions"
            
            if frame_dir.exists() and action_dir.exists():
                frame_files = sorted(frame_dir.glob("*.npy"))
                action_files = sorted(action_dir.glob("*.npy"))
                
                for frame_file, action_file in zip(frame_files, action_files):
                    frame = np.load(frame_file)
                    action = np.load(action_file)
                    frames.append(frame)
                    actions.append(action)
        
        return frames, actions
    
    def __len__(self) -> int:
        return len(self.frames)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        frame = self.frames[idx].copy()
        action = self.actions[idx].copy()
        
        # Apply augmentation if enabled
        if self.augment and self.transform:
            frame = self.transform(frame)
        
        # Convert to tensors
        if len(frame.shape) == 2:  # Grayscale
            frame = torch.from_numpy(frame).unsqueeze(0).float()
        else:  # RGB
            frame = torch.from_numpy(frame).permute(2, 0, 1).float()
        
        action = torch.from_numpy(action).float()
        
        return frame, action


def preprocess_frame(frame: np.ndarray, target_size: Tuple[int, int] = (84, 84)) -> np.ndarray:
    """
    Preprocess frame for neural network input.
    
    Args:
        frame: Input frame
        target_size: Target size (height, width)
        
    Returns:
        Preprocessed frame
    """
    # Resize
    frame = cv2.resize(frame, (target_size[1], target_size[0]))
    
    # Normalize to [0, 1]
    if frame.max() > 1.0:
        frame = frame / 255.0
    
    return frame


def save_driving_data(frames: List[np.ndarray], actions: List[np.ndarray], 
                     save_path: str, save_individual: bool = False) -> None:
    """
    Save driving data to disk.
    
    Args:
        frames: List of frames
        actions: List of actions
        save_path: Path to save the data
        save_individual: Whether to save individual files
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    if save_individual:
        # Save individual files
        frames_dir = save_path / "frames"
        actions_dir = save_path / "actions"
        frames_dir.mkdir(exist_ok=True)
        actions_dir.mkdir(exist_ok=True)
        
        for i, (frame, action) in enumerate(zip(frames, actions)):
            np.save(frames_dir / f"frame_{i:06d}.npy", frame)
            np.save(actions_dir / f"action_{i:06d}.npy", action)
    else:
        # Save as pickle files
        with open(save_path / "frames.pkl", 'wb') as f:
            pickle.dump(frames, f)
        with open(save_path / "actions.pkl", 'wb') as f:
            pickle.dump(actions, f)


def create_data_loader(dataset: Dataset, batch_size: int, shuffle: bool = True, 
                      num_workers: int = 4) -> DataLoader:
    """
    Create a data loader for the dataset.
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def normalize_actions(actions: np.ndarray, action_ranges: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """
    Normalize actions to [-1, 1] range.
    
    Args:
        actions: Actions array (N, action_dim)
        action_ranges: Dictionary of action ranges for each dimension
        
    Returns:
        Normalized actions
    """
    normalized = np.zeros_like(actions)
    
    for i, (action_name, (min_val, max_val)) in enumerate(action_ranges.items()):
        if i < actions.shape[1]:
            normalized[:, i] = 2 * (actions[:, i] - min_val) / (max_val - min_val) - 1
    
    return normalized


def denormalize_actions(normalized_actions: np.ndarray, 
                       action_ranges: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """
    Denormalize actions from [-1, 1] range to original range.
    
    Args:
        normalized_actions: Normalized actions array (N, action_dim)
        action_ranges: Dictionary of action ranges for each dimension
        
    Returns:
        Denormalized actions
    """
    denormalized = np.zeros_like(normalized_actions)
    
    for i, (action_name, (min_val, max_val)) in enumerate(action_ranges.items()):
        if i < normalized_actions.shape[1]:
            denormalized[:, i] = (normalized_actions[:, i] + 1) / 2 * (max_val - min_val) + min_val
    
    return denormalized


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics for predictions.
    
    Args:
        predictions: Predicted actions
        targets: Target actions
        
    Returns:
        Dictionary of metrics
    """
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    
    # Calculate correlation for each action dimension
    correlations = []
    for i in range(predictions.shape[1]):
        if np.std(predictions[:, i]) > 0 and np.std(targets[:, i]) > 0:
            corr = np.corrcoef(predictions[:, i], targets[:, i])[0, 1]
            correlations.append(corr)
        else:
            correlations.append(0.0)
    
    return {
        'mse': mse,
        'mae': mae,
        'correlation_mean': np.mean(correlations),
        'correlation_std': np.std(correlations)
    }