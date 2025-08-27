"""
CNN models for vision-based autonomous driving.
Includes architectures for both imitation learning and reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class DrivingCNN(nn.Module):
    """
    CNN architecture for driving behavior learning.
    Can be used for both imitation learning and as a feature extractor for RL.
    """
    
    def __init__(self, 
                 input_channels: int = 1,
                 conv_layers: List[int] = [32, 64, 128, 256],
                 fc_layers: List[int] = [512, 256, 128],
                 output_dim: int = 2,
                 dropout_rate: float = 0.3,
                 activation: str = "relu",
                 normalize_output: bool = True):
        """
        Initialize the CNN model.
        
        Args:
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
            conv_layers: List of convolution layer sizes
            fc_layers: List of fully connected layer sizes
            output_dim: Output dimension (number of actions)
            dropout_rate: Dropout rate for regularization
            activation: Activation function ("relu", "leaky_relu", "elu")
            normalize_output: Whether to normalize output to [-1, 1]
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.normalize_output = normalize_output
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels
        
        for out_channels in conv_layers:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    self._get_activation(activation),
                    nn.MaxPool2d(2, 2),
                    nn.Dropout2d(dropout_rate * 0.5)
                )
            )
            in_channels = out_channels
        
        # Calculate the size of flattened features
        # Assuming input size of 84x84, after 4 maxpool layers: 84 -> 42 -> 21 -> 10 -> 5
        self.feature_size = conv_layers[-1] * 5 * 5
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        in_features = self.feature_size
        
        for out_features in fc_layers:
            self.fc_layers.append(
                nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.BatchNorm1d(out_features),
                    self._get_activation(activation),
                    nn.Dropout(dropout_rate)
                )
            )
            in_features = out_features
        
        # Output layer
        self.output_layer = nn.Linear(in_features, output_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            return nn.LeakyReLU(0.1, inplace=True)
        elif activation == "elu":
            return nn.ELU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            Output tensor (batch_size, output_dim)
        """
        # Convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        
        # Output layer
        x = self.output_layer(x)
        
        # Normalize output to [-1, 1] if requested
        if self.normalize_output:
            x = torch.tanh(x)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the convolutional layers.
        Useful for transfer learning or feature analysis.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            Feature tensor (batch_size, feature_size)
        """
        # Convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        return x


class DrivingCNNWithAttention(nn.Module):
    """
    CNN with attention mechanism for better focus on relevant parts of the image.
    """
    
    def __init__(self, 
                 input_channels: int = 1,
                 conv_layers: List[int] = [32, 64, 128, 256],
                 fc_layers: List[int] = [512, 256, 128],
                 output_dim: int = 2,
                 dropout_rate: float = 0.3,
                 attention_dim: int = 128):
        """
        Initialize the CNN with attention.
        
        Args:
            input_channels: Number of input channels
            conv_layers: List of convolution layer sizes
            fc_layers: List of fully connected layer sizes
            output_dim: Output dimension
            dropout_rate: Dropout rate
            attention_dim: Dimension of attention mechanism
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.output_dim = output_dim
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels
        
        for out_channels in conv_layers:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Dropout2d(dropout_rate * 0.5)
                )
            )
            in_channels = out_channels
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(conv_layers[-1], attention_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(attention_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Calculate feature size
        self.feature_size = conv_layers[-1] * 5 * 5
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        in_features = self.feature_size
        
        for out_features in fc_layers:
            self.fc_layers.append(
                nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.BatchNorm1d(out_features),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout_rate)
                )
            )
            in_features = out_features
        
        # Output layer
        self.output_layer = nn.Linear(in_features, output_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            Output tensor and attention weights
        """
        # Convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Apply attention
        attention_weights = self.attention(x)
        attended_features = x * attention_weights
        
        # Flatten
        x = attended_features.view(attended_features.size(0), -1)
        
        # Fully connected layers
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        
        # Output layer
        x = self.output_layer(x)
        
        # Normalize output
        x = torch.tanh(x)
        
        return x, attention_weights


class DrivingCNNWithLSTM(nn.Module):
    """
    CNN with LSTM for temporal modeling of driving behavior.
    """
    
    def __init__(self, 
                 input_channels: int = 1,
                 conv_layers: List[int] = [32, 64, 128, 256],
                 lstm_hidden_size: int = 128,
                 lstm_num_layers: int = 2,
                 fc_layers: List[int] = [256, 128],
                 output_dim: int = 2,
                 dropout_rate: float = 0.3):
        """
        Initialize CNN with LSTM.
        
        Args:
            input_channels: Number of input channels
            conv_layers: List of convolution layer sizes
            lstm_hidden_size: LSTM hidden size
            lstm_num_layers: Number of LSTM layers
            fc_layers: List of fully connected layer sizes
            output_dim: Output dimension
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels
        
        for out_channels in conv_layers:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Dropout2d(dropout_rate * 0.5)
                )
            )
            in_channels = out_channels
        
        # Calculate feature size
        self.feature_size = conv_layers[-1] * 5 * 5
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout_rate if lstm_num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        in_features = lstm_hidden_size
        
        for out_features in fc_layers:
            self.fc_layers.append(
                nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.BatchNorm1d(out_features),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout_rate)
                )
            )
            in_features = out_features
        
        # Output layer
        self.output_layer = nn.Linear(in_features, output_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with LSTM.
        
        Args:
            x: Input tensor (batch_size, seq_len, channels, height, width)
            hidden: LSTM hidden state
            
        Returns:
            Output tensor and new hidden state
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Process each frame through CNN
        cnn_outputs = []
        for t in range(seq_len):
            frame = x[:, t]  # (batch_size, channels, height, width)
            
            # Convolutional layers
            for conv_layer in self.conv_layers:
                frame = conv_layer(frame)
            
            # Flatten
            frame = frame.view(frame.size(0), -1)  # (batch_size, feature_size)
            cnn_outputs.append(frame)
        
        # Stack CNN outputs
        cnn_outputs = torch.stack(cnn_outputs, dim=1)  # (batch_size, seq_len, feature_size)
        
        # LSTM layer
        lstm_out, hidden = self.lstm(cnn_outputs, hidden)
        
        # Take the last output
        lstm_out = lstm_out[:, -1]  # (batch_size, lstm_hidden_size)
        
        # Fully connected layers
        for fc_layer in self.fc_layers:
            lstm_out = fc_layer(lstm_out)
        
        # Output layer
        output = self.output_layer(lstm_out)
        
        # Normalize output
        output = torch.tanh(output)
        
        return output, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state."""
        return (
            torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(device),
            torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(device)
        )


def create_cnn_model(config: Dict[str, Any], model_type: str = "basic") -> nn.Module:
    """
    Factory function to create CNN models based on configuration.
    
    Args:
        config: Model configuration
        model_type: Type of model ("basic", "attention", "lstm")
        
    Returns:
        CNN model
    """
    cnn_config = config['cnn']
    
    if model_type == "basic":
        return DrivingCNN(
            input_channels=cnn_config['input_channels'],
            conv_layers=cnn_config['conv_layers'],
            fc_layers=cnn_config['fc_layers'],
            output_dim=2,  # steering, acceleration
            dropout_rate=cnn_config['dropout_rate'],
            activation=cnn_config['activation']
        )
    elif model_type == "attention":
        return DrivingCNNWithAttention(
            input_channels=cnn_config['input_channels'],
            conv_layers=cnn_config['conv_layers'],
            fc_layers=cnn_config['fc_layers'],
            output_dim=2,
            dropout_rate=cnn_config['dropout_rate']
        )
    elif model_type == "lstm":
        return DrivingCNNWithLSTM(
            input_channels=cnn_config['input_channels'],
            conv_layers=cnn_config['conv_layers'],
            fc_layers=cnn_config['fc_layers'],
            output_dim=2,
            dropout_rate=cnn_config['dropout_rate']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)