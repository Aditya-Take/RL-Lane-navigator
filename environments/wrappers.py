"""
Environment wrappers for domain randomization, reward shaping, and observation preprocessing.
"""

import gymnasium as gym
import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional, Union
from gymnasium import spaces
import random


class DomainRandomizationWrapper(gym.Wrapper):
    """
    Wrapper for applying domain randomization to observations and environment parameters.
    """
    
    def __init__(self, env: gym.Env, randomization_config: Dict[str, Any]):
        """
        Initialize domain randomization wrapper.
        
        Args:
            env: Base environment
            randomization_config: Domain randomization configuration
        """
        super().__init__(env)
        self.randomization_config = randomization_config
        self.enabled = randomization_config.get('enabled', True)
        
        # Weather randomization parameters
        self.weather_config = randomization_config.get('weather', {})
        self.brightness_range = self.weather_config.get('brightness_range', [0.7, 1.3])
        self.contrast_range = self.weather_config.get('contrast_range', [0.8, 1.2])
        self.noise_std_range = self.weather_config.get('noise_std_range', [0.0, 0.05])
        self.blur_range = self.weather_config.get('blur_range', [0.0, 2.0])
        
        # Traffic randomization parameters
        self.traffic_config = randomization_config.get('traffic', {})
        self.vehicle_speed_range = self.traffic_config.get('speed_range', [15.0, 35.0])
        self.aggression_range = self.traffic_config.get('aggression_range', [0.1, 0.9])
        
        # Current randomization parameters
        self.current_brightness = 1.0
        self.current_contrast = 1.0
        self.current_noise_std = 0.0
        self.current_blur = 0.0
        
        # Apply randomization at reset
        self._apply_randomization()
    
    def _apply_randomization(self):
        """Apply random domain randomization parameters."""
        if not self.enabled:
            return
        
        # Randomize weather parameters
        self.current_brightness = np.random.uniform(*self.brightness_range)
        self.current_contrast = np.random.uniform(*self.contrast_range)
        self.current_noise_std = np.random.uniform(*self.noise_std_range)
        self.current_blur = np.random.uniform(*self.blur_range)
        
        # Randomize traffic parameters if available
        if hasattr(self.env, 'vehicles'):
            for vehicle in self.env.vehicles:
                if hasattr(vehicle, 'speed'):
                    vehicle.speed = np.random.uniform(*self.vehicle_speed_range)
                if hasattr(vehicle, 'aggressive'):
                    vehicle.aggressive = np.random.uniform(*self.aggression_range)
    
    def _randomize_observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Apply domain randomization to observation.
        
        Args:
            observation: Input observation
            
        Returns:
            Randomized observation
        """
        if not self.enabled:
            return observation
        
        # Convert to float if needed
        if observation.dtype != np.float32:
            observation = observation.astype(np.float32)
        
        # Apply brightness adjustment
        if self.current_brightness != 1.0:
            observation = observation * self.current_brightness
            observation = np.clip(observation, 0, 255)
        
        # Apply contrast adjustment
        if self.current_contrast != 1.0:
            mean_val = np.mean(observation)
            observation = (observation - mean_val) * self.current_contrast + mean_val
            observation = np.clip(observation, 0, 255)
        
        # Apply noise
        if self.current_noise_std > 0:
            noise = np.random.normal(0, self.current_noise_std, observation.shape)
            observation = observation + noise
            observation = np.clip(observation, 0, 255)
        
        # Apply blur
        if self.current_blur > 0:
            kernel_size = max(1, int(self.current_blur * 2 + 1))
            if kernel_size % 2 == 0:
                kernel_size += 1
            observation = cv2.GaussianBlur(observation, (kernel_size, kernel_size), 0)
        
        return observation
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and apply new randomization."""
        observation, info = self.env.reset(**kwargs)
        self._apply_randomization()
        observation = self._randomize_observation(observation)
        return observation, info
    
    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step environment with randomized observation."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation = self._randomize_observation(observation)
        return observation, reward, terminated, truncated, info


class RewardShapingWrapper(gym.Wrapper):
    """
    Wrapper for custom reward shaping based on driving behavior.
    """
    
    def __init__(self, env: gym.Env, reward_config: Dict[str, Any]):
        """
        Initialize reward shaping wrapper.
        
        Args:
            env: Base environment
            reward_config: Reward configuration
        """
        super().__init__(env)
        self.reward_config = reward_config
        
        # Store previous state for calculating smoothness rewards
        self.previous_action = None
        self.previous_speed = None
        self.previous_lane = None
    
    def _calculate_safety_reward(self, info: Dict[str, Any]) -> float:
        """Calculate safety-related rewards."""
        safety_reward = 0.0
        
        # Safe distance reward
        if 'closest_vehicle_distance' in info:
            distance = info['closest_vehicle_distance']
            if distance > 20:  # Safe distance
                safety_reward += self.reward_config.get('safe_distance', 0.2)
            elif distance < 10:  # Too close
                safety_reward -= self.reward_config.get('safe_distance', 0.2)
        
        return safety_reward
    
    def _calculate_smoothness_reward(self, action: np.ndarray, info: Dict[str, Any]) -> float:
        """Calculate smoothness-related rewards."""
        smoothness_reward = 0.0
        
        if self.previous_action is not None:
            # Sudden braking penalty
            if action[1] < -2.0 and self.previous_action[1] > -1.0:
                smoothness_reward += self.reward_config.get('sudden_braking', -0.5)
            
            # Sudden acceleration penalty
            if action[1] > 2.0 and self.previous_action[1] < 1.0:
                smoothness_reward += self.reward_config.get('sudden_acceleration', -0.3)
        
        # Smooth driving reward
        if self.previous_action is not None:
            action_change = np.linalg.norm(action - self.previous_action)
            if action_change < 0.5:  # Smooth action
                smoothness_reward += self.reward_config.get('smooth_driving', 0.1)
        
        return smoothness_reward
    
    def _calculate_efficiency_reward(self, info: Dict[str, Any]) -> float:
        """Calculate efficiency-related rewards."""
        efficiency_reward = 0.0
        
        # Speed efficiency
        if 'speed' in info:
            speed = info['speed']
            target_speed_range = self.reward_config.get('reward_speed_range', [20.0, 30.0])
            if target_speed_range[0] <= speed <= target_speed_range[1]:
                efficiency_reward += self.reward_config.get('speed_efficiency', 0.3)
        
        # Progress reward
        if 'progress' in info:
            efficiency_reward += info['progress'] * self.reward_config.get('progress', 0.1)
        
        # Fuel efficiency (simplified)
        if 'acceleration' in info:
            accel = abs(info['acceleration'])
            if accel < 1.0:  # Gentle acceleration
                efficiency_reward += self.reward_config.get('fuel_efficiency', 0.05)
        
        return efficiency_reward
    
    def _calculate_lane_reward(self, info: Dict[str, Any]) -> float:
        """Calculate lane-related rewards."""
        lane_reward = 0.0
        
        # Lane keeping reward
        if 'lane_index' in info:
            current_lane = info['lane_index']
            if self.previous_lane is not None and current_lane == self.previous_lane:
                lane_reward += self.reward_config.get('lane_keeping', 0.5)
            
            # Lane violation penalty
            if current_lane < 0 or current_lane >= info.get('lanes_count', 4):
                lane_reward += self.reward_config.get('lane_violation', -1.0)
        
        return lane_reward
    
    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step environment with custom reward shaping."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Calculate additional rewards
        safety_reward = self._calculate_safety_reward(info)
        smoothness_reward = self._calculate_smoothness_reward(action, info)
        efficiency_reward = self._calculate_efficiency_reward(info)
        lane_reward = self._calculate_lane_reward(info)
        
        # Combine rewards
        shaped_reward = reward + safety_reward + smoothness_reward + efficiency_reward + lane_reward
        
        # Update previous state
        self.previous_action = action.copy() if isinstance(action, np.ndarray) else action
        if 'speed' in info:
            self.previous_speed = info['speed']
        if 'lane_index' in info:
            self.previous_lane = info['lane_index']
        
        # Add reward components to info
        info['reward_components'] = {
            'base_reward': reward,
            'safety_reward': safety_reward,
            'smoothness_reward': smoothness_reward,
            'efficiency_reward': efficiency_reward,
            'lane_reward': lane_reward,
            'total_reward': shaped_reward
        }
        
        return observation, shaped_reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and previous state."""
        observation, info = self.env.reset(**kwargs)
        self.previous_action = None
        self.previous_speed = None
        self.previous_lane = None
        return observation, info


class ObservationPreprocessingWrapper(gym.Wrapper):
    """
    Wrapper for preprocessing observations (normalization, resizing, etc.).
    """
    
    def __init__(self, env: gym.Env, target_size: Tuple[int, int] = (84, 84), normalize: bool = True):
        """
        Initialize observation preprocessing wrapper.
        
        Args:
            env: Base environment
            target_size: Target size for observations (height, width)
            normalize: Whether to normalize observations to [0, 1]
        """
        super().__init__(env)
        self.target_size = target_size
        self.normalize = normalize
        
        # Update observation space
        if hasattr(env.observation_space, 'shape'):
            if len(env.observation_space.shape) == 3:  # Image observation
                self.observation_space = spaces.Box(
                    low=0.0 if normalize else 0,
                    high=1.0 if normalize else 255,
                    shape=(env.observation_space.shape[0], *target_size),
                    dtype=np.float32 if normalize else env.observation_space.dtype
                )
    
    def _preprocess_observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Preprocess observation.
        
        Args:
            observation: Input observation
            
        Returns:
            Preprocessed observation
        """
        # Resize if needed
        if observation.shape[-2:] != self.target_size:
            if len(observation.shape) == 3:  # (C, H, W)
                resized = np.zeros((observation.shape[0], *self.target_size), dtype=observation.dtype)
                for i in range(observation.shape[0]):
                    resized[i] = cv2.resize(observation[i], self.target_size[::-1])
                observation = resized
            elif len(observation.shape) == 2:  # (H, W)
                observation = cv2.resize(observation, self.target_size[::-1])
        
        # Normalize to [0, 1]
        if self.normalize and observation.max() > 1.0:
            observation = observation.astype(np.float32) / 255.0
        
        return observation
    
    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step environment with preprocessed observation."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation = self._preprocess_observation(observation)
        return observation, reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment with preprocessed observation."""
        observation, info = self.env.reset(**kwargs)
        observation = self._preprocess_observation(observation)
        return observation, info


class FrameStackWrapper(gym.Wrapper):
    """
    Wrapper for stacking multiple frames to provide temporal information.
    """
    
    def __init__(self, env: gym.Env, num_frames: int = 4):
        """
        Initialize frame stack wrapper.
        
        Args:
            env: Base environment
            num_frames: Number of frames to stack
        """
        super().__init__(env)
        self.num_frames = num_frames
        self.frames = []
        
        # Update observation space
        if hasattr(env.observation_space, 'shape'):
            original_shape = env.observation_space.shape
            if len(original_shape) == 3:  # (C, H, W)
                new_shape = (original_shape[0] * num_frames, *original_shape[1:])
            elif len(original_shape) == 2:  # (H, W)
                new_shape = (num_frames, *original_shape)
            else:
                raise ValueError(f"Unsupported observation shape: {original_shape}")
            
            self.observation_space = spaces.Box(
                low=env.observation_space.low.min(),
                high=env.observation_space.high.max(),
                shape=new_shape,
                dtype=env.observation_space.dtype
            )
    
    def _stack_frames(self) -> np.ndarray:
        """Stack the stored frames."""
        if len(self.frames) == 0:
            return np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        
        # Pad with zeros if not enough frames
        while len(self.frames) < self.num_frames:
            self.frames.insert(0, np.zeros_like(self.frames[0]))
        
        # Keep only the last num_frames
        if len(self.frames) > self.num_frames:
            self.frames = self.frames[-self.num_frames:]
        
        # Stack frames
        if len(self.frames[0].shape) == 3:  # (C, H, W)
            stacked = np.concatenate(self.frames, axis=0)
        else:  # (H, W)
            stacked = np.stack(self.frames, axis=0)
        
        return stacked
    
    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step environment with frame stacking."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(observation)
        stacked_observation = self._stack_frames()
        return stacked_observation, reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and clear frame stack."""
        observation, info = self.env.reset(**kwargs)
        self.frames = [observation]
        stacked_observation = self._stack_frames()
        return stacked_observation, info


class MultiAgentWrapper(gym.Wrapper):
    """
    Wrapper for multi-agent scenarios with different vehicle behaviors.
    """
    
    def __init__(self, env: gym.Env, agent_types: list):
        """
        Initialize multi-agent wrapper.
        
        Args:
            env: Base environment
            agent_types: List of agent type configurations
        """
        super().__init__(env)
        self.agent_types = agent_types
        
        # Apply agent types to vehicles
        self._apply_agent_types()
    
    def _apply_agent_types(self):
        """Apply different agent types to vehicles."""
        if hasattr(self.env, 'vehicles') and self.env.vehicles:
            for i, vehicle in enumerate(self.env.vehicles):
                # Select agent type based on probability
                agent_type = random.choices(
                    self.agent_types,
                    weights=[at.get('probability', 1.0) for at in self.agent_types]
                )[0]
                
                # Apply agent type properties
                if 'speed_range' in agent_type:
                    vehicle.speed = np.random.uniform(*agent_type['speed_range'])
                
                if 'aggression' in agent_type:
                    vehicle.aggressive = agent_type['aggression']
                
                # Store agent type for reference
                vehicle.agent_type = agent_type.get('name', 'Unknown')
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and reapply agent types."""
        observation, info = self.env.reset(**kwargs)
        self._apply_agent_types()
        return observation, info


def create_wrapped_environment(env: gym.Env, 
                             wrappers_config: Dict[str, Any]) -> gym.Env:
    """
    Create a wrapped environment with multiple wrappers.
    
    Args:
        env: Base environment
        wrappers_config: Configuration for wrappers
        
    Returns:
        Wrapped environment
    """
    # Apply observation preprocessing
    if wrappers_config.get('observation_preprocessing', {}).get('enabled', True):
        obs_config = wrappers_config['observation_preprocessing']
        env = ObservationPreprocessingWrapper(
            env,
            target_size=obs_config.get('target_size', (84, 84)),
            normalize=obs_config.get('normalize', True)
        )
    
    # Apply frame stacking
    if wrappers_config.get('frame_stacking', {}).get('enabled', False):
        frame_config = wrappers_config['frame_stacking']
        env = FrameStackWrapper(env, num_frames=frame_config.get('num_frames', 4))
    
    # Apply domain randomization
    if wrappers_config.get('domain_randomization', {}).get('enabled', True):
        env = DomainRandomizationWrapper(env, wrappers_config['domain_randomization'])
    
    # Apply reward shaping
    if wrappers_config.get('reward_shaping', {}).get('enabled', True):
        env = RewardShapingWrapper(env, wrappers_config['reward_shaping'])
    
    # Apply multi-agent wrapper
    if wrappers_config.get('multi_agent', {}).get('enabled', True):
        env = MultiAgentWrapper(env, wrappers_config['multi_agent'].get('agent_types', []))
    
    return env