"""
Data collection script for gathering expert demonstrations for imitation learning.
"""

import numpy as np
import cv2
import gymnasium as gym
from typing import Dict, List, Tuple, Optional, Any
import os
import json
from pathlib import Path
import time
from tqdm import tqdm

from environments.env_configs import create_environment, get_highway_config
from environments.wrappers import create_wrapped_environment
from utils.data_utils import save_driving_data, preprocess_frame


class ExpertDataCollector:
    """
    Collect expert demonstrations for imitation learning.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the expert data collector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_collection_config = config['imitation_learning']['data_collection']
        
        # Create environment
        self.env = self._create_environment()
        
        # Data storage
        self.frames = []
        self.actions = []
        self.episode_data = []
        
        print("Expert Data Collector initialized")
    
    def _create_environment(self) -> gym.Env:
        """Create the environment for data collection."""
        env_name = self.config['environment']['name']
        
        # Create base environment
        env = create_environment(env_name)
        
        # Apply wrappers if needed
        wrappers_config = {
            'observation_preprocessing': {
                'enabled': True,
                'target_size': (84, 84),
                'normalize': True
            },
            'domain_randomization': {
                'enabled': False  # Disable for expert data collection
            },
            'reward_shaping': {
                'enabled': False  # Disable for expert data collection
            },
            'multi_agent': {
                'enabled': False  # Disable for expert data collection
            }
        }
        
        env = create_wrapped_environment(env, wrappers_config)
        
        return env
    
    def collect_expert_data(self, save_path: str, episodes: Optional[int] = None) -> Dict[str, Any]:
        """
        Collect expert demonstrations.
        
        Args:
            save_path: Path to save the collected data
            episodes: Number of episodes to collect (if None, use config default)
            
        Returns:
            Collection statistics
        """
        if episodes is None:
            episodes = self.data_collection_config['episodes']
        
        max_steps = self.data_collection_config['max_steps_per_episode']
        save_frames = self.data_collection_config['save_frames']
        frame_skip = self.data_collection_config['frame_skip']
        
        print(f"Collecting expert data for {episodes} episodes...")
        
        total_frames = 0
        total_actions = 0
        successful_episodes = 0
        
        for episode in tqdm(range(episodes), desc="Collecting episodes"):
            episode_frames, episode_actions, episode_stats = self._collect_episode(
                max_steps, save_frames, frame_skip
            )
            
            if episode_stats['successful']:
                self.frames.extend(episode_frames)
                self.actions.extend(episode_actions)
                successful_episodes += 1
                total_frames += len(episode_frames)
                total_actions += len(episode_actions)
            
            # Save episode data
            self.episode_data.append({
                'episode': episode,
                'frames_count': len(episode_frames),
                'actions_count': len(episode_actions),
                'successful': episode_stats['successful'],
                'episode_reward': episode_stats['episode_reward'],
                'episode_length': episode_stats['episode_length'],
                'collision': episode_stats['collision'],
                'off_road': episode_stats['off_road']
            })
            
            # Save data periodically
            if (episode + 1) % 100 == 0:
                self._save_data(save_path, episode + 1)
        
        # Save final data
        self._save_data(save_path, episodes)
        
        # Calculate statistics
        stats = {
            'total_episodes': episodes,
            'successful_episodes': successful_episodes,
            'success_rate': successful_episodes / episodes,
            'total_frames': total_frames,
            'total_actions': total_actions,
            'avg_frames_per_episode': total_frames / successful_episodes if successful_episodes > 0 else 0,
            'avg_actions_per_episode': total_actions / successful_episodes if successful_episodes > 0 else 0
        }
        
        print(f"Data collection completed!")
        print(f"Successful episodes: {successful_episodes}/{episodes} ({stats['success_rate']:.2%})")
        print(f"Total frames collected: {total_frames}")
        print(f"Total actions collected: {total_actions}")
        
        return stats
    
    def _collect_episode(self, max_steps: int, save_frames: bool, frame_skip: int) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, Any]]:
        """
        Collect data for a single episode.
        
        Args:
            max_steps: Maximum steps per episode
            save_frames: Whether to save frames
            frame_skip: Frame skip rate
            
        Returns:
            Episode frames, actions, and statistics
        """
        obs, _ = self.env.reset()
        episode_frames = []
        episode_actions = []
        episode_reward = 0
        collision = False
        off_road = False
        
        for step in range(max_steps):
            # Get expert action (using environment's default policy)
            action = self._get_expert_action(obs)
            
            # Save frame and action
            if step % frame_skip == 0 and save_frames:
                frame = self._preprocess_frame(obs)
                episode_frames.append(frame)
                episode_actions.append(action)
            
            # Take action
            obs, reward, terminated, truncated, info = self.env.step(action)
            episode_reward += reward
            
            # Check for termination conditions
            if terminated or truncated:
                collision = info.get('crashed', False)
                off_road = info.get('off_road', False)
                break
        
        # Episode statistics
        episode_stats = {
            'successful': not collision and not off_road and episode_reward > 0,
            'episode_reward': episode_reward,
            'episode_length': step + 1,
            'collision': collision,
            'off_road': off_road
        }
        
        return episode_frames, episode_actions, episode_stats
    
    def _get_expert_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Get expert action for the given observation.
        This uses the environment's default policy or scripted behavior.
        
        Args:
            observation: Current observation
            
        Returns:
            Expert action
        """
        # For highway-env, we can use the environment's built-in expert policy
        # or implement a simple rule-based expert
        
        # Simple rule-based expert for demonstration
        # In practice, you might want to use a more sophisticated expert or human demonstrations
        
        # Extract basic information from observation
        # This is a simplified version - in practice, you'd parse the observation properly
        if len(observation.shape) == 3:  # Image observation
            # For image observations, we'll use a simple heuristic
            # This is just for demonstration - real expert would be more sophisticated
            action = self._rule_based_expert_image(observation)
        else:
            # For state-based observations
            action = self._rule_based_expert_state(observation)
        
        return action
    
    def _rule_based_expert_image(self, observation: np.ndarray) -> np.ndarray:
        """
        Simple rule-based expert for image observations.
        
        Args:
            observation: Image observation
            
        Returns:
            Action [steering, acceleration]
        """
        # This is a very simplified expert for demonstration
        # In practice, you'd implement a more sophisticated expert or use human demonstrations
        
        # Simple lane following heuristic
        # Assume the road is in the center of the image
        height, width = observation.shape[-2:]
        center_x = width // 2
        
        # Find the road center (simplified)
        # In practice, you'd use proper lane detection
        road_center = center_x
        
        # Calculate steering based on road center
        steering_error = (road_center - center_x) / (width / 2)
        steering = np.clip(steering_error * 0.5, -0.5, 0.5)
        
        # Simple speed control
        acceleration = 0.5  # Maintain moderate speed
        
        return np.array([steering, acceleration])
    
    def _rule_based_expert_state(self, observation: np.ndarray) -> np.ndarray:
        """
        Simple rule-based expert for state observations.
        
        Args:
            observation: State observation
            
        Returns:
            Action [steering, acceleration]
        """
        # This is a very simplified expert for demonstration
        # In practice, you'd implement a more sophisticated expert
        
        # Assume observation contains: [x, y, heading, speed, lane_index, ...]
        if len(observation) >= 5:
            lane_index = observation[4]
            speed = observation[3]
            
            # Simple lane keeping
            target_lane = 1  # Stay in the middle lane
            lane_error = lane_index - target_lane
            steering = np.clip(-lane_error * 0.3, -0.5, 0.5)
            
            # Speed control
            target_speed = 25.0
            speed_error = target_speed - speed
            acceleration = np.clip(speed_error * 0.1, -5.0, 5.0)
        else:
            # Fallback to simple actions
            steering = 0.0
            acceleration = 0.5
        
        return np.array([steering, acceleration])
    
    def _preprocess_frame(self, observation: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for storage.
        
        Args:
            observation: Raw observation
            
        Returns:
            Preprocessed frame
        """
        if len(observation.shape) == 3:  # (C, H, W)
            # Convert to grayscale if needed
            if observation.shape[0] == 3:  # RGB
                frame = np.mean(observation, axis=0)  # Convert to grayscale
            else:
                frame = observation[0]  # Take first channel
        else:
            frame = observation
        
        # Normalize to [0, 1]
        if frame.max() > 1.0:
            frame = frame / 255.0
        
        return frame.astype(np.float32)
    
    def _save_data(self, save_path: str, episode_count: int) -> None:
        """
        Save collected data.
        
        Args:
            save_path: Path to save data
            episode_count: Current episode count
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save frames and actions
        if self.frames and self.actions:
            frames_array = np.array(self.frames)
            actions_array = np.array(self.actions)
            
            save_driving_data(
                frames_array, 
                actions_array, 
                save_path / f"episode_{episode_count:06d}",
                save_individual=False
            )
        
        # Save episode statistics
        with open(save_path / f"episode_stats_{episode_count:06d}.json", 'w') as f:
            json.dump(self.episode_data, f, indent=2)
        
        print(f"Data saved for {episode_count} episodes")


class HumanDataCollector:
    """
    Collect human demonstrations for imitation learning.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the human data collector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.env = self._create_environment()
        
        # Data storage
        self.frames = []
        self.actions = []
        
        print("Human Data Collector initialized")
        print("Use arrow keys to control the vehicle:")
        print("  ↑: Accelerate")
        print("  ↓: Brake")
        print("  ←: Turn left")
        print("  →: Turn right")
        print("  Space: Stop collection")
    
    def _create_environment(self) -> gym.Env:
        """Create the environment for human data collection."""
        env_name = self.config['environment']['name']
        env = create_environment(env_name)
        
        # Enable manual control
        env.config['manual_control'] = True
        env.config['real_time_rendering'] = True
        
        return env
    
    def collect_human_data(self, save_path: str, episodes: int = 10) -> Dict[str, Any]:
        """
        Collect human demonstrations.
        
        Args:
            save_path: Path to save the collected data
            episodes: Number of episodes to collect
            
        Returns:
            Collection statistics
        """
        print(f"Starting human data collection for {episodes} episodes...")
        print("Press any key to start...")
        input()
        
        total_frames = 0
        successful_episodes = 0
        
        for episode in range(episodes):
            print(f"\nEpisode {episode + 1}/{episodes}")
            print("Press any key to start episode...")
            input()
            
            episode_frames, episode_actions, success = self._collect_human_episode()
            
            if success:
                self.frames.extend(episode_frames)
                self.actions.extend(episode_actions)
                successful_episodes += 1
                total_frames += len(episode_frames)
            
            print(f"Episode {episode + 1} completed. Success: {success}")
        
        # Save collected data
        if self.frames and self.actions:
            frames_array = np.array(self.frames)
            actions_array = np.array(self.actions)
            
            save_driving_data(frames_array, actions_array, save_path)
        
        stats = {
            'total_episodes': episodes,
            'successful_episodes': successful_episodes,
            'success_rate': successful_episodes / episodes,
            'total_frames': total_frames
        }
        
        print(f"Human data collection completed!")
        print(f"Successful episodes: {successful_episodes}/{episodes}")
        print(f"Total frames collected: {total_frames}")
        
        return stats
    
    def _collect_human_episode(self) -> Tuple[List[np.ndarray], List[np.ndarray], bool]:
        """
        Collect human demonstration for one episode.
        
        Returns:
            Episode frames, actions, and success status
        """
        obs, _ = self.env.reset()
        episode_frames = []
        episode_actions = []
        
        # Initialize action
        action = np.array([0.0, 0.0])  # [steering, acceleration]
        
        while True:
            # Render environment
            self.env.render()
            
            # Get human input
            key = cv2.waitKey(1) & 0xFF
            
            # Process key input
            if key == ord('q'):
                break
            elif key == 82:  # Up arrow
                action[1] = min(action[1] + 0.1, 5.0)  # Accelerate
            elif key == 84:  # Down arrow
                action[1] = max(action[1] - 0.1, -5.0)  # Brake
            elif key == 81:  # Left arrow
                action[0] = max(action[0] - 0.1, -0.5)  # Turn left
            elif key == 83:  # Right arrow
                action[0] = min(action[0] + 0.1, 0.5)  # Turn right
            elif key == ord(' '):  # Space
                action = np.array([0.0, 0.0])  # Stop
            
            # Take action
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Store frame and action
            frame = self._preprocess_frame(obs)
            episode_frames.append(frame)
            episode_actions.append(action.copy())
            
            # Check for episode end
            if terminated or truncated:
                success = not info.get('crashed', False) and not info.get('off_road', False)
                break
        
        self.env.close()
        return episode_frames, episode_actions, success
    
    def _preprocess_frame(self, observation: np.ndarray) -> np.ndarray:
        """Preprocess frame for storage."""
        if len(observation.shape) == 3:  # (C, H, W)
            if observation.shape[0] == 3:  # RGB
                frame = np.mean(observation, axis=0)
            else:
                frame = observation[0]
        else:
            frame = observation
        
        if frame.max() > 1.0:
            frame = frame / 255.0
        
        return frame.astype(np.float32)


def main():
    """Main function for data collection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect expert data for imitation learning")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--save_path", type=str, default="./data/expert_demonstrations",
                       help="Path to save collected data")
    parser.add_argument("--episodes", type=int, default=None,
                       help="Number of episodes to collect")
    parser.add_argument("--mode", type=str, choices=["expert", "human"], default="expert",
                       help="Data collection mode")
    
    args = parser.parse_args()
    
    # Load configuration
    from utils.data_utils import load_config
    config = load_config(args.config)
    
    # Create data collector
    if args.mode == "expert":
        collector = ExpertDataCollector(config)
    else:
        collector = HumanDataCollector(config)
    
    # Collect data
    stats = collector.collect_expert_data(args.save_path, args.episodes)
    
    # Save statistics
    with open(f"{args.save_path}/collection_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("Data collection completed successfully!")


if __name__ == "__main__":
    main()