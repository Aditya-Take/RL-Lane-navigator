"""
Compatibility module for highway_env imports.
Handles different versions of the highway_env package.
"""

import gymnasium as gym
from typing import Dict, Any, Optional


def make_highway_env(env_name: str, **kwargs) -> gym.Env:
    """
    Create a highway environment with compatibility for different versions.
    
    Args:
        env_name: Name of the environment
        **kwargs: Environment configuration
        
    Returns:
        Gymnasium environment
    """
    try:
        # Try gymnasium first (newer versions)
        env = gym.make(env_name, **kwargs)
        return env
    except Exception as e1:
        try:
            # Try highway_env.make (older versions)
            from highway_env import make as highway_make
            env = highway_make(env_name, **kwargs)
            return env
        except ImportError as e2:
            try:
                # Try direct highway_env import
                import highway_env
                env = highway_env.make(env_name, **kwargs)
                return env
            except Exception as e3:
                # Fallback to basic gymnasium environment
                print(f"Warning: Could not create {env_name} with highway_env, using basic gymnasium environment")
                print(f"Errors: {e1}, {e2}, {e3}")
                env = gym.make(env_name)
                return env


def register_highway_envs():
    """Register highway environments with gymnasium."""
    try:
        # Try to register highway environments
        import highway_env
        # The environments should be automatically registered
        print("✅ Highway environments registered successfully")
    except ImportError:
        print("⚠️  highway_env not available, using basic gymnasium environments")
        # Register basic environments as fallback
        pass


# Register environments on import
register_highway_envs()