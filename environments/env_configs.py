"""
Environment configurations for highway-env scenarios.
Includes configurations for different environments and domain randomization.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import gymnasium as gym
import highway_env


def get_highway_config(vehicles_count: int = 50, 
                      observation_type: str = "GrayscaleObservation",
                      action_type: str = "ContinuousAction",
                      **kwargs) -> Dict[str, Any]:
    """
    Get configuration for highway environment.
    
    Args:
        vehicles_count: Number of vehicles in the environment
        observation_type: Type of observation ("GrayscaleObservation", "RGB", "Kinematics")
        action_type: Type of action ("ContinuousAction", "DiscreteAction")
        **kwargs: Additional configuration parameters
        
    Returns:
        Environment configuration
    """
    config = {
        "id": "highway-v0",
        "import_module": "highway_env",
        "config": {
            "observation": {
                "type": observation_type,
                "width": 84,
                "height": 84,
                "normalize": True
            },
            "action": {
                "type": action_type,
                "acceleration_range": [-5.0, 5.0],
                "steering_range": [-0.5, 0.5],
                "speed_range": [0, 30.0]
            },
            "lanes_count": 4,
            "vehicles_count": vehicles_count,
            "duration": 40,
            "initial_spacing": 2,
            "collision_reward": -100,
            "right_lane_reward": 1.0,
            "high_speed_reward": 0.5,
            "reward_speed_range": [20.0, 30.0],
            "normalize_reward": True,
            "offroad_terminal": True,
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "screen_width": 600,
            "screen_height": 400,
            "centering_position": [0.3, 0.5],
            "scaling": 5.5,
            "show_trajectories": True,
            "render_agent": True,
            "offscreen_rendering": False,
            "manual_control": False,
            "real_time_rendering": False
        }
    }
    
    # Update with additional parameters
    config["config"].update(kwargs)
    
    return config


def get_intersection_config(vehicles_count: int = 30,
                           observation_type: str = "GrayscaleObservation",
                           action_type: str = "ContinuousAction",
                           **kwargs) -> Dict[str, Any]:
    """
    Get configuration for intersection environment.
    
    Args:
        vehicles_count: Number of vehicles in the environment
        observation_type: Type of observation
        action_type: Type of action
        **kwargs: Additional configuration parameters
        
    Returns:
        Environment configuration
    """
    config = {
        "id": "intersection-v0",
        "import_module": "highway_env",
        "config": {
            "observation": {
                "type": observation_type,
                "width": 84,
                "height": 84,
                "normalize": True
            },
            "action": {
                "type": action_type,
                "acceleration_range": [-5.0, 5.0],
                "steering_range": [-0.5, 0.5],
                "speed_range": [0, 30.0]
            },
            "vehicles_count": vehicles_count,
            "duration": 40,
            "initial_spacing": 2,
            "collision_reward": -100,
            "high_speed_reward": 0.5,
            "reward_speed_range": [20.0, 30.0],
            "normalize_reward": True,
            "offroad_terminal": True,
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "screen_width": 600,
            "screen_height": 400,
            "centering_position": [0.5, 0.5],
            "scaling": 7,
            "show_trajectories": True,
            "render_agent": True,
            "offscreen_rendering": False,
            "manual_control": False,
            "real_time_rendering": False
        }
    }
    
    # Update with additional parameters
    config["config"].update(kwargs)
    
    return config


def get_roundabout_config(vehicles_count: int = 20,
                         observation_type: str = "GrayscaleObservation",
                         action_type: str = "ContinuousAction",
                         **kwargs) -> Dict[str, Any]:
    """
    Get configuration for roundabout environment.
    
    Args:
        vehicles_count: Number of vehicles in the environment
        observation_type: Type of observation
        action_type: Type of action
        **kwargs: Additional configuration parameters
        
    Returns:
        Environment configuration
    """
    config = {
        "id": "roundabout-v0",
        "import_module": "highway_env",
        "config": {
            "observation": {
                "type": observation_type,
                "width": 84,
                "height": 84,
                "normalize": True
            },
            "action": {
                "type": action_type,
                "acceleration_range": [-5.0, 5.0],
                "steering_range": [-0.5, 0.5],
                "speed_range": [0, 30.0]
            },
            "vehicles_count": vehicles_count,
            "duration": 40,
            "initial_spacing": 2,
            "collision_reward": -100,
            "high_speed_reward": 0.5,
            "reward_speed_range": [20.0, 30.0],
            "normalize_reward": True,
            "offroad_terminal": True,
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "screen_width": 600,
            "screen_height": 400,
            "centering_position": [0.5, 0.5],
            "scaling": 7,
            "show_trajectories": True,
            "render_agent": True,
            "offscreen_rendering": False,
            "manual_control": False,
            "real_time_rendering": False
        }
    }
    
    # Update with additional parameters
    config["config"].update(kwargs)
    
    return config


def get_parking_config(vehicles_count: int = 10,
                      observation_type: str = "GrayscaleObservation",
                      action_type: str = "ContinuousAction",
                      **kwargs) -> Dict[str, Any]:
    """
    Get configuration for parking environment.
    
    Args:
        vehicles_count: Number of vehicles in the environment
        observation_type: Type of observation
        action_type: Type of action
        **kwargs: Additional configuration parameters
        
    Returns:
        Environment configuration
    """
    config = {
        "id": "parking-v0",
        "import_module": "highway_env",
        "config": {
            "observation": {
                "type": observation_type,
                "width": 84,
                "height": 84,
                "normalize": True
            },
            "action": {
                "type": action_type,
                "acceleration_range": [-5.0, 5.0],
                "steering_range": [-0.5, 0.5],
                "speed_range": [0, 30.0]
            },
            "vehicles_count": vehicles_count,
            "duration": 60,
            "initial_spacing": 2,
            "collision_reward": -100,
            "high_speed_reward": 0.5,
            "reward_speed_range": [20.0, 30.0],
            "normalize_reward": True,
            "offroad_terminal": True,
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "screen_width": 600,
            "screen_height": 400,
            "centering_position": [0.5, 0.5],
            "scaling": 7,
            "show_trajectories": True,
            "render_agent": True,
            "offscreen_rendering": False,
            "manual_control": False,
            "real_time_rendering": False
        }
    }
    
    # Update with additional parameters
    config["config"].update(kwargs)
    
    return config


def apply_domain_randomization(config: Dict[str, Any], 
                             randomization_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply domain randomization to environment configuration.
    
    Args:
        config: Base environment configuration
        randomization_config: Domain randomization configuration
        
    Returns:
        Randomized configuration
    """
    if not randomization_config.get('enabled', True):
        return config
    
    # Traffic randomization
    if 'traffic' in randomization_config:
        traffic_config = randomization_config['traffic']
        
        # Randomize vehicle count
        if 'vehicle_count_range' in traffic_config:
            min_count, max_count = traffic_config['vehicle_count_range']
            config['config']['vehicles_count'] = np.random.randint(min_count, max_count + 1)
        
        # Randomize vehicle types
        if 'vehicle_types' in traffic_config:
            vehicle_types = traffic_config['vehicle_types']
            config['config']['other_vehicles_type'] = np.random.choice(vehicle_types)
    
    # Weather/visual randomization
    if 'weather' in randomization_config:
        weather_config = randomization_config['weather']
        
        # These will be applied in the environment wrapper
        config['config']['weather_randomization'] = weather_config
    
    # Road randomization
    if 'road' in randomization_config:
        road_config = randomization_config['road']
        
        # Randomize lane width
        if 'lane_width_range' in road_config:
            min_width, max_width = road_config['lane_width_range']
            config['config']['lane_width'] = np.random.uniform(min_width, max_width)
        
        # Randomize speed limit
        if 'speed_limit_range' in road_config:
            min_speed, max_speed = road_config['speed_limit_range']
            config['config']['speed_limit'] = np.random.uniform(min_speed, max_speed)
    
    return config


def create_environment(env_name: str, 
                      config: Optional[Dict[str, Any]] = None,
                      randomization_config: Optional[Dict[str, Any]] = None) -> gym.Env:
    """
    Create a highway-env environment with optional domain randomization.
    
    Args:
        env_name: Name of the environment
        config: Environment configuration
        randomization_config: Domain randomization configuration
        
    Returns:
        Gym environment
    """
    if config is None:
        # Use default configurations
        if env_name == "highway-v0":
            config = get_highway_config()
        elif env_name == "intersection-v0":
            config = get_intersection_config()
        elif env_name == "roundabout-v0":
            config = get_roundabout_config()
        elif env_name == "parking-v0":
            config = get_parking_config()
        else:
            raise ValueError(f"Unknown environment: {env_name}")
    
    # Apply domain randomization if specified
    if randomization_config is not None:
        config = apply_domain_randomization(config, randomization_config)
    
    # Create environment
    env = gym.make(env_name, **config.get('config', {}))
    
    return env


def get_multi_agent_config(env_name: str,
                          agent_types: List[Dict[str, Any]],
                          **kwargs) -> Dict[str, Any]:
    """
    Get configuration for multi-agent environment.
    
    Args:
        env_name: Base environment name
        agent_types: List of agent type configurations
        **kwargs: Additional configuration parameters
        
    Returns:
        Multi-agent environment configuration
    """
    # Get base configuration
    if env_name == "highway-v0":
        config = get_highway_config(**kwargs)
    elif env_name == "intersection-v0":
        config = get_intersection_config(**kwargs)
    elif env_name == "roundabout-v0":
        config = get_roundabout_config(**kwargs)
    elif env_name == "parking-v0":
        config = get_parking_config(**kwargs)
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    
    # Add multi-agent configuration
    config['config']['multi_agent'] = True
    config['config']['agent_types'] = agent_types
    
    return config


def get_reward_config(reward_shaping: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get reward configuration for environment.
    
    Args:
        reward_shaping: Reward shaping configuration
        
    Returns:
        Reward configuration
    """
    return {
        "collision_reward": reward_shaping.get("collision", -100),
        "right_lane_reward": reward_shaping.get("right_lane", 1.0),
        "high_speed_reward": reward_shaping.get("high_speed", 0.5),
        "speed_efficiency_reward": reward_shaping.get("speed_efficiency", 0.3),
        "safe_distance_reward": reward_shaping.get("safe_distance", 0.2),
        "smooth_driving_reward": reward_shaping.get("smooth_driving", 0.1),
        "lane_keeping_reward": reward_shaping.get("lane_keeping", 0.5),
        "progress_reward": reward_shaping.get("progress", 0.1),
        "fuel_efficiency_reward": reward_shaping.get("fuel_efficiency", 0.05),
        "sudden_braking_penalty": reward_shaping.get("sudden_braking", -0.5),
        "sudden_acceleration_penalty": reward_shaping.get("sudden_acceleration", -0.3),
        "lane_violation_penalty": reward_shaping.get("lane_violation", -1.0),
        "off_road_penalty": reward_shaping.get("off_road", -50)
    }


def get_observation_config(observation_type: str = "GrayscaleObservation",
                          width: int = 84,
                          height: int = 84,
                          normalize: bool = True) -> Dict[str, Any]:
    """
    Get observation configuration.
    
    Args:
        observation_type: Type of observation
        width: Observation width
        height: Observation height
        normalize: Whether to normalize observations
        
    Returns:
        Observation configuration
    """
    return {
        "type": observation_type,
        "width": width,
        "height": height,
        "normalize": normalize
    }


def get_action_config(action_type: str = "ContinuousAction",
                     acceleration_range: List[float] = [-5.0, 5.0],
                     steering_range: List[float] = [-0.5, 0.5],
                     speed_range: List[float] = [0, 30.0]) -> Dict[str, Any]:
    """
    Get action configuration.
    
    Args:
        action_type: Type of action
        acceleration_range: Acceleration range
        steering_range: Steering range
        speed_range: Speed range
        
    Returns:
        Action configuration
    """
    return {
        "type": action_type,
        "acceleration_range": acceleration_range,
        "steering_range": steering_range,
        "speed_range": speed_range
    }