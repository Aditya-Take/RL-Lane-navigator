# Vision-Based Autonomous Driving Agent

A comprehensive autonomous driving agent using Imitation Learning and Reinforcement Learning (PPO) in highway-env with domain randomization and multi-agent scenarios.

## 🎯 Objectives

- ✅ Develop a robust autonomous driving agent capable of handling multiple real-world traffic scenarios
- ✅ Leverage Imitation Learning to bootstrap the model using expert driving behavior
- ✅ Apply Reinforcement Learning (PPO) to refine strategies in dynamic environments
- ✅ Utilize vision-based inputs for perception and decision-making
- ✅ Perform domain randomization to enhance generalization
- ✅ Integrate multi-agent traffic scenarios for real-world complexity

## 🚗 Key Features

### Phase 1: Imitation Learning (CNN-based)
- Use highway-env's scripted expert or manual control
- Capture rendered frames and corresponding actions
- Train a Convolutional Neural Network to map visual input → action output
- Learn basic behaviors like lane following, turning, and parking

### Phase 2: Reinforcement Learning (PPO)
- Initialize PPO agent with optional pretrained CNN weights from IL
- Use kinematic state vectors or rendered frames as input
- Apply custom reward shaping for optimal behavior
- Train across diverse traffic densities and road geometries

### Multi-Agent Simulation
- Spawn multiple agents with stochastic behavior
- Implement multi-agent complexity through different vehicle types
- Navigate realistic traffic and avoid collisions

## 🛣️ Supported Scenarios

- 🚘 Lane following (highway-v0)
- 🚦 Intersection handling (intersection-v0)
- 🔁 Roundabout navigation (roundabout-v0)
- 🅿️ Parking maneuvers (parking-v0)

## 🧰 Technologies & Tools

- **Simulator**: highway-env (2D lightweight driving simulator)
- **Language**: Python
- **Libraries**:
  - PyTorch (for CNN)
  - Stable-Baselines3 (PPO)
  - Imitation (for Behavioral Cloning)
  - OpenCV (image preprocessing)
  - Gymnasium for environment handling

## 📁 Project Structure

```
├── agents/                 # Agent implementations
│   ├── cnn_agent.py       # CNN-based imitation learning agent
│   ├── ppo_agent.py       # PPO reinforcement learning agent
│   └── hybrid_agent.py    # Combined IL+RL agent
├── environments/           # Environment configurations and wrappers
│   ├── env_configs.py     # Environment configurations
│   ├── wrappers.py        # Custom environment wrappers
│   └── domain_randomization.py  # Domain randomization utilities
├── models/                # Neural network architectures
│   ├── cnn_models.py      # CNN architectures for vision
│   └── policy_networks.py # Policy networks for RL
├── training/              # Training scripts and utilities
│   ├── imitation_learning.py  # IL training pipeline
│   ├── reinforcement_learning.py  # RL training pipeline
│   └── data_collection.py # Expert demonstration collection
├── evaluation/            # Evaluation and testing
│   ├── evaluate_agent.py  # Agent evaluation scripts
│   └── metrics.py         # Performance metrics
├── utils/                 # Utility functions
│   ├── visualization.py   # Visualization tools
│   └── data_utils.py      # Data processing utilities
├── configs/               # Configuration files
│   └── training_config.yaml  # Training hyperparameters
└── scripts/               # Main execution scripts
    ├── train_il.py        # Train imitation learning agent
    ├── train_rl.py        # Train reinforcement learning agent
    └── evaluate.py        # Evaluate trained agents
```

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train Imitation Learning Agent**:
   ```bash
   python scripts/train_il.py --env highway-v0 --episodes 1000
   ```

3. **Train Reinforcement Learning Agent**:
   ```bash
   python scripts/train_rl.py --env highway-v0 --pretrained_il path/to/il_model
   ```

4. **Evaluate Agent**:
   ```bash
   python scripts/evaluate.py --agent path/to/trained_agent --env highway-v0
   ```

## 📈 Expected Outcomes

A vision-based autonomous agent capable of:
- ✅ Navigating intersections, roundabouts, and parking
- ✅ Avoiding dynamic traffic agents
- ✅ Making real-time decisions under uncertainty
- ✅ Generalizing across varied conditions
- ✅ Achieving superior performance with IL+RL compared to IL-only

## 📹 Demonstration

The agent will be evaluated on:
- Lane keeping with random traffic
- Intersection negotiation
- Roundabout entry/exit with multiple agents
- Parking in constrained space

## 🔧 Configuration

Edit `configs/training_config.yaml` to customize:
- Training hyperparameters
- Environment settings
- Reward shaping
- Domain randomization parameters
- Multi-agent configurations

## 📊 Evaluation Metrics

- Episode reward
- Number of collisions
- Average time per episode
- Lane discipline consistency
- Success rate across different scenarios