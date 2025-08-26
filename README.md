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
├── scripts/               # Main execution scripts
│   ├── train_il.py        # Train imitation learning agent
│   ├── train_rl.py        # Train reinforcement learning agent
│   └── evaluate.py        # Evaluate trained agents
├── simulation_viewer.py   # Full simulation viewer with trained agents
├── quick_simulation.py    # Quick demonstrations without trained models
├── test_simulation.py     # Test script for simulation capabilities
└── SIMULATION_GUIDE.md    # Comprehensive simulation usage guide
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

## 📹 Simulation Viewing

### Quick Demonstrations (No Trained Models Required)

View simulations of all scenarios with simple rule-based agents:

```bash
# Test simulation capabilities
python test_simulation.py

# View a single scenario
python quick_simulation.py --scenario highway --render

# View all scenarios
python quick_simulation.py --scenario all --render --save_videos

# Interactive mode
python quick_simulation.py --interactive
```

### Full Simulation Viewer (With Trained Models)

View simulations with trained agents:

```bash
# View with trained RL agent
python simulation_viewer.py --scenario highway --agent_type rl --agent_path ./models/rl_model --render

# View all scenarios with hybrid agent
python simulation_viewer.py --scenario all --agent_type hybrid --agent_path ./models/hybrid_model --save_videos

# Interactive mode
python simulation_viewer.py --interactive --agent_type rl --agent_path ./models/rl_model
```

### Supported Scenarios

The agent will be evaluated on:
- 🛣️ **Highway Driving** - Multi-lane highway navigation with traffic
- 🚦 **Intersection Handling** - Traffic light and stop sign navigation  
- 🔁 **Roundabout Navigation** - Circular intersection navigation
- 🅿️ **Parking Maneuvers** - Parallel and perpendicular parking

For detailed usage instructions, see [SIMULATION_GUIDE.md](SIMULATION_GUIDE.md).

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