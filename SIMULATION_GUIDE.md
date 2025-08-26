# Simulation Viewer Guide

This guide explains how to view final simulations for all supported autonomous driving scenarios in this repository.

## 🎯 Supported Scenarios

The repository supports four main driving scenarios:

1. **Highway Driving** (`highway-v0`) - Multi-lane highway navigation with traffic
2. **Intersection Handling** (`intersection-v0`) - Traffic light and stop sign navigation
3. **Roundabout Navigation** (`roundabout-v0`) - Circular intersection navigation
4. **Parking Maneuvers** (`parking-v0`) - Parallel and perpendicular parking

## 🚀 Quick Start

### Option 1: Quick Demonstrations (No Trained Models Required)

Use the `quick_simulation.py` script to see demonstrations of all scenarios with simple rule-based agents:

```bash
# View a single scenario
python quick_simulation.py --scenario highway --render

# View all scenarios
python quick_simulation.py --scenario all --render --save_videos

# Interactive mode
python quick_simulation.py --interactive
```

### Option 2: Full Simulation Viewer (With Trained Models)

Use the `simulation_viewer.py` script to view simulations with trained agents:

```bash
# View with trained RL agent
python simulation_viewer.py --scenario highway --agent_type rl --agent_path ./models/trained_rl_model --render

# View all scenarios with hybrid agent
python simulation_viewer.py --scenario all --agent_type hybrid --agent_path ./models/trained_hybrid_model --save_videos

# Interactive mode
python simulation_viewer.py --interactive --agent_type rl --agent_path ./models/trained_rl_model
```

## 📋 Available Scripts

### 1. `quick_simulation.py`

**Purpose**: Quick demonstrations without requiring trained models
**Best for**: Getting familiar with the environments and scenarios

**Features**:
- ✅ No trained models required
- ✅ Simple rule-based agent demonstrations
- ✅ All four scenarios supported
- ✅ Interactive mode available
- ✅ Video recording capability
- ✅ Real-time rendering

**Usage Examples**:
```bash
# Basic highway demonstration
python quick_simulation.py --scenario highway --render

# Save video of intersection demo
python quick_simulation.py --scenario intersection --render --save_videos

# Run all scenarios and save videos
python quick_simulation.py --scenario all --render --save_videos

# Interactive mode
python quick_simulation.py --interactive
```

### 2. `simulation_viewer.py`

**Purpose**: View simulations with trained agents (IL, RL, or Hybrid)
**Best for**: Evaluating trained models and seeing final performance

**Features**:
- ✅ Support for trained IL, RL, and Hybrid agents
- ✅ All four scenarios supported
- ✅ Comprehensive metrics and evaluation
- ✅ Video recording with performance overlay
- ✅ Interactive mode available
- ✅ Batch processing for all scenarios

**Usage Examples**:
```bash
# View with trained RL agent
python simulation_viewer.py --scenario highway --agent_type rl --agent_path ./models/rl_model --render

# View with trained IL agent
python simulation_viewer.py --scenario intersection --agent_type il --agent_path ./models/il_model --render

# View with hybrid agent
python simulation_viewer.py --scenario roundabout --agent_type hybrid --agent_path ./models/hybrid_model --render

# Run all scenarios with RL agent
python simulation_viewer.py --scenario all --agent_type rl --agent_path ./models/rl_model --save_videos

# Interactive mode with trained agent
python simulation_viewer.py --interactive --agent_type rl --agent_path ./models/rl_model
```

## 🎮 Interactive Controls

When running simulations with rendering enabled, you can use these controls:

- **`q`** - Quit the simulation
- **`r`** - Reset the simulation (start over)
- **Mouse** - Click and drag to adjust camera view (if supported)

## 📊 Understanding Results

### Performance Metrics

Both scripts provide comprehensive performance metrics:

- **Total Reward**: Cumulative reward for the episode
- **Average Reward**: Mean reward per step
- **Steps**: Number of steps taken
- **Success**: Whether the episode completed successfully
- **Collisions**: Number of vehicle collisions
- **Off-road**: Number of times vehicle left the road
- **Lane Violations**: Number of lane boundary violations
- **Speed Violations**: Number of speed limit violations

### Success Criteria

An episode is considered successful if:
- No collisions occurred
- Vehicle stayed on the road
- Episode completed without early termination

## 🎥 Video Recording

Both scripts support video recording:

```bash
# Save videos to default directory
python quick_simulation.py --scenario all --save_videos

# Save videos to custom directory
python simulation_viewer.py --scenario highway --agent_type rl --agent_path ./models/rl_model --save_videos --output_dir ./my_videos
```

**Video Output**:
- Format: MP4
- Frame Rate: 20 FPS
- Quality: Same as rendered display
- Naming: `{scenario}_{agent_type}_simulation.mp4`

## 🔧 Configuration

### Environment Configuration

You can customize environment settings by modifying `configs/training_config.yaml`:

```yaml
environment:
  name: "highway-v0"  # Default scenario
  config:
    vehicles:
      count: 50        # Number of vehicles
      spacing: 2.0     # Vehicle spacing
    simulation:
      duration: 40     # Episode duration
      frequency: 15    # Simulation frequency
```

### Agent Configuration

For trained agents, ensure your model files are in the correct format:

- **IL Agent**: Single model file (e.g., `il_model.pth`)
- **RL Agent**: Single model file (e.g., `rl_model.zip`)
- **Hybrid Agent**: Two model files (e.g., `hybrid_model_il.pth`, `hybrid_model_rl.zip`)

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   # Ensure you're in the project root directory
   cd /path/to/autonomous-driving-agent
   python quick_simulation.py --scenario highway --render
   ```

2. **Missing Dependencies**:
   ```bash
   # Install required packages
   pip install -r requirements.txt
   ```

3. **Model Loading Errors**:
   ```bash
   # Use quick simulation if you don't have trained models
   python quick_simulation.py --scenario highway --render
   ```

4. **Rendering Issues**:
   ```bash
   # Try without rendering first
   python quick_simulation.py --scenario highway
   
   # Or use headless mode
   export DISPLAY=:0
   python quick_simulation.py --scenario highway --render
   ```

### Performance Tips

1. **For Better Performance**:
   - Use `--max_steps 200` for shorter demonstrations
   - Disable rendering with `--no-render` for faster execution
   - Use CPU-only mode if GPU is not available

2. **For Better Quality**:
   - Increase `max_steps` for longer demonstrations
   - Enable video recording with `--save_videos`
   - Use trained models for realistic behavior

## 📈 Advanced Usage

### Batch Processing

Run multiple scenarios with different configurations:

```bash
# Create a batch script
#!/bin/bash
for scenario in highway intersection roundabout parking; do
    echo "Running $scenario simulation..."
    python simulation_viewer.py --scenario $scenario --agent_type rl --agent_path ./models/rl_model --save_videos
done
```

### Custom Agent Integration

You can integrate your own agents by modifying the agent loading code in `simulation_viewer.py`:

```python
def load_custom_agent(self, agent_path: str):
    """Load your custom agent implementation."""
    # Your custom agent loading code here
    pass
```

### Metrics Analysis

Analyze simulation results programmatically:

```python
from simulation_viewer import SimulationViewer

viewer = SimulationViewer()
results = viewer.run_all_scenarios("rl", "./models/rl_model", save_videos=True)

# Analyze results
for scenario, result in results.items():
    print(f"{scenario}: Success Rate = {result['success']}")
```

## 🎯 Next Steps

After viewing the simulations:

1. **Train Your Own Agents**:
   ```bash
   python scripts/train_il.py --env highway-v0 --episodes 1000
   python scripts/train_rl.py --env highway-v0 --pretrained_il ./models/il_model
   ```

2. **Evaluate Trained Models**:
   ```bash
   python scripts/evaluate.py --agent_type rl --agent_path ./models/rl_model --env_name highway-v0
   ```

3. **Customize Environments**:
   - Modify `environments/env_configs.py` for custom scenarios
   - Adjust reward functions in `configs/training_config.yaml`
   - Add new environment wrappers in `environments/wrappers.py`

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the main README.md for setup instructions
3. Check the requirements.txt for dependency versions
4. Ensure you're using the correct Python environment

Happy simulating! 🚗💨