# Repository Update Notes

## Updated Dependencies (Latest Compatible Versions)

The repository has been successfully updated to use the latest compatible versions of all dependencies:

### Updated Packages:
- **highway-env**: 1.8.0+ (latest: 1.10.1 compatible)
- **gymnasium**: 0.29.0+ (latest: 1.2.0 compatible)  
- **stable-baselines3**: 2.3.0+ (latest: 2.7.0)
- **torch**: 2.0.0+ (latest: 2.8.0)
- **torchvision**: 0.15.0+ (latest: 0.23.0)
- **numpy**: 1.24.0 to <2.3.0 (for opencv compatibility)
- **opencv-python**: 4.8.0+ (latest: 4.12.0.88)
- **matplotlib**: 3.7.0+ (latest: 3.10.5)
- **tensorboard**: 2.13.0+ (latest: 2.20.0)
- **scikit-learn**: 1.3.0+ (latest: 1.7.1)
- **tqdm**: 4.65.0+ (latest: 4.67.1)
- **pillow**: 10.0.0+ (latest: 11.3.0)
- **seaborn**: 0.12.0+ (latest: 0.13.2)
- **wandb**: 0.15.0+ (latest: 0.21.1)

### Code Updates:

1. **Modern Python syntax**: Updated `super()` calls from old-style `super(Class, self)` to modern `super()`
2. **Highway-env import fix**: Changed `from highway_env import make` to `import highway_env` and use `gym.make()`
3. **Highway-env API compatibility**: Updated environment configuration to use the new highway-env v1.10+ API:
   - Changed from passing config parameters to `gym.make()` to using `env.configure()` method
   - Updated `GrayscaleObservation` configuration with required parameters: `observation_shape`, `stack_size`, `weights`
   - Fixed environment initialization flow to configure after creation but before reset
4. **Security improvements**: Added `weights_only=True` parameter to `torch.load()` calls for security
5. **Python version requirement**: Updated setup.py to require Python 3.9+ (was 3.8+)

### Known Issue:

**Imitation Package**: The `imitation` package was temporarily removed from requirements due to dependency conflicts with newer gymnasium versions. The imitation package (v1.0.1) requires exactly `gymnasium~=0.29` while other packages like highway-env require `gymnasium>=1.0.0a2`. 

**Workaround**: The codebase is designed to work without the imitation package for core functionality. If imitation learning features are needed, users can:
1. Install imitation separately: `pip install imitation==1.0.1`
2. Or wait for a future imitation package update that supports newer gymnasium versions

### Compatibility Notes:

- All core functionality (PPO training, CNN models, environment wrappers) works with updated versions
- The codebase uses modern gymnasium API (5-tuple step return, 2-tuple reset return)
- PyTorch 2.8.0 with CUDA 12.8 support included
- Compatible with Python 3.9 through 3.13

### Installation:

```bash
pip install -r requirements.txt
python setup.py  # Run setup script for validation
```

All modules now import successfully and the codebase is ready for use with the latest package versions.