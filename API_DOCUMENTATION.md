# T4B Gym - Quick Reference

A reinforcement learning environment for HVAC building control using Twin4Build.

## Installation

```bash
conda env create -f environment.yml
conda activate t4b_env
pip install -r requirements.txt
```

## Quick Start

```python
import twin4build as tb
from t4b_gym.t4b_gym_env import T4BGymEnv
from boptest_model.rooms_and_ahu_model import load_model_and_params

# Load model and create environment
model = load_model_and_params()
env = T4BGymEnv(
    model=model,
    io_config_file="policy_input_output.json",
    start_time=datetime(2024, 1, 1, tzinfo=gettz("Europe/Copenhagen")),
    end_time=datetime(2024, 1, 15, tzinfo=gettz("Europe/Copenhagen")),
    episode_length=2160,
    step_size=600
)

# Use environment
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

## Core Classes

### T4BGymEnv
Main gymnasium environment for building control.

**Key Parameters:**
- `model`: Twin4Build model
- `io_config_file`: JSON config defining actions/observations
- `start_time`/`end_time`: Simulation period (with timezone)
- `episode_length`: Steps per episode
- `step_size`: Simulation step in seconds
- `forecast_horizon`: Future steps to include in observations

### GymSimulator
Extends Twin4Build Simulator with gymnasium compatibility.

**Key Methods:**
- `initialize_simulation(start_time, end_time, step_size)`
- `step_simulation(actions)` - Apply actions and advance simulation
- `get_observations()` - Get current state observations

## Configuration

IO config file (`policy_input_output.json`):
```json
{
    "actions": {
        "component_id": {
            "signal_name": {
                "signal_key": "input_parameter",
                "min": min_value,
                "max": max_value
            }
        }
    },
    "observations": {
        "component_id": {
            "signal_name": {
                "signal_key": "output_parameter",
                "min": min_value,
                "max": max_value
            }
        }
    },
    "time_embeddings": {
        "time_of_day": {"signal_key": "timeOfDay", "min": 0, "max": 24}
    },
    "forecasts": {
        "outdoor_temperature": {
            "outdoorTemperature": {"signal_key": "outdoorTemperature", "min": -20, "max": 50}
        }
    }
}
```

## Training Example

```python
from stable_baselines3 import PPO
from t4b_gym.t4b_gym_env import NormalizedObservationWrapper, NormalizedActionWrapper

# Create and wrap environment
env = T4BGymEnv(model, "config.json", start_time, end_time)
env = NormalizedObservationWrapper(env)
env = NormalizedActionWrapper(env)

# Train
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=50000)
model.save("trained_model")
```

## Custom Reward Function

```python
class CustomRewardEnv(T4BGymEnv):
    def get_reward(self, observations, action):
        temp = self.simulator.model.components["temp_sensor"].output["measuredValue"]
        setpoint = self.simulator.model.components["setpoint"].output["scheduleValue"]
        power = self.simulator.model.components["heater"].output["Power"]
        
        temp_penalty = abs(temp - setpoint)
        return -(temp_penalty * 10 + power * 0.01)
```

## Model Evaluation

```python
from use_case.model_eval import test_model, plot_results

# Test trained model
observations, rewards = test_model(env, model)

# Plot results
plot_results(env.simulator, rewards, save_plots=True)
```

## Common Issues

- **Timezone required**: Always include timezone in datetime objects
- **Component IDs**: Ensure they exist in the model
- **Action bounds**: Values are automatically clipped to min/max ranges
- **Debugging**: Use `GymSimulator(model, enable_logging=True)` for detailed logs

## Performance Tips

- Use `step_size=600` (10 minutes) for good balance of speed/accuracy
- Keep `episode_length` reasonable (1440 steps = 10 days)
- Enable `random_start=True` for training diversity
- Use wrappers for normalization in training
