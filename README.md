# T4B Gym Documentation

## Overview

T4B Gym is a reinforcement learning environment for building control systems, built on top of the Twin4Build framework. It provides a gymnasium-compatible interface for training RL agents to control HVAC systems in multi-zone buildings.

## Table of Contents

- [Installation](#installation)
- [Core Components](#core-components)
  - [GymSimulator](#gymsimulator)
  - [T4BGymEnv](#t4bgymenv)
  - [Wrappers](#wrappers)
- [Model Components](#model-components)
- [Use Cases](#use-cases)
- [Configuration](#configuration)
- [Examples](#examples)

---

## Installation

### Prerequisites

Ensure you have the following dependencies installed:

```bash
# Core dependencies
pip install gymnasium==1.1.1
pip install stable_baselines3==2.6.0
pip install torch==2.7.0+cu118
pip install numpy==1.23.5
pip install pandas==1.5.3
```

### Environment Setup

1. Create and activate a Conda environment:
```bash
conda env create -f environment.yml
conda activate t4b_env
```

2. Install additional requirements:
```bash
pip install -r requirements.txt
```

3. Install Twin4Build (prerequisite):
   Follow instructions at: [Twin4Build Repository](https://github.com/JBjoernskov/Twin4Build)

---

## Core Components

### GymSimulator

The `GymSimulator` class extends the Twin4Build `Simulator` class to provide gymnasium-compatible functionality.

#### Constructor

```python
class GymSimulator(tb.Simulator):
    def __init__(self, model, enable_logging: bool = False)
```

**Parameters:**
- `model`: A Twin4Build model instance
- `enable_logging`: Whether to enable debug logging (default: False)

#### Key Methods

##### `initialize_simulation(startTime, endTime, stepSize)`

Initialize the simulation parameters and model.

```python
def initialize_simulation(self, startTime: datetime, endTime: datetime, stepSize: int) -> None
```

**Parameters:**
- `startTime` (datetime): Start time with timezone info
- `endTime` (datetime): End time with timezone info  
- `stepSize` (int): Step size in seconds

**Example:**
```python
from datetime import datetime
from dateutil.tz import gettz

simulator = GymSimulator(model, enable_logging=True)
start_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=gettz("Europe/Copenhagen"))
end_time = datetime(2024, 1, 15, 0, 0, 0, tzinfo=gettz("Europe/Copenhagen"))
simulator.initialize_simulation(start_time, end_time, 600)
```

##### `add_control_input(component_id, input_name)`

Add a control input to the simulator.

```python
def add_control_input(self, component_id: str, input_name: str) -> None
```

**Parameters:**
- `component_id`: ID of the component to control
- `input_name`: Name of the input parameter to control

**Example:**
```python
simulator.add_control_input("supply_air_temp_setpoint_sensor", "measuredValue")
```

##### `add_observation_output(component_id, output_name)`

Add an observation output to monitor.

```python
def add_observation_output(self, component_id: str, output_name: str) -> None
```

**Parameters:**
- `component_id`: ID of the component to observe
- `output_name`: Name of the output parameter to monitor

**Example:**
```python
simulator.add_observation_output("core_indoor_temp_sensor", "measuredValue")
```

##### `step_simulation(actions)`

Perform one simulation step with given actions.

```python
def step_simulation(self, actions: Optional[np.ndarray] = None) -> bool
```

**Parameters:**
- `actions`: Control actions as numpy array

**Returns:**
- `bool`: True if simulation is complete

**Example:**
```python
actions = np.array([22.0, 0.7, 0.3])  # Example actions
done = simulator.step_simulation(actions)
```

##### `get_observations()`

Get current observations from monitored outputs.

```python
def get_observations(self) -> Dict[str, Dict[str, float]]
```

**Returns:**
- Dictionary mapping component IDs to their output values

**Example:**
```python
observations = simulator.get_observations()
# Returns: {"core_indoor_temp_sensor": {"measuredValue": 22.5}}
```

---

### T4BGymEnv

The main gymnasium environment class for building control.

#### Constructor

```python
class T4BGymEnv(gym.Env):
    def __init__(self, 
                 model: tb.Model, 
                 io_config_file: str,
                 start_time: datetime = None,
                 end_time: datetime = None,
                 episode_length: int = None,
                 random_start: bool = False,
                 excluding_periods: List[Tuple[datetime, datetime]] = None,
                 forecast_horizon: int = 0,
                 step_size: int = 600,
                 warmup_period: int = 0)
```

**Parameters:**
- `model`: Twin4Build model instance
- `io_config_file`: Path to JSON file defining actions and observations
- `start_time`: Simulation start time (with timezone)
- `end_time`: Simulation end time (with timezone)
- `episode_length`: Episode length in steps
- `random_start`: Whether to randomize episode start times
- `excluding_periods`: List of (start, end) datetime tuples to exclude
- `forecast_horizon`: Number of forecast steps to include in observations
- `step_size`: Simulation step size in seconds
- `warmup_period`: Number of warmup steps (not implemented)

#### Key Methods

##### `reset(seed, options)`

Reset the environment to initial state.

```python
def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]
```

**Returns:**
- Tuple of (initial_observations, info_dict)

**Example:**
```python
env = T4BGymEnv(model, "config.json", start_time, end_time)
observations, info = env.reset()
```

##### `step(action)`

Take a step in the environment.

```python
def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]
```

**Parameters:**
- `action`: Control actions as numpy array

**Returns:**
- Tuple of (observation, reward, terminated, truncated, info)

**Example:**
```python
action = np.array([22.0, 0.7, 0.3])
obs, reward, terminated, truncated, info = env.step(action)
```

##### `get_reward(observations, action)`

Calculate reward (should be overridden in subclasses).

```python
def get_reward(self, observations, action) -> float
```

**Parameters:**
- `observations`: Current state observations
- `action`: Control actions applied

**Returns:**
- `float`: Reward value

**Example Implementation:**
```python
class CustomRewardEnv(T4BGymEnv):
    def get_reward(self, observations, action):
        # Temperature violation penalty
        temp = self.simulator.model.components["temp_sensor"].output["measuredValue"]
        setpoint = self.simulator.model.components["setpoint"].output["scheduleValue"]
        temp_penalty = abs(temp - setpoint)
        
        # Energy consumption penalty
        power = self.simulator.model.components["heater"].output["Power"]
        
        return -(temp_penalty * 10 + power * 0.01)
```

---

### Wrappers

#### NormalizedObservationWrapper

Normalizes observations to the range [-1, 1].

```python
class NormalizedObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env)
```

**Example:**
```python
env = T4BGymEnv(model, config_file, start_time, end_time)
env = NormalizedObservationWrapper(env)
```

#### NormalizedActionWrapper

Normalizes actions to the range [-1, 1].

```python
class NormalizedActionWrapper(gym.ActionWrapper):
    def __init__(self, env)
```

**Example:**
```python
env = T4BGymEnv(model, config_file, start_time, end_time)
env = NormalizedActionWrapper(env)
```

---

## Model Components

### Building Model Functions

#### `load_model_and_params(model_id)`

Load a pre-configured building model with parameters.

```python
def load_model_and_params(model_id="rooms_and_ahu_model") -> tb.Model
```

**Parameters:**
- `model_id`: Model identifier (default: "rooms_and_ahu_model")

**Returns:**
- Configured Twin4Build model

**Example:**
```python
from boptest_model.rooms_and_ahu_model import load_model_and_params

model = load_model_and_params()
```

#### Model Functions

The building model is constructed using three main functions:

##### `envelope_fcn(self)`

Adds building envelope components (rooms, sensors, outdoor environment).

**Components Added:**
- Building spaces (core, north, south, east, west)
- Temperature and CO2 sensors
- Occupancy profiles
- Outdoor environment connections

##### `vavs_fcn(self)` 

Adds VAV (Variable Air Volume) system components.

**Components Added:**
- Temperature setpoint schedules
- VAV reheat controllers
- Reheat coils
- Supply dampers
- Sensors for control feedback

##### `ahu_fcn(self)`

Adds Air Handling Unit (AHU) components.

**Components Added:**
- Supply and return fans
- Heating and cooling coils
- Dampers for air mixing
- Temperature and flow sensors

---

## Configuration

### IO Configuration File

The IO configuration file defines the action and observation spaces using JSON format:

```json
{
    "actions": {
        "component_id": {
            "signal_name": {
                "signal_key": "input_parameter_name",
                "min": min_value,
                "max": max_value,
                "description": "Description of the action"
            }
        }
    },
    "observations": {
        "component_id": {
            "signal_name": {
                "signal_key": "output_parameter_name", 
                "min": min_value,
                "max": max_value,
                "description": "Description of the observation"
            }
        }
    },
    "time_embeddings": {
        "time_of_day": {
            "signal_key": "timeOfDay",
            "min": 0,
            "max": 24,
            "description": "Time of day in hours"
        }
    },
    "forecasts": {
        "outdoor_temperature": {
            "outdoorTemperature": {
                "signal_key": "outdoorTemperature",
                "min": -20,
                "max": 50,
                "description": "Outdoor temperature forecast"
            }
        }
    }
}
```

#### Configuration Sections

1. **Actions**: Define controllable inputs
2. **Observations**: Define observable outputs  
3. **Time Embeddings**: Add temporal features (sin/cos encoded)
4. **Forecasts**: Include weather and schedule forecasts

---

## Use Cases

### Model Evaluation

The `model_eval.py` module provides functions for testing and evaluating trained models.

#### `test_model(env, model)`

Test a trained model in the environment.

```python
def test_model(env, model) -> Tuple[List, List]
```

**Parameters:**
- `env`: T4BGymEnv environment instance
- `model`: Trained RL model

**Returns:**
- Tuple of (observations_list, rewards_list)

**Example:**
```python
from use_case.model_eval import test_model
from stable_baselines3 import PPO

# Load trained model
model = PPO.load("path/to/model.zip")

# Test model
observations, rewards = test_model(env, model)
```

#### `plot_results(simulator, rewards, save_plots)`

Generate plots of simulation results.

```python
def plot_results(simulator: tb.Simulator, 
                rewards=None, 
                plotting_stepSize=600, 
                save_plots=False)
```

**Parameters:**
- `simulator`: GymSimulator instance
- `rewards`: List of rewards (optional)
- `plotting_stepSize`: Time step for plotting
- `save_plots`: Whether to save plots to files

**Example:**
```python
from use_case.model_eval import plot_results

plot_results(env.simulator, rewards, save_plots=True)
```

### Training Examples

#### PPO Training

```python
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

# Create environment
env = T4BGymEnv(model, config_file, start_time, end_time)
env = NormalizedObservationWrapper(env)
env = NormalizedActionWrapper(env)
env = Monitor(env, filename="monitor.csv")

# Create PPO model
model = PPO('MlpPolicy', env, 
           verbose=1, 
           gamma=0.99,
           learning_rate=5e-4, 
           batch_size=50, 
           n_steps=200)

# Create evaluation callback
callback = EvalCallback(env, 
                       best_model_save_path="./logs/", 
                       eval_freq=1000, 
                       n_eval_episodes=5)

# Train model
model.learn(total_timesteps=100000, callback=callback)
```

#### Custom Reward Function

```python
class CustomRewardEnv(T4BGymEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.previous_objective = 0.0
        
    def get_reward(self, observations, action):
        # Get temperature measurements
        temp = self.simulator.model.components["temp_sensor"].output["measuredValue"]
        setpoint = self.simulator.model.components["setpoint"].output["scheduleValue"]
        
        # Calculate violations
        temp_violation = max(0, abs(temp - setpoint) - 1.0)  # 1°C deadband
        
        # Get power consumption
        power = self.simulator.model.components["heater"].output["Power"]
        
        # Combined objective
        current_objective = temp_violation * 1000 + power * 0.01
        
        # Reward as negative change in objective
        reward = -(current_objective - self.previous_objective)
        self.previous_objective = current_objective
        
        return reward
```

---

## Examples

### Basic Usage

```python
import twin4build as tb
import datetime
from dateutil.tz import gettz
from t4b_gym.t4b_gym_env import T4BGymEnv
from boptest_model.rooms_and_ahu_model import load_model_and_params

# Load model
model = load_model_and_params()

# Define time period
start_time = datetime.datetime(2024, 1, 1, tzinfo=gettz("Europe/Copenhagen"))
end_time = datetime.datetime(2024, 1, 15, tzinfo=gettz("Europe/Copenhagen"))

# Create environment
env = T4BGymEnv(
    model=model,
    io_config_file="policy_input_output.json",
    start_time=start_time,
    end_time=end_time,
    episode_length=2160,  # 15 days at 10-minute steps
    step_size=600,
    forecast_horizon=10
)

# Reset environment
obs, info = env.reset()

# Take steps
for i in range(100):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()
```

### Advanced Usage with Forecasts

```python
# Configuration with forecasts and time embeddings
config = {
    "actions": {
        "supply_air_temp_setpoint_sensor": {
            "measuredValue": {"signal_key": "measuredValue", "min": 12, "max": 35}
        }
    },
    "observations": {
        "core_indoor_temp_sensor": {
            "measuredValue": {"signal_key": "measuredValue", "min": 0, "max": 40}
        }
    },
    "time_embeddings": {
        "time_of_day": {"signal_key": "timeOfDay", "min": 0, "max": 24},
        "day_of_week": {"signal_key": "dayOfWeek", "min": 0, "max": 7}
    },
    "forecasts": {
        "outdoor_temperature": {
            "outdoorTemperature": {"signal_key": "outdoorTemperature", "min": -20, "max": 50}
        }
    }
}

# Save config and create environment
import json
with open("advanced_config.json", "w") as f:
    json.dump(config, f)

env = T4BGymEnv(
    model=model,
    io_config_file="advanced_config.json",
    start_time=start_time,
    end_time=end_time,
    forecast_horizon=24,  # 24-step forecast
    step_size=600
)
```

### Training with Stable Baselines3

```python
from stable_baselines3 import PPO
from t4b_gym.t4b_gym_env import NormalizedObservationWrapper, NormalizedActionWrapper

# Create and wrap environment
env = T4BGymEnv(model, "config.json", start_time, end_time)
env = NormalizedObservationWrapper(env)
env = NormalizedActionWrapper(env)

# Create and train model
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=50000)

# Save model
model.save("trained_model")

# Load and test
model = PPO.load("trained_model")
obs, info = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

---

## Error Handling

### Common Issues

1. **Missing Timezone Information**
   ```python
   # ❌ Wrong
   start_time = datetime.datetime(2024, 1, 1)
   
   # ✅ Correct
   start_time = datetime.datetime(2024, 1, 1, tzinfo=gettz("Europe/Copenhagen"))
   ```

2. **Invalid Action Values**
   ```python
   # Actions are automatically clipped to action space bounds
   # But NaN values will be converted to 0
   action = np.array([np.nan, 22.0])  # First value becomes 0
   ```

3. **Configuration File Errors**
   ```python
   # Ensure component IDs exist in the model
   # Ensure signal keys match component input/output names
   ```

### Debugging

Enable logging for detailed debugging:

```python
simulator = GymSimulator(model, enable_logging=True)
```

This will log:
- Component connections
- Control input applications  
- Observation recordings
- Simulation progress

---

## Performance Considerations

### Memory Usage

- Long episodes with many observations can consume significant memory
- Consider shorter episodes for training
- Use `forecast_horizon` judiciously

### Computational Performance

- Larger `step_size` values reduce computational load
- FMU components are more computationally expensive
- Consider parallel environments for training

### Recommended Settings

For training:
```python
env = T4BGymEnv(
    model=model,
    io_config_file="config.json", 
    episode_length=1440,  # 10 days at 10-minute steps
    step_size=600,        # 10 minutes
    forecast_horizon=24,  # 4 hours ahead
    random_start=True
)
```

For evaluation:
```python
env = T4BGymEnv(
    model=model,
    io_config_file="config.json",
    start_time=fixed_start_time,
    end_time=fixed_end_time,
    step_size=600,
    forecast_horizon=24,
    random_start=False
)
```
