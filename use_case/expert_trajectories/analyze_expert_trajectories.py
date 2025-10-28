"""
This script loads and analyzes the expert_trajectories.npz file, plotting each action dimension over time in a separate plot, using action names from the environment if available.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(MAIN_DIR)

from use_case.multizone_simple_air_RL_SAC_DDPG import get_custom_env
import datetime
from dateutil.tz import gettz

EXPERT_PATH = os.path.join(os.path.dirname(__file__), "expert_trajectories.npz")
POLICY_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "policy_input_output.json")

stepSize = 600
start_time = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))
end_time = datetime.datetime(year=2024, month=1, day=15, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))

def get_action_names_from_env():
    # Try to get action names from the environment's action space definition
    import json
    with open(POLICY_CONFIG_PATH, 'r') as f:
        policy_config = json.load(f)
    action_names = []
    for component_id, actions in policy_config['actions'].items():
        for action_name, action_config in actions.items():
            # Use component_id and action_name for clarity
            action_names.append(f"{component_id}:{action_name}")
    return action_names

def get_observation_names_from_env():
    # Get observation names from the policy config file
    import json
    with open(POLICY_CONFIG_PATH, 'r') as f:
        policy_config = json.load(f)
    
    observation_names = []
    
    # Add observations (18 values)
    for component_id, observations in policy_config['observations'].items():
        for obs_name, obs_config in observations.items():
            observation_names.append(f"{component_id}:{obs_name}")
    
    # Add time embeddings (6 values: sin and cos for each of 3 time features)
    for time_key, time_config in policy_config['time_embeddings'].items():
        observation_names.append(f"time:{time_key}:sin")
        observation_names.append(f"time:{time_key}:cos")
    
    # Add first forecast value for each forecast type (forecast horizon = 50)
    forecast_horizon = 50
    for forecast_key, forecast_config in policy_config['forecasts'].items():
        for forecast_name, forecast_detail in forecast_config.items():
            observation_names.append(f"forecast:{forecast_key}:{forecast_name}")
    
    return observation_names

def plot_all_actions_separate(acts_arr, action_names=None, save_plots=False, save_dir=None):
    num_steps, num_actions = acts_arr.shape
    for i in range(num_actions):
        plt.figure(figsize=(12, 3))
        plt.plot(acts_arr[:, i])
        label = action_names[i] if action_names and i < len(action_names) else f"Action {i}"
        plt.ylabel(label)
        plt.title(f"{label} over Time")
        plt.xlabel("Timestep")
        plt.grid(True)
        plt.tight_layout()
        
        if save_plots and save_dir:
            # Clean filename by replacing special characters
            safe_label = label.replace(':', '_').replace('/', '_').replace('\\', '_')
            filename = os.path.join(save_dir, f"action_{i:02d}_{safe_label}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def plot_first_observations_separate(obs_arr, num_dims=18):
    num_steps = obs_arr.shape[0]
    for i in range(min(num_dims, obs_arr.shape[1])):
        plt.figure(figsize=(12, 3))
        plt.plot(obs_arr[:, i])
        plt.ylabel(f"Obs {i}")
        plt.title(f"Observation Dimension {i} over Time")
        plt.xlabel("Timestep")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def plot_observations_with_names(obs_arr, observation_names=None, save_plots=False, save_dir=None):
    """Plot observations with names from policy config, including first forecast values"""
    num_steps, num_obs = obs_arr.shape
    
    # Plot first 24 observations (18 observations + 6 time embeddings)
    for i in range(min(24, num_obs)):
        plt.figure(figsize=(12, 3))
        plt.plot(obs_arr[:, i])
        label = observation_names[i] if observation_names and i < len(observation_names) else f"Obs {i}"
        plt.ylabel(label)
        plt.title(f"{label} over Time")
        plt.xlabel("Timestep")
        plt.grid(True)
        plt.tight_layout()
        
        if save_plots and save_dir:
            # Clean filename by replacing special characters
            safe_label = label.replace(':', '_').replace('/', '_').replace('\\', '_')
            filename = os.path.join(save_dir, f"observation_{i:02d}_{safe_label}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def plot_first_forecast_values(obs_arr, observation_names=None, save_plots=False, save_dir=None):
    """Plot the first forecast value for each forecast type"""
    import json
    with open(POLICY_CONFIG_PATH, 'r') as f:
        policy_config = json.load(f)
    
    # Calculate positions for first forecast values
    # First 18: observations, next 6: time embeddings (sin/cos), next 51: outdoor_temperature, next 51: global_irradiation, etc.
    forecast_horizon = 51
    base_index = 24  # 18 observations + 6 time embeddings
    
    forecast_indices = []
    forecast_labels = []
    
    for i, (forecast_key, forecast_config) in enumerate(policy_config['forecasts'].items()):
        for forecast_name, forecast_detail in forecast_config.items():
            # First forecast value is at base_index + i * forecast_horizon
            forecast_index = base_index + i * forecast_horizon
            if forecast_index < obs_arr.shape[1]:
                forecast_indices.append(forecast_index)
                forecast_labels.append(f"forecast:{forecast_key}:{forecast_name}")
    
    # Plot first forecast value for each forecast type
    for i, (idx, label) in enumerate(zip(forecast_indices, forecast_labels)):
        plt.figure(figsize=(12, 3))
        plt.plot(obs_arr[:, idx])
        plt.ylabel(label)
        plt.title(f"First Forecast Value: {label} over Time")
        plt.xlabel("Timestep")
        plt.grid(True)
        plt.tight_layout()
        
        if save_plots and save_dir:
            # Clean filename by replacing special characters
            safe_label = label.replace(':', '_').replace('/', '_').replace('\\', '_')
            filename = os.path.join(save_dir, f"forecast_{i:02d}_{safe_label}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def main():
    parser = argparse.ArgumentParser(description="Analyze expert trajectories and generate plots")
    parser.add_argument('--save', action='store_true', help='Save plots to plots_trajectory folder instead of showing them')
    args = parser.parse_args()
    
    # Create save directory if needed
    save_dir = None
    if args.save:
        save_dir = os.path.join(os.path.dirname(__file__), "plots_trajectory")
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving plots to: {save_dir}")
    
    data = np.load(EXPERT_PATH, allow_pickle=True)
    acts_arr = data['acts']
    print(f"Loaded actions array with shape: {acts_arr.shape}")
    action_names = get_action_names_from_env()
    plot_all_actions_separate(acts_arr, action_names, save_plots=args.save, save_dir=save_dir)
    
    # Plot observations with names
    obs_arr = data['obs']
    print(f"Loaded observations array with shape: {obs_arr.shape}")
    observation_names = get_observation_names_from_env()
    print(f"Generated {len(observation_names)} observation names")
    
    # Plot first 24 observations (18 observations + 6 time embeddings)
    plot_observations_with_names(obs_arr, observation_names, save_plots=args.save, save_dir=save_dir)
    
    # Plot first forecast value for each forecast type
    plot_first_forecast_values(obs_arr, observation_names, save_plots=args.save, save_dir=save_dir)

if __name__ == "__main__":
    main() 