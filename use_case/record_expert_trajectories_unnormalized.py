"""
This script runs the baseline simulation for the multizone_simple_air model WITHOUT normalization,
records the observations and actions at each step, and saves them in the expert trajectory format.
This avoids double normalization issues when using autoencoders.

Usage:
    python record_expert_trajectories_unnormalized.py
"""
import os
import sys
import json
import numpy as np
import datetime
from dateutil.tz import gettz
import twin4build as tb
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(MAIN_DIR)

from boptest_model.rooms_and_ahu_model import load_model_and_params
from t4b_gym.t4b_gym_env import T4BGymEnv

POLICY_CONFIG_PATH = os.path.join(SCRIPT_DIR, "policy_input_output.json")
EXPERT_SAVE_PATH = os.path.join(SCRIPT_DIR, "expert_trajectories_unnormalized.npz")

stepSize = 600
start_time = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))
end_time = datetime.datetime(year=2024, month=1, day=15, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))


class T4BGymEnvCustomReward(T4BGymEnv):
    """Custom environment with reward function for baseline recording."""
    
    def get_reward(self, observations, action):
        """Calculate the reward based on the observations and action."""
        # Placeholder reward function for baseline recording
        return 0.0


def get_action_from_baseline(model, policy_config):
    """Extract baseline actions in the same order as the environment's action space."""
    action = []
    
    # Create a mapping from component_id to component for faster lookup
    component_map = {comp.id: comp for comp in model.components.values()}
    
    # Iterate through actions in the same order as defined in the environment
    for component_id, actions in policy_config['actions'].items():
        for action_name, action_config in actions.items():
            # Try to get the value from the model's components
            if component_id in component_map:
                comp = component_map[component_id]
                signal_key = action_config['signal_key']
                # Try input first, then output
                if signal_key in comp.input:
                    value = comp.input[signal_key].get()
                elif signal_key in comp.output:
                    value = comp.output[signal_key].get()
                else:
                    value = 0.0  # fallback
            else:
                value = 0.0  # fallback if component not found
            action.append(value)
    
    return np.array(action, dtype=np.float32)


def get_custom_env_unnormalized(stepSize, start_time, end_time):
    """Create environment WITHOUT normalization wrappers."""
    # Load model
    model = load_model_and_params("rooms_and_ahu_model")
    
    # Create environment WITHOUT normalization
    env = T4BGymEnvCustomReward(
        model=model, 
        io_config_file=POLICY_CONFIG_PATH,
        start_time=start_time,
        end_time=end_time,
        episode_length=int(3600*24*5 / stepSize),  # 5 days
        random_start=True, 
        excluding_periods=None, 
        forecast_horizon=50,
        step_size=stepSize,
        warmup_period=0
    )
    
    # IMPORTANT: NO normalization wrappers applied
    return env


def main():
    print("Recording expert trajectories from UNNORMALIZED environment...")
    
    # Load policy config
    with open(POLICY_CONFIG_PATH, 'r') as f:
        policy_config = json.load(f)
    
    # Create environment in baseline mode WITHOUT normalization
    env = get_custom_env_unnormalized(stepSize, start_time, end_time)
    env.unwrapped.baseline_mode = True
    env.unwrapped.episode_length = None
    env.unwrapped.random_start = False
    
    # Reset environment
    obs, _ = env.reset()
    done = False
    obs_list = []
    acts_list = []
    next_obs_list = []
    dones_list = []
    infos_list = []
    
    pbar = tqdm(total=int((end_time-start_time).total_seconds()//stepSize), desc="Recording expert trajectories")
    
    while not done:
        # Record current observation using the environment's _get_obs method
        # This ensures we get observations in the correct order and format
        obs_vec = env.unwrapped._get_obs()
        
        # Step environment in baseline mode (pass dummy action)
        dummy_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        next_obs, reward, terminated, truncated, info = env.step(dummy_action)
        
        # After stepping, extract the action that was actually applied by the baseline controller
        action_vec = get_action_from_baseline(env.unwrapped.simulator.model, policy_config)
        
        # Verify action dimensions match environment action space
        if action_vec.shape != env.action_space.shape:
            print(f"Warning: Action shape mismatch. Expected {env.action_space.shape}, got {action_vec.shape}")
            # Pad or truncate to match
            if len(action_vec) < env.action_space.shape[0]:
                action_vec = np.pad(action_vec, (0, env.action_space.shape[0] - len(action_vec)), 'constant')
            else:
                action_vec = action_vec[:env.action_space.shape[0]]
        
        obs_list.append(obs_vec)
        acts_list.append(action_vec)
        next_obs_vec = env.unwrapped._get_obs()
        next_obs_list.append(next_obs_vec)
        dones_list.append(terminated or truncated)
        infos_list.append(info)
        obs = next_obs
        done = terminated or truncated
        pbar.update(1)
    
    pbar.close()
    
    # Convert to arrays
    obs_arr = np.stack(obs_list)
    acts_arr = np.stack(acts_list)
    next_obs_arr = np.stack(next_obs_list)
    dones_arr = np.array(dones_list, dtype=bool)
    
    # Print information about the recorded data
    print(f"Recorded expert trajectories (UNNORMALIZED):")
    print(f"  Observations shape: {obs_arr.shape}")
    print(f"  Actions shape: {acts_arr.shape}")
    print(f"  Observation range: [{obs_arr.min():.3f}, {obs_arr.max():.3f}]")
    print(f"  Action range: [{acts_arr.min():.3f}, {acts_arr.max():.3f}]")
    
    # Save in imitation-compatible format
    np.savez(EXPERT_SAVE_PATH, obs=obs_arr, acts=acts_arr, next_obs=next_obs_arr, dones=dones_arr, infos=infos_list)
    print(f"Unnormalized expert trajectories saved to {EXPERT_SAVE_PATH}")
    print("Use these trajectories with autoencoders to avoid double normalization issues!")


if __name__ == "__main__":
    main() 