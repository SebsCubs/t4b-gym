"""
This script runs the baseline simulation for the multizone_simple_air model, records the observations and actions at each step (using the mapping from policy_input_output.json), and saves them in the expert trajectory format required for the imitation library (obs, acts, dones, next_obs, infos). The custom reward logic is imported from the T4BGymEnvCustomReward class in the RL script.
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

# Import the custom reward class from the RL script
from use_case.multizone_simple_air_RL_control import get_custom_env

POLICY_CONFIG_PATH = os.path.join(SCRIPT_DIR, "policy_input_output_co2sets.json")
EXPERT_SAVE_PATH = os.path.join(SCRIPT_DIR, "expert_trajectories_extended.npz")

stepSize = 600
start_time = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))
end_time = datetime.datetime(year=2024, month=1, day=15, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))

def get_action_from_baseline(model, policy_config):
    """Extract baseline actions in the same order as the environment's action space."""
    # Get the action space structure from the environment to ensure consistent ordering
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


def run_single_simulation(start_time, end_time, policy_config, pbar=None):
    """Run a single simulation and return the trajectory data."""
    # Create environment in baseline mode
    env = get_custom_env(stepSize, start_time, end_time)
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
    
    total_steps = int((end_time-start_time).total_seconds()//stepSize)
    
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
        
        if pbar:
            pbar.update(1)
    
    return obs_list, acts_list, next_obs_list, dones_list, infos_list


def main(time_periods=None):
    """
    Run expert trajectory recording for multiple time periods.
    
    Args:
        time_periods: List of tuples (start_time, end_time) defining simulation periods.
                     If None, uses the default single period.
    
    Example:
        # Single time period (default behavior)
        main()
        
        # Multiple time periods
        time_periods = [
            (datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=gettz("Europe/Copenhagen")),
             datetime.datetime(2024, 1, 7, 0, 0, 0, tzinfo=gettz("Europe/Copenhagen"))),
            (datetime.datetime(2024, 1, 15, 0, 0, 0, tzinfo=gettz("Europe/Copenhagen")),
             datetime.datetime(2024, 1, 22, 0, 0, 0, tzinfo=gettz("Europe/Copenhagen"))),
        ]
        main(time_periods)
    """
    # Load policy config
    with open(POLICY_CONFIG_PATH, 'r') as f:
        policy_config = json.load(f)
    
    # Use default time period if none provided
    if time_periods is None:
        time_periods = [(start_time, end_time)]
    
    # Calculate total steps across all periods for progress tracking
    total_steps = sum(int((end - start).total_seconds()//stepSize) for start, end in time_periods)
    
    # Initialize aggregated lists
    all_obs_list = []
    all_acts_list = []
    all_next_obs_list = []
    all_dones_list = []
    all_infos_list = []
    
    pbar = tqdm(total=total_steps, desc="Recording expert trajectories")
    
    # Run simulations for each time period
    for i, (period_start, period_end) in enumerate(time_periods):
        print(f"\nRunning simulation {i+1}/{len(time_periods)}: {period_start} to {period_end}")
        
        obs_list, acts_list, next_obs_list, dones_list, infos_list = run_single_simulation(
            period_start, period_end, policy_config, pbar
        )
        
        # Append to aggregated lists
        all_obs_list.extend(obs_list)
        all_acts_list.extend(acts_list)
        all_next_obs_list.extend(next_obs_list)
        all_dones_list.extend(dones_list)
        all_infos_list.extend(infos_list)
    
    pbar.close()
    
    # Convert to arrays
    obs_arr = np.stack(all_obs_list)
    acts_arr = np.stack(all_acts_list)
    next_obs_arr = np.stack(all_next_obs_list)
    dones_arr = np.array(all_dones_list, dtype=bool)
    
    # Print information about the recorded data
    print(f"\nRecorded expert trajectories from {len(time_periods)} simulation(s):")
    print(f"  Total observations shape: {obs_arr.shape}")
    print(f"  Total actions shape: {acts_arr.shape}")
    print(f"  Observation range: [{obs_arr.min():.3f}, {obs_arr.max():.3f}]")
    print(f"  Action range: [{acts_arr.min():.3f}, {acts_arr.max():.3f}]")
    print(f"  Total steps: {len(all_obs_list)}")
    
    # Save in imitation-compatible format
    np.savez(EXPERT_SAVE_PATH, obs=obs_arr, acts=acts_arr, next_obs=next_obs_arr, dones=dones_arr, infos=all_infos_list)
    print(f"Expert trajectories saved to {EXPERT_SAVE_PATH}")

if __name__ == "__main__":
    #main() 
    # Multiple time periods
    time_periods = [
        # Typical heat day: January 11-25, 2024
        (datetime.datetime(year=2024, month=1, day=11, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen")),
            datetime.datetime(year=2024, month=1, day=25, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))),
        
        # Mix day: March 17 - March 31, 2024
        (datetime.datetime(year=2024, month=3, day=17, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen")),
            datetime.datetime(year=2024, month=3, day=31, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))),
        
        # Typical cool day: May 17-31, 2024
        (datetime.datetime(year=2024, month=5, day=17, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen")),
            datetime.datetime(year=2024, month=5, day=31, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen")))
    ]       

    time_periods = [
        # Typical heat day: January 11-25, 2024
        (datetime.datetime(year=2024, month=1, day=11, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen")),
            datetime.datetime(year=2024, month=1, day=25, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))),
        
        # Mix day: March 17 - March 31, 2024
        (datetime.datetime(year=2024, month=3, day=17, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen")),
            datetime.datetime(year=2024, month=3, day=31, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))),
        
    ]     

    main(time_periods)