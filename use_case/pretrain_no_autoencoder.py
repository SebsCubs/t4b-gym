"""
This script loads expert trajectories, visualizes the observations, and pretrains a PPO or A2C policy using behavioral cloning (imitation learning) for further RL training.

Usage Examples:
==============

1. Pretrain a PPO policy with behavioral cloning (default large network):
   python pretrain_no_autoencoder.py

2. Pretrain an A2C policy with behavioral cloning:
   python pretrain_no_autoencoder.py --algo a2c

3. Pretrain PPO policy with extra-large network:
   python pretrain_no_autoencoder.py --network-size xlarge

4. Pretrain A2C policy with small network:
   python pretrain_no_autoencoder.py --algo a2c --network-size small

5. Pretrain PPO policy and test it after training:
   python pretrain_no_autoencoder.py --test

6. Pretrain A2C policy with medium network and test it:
   python pretrain_no_autoencoder.py --algo a2c --network-size medium --test

7. Pretrain with sanity checks and save plots:
   python pretrain_no_autoencoder.py --save-plots


Network Sizes:
=============
- small: [64, 64] (2 layers, 64 units each)
- medium: [256, 256] (2 layers, 256 units each)  
- large: [512, 512, 256, 256] (4 layers, default)
- xlarge: [1024, 1024, 512, 512, 256] (5 layers)

Normalization Handling:
======================
- Uses NormalizedObservationWrapper (automatically sets observation space to [-1, 1])
- Action normalization is always applied via NormalizedActionWrapper
- Expert trajectory observations are pre-normalized to match NormalizedObservationWrapper
- Expert trajectory actions are pre-normalized to match NormalizedActionWrapper

Recommended Workflow:
====================
1. Record expert trajectories:
   python record_expert_trajectories.py (original observations and actions)

2. Pretrain without autoencoder:
   python pretrain_no_autoencoder.py --algo ppo

Output Files:
============
- ppo_pretrained_bc.zip: Pretrained PPO policy weights
- a2c_pretrained_bc.zip: Pretrained A2C policy weights

Note: This script requires one of the following expert trajectory files:
- expert_trajectories.npz (original observations and actions)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(MAIN_DIR)

from t4b_gym.t4b_gym_env import T4BGymEnv
import datetime
from dateutil.tz import gettz
from use_case.model_eval import test_model

# Imitation library imports
from imitation.algorithms.bc import BC
from imitation.data.types import Transitions

EXPERT_PATH = os.path.join(os.path.dirname(__file__), "expert_trajectories_merged.npz")
POLICY_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "policy_input_output_co2sets.json")
PRETRAINED_MODEL_PATHS = {
    'ppo': os.path.join(os.path.dirname(__file__), "ppo_pretrained_bc.zip"),
    'a2c': os.path.join(os.path.dirname(__file__), "a2c_pretrained_bc.zip"),
}

stepSize = 600
start_time = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))
end_time = datetime.datetime(year=2024, month=1, day=15, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))

def create_large_ppo_policy(env, network_size='large', device='cpu'):
    """Create PPO with customizable MLP policy network size."""
    if network_size == 'small':
        policy_kwargs = {
            "net_arch": {
                "pi": [64, 64],      # Policy network: 2 layers with 64 units each
                "vf": [64, 64]       # Value function network: same architecture
            },
            "activation_fn": torch.nn.ReLU
        }
    elif network_size == 'medium':
        policy_kwargs = {
            "net_arch": {
                "pi": [256, 256],    # Policy network: 2 layers with 256 units each
                "vf": [256, 256]     # Value function network: same architecture
            },
            "activation_fn": torch.nn.ReLU
        }
    elif network_size == 'large':
        policy_kwargs = {
            "net_arch": {
                "pi": [512, 512, 256, 256],  # Policy network: 4 layers
                "vf": [512, 512, 256, 256]   # Value function network: same architecture
            },
            "activation_fn": torch.nn.ReLU
        }
    elif network_size == 'xlarge':
        policy_kwargs = {
            "net_arch": {
                "pi": [1024, 1024, 512, 512, 256],  # Policy network: 5 layers
                "vf": [1024, 1024, 512, 512, 256]   # Value function network: same architecture
            },
            "activation_fn": torch.nn.ReLU
        }
    else:
        raise ValueError(f"Unknown network size: {network_size}")
    
    return PPO(
        'MlpPolicy', 
        env, 
        verbose=1,
        policy_kwargs=policy_kwargs,
        device=device
    )


def create_large_a2c_policy(env, network_size='large', device='cpu'):
    """Create A2C with customizable MLP policy network size."""
    if network_size == 'small':
        policy_kwargs = {
            "net_arch": [64, 64],        # Shared network: 2 layers
            "activation_fn": torch.nn.ReLU
        }
    elif network_size == 'medium':
        policy_kwargs = {
            "net_arch": [256, 256],      # Shared network: 2 layers
            "activation_fn": torch.nn.ReLU
        }
    elif network_size == 'large':
        policy_kwargs = {
            "net_arch": [512, 512, 256, 256],  # Shared network: 4 layers
            "activation_fn": torch.nn.ReLU
        }
    elif network_size == 'xlarge':
        policy_kwargs = {
            "net_arch": [1024, 1024, 512, 512, 256],  # Shared network: 5 layers
            "activation_fn": torch.nn.ReLU
        }
    else:
        raise ValueError(f"Unknown network size: {network_size}")
    
    return A2C(
        'MlpPolicy', 
        env, 
        verbose=1,
        policy_kwargs=policy_kwargs,
        device=device
    )

def test_pretrained_policy(policy_path, env, algo='ppo', num_episodes=1):
    """Test a pre-trained policy in the environment and plot rewards."""
    print(f"Testing policy from {policy_path}...")
    if algo == 'ppo':
        policy = PPO.load(policy_path, env=env)
    elif algo == 'a2c':
        policy = A2C.load(policy_path, env=env)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    episode_rewards = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        rewards = []
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            total_reward += reward
            done = terminated or truncated
        episode_rewards.append(total_reward)
        plt.plot(rewards, label=f"Episode {ep+1}")
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.title(f"Pretrained {algo.upper()} Policy Reward per Episode")
    plt.legend()
    plt.tight_layout()
    plt.show()
    print(f"Average reward over {num_episodes} episode(s): {np.mean(episode_rewards):.2f}")


def check_time_alignment_and_episode_cuts(obs_arr, next_obs_arr, dones_arr):
    """Check that next_obs[t] == obs[t+1] except where done[t] is True."""
    print("=== Time-alignment & Episode Cuts Check ===")
    
    # Check for misalignments
    misalignments = []
    for t in range(len(obs_arr) - 1):
        if not dones_arr[t]:  # If not done, next_obs[t] should equal obs[t+1]
            if not np.allclose(obs_arr[t+1], next_obs_arr[t], rtol=1e-5, atol=1e-8):
                misalignments.append(t)
    
    if misalignments:
        print(f"[ERROR] Found {len(misalignments)} time-alignment issues!")
        print(f"   First few misalignments at timesteps: {misalignments[:5]}")
        return False
    else:
        print("[SUCCESS] Time-alignment is correct")
        return True


def check_scaling_and_units(obs_arr, acts_arr):
    """Check observation and action scaling and identify potential issues."""
    print("\n=== Scaling & Units Check ===")
    
    # Observation statistics
    obs_mean = np.mean(obs_arr, axis=0)
    obs_std = np.std(obs_arr, axis=0)
    obs_min = np.min(obs_arr, axis=0)
    obs_max = np.max(obs_arr, axis=0)
    
    # Action statistics
    acts_mean = np.mean(acts_arr, axis=0)
    acts_std = np.std(acts_arr, axis=0)
    acts_min = np.min(acts_arr, axis=0)
    acts_max = np.max(acts_arr, axis=0)
    
    print(f"Observations shape: {obs_arr.shape}")
    print(f"Actions shape: {acts_arr.shape}")
    
    # Check for extreme values
    obs_extreme = np.sum(np.abs(obs_arr) > 100, axis=0)
    acts_extreme = np.sum(np.abs(acts_arr) > 10, axis=0)
    
    print(f"\nObservation statistics:")
    print(f"  Mean range: [{obs_mean.min():.3f}, {obs_mean.max():.3f}]")
    print(f"  Std range: [{obs_std.min():.3f}, {obs_std.max():.3f}]")
    print(f"  Value range: [{obs_min.min():.3f}, {obs_max.max():.3f}]")
    print(f"  Extreme values (>100): {obs_extreme.sum()} total")
    
    print(f"\nAction statistics:")
    print(f"  Mean range: [{acts_mean.min():.3f}, {acts_mean.max():.3f}]")
    print(f"  Std range: [{acts_std.min():.3f}, {acts_std.max():.3f}]")
    print(f"  Value range: [{acts_min.min():.3f}, {acts_max.max():.3f}]")
    print(f"  Extreme values (>10): {acts_extreme.sum()} total")
    
    # Check for zero variance (constant features)
    obs_zero_var = np.sum(obs_std < 1e-6)
    acts_zero_var = np.sum(acts_std < 1e-6)
    
    if obs_zero_var > 0:
        print(f"[WARNING] {obs_zero_var} observation features have zero variance")
    if acts_zero_var > 0:
        print(f"[WARNING] {acts_zero_var} action features have zero variance")
    
    return True


def check_coverage_and_balance(obs_arr, acts_arr, save_plots=False, save_dir=None):
    """Check data coverage and balance using histograms and basic statistics."""
    print("\n=== Coverage & Balance Check ===")
    
    num_obs_features = obs_arr.shape[1]
    num_act_features = acts_arr.shape[1]
    
    # Plot histograms for a subset of features
    num_to_plot = min(8, num_obs_features)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(num_to_plot):
        axes[i].hist(obs_arr[:, i], bins=50, alpha=0.7)
        axes[i].set_title(f'Obs Feature {i}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    if save_plots and save_dir:
        plt.savefig(os.path.join(save_dir, 'observation_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # Plot action distributions
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(min(8, num_act_features)):
        axes[i].hist(acts_arr[:, i], bins=50, alpha=0.7)
        axes[i].set_title(f'Action Feature {i}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    if save_plots and save_dir:
        plt.savefig(os.path.join(save_dir, 'action_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # Check for data sparsity
    obs_sparse = np.sum(obs_arr == 0, axis=0) / len(obs_arr)
    acts_sparse = np.sum(acts_arr == 0, axis=0) / len(acts_arr)
    
    print(f"Observation sparsity (fraction of zeros):")
    for i, sparsity in enumerate(obs_sparse):
        if sparsity > 0.5:
            print(f"  Feature {i}: {sparsity:.3f} (high sparsity)")
    
    print(f"Action sparsity (fraction of zeros):")
    for i, sparsity in enumerate(acts_sparse):
        if sparsity > 0.5:
            print(f"  Feature {i}: {sparsity:.3f} (high sparsity)")
    
    return True


def check_data_quality(obs_arr, acts_arr, dones_arr):
    """Comprehensive data quality checks."""
    print("\n=== Data Quality Check ===")
    
    # Check for NaN values
    obs_nans = np.isnan(obs_arr).sum()
    acts_nans = np.isnan(acts_arr).sum()
    
    if obs_nans > 0:
        print(f"[ERROR] Found {obs_nans} NaN values in observations")
        return False
    else:
        print("[SUCCESS] No NaN values in observations")
    
    if acts_nans > 0:
        print(f"[ERROR] Found {acts_nans} NaN values in actions")
        return False
    else:
        print("[SUCCESS] No NaN values in actions")
    
    # Check for infinite values
    obs_inf = np.isinf(obs_arr).sum()
    acts_inf = np.isinf(acts_arr).sum()
    
    if obs_inf > 0:
        print(f"[ERROR] Found {obs_inf} infinite values in observations")
        return False
    else:
        print("[SUCCESS] No infinite values in observations")
    
    if acts_inf > 0:
        print(f"[ERROR] Found {acts_inf} infinite values in actions")
        return False
    else:
        print("[SUCCESS] No infinite values in actions")
    
    # Check episode structure
    num_episodes = np.sum(dones_arr)
    print(f"Number of episodes: {num_episodes}")
    print(f"Total timesteps: {len(obs_arr)}")
    print(f"Average episode length: {len(obs_arr) / max(num_episodes, 1):.1f} steps")
    
    return True


def run_sanity_checks(obs_arr, acts_arr, next_obs_arr, dones_arr, save_plots=False, save_dir=None):
    """Run all sanity checks on the expert trajectory data."""
    print("Running sanity checks on expert trajectory data...")
    print("=" * 60)
    
    checks_passed = True
    
    # Run all checks
    checks_passed &= check_data_quality(obs_arr, acts_arr, dones_arr)
    checks_passed &= check_time_alignment_and_episode_cuts(obs_arr, next_obs_arr, dones_arr)
    checks_passed &= check_scaling_and_units(obs_arr, acts_arr)
    checks_passed &= check_coverage_and_balance(obs_arr, acts_arr, save_plots, save_dir)
    
    print("\n" + "=" * 60)
    if checks_passed:
        print("[SUCCESS] All sanity checks passed! Data looks good for training.")
    else:
        print("[ERROR] Some sanity checks failed. Please review the data before training.")
    
    return checks_passed

def get_env(stepSize, start_time, end_time):
    """Create environment with custom reward function for pretraining."""
    from use_case.multizone_simple_air_RL_control import load_model_and_params
    from t4b_gym.t4b_gym_env import NormalizedObservationWrapper, NormalizedActionWrapper
    model = load_model_and_params()
    
    class T4BGymEnvCustomReward(T4BGymEnv):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
        def get_reward(self, action, observation):
            """
            Custom reward function for pretraining.
            Simplified reward that encourages staying within comfortable temperature ranges.
            """
            # Extract temperature-related observations (assuming they're in the first few positions)
            # This is a simplified reward - adjust based on your actual observation structure
            temp_obs = observation[:5]  # Assuming first 5 observations are temperatures
            
            # Reward for staying within comfortable range (e.g., 20-24°C)
            # Normalize temperatures to reasonable range (assuming they're in °C)
            temp_penalty = 0
            for temp in temp_obs:
                if temp < 18 or temp > 26:  # Penalty for uncomfortable temperatures
                    temp_penalty += abs(temp - 22) * 0.1  # 22°C as target
            
            # Small penalty for large actions to encourage smooth control
            action_penalty = np.mean(np.abs(action)) * 0.01
            
            # Base reward minus penalties
            reward = 1.0 - temp_penalty - action_penalty
            
            return reward
    
    env = T4BGymEnvCustomReward(
        model=model, 
        io_config_file=POLICY_CONFIG_PATH,
        start_time=start_time,
        end_time=end_time,
        episode_length=int(3600*24*5 / stepSize),
        random_start=True, 
        excluding_periods=None, 
        forecast_horizon=50,
        step_size=stepSize,
        warmup_period=0
    )

    env = NormalizedObservationWrapper(env)
    env = NormalizedActionWrapper(env)
    
    return env


def normalize_expert_trajectories(obs_arr, acts_arr, next_obs_arr, stepSize, start_time, end_time):
    """
    Normalize expert trajectory observations and actions to match environment normalization.
    
    Args:
        obs_arr: Expert trajectory observations
        acts_arr: Expert trajectory actions  
        next_obs_arr: Expert trajectory next observations
        stepSize: Environment step size
        start_time: Environment start time
        end_time: Environment end time
        
    Returns:
        tuple: (normalized_obs_arr, normalized_acts_arr, normalized_next_obs_arr)
    """
    # For BC training without autoencoder, we need to normalize expert trajectories
    # to match what the environment will provide (after NormalizedObservationWrapper)

    # Get the original observation space bounds (before normalization wrapper)
    # We need to create a temporary environment without wrappers to get the original bounds
    from use_case.multizone_simple_air_RL_control import load_model_and_params
    from t4b_gym.t4b_gym_env import T4BGymEnv
    
    # Create base environment without wrappers
    model = load_model_and_params()
    temp_env = T4BGymEnv(
        model=model, 
        io_config_file=POLICY_CONFIG_PATH,
        start_time=start_time,
        end_time=end_time,
        episode_length=int(3600*24*5 / stepSize),
        random_start=True, 
        excluding_periods=None, 
        forecast_horizon=50,
        step_size=stepSize,
        warmup_period=0
    )

    if 'boptest' in EXPERT_PATH:
        #Get the action and observation space indeces of the values with temperatures
        observation_keys = temp_env._observations
        observation_keys_indices = [i for i, key in enumerate(observation_keys) if "temp" in key]

        action_keys = temp_env._actions
        action_keys_indices = [i for i, key in enumerate(action_keys) if "temp" in key]

        #Convert Kelvin to Celsius for the acts and obs with temperatures
        # Apply conversion row-wise (for each time step) to the feature dimension
        obs_arr[:, observation_keys_indices] = obs_arr[:, observation_keys_indices] - 273.15
        next_obs_arr[:, observation_keys_indices] = next_obs_arr[:, observation_keys_indices] - 273.15
        acts_arr[:, action_keys_indices] = acts_arr[:, action_keys_indices] - 273.15

    
    
    # Get original bounds
    obs_low = temp_env.observation_space.low
    obs_high = temp_env.observation_space.high
    act_low = temp_env.action_space.low
    act_high = temp_env.action_space.high
    
    # Apply the same normalization as NormalizedObservationWrapper: 2*(obs - low)/(high - low) - 1
    normalized_obs_arr = 2 * (obs_arr - obs_low) / (obs_high - obs_low) - 1
    normalized_next_obs_arr = 2 * (next_obs_arr - obs_low) / (obs_high - obs_low) - 1
    
    # Apply the same normalization as NormalizedActionWrapper: 2*(acts - low)/(high - low) - 1
    normalized_acts_arr = 2 * (acts_arr - act_low) / (act_high - act_low) - 1
    
    # Clip to [-1, 1] bounds (same as wrappers)
    normalized_obs_arr = np.clip(normalized_obs_arr, -1, 1)
    normalized_next_obs_arr = np.clip(normalized_next_obs_arr, -1, 1)
    normalized_acts_arr = np.clip(normalized_acts_arr, -1, 1)
    
    print(f"Expert trajectory observations normalized to range [{normalized_obs_arr.min():.3f}, {normalized_obs_arr.max():.3f}]")
    print(f"Expert trajectory actions normalized to range [{normalized_acts_arr.min():.3f}, {normalized_acts_arr.max():.3f}]")
    
    # Close temporary environment
    temp_env.close()
    
    return normalized_obs_arr, normalized_acts_arr, normalized_next_obs_arr


def main():
    parser = argparse.ArgumentParser(description="Pretrain RL policy with behavioral cloning using expert trajectories.")
    parser.add_argument('--algo', choices=['ppo', 'a2c'], default='ppo', help='RL algorithm to use (default: ppo)')
    parser.add_argument('--test', action='store_true', help='Test the pretrained policy after training')
    parser.add_argument('--skip-checks', action='store_true', help='Skip sanity checks (not recommended)')
    parser.add_argument('--save-plots', action='store_true', default=True, help='Save sanity check plots to plots_sanity_checks folder')
    parser.add_argument('--network-size', choices=['small', 'medium', 'large', 'xlarge'], default='large', 
                       help='Network size for the policy (default: large)')
    args = parser.parse_args()
    algo = args.algo
    model_path = PRETRAINED_MODEL_PATHS[algo]

    # Choose expert trajectory source
    expert_path = EXPERT_PATH
    print("Using original expert trajectories")

    # Create save directory for plots if needed
    save_dir = None
    if args.save_plots:
        save_dir = os.path.join(os.path.dirname(__file__), "plots_sanity_checks")
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving sanity check plots to: {save_dir}")

    # Load expert trajectories
    data = np.load(expert_path, allow_pickle=True)
    obs_arr = data['obs']
    acts_arr = data['acts']
    next_obs_arr = data['obs_next']
    dones_arr = data['dones']
    
    print("Normalizing expert trajectory observations and actions to match environment normalization...")
    
    # Normalize expert trajectories to match environment normalization
    obs_arr, acts_arr, next_obs_arr = normalize_expert_trajectories(
        obs_arr, acts_arr, next_obs_arr, stepSize, start_time, end_time
    )

    # Run sanity checks
    if not args.skip_checks:
        checks_passed = run_sanity_checks(obs_arr, acts_arr, next_obs_arr, dones_arr, 
                                        save_plots=args.save_plots, save_dir=save_dir)
        if not checks_passed:
            print("Sanity checks failed. Consider fixing the data or use --skip-checks to proceed anyway.")
            return
    else:
        print("Skipping sanity checks (not recommended)")

    # Visualize observations
    #visualize_observations(obs_arr)

    # Prepare environment (for policy and BC trainer)
    # Use the same environment setup as regular RL training for consistency
    env = get_env(stepSize, start_time, end_time)
    
    
    # Create an RL policy (untrained) with specified network size
    print(f"Creating {algo.upper()} policy with {args.network_size} network size...")
    if algo == 'ppo':
        policy = create_large_ppo_policy(env, network_size=args.network_size)
    elif algo == 'a2c':
        policy = create_large_a2c_policy(env, network_size=args.network_size)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    # Prepare transitions for BC
    infos_arr = [{} for _ in range(len(obs_arr))]
    transitions = Transitions(
        obs=obs_arr,
        acts=acts_arr,
        infos=infos_arr,
        next_obs=next_obs_arr,
        dones=dones_arr
    )

    # Pretrain with Behavioral Cloning
    rng = np.random.default_rng(0)
    bc_trainer = BC( 
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        policy=policy.policy,  # Use the RL policy's network
        device='cpu',
        rng=rng
    )
    print(f"Pretraining {algo.upper()} policy with behavioral cloning...")
    bc_trainer.train(n_epochs=50)
    print("Pretraining complete.")

    # Save the pretrained policy weights for later RL training
    policy.save(model_path)
    print(f"Pretrained {algo.upper()} policy saved to {model_path}")

    # Test the pretrained policy
    if args.test:
        test_pretrained_policy(model_path, env, algo=algo, num_episodes=1)
    else:
        # Use the same environment for testing (consistent with training)
        test_model(env, policy)

if __name__ == "__main__":
    main() 