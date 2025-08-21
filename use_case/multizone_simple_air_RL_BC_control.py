"""
This script uses the t4b_gym environment with the t4b model of the BOPTEST model multizone_simple_air 
It defines a custom reward function and uses the PPO algorithm to control different setpoints of the model. 
"""

import twin4build as tb
import datetime
import twin4build.examples.utils as utils
from dateutil.tz import gettz 
import sys
import os
import logging
from stable_baselines3 import PPOKL4BC, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(MAIN_DIR)
from t4b_gym.t4b_gym_env import T4BGymEnv, NormalizedObservationWrapper, NormalizedActionWrapper
from boptest_model.rooms_and_ahu_model import load_model_and_params
from use_case.model_eval import test_model

log_dir = os.path.join(SCRIPT_DIR, 'logs_finetune_bc')
os.makedirs(log_dir, exist_ok=True)


POLICY_CONFIG_PATH = os.path.join(SCRIPT_DIR, "policy_input_output.json")
device = 'cpu'
bc_model_path = os.path.join(os.path.dirname(__file__), "ppo_pretrained_bc.zip")


def get_custom_env(stepSize, start_time, end_time):
    model = load_model_and_params()
    class T4BGymEnvCustomReward(T4BGymEnv):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.previous_objective = 0.0
        def get_reward(self, action, observation):
            zones = ['core', 'north', 'east', 'south', 'west']
            temp_violations = []
            for zone in zones:
                temp = self.simulator.model.components[f"{zone}_indoor_temp_sensor"].output["measuredValue"]
                heating_setpoint = self.simulator.model.components[f"{zone}_temperature_heating_setpoint"].output["scheduleValue"]
                cooling_setpoint = self.simulator.model.components[f"{zone}_temperature_cooling_setpoint"].output["scheduleValue"]
                heating_violation = max(0, heating_setpoint - temp)
                cooling_violation = max(0, temp - cooling_setpoint)
                zone_violation = (1+heating_violation)**2 + (1+cooling_violation)**2
                temp_violations.append(zone_violation)
            temp_violation_penalty = 1000 * sum(temp_violations)
            coils_power_consumption = []
            for zone in zones:
                airflow_rate = self.simulator.model.components[f"{zone}_reheat_coil"].input["airFlowRate"]
                inlet_air_temp = self.simulator.model.components[f"{zone}_reheat_coil"].input["inletAirTemperature"]
                outlet_air_temp = self.simulator.model.components[f"{zone}_reheat_coil"].output["outletAirTemperature"]
                tol = 1e-5
                specificHeatCapacityAir = 1005 #J/kg/K
                if airflow_rate>tol:
                    if inlet_air_temp < outlet_air_temp:
                        Q = airflow_rate*specificHeatCapacityAir*(outlet_air_temp - inlet_air_temp)
                        if np.isnan(Q):
                            raise ValueError("Q is not a number")
                        coils_power_consumption.append(Q)
                    else:
                        Q = 0
                        coils_power_consumption.append(Q)
                else:
                    coils_power_consumption.append(0)
            coils_power_consumption_penalty = 0.01 * sum(coils_power_consumption)
            fan_power = self.simulator.model.components["vent_power_sensor"].output["measuredValue"]
            supply_cooling_coil_power = self.simulator.model.components["supply_cooling_coil"].output["Power"]
            supply_heating_coil_power = self.simulator.model.components["supply_heating_coil"].output["Power"]
            ahu_power_consumption_penalty = 0.01 * (fan_power + supply_cooling_coil_power + supply_heating_coil_power)
            reward = temp_violation_penalty + coils_power_consumption_penalty + ahu_power_consumption_penalty
            reward = reward/1000
            if np.isnan(reward):
                raise ValueError("Reward is not a number")
            return -reward
    env = T4BGymEnvCustomReward(
        model = model, 
        io_config_file = POLICY_CONFIG_PATH,
        start_time = start_time,
        end_time = end_time,
        episode_length= int(3600*24*5 / stepSize),  # 5 days
        random_start=True, 
        excluding_periods=None, 
        forecast_horizon=50,
        step_size=stepSize,
        warmup_period=0) 

    return env

class SingleLineProgressCallback(BaseCallback):
    """
    Custom callback for printing training progress in a single line.
    Displays model's cumulative timesteps against an overall target.
    """
    def __init__(self, overall_target_agent_steps, verbose=0):
        super().__init__(verbose)
        self.overall_target_agent_steps = overall_target_agent_steps
        self.target_reached_printed_newline = False # Flag to ensure newline is printed only once

    def _init_callback(self) -> None:
        super()._init_callback() # Call parent's _init_callback if it exists and is needed
        # self.model is now available, and self.model.num_timesteps is its current cumulative step count.
        if self.model.num_timesteps > 0:
            print(f"\nCallback: Continuing training. Model has {self.model.num_timesteps} of {self.overall_target_agent_steps} timesteps.")
        else:
            print(f"\nCallback: Starting training from 0. Target: {self.overall_target_agent_steps} timesteps.")
        
        # If already at/beyond target when starting, mark newline as (conceptually) done for _on_step
        if self.model.num_timesteps >= self.overall_target_agent_steps:
            print() # Ensure this initial status message is on its own line if target already met
            self.target_reached_printed_newline = True
        else:
            self.target_reached_printed_newline = False

    def _on_step(self) -> bool:
        current_cumulative_steps = self.num_timesteps
        progress_percent = (current_cumulative_steps / self.overall_target_agent_steps) * 100 if self.overall_target_agent_steps > 0 else 0
        print(
            f"\rTraining progress: {current_cumulative_steps}/{self.overall_target_agent_steps} steps "
            f"({progress_percent:.1f}%)",
            end="",
        )
        
        # Add a newline only once when the overall target is first reached or exceeded.
        if current_cumulative_steps >= self.overall_target_agent_steps and not self.target_reached_printed_newline:
            print() # Print the newline
            self.target_reached_printed_newline = True # Set flag to prevent further newlines
        return True


class PeriodicSaveCallback(BaseCallback):
    """
    Callback for saving a model periodically during training.
    Fixed to work properly with vectorized environments.
    """
    def __init__(self, save_freq: int, save_path: str, verbose: int = 0):
        """
        Args:
            save_freq: Save frequency in total timesteps (not callback calls)
            save_path: Path to save the model
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.last_save_step = 0

    def _on_step(self) -> bool:
        current_steps = self.model.num_timesteps
        if current_steps - self.last_save_step >= self.save_freq:
            model_path = os.path.join(self.save_path, f'model_{current_steps}')
            try:
                self.model.save(model_path)
                self.last_save_step = current_steps
                if self.verbose > 0:
                    print(f"\nSaving model checkpoint to {model_path}")
            except Exception as e:
                if self.verbose > 0:
                    print(f"\nWarning: Failed to save model at step {current_steps}: {e}")
        return True


class AdaptKLtoBC(BaseCallback):
    def __init__(self, target_kl=0.02, up=1.5, down=0.7, min_beta=1e-4, max_beta=5.0, verbose=0):
        super().__init__(verbose)
        self.target_kl, self.up, self.down = target_kl, up, down
        self.min_beta, self.max_beta = min_beta, max_beta

    def _on_step(self) -> bool:
        # This method is called on every step, but we only want to adapt KL on rollout end
        # So we just return True to continue training
        return True

    def _on_rollout_end(self) -> bool:
        kl = self.model.logger.name_to_value.get("train/kl_bc", None)
        if kl is None:
            return True
        beta = self.model.beta_kl
        if kl > 2 * self.target_kl:
            beta *= self.up
        elif kl < 0.5 * self.target_kl:
            beta *= self.down
        beta = float(max(self.min_beta, min(self.max_beta, beta)))
        self.model.set_beta_kl(beta)
        if self.verbose:
            print(f"[KL-BC] kl={kl:.4f} -> beta={beta:.4f}")
        return True


class ActionBoundsMonitorCallback(BaseCallback):
    """Monitor action bounds during training to catch when actions go out of bounds."""
    
    def __init__(self, check_freq=1000, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.step_count = 0
        
    def _on_step(self) -> bool:
        self.step_count += 1
        
        # Check action bounds periodically
        if self.step_count % self.check_freq == 0:
            # Get a sample observation and test policy output
            obs = self.training_env.observation_space.sample()
            with torch.no_grad():
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Check if actions are within bounds
                if action.min() < -1.1 or action.max() > 1.1:  # Allow small tolerance
                    print(f"\nWARNING: Actions out of bounds at step {self.step_count}!")
                    print(f"Action range: [{action.min():.4f}, {action.max():.4f}]")
                    print(f"Action shape: {action.shape}")
                    
                    # Log to tensorboard if available
                    if hasattr(self.model, 'logger'):
                        self.model.logger.record("train/action_min", action.min())
                        self.model.logger.record("train/action_max", action.max())
        
        return True


class RobustNormalizedActionWrapper(gym.ActionWrapper):
    """Robust version of NormalizedActionWrapper that handles out-of-bounds actions gracefully."""
    
    def __init__(self, env):
        super().__init__(env)
        
        # Assert that original action space is a Box space
        assert isinstance(self.unwrapped.action_space, gym.spaces.Box), 'This wrapper only works with continuous action space (spaces.Box)'
        
        # Store low and high bounds of action space
        self.low = self.unwrapped.action_space.low
        self.high = self.unwrapped.action_space.high
        
        # Check for invalid action space bounds
        if np.any(self.low >= self.high):
            raise ValueError("Action space bounds are invalid: low >= high")
            
        # Check for NaN values in action space bounds
        if np.isnan(self.low).any() or np.isnan(self.high).any():
            raise ValueError("NaN values detected in action space bounds")
        
        # Redefine action space to lie between [-1,1]
        self.action_space = gym.spaces.Box(low=-1, high=1,
                                          shape=self.unwrapped.action_space.shape, 
                                          dtype=np.float32)
        
        print("Applied RobustNormalizedActionWrapper with automatic action clipping")
        
    def action(self, action_wrapper):
        """Convert normalized actions back to original space with automatic clipping."""
        # Check for NaN values in input action
        if np.isnan(action_wrapper).any():
            raise ValueError("NaN values detected in input action")
            
        # Clip actions to [-1, 1] bounds if they're outside
        if not np.all(action_wrapper >= -1) or not np.all(action_wrapper <= 1):
            # Log clipping (but limit frequency)
            if not hasattr(self, '_clip_count'):
                self._clip_count = 0
            self._clip_count += 1
            if self._clip_count % 1000 == 1:  # Print first occurrence and every 1000th after
                print(f"WARNING: Actions clipped from [{action_wrapper.min():.4f}, {action_wrapper.max():.4f}] to [-1.0, 1.0] (occurrence #{self._clip_count})")
            
            # Clip the actions
            action_wrapper = np.clip(action_wrapper, -1.0, 1.0)
            
        # Convert to original action space
        action = self.low + (0.5 * (action_wrapper + 1.0) * (self.high - self.low))
        
        # Check for NaN values in output action
        if np.isnan(action).any():
            raise ValueError("NaN values detected in output action")
            
        # Final safety check - clip to original bounds if needed
        action = np.clip(action, self.low, self.high)
        
        return action


class ActionClippingWrapper(gym.ActionWrapper):
    """Wrapper to clip actions to valid bounds before passing to environment."""
    
    def __init__(self, env):
        super().__init__(env)
        print("Applied ActionClippingWrapper to ensure actions stay within bounds")
        
    def action(self, action):
        """Clip actions to valid bounds."""
        # Clip actions to [-1, 1] range
        clipped_action = np.clip(action, -1.0, 1.0)
        
        # Log if clipping occurred (but limit frequency to avoid spam)
        if not np.array_equal(action, clipped_action):
            # Only print every 1000th clipping event to avoid spam
            if not hasattr(self, '_clip_count'):
                self._clip_count = 0
            self._clip_count += 1
            if self._clip_count % 1000 == 1:  # Print first occurrence and every 1000th after
                print(f"WARNING: Actions clipped from [{action.min():.4f}, {action.max():.4f}] to [{clipped_action.min():.4f}, {clipped_action.max():.4f}] (occurrence #{self._clip_count})")
        
        return clipped_action

def create_large_ppo_policy_kwargs(network_size='large'):
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
    
    return policy_kwargs

def PPO_BC_finetune_training(test_model_flag=False, reload_model_flag=False,  total_timesteps=100000,
                 load_pretrained_bc=False):
    """
    Train PPO with optional autoencoder support.
    
    Args:
        test_model_flag: If True, test the model instead of training
        reload_model_flag: If True, reload an existing model for continued training
        load_pretrained_bc: If True, load pretrained behavioral cloning model
    """
    
    stepSize = 600 #Seconds
    #Define the range of available data
    start_time = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))
    end_time = datetime.datetime(year=2024, month=1, day=15, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))        

    env = get_custom_env(stepSize, start_time, end_time)
    env = NormalizedObservationWrapper(env)
    env = RobustNormalizedActionWrapper(env)  # Use robust wrapper that handles out-of-bounds actions
    env = Monitor(env=env, filename=os.path.join(log_dir,'monitor.csv'))

    policy_kwargs = create_large_ppo_policy_kwargs()

    if test_model_flag:
        model_path = os.path.join(log_dir, "ppo_model.zip")
        bc_model = PPO.load(bc_model_path, env=env, device=device)
        bc_model.policy.to(device).eval()
        model = PPOKL4BC.load(model_path, env=env, bc_policy=bc_model.policy, policy_kwargs=policy_kwargs, device=device)
        print(f"Training steps: {model.num_timesteps}")
        test_model(env, model)
        return

    # Load pretrained behavioral cloning model if requested

    
    if os.path.exists(bc_model_path):
        print(f"Loading pretrained behavioral cloning model from {bc_model_path}")
        bc_model = PPO.load(bc_model_path, env=env, device=device)
        print(f"Loaded pretrained model with {bc_model.num_timesteps} timesteps")
        
        model = PPOKL4BC(
        "MlpPolicy", env, bc_policy=bc_model.policy, policy_kwargs=policy_kwargs, beta_kl=1.0,
        n_steps=4096, batch_size=256, n_epochs=10,
        gamma=0.997, gae_lambda=0.95, learning_rate=1e-4, clip_range=0.15,  # Increased learning rate
        ent_coef=0.0, vf_coef=0.5, device=device, verbose=1
        )
        
        # Debug: Check action space bounds
        print(f"Environment action space: {env.action_space}")
        print(f"Environment action space bounds: [{env.action_space.low}, {env.action_space.high}]")
        
        # Debug: Test policy output bounds
        test_obs = env.observation_space.sample()
        with torch.no_grad():
            test_action, _ = model.predict(test_obs, deterministic=True)
            print(f"Test policy output bounds (deterministic): [{test_action.min():.4f}, {test_action.max():.4f}]")
            if test_action.min() < -1 or test_action.max() > 1:
                print(f"WARNING: Policy producing actions outside [-1, 1] bounds!")
                print(f"Action range: [{test_action.min():.4f}, {test_action.max():.4f}]")
            
            # Test stochastic actions (like during training)
            test_action_stoch, _ = model.predict(test_obs, deterministic=False)
            print(f"Test policy output bounds (stochastic): [{test_action_stoch.min():.4f}, {test_action_stoch.max():.4f}]")
            if test_action_stoch.min() < -1 or test_action_stoch.max() > 1:
                print(f"WARNING: Stochastic policy producing actions outside [-1, 1] bounds!")
                print(f"Stochastic action range: [{test_action_stoch.min():.4f}, {test_action_stoch.max():.4f}]")
            
            # Test multiple random observations to get a better sense of action distribution
            print("Testing action bounds with multiple random observations...")
            action_mins = []
            action_maxs = []
            stoch_action_mins = []
            stoch_action_maxs = []
            for i in range(10):
                test_obs = env.observation_space.sample()
                test_action, _ = model.predict(test_obs, deterministic=True)
                test_action_stoch, _ = model.predict(test_obs, deterministic=False)
                action_mins.append(test_action.min())
                action_maxs.append(test_action.max())
                stoch_action_mins.append(test_action_stoch.min())
                stoch_action_maxs.append(test_action_stoch.max())
            
            print(f"Deterministic action range across 10 samples: [{min(action_mins):.4f}, {max(action_maxs):.4f}]")
            print(f"Stochastic action range across 10 samples: [{min(stoch_action_mins):.4f}, {max(stoch_action_maxs):.4f}]")
            print(f"Average deterministic action range: [{np.mean(action_mins):.4f}, {np.mean(action_maxs):.4f}]")
            print(f"Average stochastic action range: [{np.mean(stoch_action_mins):.4f}, {np.mean(stoch_action_maxs):.4f}]")
        
        # Set learning rate for fine-tuning from pretrained model
        fine_tune_lr = 1e-4  # Increased learning rate for better stability
        model.learning_rate = fine_tune_lr
        print(f"Set learning rate to {fine_tune_lr} for fine-tuning from pretrained model")
    else:
        raise FileNotFoundError(f"Pretrained behavioral cloning model not found at {bc_model_path}. "
                                f"Please run the pretraining script first to generate the required model file.")


    # Set up callback for BC fine-tuning

    AdaptKLtoBC_callback = AdaptKLtoBC(target_kl=0.005, verbose=1)
    progress_callback = SingleLineProgressCallback(total_timesteps, verbose=0)
    periodic_save_callback = PeriodicSaveCallback(save_freq=100000, save_path=log_dir, verbose=0)
    action_bounds_callback = ActionBoundsMonitorCallback(check_freq=1000, verbose=1)
    callback = CallbackList([AdaptKLtoBC_callback, progress_callback, periodic_save_callback, action_bounds_callback])

    # Train the model
    print(f"\n{'='*60}")
    print("STARTING TRAINING")
    print(f"{'='*60}")
    
    print(f"Total timesteps to train: {total_timesteps:,}")
    print(f"Environment: {env.__class__.__name__}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Device: {device}")
    print(f"Log directory: {log_dir}")
    print(f"{'='*60}\n")
    
    if reload_model_flag:
        model_path = os.path.join(log_dir, "100k.zip")
        print(f"Reloading existing model from {model_path}")
        model = PPO.load(model_path, env=env, device=device)
        print(f"Loaded model with {model.num_timesteps} previous timesteps")

        # Set lower learning rate for fine-tuning from pretrained model
        fine_tune_lr = 1e-6  # 10x lower than default for fine-tuning
        model.learning_rate = fine_tune_lr
        print(f"Set learning rate to {fine_tune_lr} for fine-tuning from pretrained model")

        new_logger = configure(log_dir, ['csv'])
        model.set_logger(new_logger)

        print("Continuing training with existing model...")
        model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=False)
    else:
        new_logger = configure(log_dir, ['csv'])
        model.set_logger(new_logger)

        print("Starting fresh training...")
        model.learn(total_timesteps=total_timesteps, callback=callback)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Final model timesteps: {model.num_timesteps}")
    print(f"{'='*60}\n")

    # Save the model
    model_save_path = os.path.join(log_dir, "ppo_model")
    print(f"Saving trained model to {model_save_path}")
    model.save(model_save_path)
    print(f"Model saved successfully!")
    print(f"Model file size: {os.path.getsize(model_save_path + '.zip') / (1024*1024):.2f} MB")

    #Test the model
    test_model(env, model)


if __name__ == "__main__":
    PPO_BC_finetune_training(test_model_flag=False, reload_model_flag=False, load_pretrained_bc=True, total_timesteps=500000)


