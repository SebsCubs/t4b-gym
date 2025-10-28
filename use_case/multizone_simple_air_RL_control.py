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
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import numpy as np
import gymnasium as gym

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(MAIN_DIR)
from t4b_gym.t4b_gym_env import T4BGymEnv, NormalizedObservationWrapper, NormalizedActionWrapper
from boptest_model.rooms_and_ahu_model import load_model_and_params
from use_case.model_eval import test_model

log_dir = os.path.join(SCRIPT_DIR, 'logs')
os.makedirs(log_dir, exist_ok=True)


POLICY_CONFIG_PATH = os.path.join(SCRIPT_DIR, "policy_input_output_co2sets.json")
device = 'cpu'



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
    env = NormalizedObservationWrapper(env)
    env = NormalizedActionWrapper(env)  
    return env

def PPO_training(test_model_flag=False, reload_model_flag=False, use_autoencoder=False, total_timesteps=100000,
                 latent_dim=64, network_size='large', load_pretrained_bc=False):
    """
    Train PPO with optional autoencoder support.
    
    Args:
        test_model_flag: If True, test the model instead of training
        reload_model_flag: If True, reload an existing model for continued training
        use_autoencoder: If True, use autoencoder for observation compression
        latent_dim: Latent dimension for autoencoder (default: 64)
        network_size: Autoencoder network size (small/medium/large/xlarge)
        load_pretrained_bc: If True, load pretrained behavioral cloning model
    """
    
    # Create a new model
    model = load_model_and_params()

    stepSize = 600 #Seconds
    #Define the range of available data
    start_time = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))
    end_time = datetime.datetime(year=2024, month=1, day=15, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))        

    class T4BGymEnvCustomReward(T4BGymEnv):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.previous_objective = 0.0
            # Initialize imitation learning components
            self.expert_actions = None
            self.imitation_weight = 0.0
            self.training_phase = 'regular'
            
            
        def get_reward(self, observations, action):
            # Temperature violations for all zones
            zones = ['core', 'north', 'east', 'south', 'west']
            temp_violations = []
            
            for zone in zones:
                temp = self.simulator.model.components[f"{zone}_indoor_temp_sensor"].output["measuredValue"]
                heating_setpoint = self.simulator.model.components[f"{zone}_temperature_heating_setpoint"].output["scheduleValue"]
                cooling_setpoint = self.simulator.model.components[f"{zone}_temperature_cooling_setpoint"].output["scheduleValue"]
                
                # Calculate violations with deadband
                heating_violation = max(0, heating_setpoint - temp)
                cooling_violation = max(0, temp - cooling_setpoint)
                
                # Use quadratic penalty instead of exponential for stability
                zone_violation = (1+heating_violation)**2 + (1+cooling_violation)**2
                temp_violations.append(zone_violation)
            
            # Balanced temperature penalty
            temp_violation_penalty = 1000 * sum(temp_violations)  # Reduced from 10000
            
            # Estimated coil power consumption
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
            
            # Moderate coil power consumption penalty
            coils_power_consumption_penalty = 0.01 * sum(coils_power_consumption)
            
            # AHU power consumption
            fan_power = self.simulator.model.components["vent_power_sensor"].output["measuredValue"]
            supply_cooling_coil_power = self.simulator.model.components["supply_cooling_coil"].output["Power"]
            supply_heating_coil_power = self.simulator.model.components["supply_heating_coil"].output["Power"]
            ahu_power_consumption_penalty = 0.01 * (fan_power + supply_cooling_coil_power + supply_heating_coil_power)
            
            # Penalty for closing the dampers below 15%
            damper_position_penalty = 0
            for zone in zones:
                damper_position = self.simulator.model.components[f"{zone}_supply_damper_position_sensor"].output["measuredValue"]
                if damper_position < 0.15:
                    damper_position_penalty += 1000 * (0.15 - damper_position)

            # Imitation learning component - encourage staying close to expert demonstrations
            imitation_penalty = 0.0
            if self.expert_actions is not None:
                # Get current timestep in episode
                current_step = getattr(self.simulator, 'current_step', 0)
                
                if current_step < len(self.expert_actions):
                    expert_action = self.expert_actions[current_step]
                    # Imitation penalty - encourage staying close to expert
                    imitation_penalty = np.sum((action - expert_action)**2)
                else:
                    # If we've exceeded the expert trajectory length, use the last expert action
                    # This handles cases where episodes are longer than the expert data
                    expert_action = self.expert_actions[-1]
                    imitation_penalty = np.sum((action - expert_action)**2)

            # Weighted combination: start with high imitation weight, gradually reduce
            if self.training_phase == 'bc_fine_tune':
                # During BC fine-tuning, use imitation learning with decay
                imitation_weight = max(0.01, 1.0 - getattr(self.simulator, 'current_step', 0) / 1000)  # Decay over time
                task_weight = 1.0 - imitation_weight
            else:
                # During regular training, no imitation
                imitation_weight = 0.0
                task_weight = 1.0

            # Total objective with imitation learning
            task_reward = temp_violation_penalty + coils_power_consumption_penalty + ahu_power_consumption_penalty + damper_position_penalty
            task_reward = task_reward/1000 #Making the number smaller for readability
            
            reward = task_weight * (-task_reward) + imitation_weight * (-imitation_penalty)
            
            if np.isnan(reward):
                raise ValueError("Reward is not a number")
            
            
            return reward
        
        def load_expert_demonstrations(self, expert_data_path=None):
            """
            Load expert demonstrations for imitation learning from expert_trajectories.npz.
            The expert data is unnormalized, so we need to normalize it to match the environment's action space.
            """
            if expert_data_path is None:
                expert_data_path = os.path.join(os.path.dirname(__file__), "expert_trajectories_extended.npz")
            
            if not os.path.exists(expert_data_path):
                print(f"Warning: Expert data file not found at {expert_data_path}")
                print("Using dummy expert actions as fallback.")
                # Create dummy expert actions that are reasonable for HVAC control
                episode_length = int(3600*24*5 / 600)  # 5 days with 600s steps
                action_dim = self.action_space.shape[0]  # Use actual action space dimension
                self.expert_actions = np.random.uniform(0.2, 0.8, (episode_length, action_dim))
                self.training_phase = 'bc_fine_tune'
                return
            
            print(f"Loading expert demonstrations from {expert_data_path}")
            
            # Load expert trajectories
            data = np.load(expert_data_path, allow_pickle=True)
            expert_acts = data['acts']  # Shape: (timesteps, action_dim)
            
            print(f"Loaded expert actions with shape: {expert_acts.shape}")
            print(f"Expert action range: [{expert_acts.min():.3f}, {expert_acts.max():.3f}]")
            
            # Normalize expert actions to match the environment's action space
            # The environment uses NormalizedActionWrapper which maps actions to [-1, 1]
            # We need to apply the same normalization as the wrapper
            
            # Get the original action space bounds (before normalization wrapper)
            # Create a temporary environment without wrappers to get the original bounds
            from boptest_model.rooms_and_ahu_model import load_model_and_params
            from t4b_gym.t4b_gym_env import T4BGymEnv
            
            model = load_model_and_params()
            temp_env = T4BGymEnv(
                model=model, 
                io_config_file=POLICY_CONFIG_PATH,
                start_time=datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen")),
                end_time=datetime.datetime(year=2024, month=1, day=15, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen")),
                episode_length=int(3600*24*5 / 600),
                random_start=True, 
                excluding_periods=None, 
                forecast_horizon=50,
                step_size=600,
                warmup_period=0
            )
            
            # Get original action bounds
            act_low = temp_env.action_space.low
            act_high = temp_env.action_space.high
            
            # Apply the same normalization as NormalizedActionWrapper: 2*(acts - low)/(high - low) - 1
            normalized_expert_acts = 2 * (expert_acts - act_low) / (act_high - act_low) - 1
            
            # Clip to [-1, 1] bounds (same as wrapper)
            normalized_expert_acts = np.clip(normalized_expert_acts, -1, 1)
            
            print(f"Normalized expert actions to range: [{normalized_expert_acts.min():.3f}, {normalized_expert_acts.max():.3f}]")
            
            # Store the normalized expert actions
            self.expert_actions = normalized_expert_acts
            self.training_phase = 'bc_fine_tune'
            
            # Close temporary environment
            temp_env.close()
            
            print(f"Successfully loaded {len(self.expert_actions)} expert action timesteps for imitation learning")

    # Create environment with conditional normalization
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
    
    # Apply normalization wrappers conditionally
    if not use_autoencoder:
        # Apply observation normalization only when NOT using autoencoder
        env = NormalizedObservationWrapper(env)
        print("Applied observation normalization wrapper")
    else:
        print("Skipped observation normalization (autoencoder will handle it)")
    
    # Always apply action normalization (autoencoder doesn't affect actions)
    env = NormalizedActionWrapper(env)
    print("Applied action normalization wrapper")

    # Apply autoencoder if requested
    if use_autoencoder:
        print(f"Setting up autoencoder with latent_dim={latent_dim}, network_size={network_size}")
        
        # Import autoencoder functions from pretrain script
        from pretrain_with_expert import create_autoencoder_env, diagnose_autoencoder_outputs
        
        # Create autoencoder environment
        env, encoder, decoder = create_autoencoder_env(
            env, 
            network_size=network_size, 
            latent_dim=latent_dim
        )
        
        print(f"Observation space compressed to {latent_dim} dimensions")

    # Modify the environment to include the callback
    env = Monitor(env=env, filename=os.path.join(log_dir,'monitor.csv'))

    if test_model_flag:
        model_path = os.path.join(log_dir, "ppo_model.zip")
        model = PPO.load(model_path, env=env, device=device)
        print(f"Training steps: {model.num_timesteps}")
        test_model(env, model)
        return

    # Load pretrained behavioral cloning model if requested
    if load_pretrained_bc:
        bc_model_path = os.path.join(os.path.dirname(__file__), "ppo_pretrained_bc.zip")
        if os.path.exists(bc_model_path):
            print(f"Loading pretrained behavioral cloning model from {bc_model_path}")
            model = PPO.load(bc_model_path, env=env, device=device)
            print(f"Loaded pretrained model with {model.num_timesteps} timesteps")
            
            # Set up imitation learning for BC fine-tuning
            print("Setting up imitation learning for BC fine-tuning...")
            env.unwrapped.load_expert_demonstrations()  # Load expert demonstrations
            env.unwrapped.training_phase = 'bc_fine_tune'
            
            # Set lower learning rate for fine-tuning from pretrained model
            fine_tune_lr = 1e-4  # 10x lower than default for fine-tuning
            model.learning_rate = fine_tune_lr
            print(f"Set learning rate to {fine_tune_lr} for fine-tuning from pretrained model")
            
            # Set high verbosity for detailed training output
            #model.target_kl = 0.01
        else:
            raise FileNotFoundError(f"Pretrained behavioral cloning model not found at {bc_model_path}. "
                                  f"Please run the pretraining script first to generate the required model file.")
    else:
        # Create new model
        model = PPO('MlpPolicy', env, verbose=2, gamma=0.99,      
            learning_rate=1e-5, batch_size=int(50), n_steps=int(200),      
            n_epochs=10, clip_range=0.2, max_grad_norm=0.5, tensorboard_log=log_dir, device=device)
        print("Set training verbosity to maximum (2) for detailed progress")

    # Create the callback
    #Disable evaluation for now
    callback = EvalCallback(env, best_model_save_path=log_dir, log_path=log_dir, eval_freq=1000000, n_eval_episodes=5)

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
    # Example usage scenarios:
    
    # 1. Standard PPO training
    # PPO_training(test_model_flag=False, reload_model_flag=False)
    
    # 2. PPO training with autoencoder
    # PPO_training(test_model_flag=False, reload_model_flag=False, use_autoencoder=True, latent_dim=64, network_size='large')
    
    # 3. Load pretrained behavioral cloning model and continue RL fine-tuning
    # PPO_training(test_model_flag=False, reload_model_flag=False, use_autoencoder=True, latent_dim=64, network_size='large', load_pretrained_bc=True)
    
    # 4. Test trained model
    # PPO_training(test_model_flag=True, reload_model_flag=False, use_autoencoder=True, latent_dim=64, network_size='large')
    
    # Default: Standard PPO training
    #PPO_training(test_model_flag=False, reload_model_flag=False)

    #Test a finetuned model with autoencoder
    #PPO_training(test_model_flag=True, reload_model_flag=False, use_autoencoder=True, latent_dim=64, network_size='large')

    # Fine tune a pretrained bc model without autoencoder
    PPO_training(test_model_flag=False, reload_model_flag=False, use_autoencoder=False, load_pretrained_bc=True, total_timesteps=500000)


