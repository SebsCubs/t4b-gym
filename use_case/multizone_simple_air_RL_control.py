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


POLICY_CONFIG_PATH = os.path.join(SCRIPT_DIR, "policy_input_output.json")
device = 'cpu'



def PPO_training(test_model_flag=False, reload_model_flag=False, use_autoencoder=False, 
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
            
            
        def get_reward(self, action, observation):
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
            
            # Total objective (static penalty)
            reward = temp_violation_penalty + coils_power_consumption_penalty + ahu_power_consumption_penalty
            reward = reward/1000 #Making the number smaller for readability
            
            if np.isnan(reward):
                raise ValueError("Reward is not a number")
            
            
            return -reward

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
        model_path = os.path.join(log_dir, "ppo_pretrained_bc.zip")
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
        else:
            raise FileNotFoundError(f"Pretrained behavioral cloning model not found at {bc_model_path}. "
                                  f"Please run the pretraining script first to generate the required model file.")
    else:
        # Create new model
        model = PPO('MlpPolicy', env, verbose=1, gamma=0.99,      
            learning_rate=1e-5, batch_size=int(50), n_steps=int(200),      
            n_epochs=10, clip_range=0.2, max_grad_norm=0.5, tensorboard_log=log_dir, device=device)

    # Create the callback
    #Disable evaluation for now
    callback = EvalCallback(env, best_model_save_path=log_dir, log_path=log_dir, eval_freq=1000000, n_eval_episodes=5)

    # Train the model
    if reload_model_flag:
        model_path = os.path.join(log_dir, "ppo_model.zip")
        model = PPO.load(model_path, env=env, device=device)

        new_logger = configure(log_dir, ['csv'])
        model.set_logger(new_logger)

        model.learn(total_timesteps=10000, callback=callback, reset_num_timesteps=False)
    else:
        new_logger = configure(log_dir, ['csv'])
        model.set_logger(new_logger)

        model.learn(total_timesteps=10000, callback=callback)

    # Save the model
    model.save(os.path.join(log_dir, "ppo_model"))


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
    PPO_training(test_model_flag=False, reload_model_flag=False, use_autoencoder=False, load_pretrained_bc=True)


