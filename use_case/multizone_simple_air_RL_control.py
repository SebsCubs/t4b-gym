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



def PPO_training(test_model_flag=False, reload_model_flag=False):
        
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
                    zone_violation = heating_violation**2 + cooling_violation**2
                    temp_violations.append(zone_violation)
                
                # Balanced temperature penalty
                temp_violation_penalty = 100 * sum(temp_violations)  # Reduced from 10000
                
                # Water temperature differences (energy efficiency proxy)
                inlet_water_temp = self.simulator.model.components["reheat_coils_supply_water_temperature"].output["measuredValue"]
                water_temp_differences = []
                
                for zone in zones:
                    outlet_temp = self.simulator.model.components[f"{zone}_reheat_coil"].output["outletWaterTemperature"]
                    temp_diff = abs(outlet_temp - inlet_water_temp)
                    water_temp_differences.append(temp_diff)
                
                # Moderate water temperature penalty
                room_water_temp_difference_penalty = 10 * sum(water_temp_differences)
                
                # AHU power consumption
                fan_power = self.simulator.model.components["vent_power_sensor"].output["measuredValue"]
                supply_cooling_coil_power = self.simulator.model.components["supply_cooling_coil"].output["Power"]
                supply_heating_coil_power = self.simulator.model.components["supply_heating_coil"].output["Power"]
                ahu_power_consumption_penalty = 0.01 * (fan_power + supply_cooling_coil_power + supply_heating_coil_power)
                
                # Total objective (static penalty)
                reward = temp_violation_penalty + room_water_temp_difference_penalty + ahu_power_consumption_penalty
                
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

        # Modify the environment to include the callback
        env = Monitor(env=env, filename=os.path.join(log_dir,'monitor.csv'))

        if test_model_flag:
            model_path = os.path.join(log_dir, "b_1000k.zip")
            model = PPO.load(model_path, env=env, device=device)
            #print training steps
            print(f"Training steps: {model.num_timesteps}")
            test_model(env, model)
            return

        # Save the model
        model = PPO('MlpPolicy', env, verbose=1, gamma=0.99,      
            learning_rate=1e-5, batch_size=int(50), n_steps=int(200),      
            n_epochs=10, clip_range=0.2, max_grad_norm=0.5, tensorboard_log=log_dir, device=device)

        # Create the callback
        callback = EvalCallback(env, best_model_save_path=log_dir, log_path=log_dir, eval_freq=5000, n_eval_episodes=5)

        # Train the model
        if reload_model_flag:
            model_path = os.path.join(log_dir, "500k.zip")
            model = PPO.load(model_path, env=env, device=device)

            new_logger = configure(log_dir, ['csv'])
            model.set_logger(new_logger)

            model.learn(total_timesteps=1000000, callback=callback, reset_num_timesteps=False)
        else:
            new_logger = configure(log_dir, ['csv'])
            model.set_logger(new_logger)

            model.learn(total_timesteps=1000000, callback=callback)

        # Save the model
        model.save(os.path.join(log_dir, "ppo_model"))


if __name__ == "__main__":
    PPO_training(test_model_flag=False, reload_model_flag=False)
