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



def PPO_training(test_model_flag=False):
        
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
                core_temperature = self.simulator.model.components["core_indoor_temp_sensor"].output["measuredValue"]
                core_heating_temperature_setpoint = self.simulator.model.components["core_temperature_heating_setpoint"].output["scheduleValue"]
                core_cooling_temperature_setpoint = self.simulator.model.components["core_temperature_cooling_setpoint"].output["scheduleValue"]
                core_temp_set_violation = max(0, core_heating_temperature_setpoint - core_temperature) + max(0, core_temperature - core_cooling_temperature_setpoint)

                north_temperature = self.simulator.model.components["north_indoor_temp_sensor"].output["measuredValue"]
                north_heating_temperature_setpoint = self.simulator.model.components["north_temperature_heating_setpoint"].output["scheduleValue"]
                north_cooling_temperature_setpoint = self.simulator.model.components["north_temperature_cooling_setpoint"].output["scheduleValue"]
                north_temp_set_violation = max(0, north_heating_temperature_setpoint - north_temperature) + max(0, north_temperature - north_cooling_temperature_setpoint)

                east_temperature = self.simulator.model.components["east_indoor_temp_sensor"].output["measuredValue"]
                east_heating_temperature_setpoint = self.simulator.model.components["east_temperature_heating_setpoint"].output["scheduleValue"]
                east_cooling_temperature_setpoint = self.simulator.model.components["east_temperature_cooling_setpoint"].output["scheduleValue"]
                east_temp_set_violation = max(0, east_heating_temperature_setpoint - east_temperature) + max(0, east_temperature - east_cooling_temperature_setpoint)

                south_temperature = self.simulator.model.components["south_indoor_temp_sensor"].output["measuredValue"]
                south_heating_temperature_setpoint = self.simulator.model.components["south_temperature_heating_setpoint"].output["scheduleValue"]
                south_cooling_temperature_setpoint = self.simulator.model.components["south_temperature_cooling_setpoint"].output["scheduleValue"]
                south_temp_set_violation = max(0, south_heating_temperature_setpoint - south_temperature) + max(0, south_temperature - south_cooling_temperature_setpoint)

                west_temperature = self.simulator.model.components["west_indoor_temp_sensor"].output["measuredValue"]
                west_heating_temperature_setpoint = self.simulator.model.components["west_temperature_heating_setpoint"].output["scheduleValue"]
                west_cooling_temperature_setpoint = self.simulator.model.components["west_temperature_cooling_setpoint"].output["scheduleValue"]
                west_temp_set_violation = max(0, west_heating_temperature_setpoint - west_temperature) + max(0, west_temperature - west_cooling_temperature_setpoint)

                temp_violation_penalty = 10 * (core_temp_set_violation + north_temp_set_violation + east_temp_set_violation + south_temp_set_violation + west_temp_set_violation)

                #power consumption penalty
                core_outlet_water_temperature = self.simulator.model.components["core_reheat_coil"].output["outletWaterTemperature"]
                north_outlet_water_temperature = self.simulator.model.components["north_reheat_coil"].output["outletWaterTemperature"]
                east_outlet_water_temperature = self.simulator.model.components["east_reheat_coil"].output["outletWaterTemperature"]
                south_outlet_water_temperature = self.simulator.model.components["south_reheat_coil"].output["outletWaterTemperature"]
                west_outlet_water_temperature = self.simulator.model.components["west_reheat_coil"].output["outletWaterTemperature"]

                inlet_water_temperature = self.simulator.model.components["reheat_coils_supply_water_temperature"].output["measuredValue"]
                
                #rooms water temp difference:
                core_room_water_temp_difference = abs(core_outlet_water_temperature - inlet_water_temperature)
                north_room_water_temp_difference = abs(north_outlet_water_temperature - inlet_water_temperature)
                east_room_water_temp_difference = abs(east_outlet_water_temperature - inlet_water_temperature)
                south_room_water_temp_difference = abs(south_outlet_water_temperature - inlet_water_temperature)
                west_room_water_temp_difference = abs(west_outlet_water_temperature - inlet_water_temperature)

                room_water_temp_difference_penalty = (core_room_water_temp_difference + north_room_water_temp_difference + east_room_water_temp_difference + south_room_water_temp_difference + west_room_water_temp_difference)
                
                #ahu power consumption penalty
                fan_power = self.simulator.model.components["vent_power_sensor"].output["measuredValue"]
                supply_cooling_coil_power = self.simulator.model.components["supply_cooling_coil"].output["Power"]
                supply_heating_coil_power = self.simulator.model.components["supply_heating_coil"].output["Power"]
                ahu_power_consumption_penalty = fan_power + supply_cooling_coil_power + supply_heating_coil_power
                #reward
                reward = - temp_violation_penalty - room_water_temp_difference_penalty - ahu_power_consumption_penalty

                if reward > 0:
                    print("Reward is positive")

                if np.isnan(reward):
                    raise ValueError("Reward is not a number")
                #print(f"step: {self.simulator.current_step}, reward: {reward}")
                return reward

        env = T4BGymEnvCustomReward(                 
                 model = model, 
                 io_config_file = POLICY_CONFIG_PATH,
                 start_time = start_time,
                 end_time = end_time,
                 episode_length= int(3600*24*5 / stepSize),  # 5 days
                 random_start=True, 
                 excluding_periods=None, 
                 forecast_horizon=40,
                 step_size=stepSize,
                 warmup_period=0) 
  
        env = NormalizedObservationWrapper(env)
        env = NormalizedActionWrapper(env)  

        # Modify the environment to include the callback
        env = Monitor(env=env, filename=os.path.join(log_dir,'monitor.csv'))

        if test_model_flag:
            model_path = os.path.join(log_dir, "best_model.zip")
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
        callback = EvalCallback(env, best_model_save_path=log_dir, log_path=log_dir, eval_freq=1000, n_eval_episodes=5)

        # Train the model
        model.learn(total_timesteps=100000, callback=callback)

        # Save the model
        model.save("ppo_model")




if __name__ == "__main__":
    PPO_training(test_model_flag=False)
