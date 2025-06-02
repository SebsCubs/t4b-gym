import twin4build as tb
import datetime
import twin4build.examples.utils as utils
import torch.nn as nn
import torch
import json
from dateutil.tz import gettz 
import twin4build.utils.plot.plot as plot
import twin4build.utils.input_output_types as tps
import unittest
import gymnasium as gym
import numpy as np
from datetime import timezone, timedelta
import sys
import os
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(SCRIPT_DIR)  # This will point to t4b_gym folder
#Add testing directory to sys.path
sys.path.append(MAIN_DIR)
from t4b_gym_env import T4BGymEnv, NormalizedObservationWrapper, NormalizedActionWrapper
from stable_baselines3.common.monitor import Monitor
# Configure logging to write to a file
log_dir = os.path.join(SCRIPT_DIR, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'ppo_training.log')
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file)
    ]
)

POLICY_CONFIG_PATH = os.path.join(SCRIPT_DIR, "policy_input_output.json")
OUTDOOR_DATA_PATH = os.path.join(MAIN_DIR, "testing", "outdoor_data.csv")
device = 'cpu'

def fcn(self):


    self.remove_component(self.components["020B_occupancy_profile"])

    occupancy_schedule = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 0,
            "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_start_hour": [6, 7, 8, 12, 14, 16, 18],
            "ruleset_end_hour": [7, 8, 12, 14, 16, 18, 22],
            "ruleset_value": [3, 5, 20, 25, 27, 7, 3]},
        add_noise=True,
        saveSimulationResult=True,
        id="020B_occupancy_profile")
    
    self.add_connection(occupancy_schedule, self.components["[020B][020B_space_heater]"], "scheduleValue", "numberOfPeople")

        
    supply_water_schedule = tb.ScheduleSystem(
    weekDayRulesetDict = {
        "ruleset_default_value": 60,
        "ruleset_start_minute": [],
        "ruleset_end_minute": [],
        "ruleset_start_hour": [],
        "ruleset_end_hour": [],
        "ruleset_value": []
    },
    id="supply_water_schedule"
    )
    self.add_connection(supply_water_schedule, self.components["[020B][020B_space_heater]"], "scheduleValue", "supplyWaterTemperature") # Add missing input
    self.components["020B_temperature_sensor"].filename = utils.get_path(["parameter_estimation_example", "temperature_sensor.csv"])
    self.components["020B_co2_sensor"].filename = utils.get_path(["parameter_estimation_example", "co2_sensor.csv"])
    self.components["020B_valve_position_sensor"].filename = utils.get_path(["parameter_estimation_example", "valve_position_sensor.csv"])
    self.components["020B_damper_position_sensor"].filename = utils.get_path(["parameter_estimation_example", "damper_position_sensor.csv"])
    self.components["BTA004"].filename = utils.get_path(["parameter_estimation_example", "supply_air_temperature.csv"])
    self.components["020B_co2_setpoint"].weekDayRulesetDict = {"ruleset_default_value": 900,
                                                                    "ruleset_start_minute": [],
                                                                    "ruleset_end_minute": [],
                                                                    "ruleset_start_hour": [],
                                                                    "ruleset_end_hour": [],
                                                                    "ruleset_value": []}
    self.components["020B_temperature_heating_setpoint"].useFile = True
    self.components["020B_temperature_heating_setpoint"].filename = utils.get_path(["parameter_estimation_example", "temperature_heating_setpoint.csv"])
    self.components["outdoor_environment"].filename = utils.get_path(["parameter_estimation_example", "outdoor_environment.csv"])

    # Add all parameters to the space heater model
    space_heater = self.components["[020B][020B_space_heater]"]
    space_heater.C_wall = 1702351.9925186099
    space_heater.C_air = 1687166.9144215034
    space_heater.C_boundary = 19703.38320099749
    space_heater.R_out = 0.023593210668595075
    space_heater.R_in = 0.017637265946071935
    space_heater.R_boundary = 0.0009900000001525845
    space_heater.f_wall = 0.2177516281153
    space_heater.f_air = 0.6273589585754279
    space_heater.Q_occ_gain = 66.51910539690117
    space_heater.T_boundary = 21.10859741873187
    space_heater.a = 2.750896350625229
    space_heater.a = 5.410976498514536
    space_heater.infiltration = 0.0886639458678181
    space_heater.CO2_occ_gain = 9.13404775138699e-06
    space_heater.Q_flow_nominal_sh = 422.77276126759847
    space_heater.n_sh = 1.516391815540631
    space_heater.C_supply = 363.3629915317287
    space_heater.CO2_start = 400
    space_heater.fraRad_sh = 0.35
    space_heater.T_a_nominal_sh = 333.15
    space_heater.T_b_nominal_sh = 303.15
    space_heater.TAir_nominal_sh = 293.15
        
    heating_controller = self.components["020B_temperature_heating_controller"]
    co2_controller = self.components["020B_co2_controller"]
    space_heater_valve = self.components["020B_space_heater_valve"]
    supply_damper = self.components["020B_room_supply_damper"]
    exhaust_damper = self.components["020B_room_exhaust_damper"]

    heating_controller.kp = 0.0008094283781759571
    heating_controller.Ti = 5.6327636661197555
    co2_controller.kp = 6.175444394258585e-05
    co2_controller.Ti = 7.1632486356880065

    space_heater_valve.dpFixed_nominal = 0.010280806699151766
    space_heater_valve.m_flow_nominal = 0.0061715357510265575

    supply_damper.a = 2.750896350625229
    exhaust_damper.a = 5.410976498514536


def PPO_training():
        
        # Create a new model
        model = tb.Model(id="mymodel")
        filename = utils.get_path(["parameter_estimation_example", "one_room_example_model.xlsm"])
        model.load(semantic_model_filename=filename, fcn=fcn, verbose=False)

        stepSize = 600 #Seconds
        #Define the range of available data
        start_time = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))
        end_time = datetime.datetime(year=2024, month=1, day=15, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))        

        class T4BGymEnvCustomReward(T4BGymEnv):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.previous_objective = 0.0
                
            def get_reward(self, action, observation):
                # Difference between indoor temperature and setpoint
                indoor_temperature = self.simulator.model.components["[020B][020B_space_heater]"].output["indoorTemperature"]
                indoor_temperature_setpoint = self.simulator.model.components["020B_temperature_heating_setpoint"].output["scheduleValue"]
                temperature_difference = abs(indoor_temperature - indoor_temperature_setpoint)

                # Difference between CO2 concentration and setpoint
                co2_concentration = self.simulator.model.components["020B_co2_sensor"].output["measuredValue"]
                co2_concentration_setpoint = self.simulator.model.components["020B_co2_setpoint"].output["scheduleValue"]
                # Only penalize when CO2 is above setpoint
                co2_difference = max(0, co2_concentration - co2_concentration_setpoint)

                # Power consumption of space heater
                space_heater_power = self.simulator.model.components["[020B][020B_space_heater]"].output["spaceHeaterPower"]

                # Calculate current objective value (integrand)
                current_objective = (
                    0.01 * space_heater_power +  # Energy cost
                    10 * temperature_difference +  # Comfort violation
                    10 * co2_difference  # IAQ violation
                )

                # Calculate reward as negative change in objective
                reward = -(current_objective - self.previous_objective)
                
                # Store current objective for next step
                self.previous_objective = current_objective

                return reward

        env = T4BGymEnvCustomReward(                 
                 model = model, 
                 io_config_file = POLICY_CONFIG_PATH,
                 start_time = start_time,
                 end_time = end_time,
                 episode_length= int(3600*24*3 / stepSize),  # 3 days
                 random_start=True, 
                 excluding_periods=None, 
                 forecast_horizon=10,
                 step_size=stepSize,
                 warmup_period=0) 
  
        env = NormalizedObservationWrapper(env)
        env = NormalizedActionWrapper(env)  

        # Modify the environment to include the callback
        env = Monitor(env=env, filename=os.path.join(log_dir,'monitor.csv'))

        # Save the model
        model = PPO('MlpPolicy', env, verbose=1, gamma=0.99,      
            learning_rate=5e-4, batch_size=int(50), n_steps=int(200),      
            n_epochs=10, clip_range=0.2, tensorboard_log=log_dir, device=device)

        # Create the callback
        callback = EvalCallback(env, best_model_save_path=log_dir, log_path=log_dir, eval_freq=1000, n_eval_episodes=5)

        # Train the model
        model.learn(total_timesteps=10000, callback=callback)

        # Save the model
        model.save("ppo_model")

        # Close the environment
        env.close()

if __name__ == "__main__":
    PPO_training()
