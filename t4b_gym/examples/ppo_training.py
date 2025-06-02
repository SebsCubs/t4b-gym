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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(SCRIPT_DIR)  # This will point to t4b_gym folder
#Add testing directory to sys.path
sys.path.append(MAIN_DIR)
from t4b_gym_env import T4BGymEnv

# Configure logging to write to a file
log_file = os.path.join(SCRIPT_DIR, 'ppo_training.log')
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file)
    ]
)

POLICY_CONFIG_PATH = os.path.join(MAIN_DIR, "testing", "policy_input_output.json")
OUTDOOR_DATA_PATH = os.path.join(MAIN_DIR, "testing", "outdoor_data.csv")


def fcn(self):
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
            def get_reward(self, action, observation):
                return 0

        env = T4BGymEnvCustomReward(                 
                 model = model, 
                 io_config_file = POLICY_CONFIG_PATH,
                 start_time = start_time,
                 end_time = end_time,
                 episode_length=None, 
                 random_start=False, 
                 excluding_periods=None, 
                 forecast_horizon=20,
                 step_size=stepSize,
                 warmup_period=0) 
  
