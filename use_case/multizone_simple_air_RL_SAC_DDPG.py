"""
This script uses the t4b_gym environment with the t4b model of the BOPTEST model multizone_simple_air 
It defines a custom reward function and uses the SAC and DDPG algorithms to control different setpoints of the model. 

USAGE EXAMPLES:

# Train a new SAC agent:
python use_case/multizone_simple_air_RL_SAC_DDPG.py --algo sac

# Train a new DDPG agent:
python use_case/multizone_simple_air_RL_SAC_DDPG.py --algo ddpg

# Test a trained SAC agent:
python use_case/multizone_simple_air_RL_SAC_DDPG.py --algo sac --test

# Test a trained DDPG agent:
python use_case/multizone_simple_air_RL_SAC_DDPG.py --algo ddpg --test

# Continue training a SAC agent from the last checkpoint:
python use_case/multizone_simple_air_RL_SAC_DDPG.py --algo sac --continue

# Continue training a DDPG agent from the last checkpoint:
python use_case/multizone_simple_air_RL_SAC_DDPG.py --algo ddpg --continue

# Continue training and then test the SAC agent:
python use_case/multizone_simple_air_RL_SAC_DDPG.py --algo sac --continue --test

# Continue training and then test the DDPG agent:
python use_case/multizone_simple_air_RL_SAC_DDPG.py --algo ddpg --continue --test

"""

import twin4build as tb
import datetime
import twin4build.examples.utils as utils
from dateutil.tz import gettz 
import sys
import os
import logging
from stable_baselines3 import SAC, DDPG
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import numpy as np
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(MAIN_DIR)
from t4b_gym.t4b_gym_env import T4BGymEnv, NormalizedObservationWrapper, NormalizedActionWrapper
from boptest_model.rooms_and_ahu_model import load_model_and_params
from use_case.model_eval import test_model

SAC_LOG_DIR = os.path.join(SCRIPT_DIR, 'logs_sac')
DDPG_LOG_DIR = os.path.join(SCRIPT_DIR, 'logs_ddpg')
os.makedirs(SAC_LOG_DIR, exist_ok=True)
os.makedirs(DDPG_LOG_DIR, exist_ok=True)

POLICY_CONFIG_PATH = os.path.join(SCRIPT_DIR, "policy_input_output.json")
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

def get_monitor_and_progress_filenames(log_dir, continue_flag):
    if continue_flag:
        # Find a unique filename for continued runs
        monitor_files = glob.glob(os.path.join(log_dir, 'monitor_continued*.csv'))
        progress_files = glob.glob(os.path.join(log_dir, 'progress_continued*.csv'))
        monitor_idx = len(monitor_files) + 1
        progress_idx = len(progress_files) + 1
        monitor_file = os.path.join(log_dir, f'monitor_continued{monitor_idx}.csv')
        progress_file = os.path.join(log_dir, f'progress_continued{progress_idx}.csv')
    else:
        monitor_file = os.path.join(log_dir, 'monitor.csv')
        progress_file = os.path.join(log_dir, 'progress.csv')
    return monitor_file, progress_file


def SAC_training(test_model_flag=False, reload_model_flag=False, continue_flag=False):
    stepSize = 600
    start_time = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))
    end_time = datetime.datetime(year=2024, month=1, day=15, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))
    monitor_file, progress_file = get_monitor_and_progress_filenames(SAC_LOG_DIR, continue_flag)
    env = get_custom_env(stepSize, start_time, end_time)
    env = Monitor(env=env, filename=monitor_file)
    model_path = os.path.join(SAC_LOG_DIR, "sac_model.zip")
    if test_model_flag:
        model = SAC.load(model_path, env=env, device=device)
        print(f"Training steps: {model.num_timesteps}")
        test_model(env, model)
        return
    if reload_model_flag or continue_flag:
        model = SAC.load(model_path, env=env, device=device)
        new_logger = configure(SAC_LOG_DIR, ['csv'])
        model.set_logger(new_logger)
        callback = EvalCallback(env, best_model_save_path=SAC_LOG_DIR, log_path=SAC_LOG_DIR, eval_freq=10000, n_eval_episodes=3)
        model.learn(total_timesteps=1000000, callback=callback, reset_num_timesteps=False)
    else:
        model = SAC('MlpPolicy', env, verbose=1, gamma=0.99,
            learning_rate=1e-5, batch_size=50, buffer_size=100000, tau=0.005,
            train_freq=1, gradient_steps=1, tensorboard_log=SAC_LOG_DIR, device=device)
        new_logger = configure(SAC_LOG_DIR, ['csv'])
        model.set_logger(new_logger)
        callback = EvalCallback(env, best_model_save_path=SAC_LOG_DIR, log_path=SAC_LOG_DIR, eval_freq=10000, n_eval_episodes=3)
        model.learn(total_timesteps=1000000, callback=callback)
    model.save(os.path.join(SAC_LOG_DIR, "sac_model"))


def DDPG_training(test_model_flag=False, reload_model_flag=False, continue_flag=False):
    stepSize = 600
    start_time = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))
    end_time = datetime.datetime(year=2024, month=1, day=15, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))
    monitor_file, progress_file = get_monitor_and_progress_filenames(DDPG_LOG_DIR, continue_flag)
    env = get_custom_env(stepSize, start_time, end_time)
    env = Monitor(env=env, filename=monitor_file)
    model_path = os.path.join(DDPG_LOG_DIR, "ddpg_model.zip")
    if test_model_flag:
        model = DDPG.load(model_path, env=env, device=device)
        print(f"Training steps: {model.num_timesteps}")
        test_model(env, model)
        return
    if reload_model_flag or continue_flag:
        model = DDPG.load(model_path, env=env, device=device)
        new_logger = configure(DDPG_LOG_DIR, ['csv'])
        model.set_logger(new_logger)
        callback = EvalCallback(env, best_model_save_path=DDPG_LOG_DIR, log_path=DDPG_LOG_DIR, eval_freq=10000, n_eval_episodes=3)
        model.learn(total_timesteps=1000000, callback=callback, reset_num_timesteps=False)
    else:
        model = DDPG('MlpPolicy', env, verbose=1, gamma=0.99,
            learning_rate=1e-5, batch_size=50, buffer_size=100000, tau=0.005,
            train_freq=1, gradient_steps=1, tensorboard_log=DDPG_LOG_DIR, device=device)
        new_logger = configure(DDPG_LOG_DIR, ['csv'])
        model.set_logger(new_logger)
        callback = EvalCallback(env, best_model_save_path=DDPG_LOG_DIR, log_path=DDPG_LOG_DIR, eval_freq=10000, n_eval_episodes=3)
        model.learn(total_timesteps=1000000, callback=callback)
    model.save(os.path.join(DDPG_LOG_DIR, "ddpg_model"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train RL agent with SAC or DDPG.")
    parser.add_argument('--algo', choices=['sac', 'ddpg'], required=True, help='Algorithm to use: sac or ddpg')
    parser.add_argument('--test', action='store_true', help='Test the trained model instead of training')
    parser.add_argument('--continue', dest='continue_flag', action='store_true', help='Continue training from a previously saved model')
    args = parser.parse_args()
    if args.algo == 'sac':
        SAC_training(test_model_flag=args.test, continue_flag=args.continue_flag)
    elif args.algo == 'ddpg':
        DDPG_training(test_model_flag=args.test, continue_flag=args.continue_flag) 