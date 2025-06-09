import sys
import os
import datetime
from dateutil.tz import gettz 
from gymnasium.core import Wrapper
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import twin4build as tb 
from tqdm import tqdm
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(MAIN_DIR)

from boptest_model.rooms_and_ahu_model import load_model_and_params

POLICY_CONFIG_PATH = os.path.join(SCRIPT_DIR, "policy_input_output.json")
device = 'cpu'

def get_baseline(model):
        stepSize = 60 #Seconds
        #Define the range of available data
        start_time = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))
        end_time = datetime.datetime(year=2024, month=1, day=15, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))        

        simulator = tb.Simulator()

        simulator.simulate(model, start_time, end_time, stepSize)
      
        plot_results(simulator, save_plots=True)



def test_model(env, model):

        stepSize = 600 #Seconds
        #Define the range of available data
        start_time = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))
        #end_time = datetime.datetime(year=2024, month=1, day=15, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))        
        episode_length = int(3600*24*15 / stepSize)  # 15 days
        warmup_period = 0

        # Set a fixed start time
        if isinstance(env,Wrapper): 
                env.unwrapped.random_start = False
                env.unwrapped.global_start_time = start_time
                env.unwrapped.episode_length = episode_length
                env.unwrapped.warmup_period = warmup_period
        else:
                env.random_start   = False
                env.global_start_time   = start_time
                env.episode_length  = episode_length
                env.warmup_period = warmup_period

            # Reset environment
        obs, _ = env.reset()
        
        # Simulation loop
        done = False
        observations = [obs]
        rewards = []
        print('Simulating...')

        # Create progress bar
        pbar = tqdm(total=episode_length, desc="Simulation Progress")
        
        while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                observations.append(obs)
                rewards.append(reward)
                done = (terminated or truncated)
                pbar.update(1)
        
        pbar.close()
        plot_results(env.unwrapped.simulator, rewards, save_plots=True)

        return observations, rewards

def plot_results(simulator: tb.Simulator, rewards = None, plotting_stepSize=600, save_plots=False):
        # Convert actions and rewards to pandas DataFrames
        if rewards is not None:
                rewards_df = pd.DataFrame(rewards)

        """
        What do I want to plot?
        - Temperature in the rooms
        - Temp setpoints in the rooms
        - CO2 in the rooms
        - Total energy consumption (Coils consumption + Fan consumption)
        - Some actions taken by the model
        - The rewards for the episode
        """
        model_output_points = [
            {
                'component_id': 'core_indoor_temp_sensor',
                'output_value': 'measuredValue'
            },
            {
                'component_id': 'core_co2_sensor',
                'output_value': 'measuredValue'
            },
            {
                'component_id': 'core_temperature_heating_setpoint',
                'output_value': 'scheduleValue'
            },
            {
                'component_id': 'core_temperature_cooling_setpoint',
                'output_value': 'scheduleValue'
            },
            {
                'component_id': 'north_indoor_temp_sensor',
                'output_value': 'measuredValue'
            },
            {
                'component_id': 'north_co2_sensor',
                'output_value': 'measuredValue'
            },
            {
                'component_id': 'north_temperature_heating_setpoint',
                'output_value': 'scheduleValue'
            },
            {
                'component_id': 'north_temperature_cooling_setpoint',
                'output_value': 'scheduleValue'
            },
            {
                'component_id': 'south_indoor_temp_sensor',
                'output_value': 'measuredValue'
            },
            {
                'component_id': 'south_co2_sensor',
                'output_value': 'measuredValue'
            },
            {
                'component_id': 'south_temperature_heating_setpoint',
                'output_value': 'scheduleValue'
            },
            {
                'component_id': 'south_temperature_cooling_setpoint',
                'output_value': 'scheduleValue'
            },
            {
                'component_id': 'east_indoor_temp_sensor',
                'output_value': 'measuredValue'
            },
            {
                'component_id': 'east_co2_sensor',
                'output_value': 'measuredValue'
            },
            {
                'component_id': 'east_temperature_heating_setpoint',
                'output_value': 'scheduleValue'
            },
            {
                'component_id': 'east_temperature_cooling_setpoint',
                'output_value': 'scheduleValue'
            },
            {
                'component_id': 'west_indoor_temp_sensor',
                'output_value': 'measuredValue'
            },
            {
                'component_id': 'west_co2_sensor',
                'output_value': 'measuredValue'
            },
            {
                'component_id': 'west_temperature_heating_setpoint',
                'output_value': 'scheduleValue'
            },
            {
                'component_id': 'west_temperature_cooling_setpoint',
                'output_value': 'scheduleValue'
            },
     
        ]
        
        # Create plots for each room
        rooms = ['core', 'north', 'south', 'east', 'west']
        for room in rooms:
            # Get indices for this room's components
            base_idx = rooms.index(room) * 4
            temp_sensor = model_output_points[base_idx]
            co2_sensor = model_output_points[base_idx + 1]
            heating_setpoint = model_output_points[base_idx + 2]
            cooling_setpoint = model_output_points[base_idx + 3]

            # Get simulation data for temperature and setpoints
            temp_data = simulator.model.components[temp_sensor['component_id']].savedOutput[temp_sensor['output_value']]
            heating_data = simulator.model.components[heating_setpoint['component_id']].savedOutput[heating_setpoint['output_value']]
            cooling_data = simulator.model.components[cooling_setpoint['component_id']].savedOutput[cooling_setpoint['output_value']]
            sim_times = simulator.dateTimeSteps

            # Create temperature plot
            plt.figure(figsize=(12, 6))
            temp_df = pd.Series(data=temp_data, index=sim_times)
            heating_df = pd.Series(data=heating_data, index=sim_times)
            cooling_df = pd.Series(data=cooling_data, index=sim_times)

            # Convert timezone without changing the actual timestamps
            temp_df.index = temp_df.index.tz_convert('Europe/Copenhagen')
            heating_df.index = heating_df.index.tz_convert('Europe/Copenhagen')
            cooling_df.index = cooling_df.index.tz_convert('Europe/Copenhagen')

            # Resample to common timestep
            temp_df = temp_df.resample(pd.Timedelta(seconds=plotting_stepSize)).mean()
            heating_df = heating_df.resample(pd.Timedelta(seconds=plotting_stepSize)).mean()
            cooling_df = cooling_df.resample(pd.Timedelta(seconds=plotting_stepSize)).mean()

            plt.plot(temp_df.index, temp_df.values, label='Indoor Temperature', linewidth=2)
            plt.plot(heating_df.index, heating_df.values, label='Heating Setpoint', linestyle='--', linewidth=2)
            plt.plot(cooling_df.index, cooling_df.values, label='Cooling Setpoint', linestyle='--', linewidth=2)
            
            plt.title(f'{room.capitalize()} Room - Temperature and Setpoints')
            plt.xlabel('Time')
            plt.ylabel('Temperature (°C)')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_plots:
                    os.makedirs('plots', exist_ok=True)
                    plt.savefig(f'plots/{room}_temperature_setpoints.png')
            #plt.show()

            # Create CO2 plot
            plt.figure(figsize=(12, 6))
            co2_data = simulator.model.components[co2_sensor['component_id']].savedOutput[co2_sensor['output_value']]
            co2_df = pd.Series(data=co2_data, index=sim_times)
            co2_df.index = co2_df.index.tz_convert('Europe/Copenhagen')
            co2_df = co2_df.resample(pd.Timedelta(seconds=plotting_stepSize)).mean()

            plt.plot(co2_df.index, co2_df.values, label='CO2 Concentration', linewidth=2)
            plt.title(f'{room.capitalize()} Room - CO2 Concentration')
            plt.xlabel('Time')
            plt.ylabel('CO2 (ppm)')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_plots:
                    os.makedirs('plots', exist_ok=True)
                    plt.savefig(f'plots/{room}_co2.png')
            #plt.show()

        # Calculate the energy consumption
        core_outlet_water_temperature = np.array(simulator.model.components["core_reheat_coil"].savedOutput["outletWaterTemperature"])
        north_outlet_water_temperature = np.array(simulator.model.components["north_reheat_coil"].savedOutput["outletWaterTemperature"])
        east_outlet_water_temperature = np.array(simulator.model.components["east_reheat_coil"].savedOutput["outletWaterTemperature"])
        south_outlet_water_temperature = np.array(simulator.model.components["south_reheat_coil"].savedOutput["outletWaterTemperature"])
        west_outlet_water_temperature = np.array(simulator.model.components["west_reheat_coil"].savedOutput["outletWaterTemperature"])

        inlet_water_temperature = np.array(simulator.model.components["reheat_coils_supply_water_temperature"].savedOutput["measuredValue"])
        
        #rooms water temp difference:
        core_room_water_temp_difference = abs(core_outlet_water_temperature - inlet_water_temperature)
        north_room_water_temp_difference = abs(north_outlet_water_temperature - inlet_water_temperature)
        east_room_water_temp_difference = abs(east_outlet_water_temperature - inlet_water_temperature)
        south_room_water_temp_difference = abs(south_outlet_water_temperature - inlet_water_temperature)
        west_room_water_temp_difference = abs(west_outlet_water_temperature - inlet_water_temperature)

        room_water_temp_difference_penalty = (core_room_water_temp_difference + north_room_water_temp_difference + east_room_water_temp_difference + south_room_water_temp_difference + west_room_water_temp_difference)

        print(f"Sum of room water temp difference: {np.average(room_water_temp_difference_penalty)}")

        #plot the room water temp difference
        plt.figure(figsize=(12, 6))
        plt.plot(sim_times, room_water_temp_difference_penalty, label='Room Water Temp Difference', linewidth=2)
        plt.title('Room Water Temp Difference')
        plt.xlabel('Time')
        plt.ylabel('Water Temp Difference (°C)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        #plt.show()

        # AHU power consumption
        fan_power = np.array(simulator.model.components["vent_power_sensor"].savedOutput["measuredValue"])
        print(f"Fan power: {np.average(fan_power)}")
        supply_cooling_coil_power = np.array(simulator.model.components["supply_cooling_coil"].savedOutput["Power"])
        print(f"Supply cooling coil power: {np.average(supply_cooling_coil_power)}")
        supply_heating_coil_power = np.array(simulator.model.components["supply_heating_coil"].savedOutput["Power"])
        print(f"Supply heating coil power: {np.average(supply_heating_coil_power)}")
        ahu_power_consumption_penalty = fan_power + supply_cooling_coil_power + supply_heating_coil_power
        print(f"AHU power consumption: {np.average(ahu_power_consumption_penalty)}")

        #plot the fan power consumption
        plt.figure(figsize=(12, 6))
        plt.plot(sim_times, fan_power, label='Fan Power Consumption', linewidth=2)
        plt.title('Fan Power Consumption')
        plt.xlabel('Time')
        plt.ylabel('AHU Power Consumption (W)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        #plt.show()

        #plot the supply cooling and heating coil power consumption
        plt.figure(figsize=(12, 6))
        plt.plot(sim_times, supply_cooling_coil_power, label='Supply Cooling Coil Power', linewidth=2)
        plt.plot(sim_times, supply_heating_coil_power, label='Supply Heating Coil Power', linewidth=2)
        plt.title('Supply Coils Power Consumption')
        plt.xlabel('Time')
        plt.ylabel('Power Consumption (W)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        #plt.show()

        #Plot some actions
        #Load the policy_input_output.json file and get the component ids and the signal keys for the actions
        with open(POLICY_CONFIG_PATH, 'r') as f:
                policy_config = json.load(f)

        #Get the component ids and the signal keys for the actions
        component_ids = []
        signal_keys = []
        for component_id, actions in policy_config['actions'].items():
            for action_name, action_config in actions.items():
                component_ids.append(component_id)
                signal_keys.append(action_config['signal_key'])

        #Get the actions from the simulator
        actions = [simulator.model.components[component_id].savedInput[signal_key] for component_id, signal_key in zip(component_ids, signal_keys)]

        #Plot each action separately
        for action, component_id, signal_key in zip(actions, component_ids, signal_keys):
            plt.figure(figsize=(12, 6))
            plt.plot(sim_times, action, label=f'{component_id} - {signal_key}', linewidth=2)
            plt.title(f'Action: {component_id} - {signal_key}')
            plt.xlabel('Time')
            plt.ylabel('Action Value')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            # Format Y-axis to show actual numbers instead of scientific notation
            plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
            plt.tight_layout()
            if save_plots:
                os.makedirs('plots', exist_ok=True)
                plt.savefig(f'plots/action_{component_id}_{signal_key}.png')
            #plt.show()

if __name__ == "__main__":
        model = load_model_and_params()
        get_baseline(model)