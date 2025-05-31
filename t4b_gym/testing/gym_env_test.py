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

# Configure logging to write to a file
log_file = os.path.join(os.path.dirname(__file__), 'simulation_tests.log')
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        #logging.StreamHandler()  # Keep console output as well
    ]
)

uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
file_path = os.path.join(uppath(os.path.abspath(__file__), 3), "t4b_gym")
sys.path.append(file_path)
from t4b_gym_env import T4BGymEnv, GymSimulator
from tqdm import tqdm

def fcn(self):
    ##############################################################
    ################## First, define components ##################
    ##############################################################
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
        id="Occupancy schedule")
    
    co2_setpoint_schedule = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 900,
            "ruleset_start_minute": [],
            "ruleset_end_minute": [],
            "ruleset_start_hour": [],
            "ruleset_end_hour": [],
            "ruleset_value": []},
        saveSimulationResult=True,
        id="CO2 setpoint schedule")

    co2_controller = tb.PIDControllerSystem(
        K_p=-0.001,
        K_i=-0.001,
        K_d=0,
        saveSimulationResult=True,
        id="CO2 controller")

    supply_damper = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=1.6),
        a=5,
        saveSimulationResult=True,
        id="Supply damper")

    return_damper = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=1.6),
        a=5,
        saveSimulationResult=True,
        id="Return damper")

    space = tb.BuildingSpaceCo2System(
        airVolume=466.54,
        outdoorCo2Concentration=500,
        infiltration=0.005,
        generationCo2Concentration=0.0042*1000*1.225,
        saveSimulationResult=True,
        id="Space")
    
    outdoor_environment = tb.OutdoorEnvironmentSystem(
        filename=r"C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/outdoor_env_data.csv",
        saveSimulationResult=True,
        id="Outdoor_environment")
    
    outdoor_temp_sensor = tb.SensorSystem(
        saveSimulationResult=True,
        id="Outdoor_temp_sensor")
    global_irradiation = tb.SensorSystem(
        saveSimulationResult=True,
        id="globalIrradiance")

    #################################################################
    ################## Add connections to the model #################
    #################################################################
    self.add_connection(co2_controller, supply_damper,
                         "inputSignal", "damperPosition")
    self.add_connection(co2_controller, return_damper,
                         "inputSignal", "damperPosition")
    self.add_connection(supply_damper, space,
                         "airFlowRate", "supplyAirFlowRate")
    self.add_connection(return_damper, space,
                         "airFlowRate", "returnAirFlowRate")
    self.add_connection(occupancy_schedule, space,
                         "scheduleValue", "numberOfPeople")
    self.add_connection(space, co2_controller,
                         "indoorCo2Concentration", "actualValue")
    self.add_connection(co2_setpoint_schedule, co2_controller,
                         "scheduleValue", "setpointValue")
    self.add_connection(outdoor_environment, outdoor_temp_sensor,
                         "outdoorTemperature", "outdoorTemperature")
    self.add_connection(outdoor_environment, global_irradiation,
                         "globalIrradiation", "globalIrradiation")
    

def load_test_model():
    model = tb.Model(id="simple_co2_control")
    model.load(fcn=fcn, verbose=False)
    return model


class TestCustomGymSimulator(unittest.TestCase):
    def setUp(self):
        self.model = load_test_model()
        self.simulator = GymSimulator(self.model, enable_logging=True)
        self.stepSize = 600 #Seconds
        self.start_time = datetime.datetime(year=2024, month=1, day=10, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))
        self.end_time = datetime.datetime(year=2024, month=1, day=12, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))

        self.simulator.initialize_simulation(self.start_time, self.end_time, self.stepSize)
        self.show_progress_bar = True

    def test_non_controlled_simulation(self):
        """Test that the simulation runs without any control actions"""
        #SecondTime is the timesteps in seconds, dateTime is the timesteps in datetime
        #They are generated by the initialize_simulation method
        if self.show_progress_bar:
            for self.secondTime, self.dateTime in tqdm(zip(self.simulator.secondTimeSteps,self.simulator.dateTimeSteps), total=len(self.simulator.dateTimeSteps)):
                self.simulator.step_simulation()
        else:
            for self.secondTime, self.dateTime in zip(self.simulator.secondTimeSteps,self.simulator.dateTimeSteps):
                self.simulator.step_simulation()
        #Check that the current_step is correct
        self.assertEqual(self.simulator.current_step, len(self.simulator.dateTimeSteps))

    def test_controlled_simulation(self):
        """Test that the simulation runs with control actions"""
        #Create a controlled simulation
        self.simulator.add_control_input("Supply damper", "damperPosition")
        self.simulator.add_control_input("Return damper", "damperPosition")
        self.simulator.add_observation_output("CO2 setpoint schedule", "scheduleValue")
        self.simulator.add_observation_output("Space", "indoorCo2Concentration")
        
        def control_action_function(observation):
            #Get the setpoint value
            setpoint_value = observation["CO2 setpoint schedule"]
            #Get the actual value
            actual_value = observation["Space"]
            #Calculate the error
            error = setpoint_value - actual_value
            #Calculate the control action with a simple P controller
            control_action = error * 0.001
            #Return the control action with the correct format
            return {"Supply damper": {"damperPosition": control_action}, "Return damper": {"damperPosition": control_action}}

        #Run the simulation
        if self.show_progress_bar:
            for self.secondTime, self.dateTime in tqdm(zip(self.simulator.secondTimeSteps,self.simulator.dateTimeSteps), total=len(self.simulator.dateTimeSteps)):
                control_action = control_action_function(self.simulator.get_observations())
                self.simulator.step_simulation(control_action)
        else:
            for self.secondTime, self.dateTime in zip(self.simulator.secondTimeSteps,self.simulator.dateTimeSteps):
                control_action = control_action_function(self.simulator.get_observations())
                self.simulator.step_simulation(control_action)
        
        #Check that the current_step is correct
        self.assertEqual(self.simulator.current_step, len(self.simulator.dateTimeSteps))

class TestT4BGymEnv(unittest.TestCase):
    def setUp(self):
        # Load any T4B model for testing
        self.model = load_test_model()  # You define this based on your needs
        # Create environment

        self.stepSize = 600 #Seconds
        #Define the range of available data
        self.start_time = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))
        self.end_time = datetime.datetime(year=2024, month=1, day=15, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))        
        self.show_progress_bar = True

        self.env = T4BGymEnv(                 
                 model = self.model, 
                 io_config_file = r"C:\Users\asces\OneDriveUni\Projects\RL_control\t4b_gym\testing\policy_input_output.json",
                 start_time = self.start_time,
                 end_time = self.end_time,
                 episode_length=None, #Not implemented yet
                 random_start=False, #Not implemented yet
                 excluding_periods=None, #Not implemented yet
                 forecast_horizon=20,
                 step_size=self.stepSize,
                 warmup_period=0) #Not implemented yet

    def test_space_creation(self):
        """Test if observation and action spaces are created correctly"""
        io_config = self.env.io_config_dict
        
        # Test observation space
        # The observation space should be a single Box space with dimensions matching the total number of observations
        self.assertIsInstance(self.env.observation_space, gym.spaces.Box)
        
        # Calculate expected number of dimensions
        expected_dims = 0
        
        # Count observations from components
        for key, config in io_config['observations'].items():
            expected_dims += 1  # One dimension per observation
        
        # Count time embeddings (each has sin and cos components)
        if 'time_embeddings' in io_config:
            for key, config in io_config['time_embeddings'].items():
                expected_dims += 2  # Two dimensions per time embedding (sin and cos)
        
        # Count forecasts
        if 'forecasts' in io_config:
            for key, config in io_config['forecasts'].items():
                expected_dims += (self.env.forecast_horizon + 1)  # Current value + forecast_horizon future values
        
        # Check total dimensions
        self.assertEqual(self.env.observation_space.shape[0], expected_dims)
        
        # Check bounds for each dimension
        current_dim = 0
        
        # Check component observation bounds
        for key, config in io_config['observations'].items():
            self.assertEqual(self.env.observation_space.low[current_dim], config['min'])
            self.assertEqual(self.env.observation_space.high[current_dim], config['max'])
            current_dim += 1
        
        # Check time embedding bounds
        if 'time_embeddings' in io_config:
            for key, config in io_config['time_embeddings'].items():
                # Both sin and cos components are bounded by [-1, 1]
                self.assertEqual(self.env.observation_space.low[current_dim], -1)
                self.assertEqual(self.env.observation_space.high[current_dim], 1)
                current_dim += 1
                self.assertEqual(self.env.observation_space.low[current_dim], -1)
                self.assertEqual(self.env.observation_space.high[current_dim], 1)
                current_dim += 1
        
        # Check forecast bounds
        if 'forecasts' in io_config:
            for key, config in io_config['forecasts'].items():
                # Check bounds for each forecast value (current + future)
                for _ in range(self.env.forecast_horizon + 1):
                    self.assertEqual(self.env.observation_space.low[current_dim], config['min'])
                    self.assertEqual(self.env.observation_space.high[current_dim], config['max'])
                    current_dim += 1

        # Test action space
        # The action space should be a single Box space with dimensions matching the total number of actions
        self.assertIsInstance(self.env.action_space, gym.spaces.Box)
        
        # Calculate expected number of action dimensions
        expected_action_dims = len(io_config['actions'])
        self.assertEqual(self.env.action_space.shape[0], expected_action_dims)
        
        # Check action bounds
        current_dim = 0
        for comp_id, config in io_config['actions'].items():
            self.assertEqual(self.env.action_space.low[current_dim], config['min'])
            self.assertEqual(self.env.action_space.high[current_dim], config['max'])
            current_dim += 1
 
    def test_get_observations(self):
        """Test the _get_obs method for different observation types"""
        # Reset environment to get initial state
        self.env.reset()
        
        # Get observations
        obs = self.env._get_obs()
        
        # Test 1: Check array format and length
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape[0], self.env.observation_space.shape[0])
        
        # Test 2: Check component observations
        model_obs = self.env.simulator.get_observations()
        current_dim = 0
        for component_id, outputs in self.env.simulator.observation_outputs.items():
            for output_name in outputs:
                expected_value = model_obs[component_id]
                self.assertEqual(obs[current_dim], expected_value)
                current_dim += 1
        
        # Test 3: Check time embeddings if present
        if 'time_embeddings' in self.env.io_config_dict:
            current_time = self.env.simulator.dateTime
            time_embedding_keys = list(self.env.io_config_dict['time_embeddings'].keys())
            
            if "time_of_day" in time_embedding_keys:
                expected_sin = np.sin(2 * np.pi * current_time.hour / 24)
                expected_cos = np.cos(2 * np.pi * current_time.hour / 24)
                self.assertAlmostEqual(obs[current_dim], expected_sin)
                self.assertAlmostEqual(obs[current_dim + 1], expected_cos)
                current_dim += 2
                
            if "day_of_week" in time_embedding_keys:
                expected_sin = np.sin(2 * np.pi * current_time.weekday() / 7)
                expected_cos = np.cos(2 * np.pi * current_time.weekday() / 7)
                self.assertAlmostEqual(obs[current_dim], expected_sin)
                self.assertAlmostEqual(obs[current_dim + 1], expected_cos)
                current_dim += 2
                
            if "month_of_year" in time_embedding_keys:
                expected_sin = np.sin(2 * np.pi * current_time.month / 12)
                expected_cos = np.cos(2 * np.pi * current_time.month / 12)
                self.assertAlmostEqual(obs[current_dim], expected_sin)
                self.assertAlmostEqual(obs[current_dim + 1], expected_cos)
                current_dim += 2
        
        # Test 4: Check global forecasts if present
        if 'forecasts' in self.env.io_config_dict:
            # Find OutdoorEnvironment component
            outdoor_env_component = None
            for component in self.env.simulator.model.components:
                #TODO: Make this more robust
                if component == "Outdoor_environment": 
                    outdoor_env_component = self.env.simulator.model.components[component]
                    break
            
            if outdoor_env_component is not None:
                df = outdoor_env_component.df
                forecast_keys = list(self.env.io_config_dict['forecasts'].keys())
                
                # Test global forecasts
                global_forecast_columns = {
                    "outdoor_temperature": "outdoorTemperature",
                    "global_irradiation": "globalIrradiation",
                }
                
                for key in forecast_keys:
                    if key in global_forecast_columns:
                        # Get the forecast values for the current horizon
                        forecast = self.env._get_forecast(df, global_forecast_columns[key])
                        # Check each value in the forecast
                        for i in range(self.env.forecast_horizon + 1):
                            self.assertEqual(obs[current_dim + i], forecast.iloc[i])
                        current_dim += self.env.forecast_horizon + 1
        
        # Test 5: Check component-specific forecasts if present
        if 'forecasts' in self.env.io_config_dict:
            forecast_keys = list(self.env.io_config_dict['forecasts'].keys())
            for key in forecast_keys:
                if key not in global_forecast_columns:
                    component = self.env.simulator.model.components[key]
                    signal_key = self.env.io_config_dict['forecasts'][key]["signal_key"]
                    
                    if hasattr(component, 'df'):
                        df = component.df
                        forecast = self.env._get_forecast(df, signal_key)
                        for i in range(self.env.forecast_horizon + 1):
                            self.assertEqual(obs[current_dim + i], forecast.iloc[i])
                    else:
                        # For schedule components, check each forecast value
                        current_time = self.env.simulator.dateTime
                        for i in range(self.env.forecast_horizon + 1):
                            forecast_time = current_time + timedelta(seconds=i*self.env.step_size)
                            expected_value = component.get_schedule_value(forecast_time)
                            self.assertEqual(obs[current_dim + i], expected_value)
                    current_dim += self.env.forecast_horizon + 1
        
        # Test 6: Check all values are within observation space bounds
        self.assertTrue(np.all(obs >= self.env.observation_space.low))
        self.assertTrue(np.all(obs <= self.env.observation_space.high))

    def test_reset(self):
        """Test reset method with various parameter combinations"""
        # Test case 1: Fixed start time, no episode length
        env1 = T4BGymEnv(
            model=self.model,
            io_config_file=r"C:\Users\asces\OneDriveUni\Projects\RL_control\t4b_gym\testing\policy_input_output.json",
            start_time=self.start_time,
            end_time=self.end_time,
            episode_length=None,
            random_start=False,
            excluding_periods=None,
            step_size=self.stepSize
        )
        env1.reset()
        self.assertEqual(env1.sim_start_time, self.start_time)
        self.assertEqual(env1.sim_end_time, self.end_time)

        # Test case 2: Fixed start time with episode length
        episode_length = 24  # 24 steps
        env2 = T4BGymEnv(
            model=self.model,
            io_config_file=r"C:\Users\asces\OneDriveUni\Projects\RL_control\t4b_gym\testing\policy_input_output.json",
            start_time=self.start_time,
            end_time=self.end_time,
            episode_length=episode_length,
            random_start=False,
            excluding_periods=None,
            step_size=self.stepSize
        )
        env2.reset()
        self.assertEqual(env2.sim_start_time, self.start_time)
        expected_end = self.start_time + timedelta(seconds=episode_length * self.stepSize)
        self.assertEqual(env2.sim_end_time, expected_end)

        # Test case 3: Random start with episode length
        env3 = T4BGymEnv(
            model=self.model,
            io_config_file=r"C:\Users\asces\OneDriveUni\Projects\RL_control\t4b_gym\testing\policy_input_output.json",
            start_time=self.start_time,
            end_time=self.end_time,
            episode_length=episode_length,
            random_start=True,
            excluding_periods=None,
            step_size=self.stepSize
        )
        env3.reset()
        # Check that start time is within valid range
        self.assertGreaterEqual(env3.sim_start_time, self.start_time)
        self.assertLessEqual(env3.sim_start_time, self.end_time - timedelta(seconds=episode_length * self.stepSize))
        # Check that end time is correct
        self.assertEqual(env3.sim_end_time, env3.sim_start_time + timedelta(seconds=episode_length * self.stepSize))

        # Test case 4: Random start with excluding periods
        excluding_periods = [
            (self.start_time + timedelta(hours=2), self.start_time + timedelta(hours=4)),
            (self.start_time + timedelta(hours=6), self.start_time + timedelta(hours=8))
        ]
        env4 = T4BGymEnv(
            model=self.model,
            io_config_file=r"C:\Users\asces\OneDriveUni\Projects\RL_control\t4b_gym\testing\policy_input_output.json",
            start_time=self.start_time,
            end_time=self.end_time,
            episode_length=episode_length,
            random_start=True,
            excluding_periods=excluding_periods,
            step_size=self.stepSize
        )
        env4.reset()
        # Check that start time is not in excluding periods
        for start, end in excluding_periods:
            self.assertFalse(start <= env4.sim_start_time < end)
            self.assertFalse(start < env4.sim_end_time <= end)
            self.assertFalse(env4.sim_start_time <= start and env4.sim_end_time >= end)
        # Check that end time is correct
        self.assertEqual(env4.sim_end_time, env4.sim_start_time + timedelta(seconds=episode_length * self.stepSize))

        # Test case 5: Multiple resets with random start
        env5 = T4BGymEnv(
            model=self.model,
            io_config_file=r"C:\Users\asces\OneDriveUni\Projects\RL_control\t4b_gym\testing\policy_input_output.json",
            start_time=self.start_time,
            end_time=self.end_time,
            episode_length=episode_length,
            random_start=True,
            excluding_periods=excluding_periods,
            step_size=self.stepSize
        )
        # Test multiple resets to ensure different random start times
        start_times = set()
        for _ in range(10):
            env5.reset()
            start_times.add(env5.sim_start_time)
        # Check that we got different start times
        self.assertGreater(len(start_times), 1, "Random start times should be different")

        # Test case 6: Episode length too long
        with self.assertRaises(ValueError):
            env6 = T4BGymEnv(
                model=self.model,
                io_config_file=r"C:\Users\asces\OneDriveUni\Projects\RL_control\t4b_gym\testing\policy_input_output.json",
                start_time=self.start_time,
                end_time=self.end_time,
                episode_length=1000000,  # Too long
                random_start=False,
                excluding_periods=None,
                step_size=self.stepSize
            )

        # Test case 7: Fragmented time chunks
        # Create excluding periods that fragment the time into chunks smaller than episode length
        episode_length = 50  # 50 steps
        excluding_periods = [
            (self.start_time + timedelta(seconds=20*self.stepSize), self.start_time + timedelta(seconds=30*self.stepSize)),
            (self.start_time + timedelta(seconds=50*self.stepSize), self.start_time + timedelta(seconds=60*self.stepSize))
        ]
        
        with self.assertRaises(ValueError):
            env7 = T4BGymEnv(
                model=self.model,
                io_config_file=r"C:\Users\asces\OneDriveUni\Projects\RL_control\t4b_gym\testing\policy_input_output.json",
                start_time=self.start_time,
                end_time=self.start_time + timedelta(seconds=100*self.stepSize),  # 100 steps total
                episode_length=episode_length,
                random_start=True,
                excluding_periods=excluding_periods,
                step_size=self.stepSize
            )

        # Test case 8: Timeout scenario
        # Create excluding periods that leave very little room for the episode
        excluding_periods = [
            (self.start_time + timedelta(seconds=0*self.stepSize), self.start_time + timedelta(seconds=45*self.stepSize)),
            (self.start_time + timedelta(seconds=55*self.stepSize), self.start_time + timedelta(seconds=100*self.stepSize))
        ]
        
        # First test that initialization fails when excluding periods fragment time too much
        with self.assertRaises(ValueError):
            env8 = T4BGymEnv(
                model=self.model,
                io_config_file=r"C:\Users\asces\OneDriveUni\Projects\RL_control\t4b_gym\testing\policy_input_output.json",
                start_time=self.start_time,
                end_time=self.start_time + timedelta(seconds=100*self.stepSize),  # 100 steps total
                episode_length=episode_length,
                random_start=True,
                excluding_periods=excluding_periods,
                step_size=self.stepSize
            )

        # Test case 9: Verify random start time generation
        # Create excluding periods that leave room at the end of the simulation
        excluding_periods = [
            (self.start_time + timedelta(seconds=0*self.stepSize), self.start_time + timedelta(seconds=30*self.stepSize))
        ]
        
        # Create environment with valid parameters
        env9 = T4BGymEnv(
            model=self.model,
            io_config_file=r"C:\Users\asces\OneDriveUni\Projects\RL_control\t4b_gym\testing\policy_input_output.json",
            start_time=self.start_time,
            end_time=self.start_time + timedelta(seconds=100*self.stepSize),  # 100 steps total
            episode_length=episode_length,
            random_start=True,
            excluding_periods=excluding_periods,
            step_size=self.stepSize
        )
        
        # Test multiple resets to ensure we get start times in the later part of the simulation
        start_times = set()
        for _ in range(10):
            env9.reset()
            start_times.add(env9.sim_start_time)
        
        # Check that we got some start times after the excluding period
        later_start_times = {t for t in start_times if t > self.start_time + timedelta(seconds=30*self.stepSize)}
        self.assertGreater(len(later_start_times), 0, "Should get some start times after the excluding period")



if __name__ == "__main__":
    
    unittest.main()
    """
    # Create an instance of the test class and run the specific test
    test_case = TestT4BGymEnv()
    test_case.setUp()
    test_case.test_reset()
    """
