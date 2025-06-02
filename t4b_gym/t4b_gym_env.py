"""
Gym environment wrapper for T4B models, 
It implements the step and reset functions for the environment.
Each environment requires a defined model and creates a vectorized object 
that inherits from the tb.Simulator class. This class allows for the simulation to 
receive actions and return the next state, reward, done, and info every step.

Author: Sebastian Cubides @SebsCubs in github

"""
import twin4build as tb
import gymnasium as gym
import numpy as np
import random
from typing import Dict, List, Tuple, Any, Optional
from twin4build.saref4syst.system import System
from datetime import datetime, timedelta
from gymnasium import spaces
import logging
import json
from dateutil.tz import gettz 
import os
import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)

class GymSimulator(tb.Simulator):
    def __init__(self, model, enable_logging: bool = True):
        """Initialize the gym simulator with a twin4build model.
        
        Args:
            model: A twin4build model instance
            enable_logging: Whether to enable debug logging. Defaults to True.
        """
        super().__init__(model)
        self.control_inputs: Dict[str, Dict[str, Any]] = {}  # Maps component_id to {input_name: current_value}
        self.observation_outputs: Dict[str, Dict[str, Any]] = {}  # Maps component_id to {output_name: current_value}
        self.current_step = 0
        self.model = model
        # Configure logging
        self.enable_logging = enable_logging
        if enable_logging:
            logger.setLevel(logging.INFO)
            logger.info(f"Initialized gym_simulator with model: {model.__class__.__name__}")
        else:
            logger.setLevel(logging.WARNING)
        
    def initialize_simulation(self, startTime: datetime, endTime: datetime, stepSize: int) -> None:
        """Initialize the simulation parameters and model.
        
        Args:
            startTime (datetime): Start time of the simulation (must have timezone)
            endTime (datetime): End time of the simulation (must have timezone)
            stepSize (int): Step size in seconds
            
        Raises:
            AssertionError: If input parameters are invalid or missing timezone info
        """
        assert startTime.tzinfo is not None, "The argument startTime must have a timezone"
        assert endTime.tzinfo is not None, "The argument endTime must have a timezone"
        assert isinstance(stepSize, int), "The argument stepSize must be an integer"
        
        self.startTime = startTime
        self.endTime = endTime
        self.stepSize = stepSize
        self.current_step = 0
        
        # Initialize model and generate timesteps
        self.model.initialize(startTime=startTime, endTime=endTime, stepSize=stepSize)
        self.get_simulation_timesteps(startTime, endTime, stepSize)

        self.secondTime = self.secondTimeSteps[self.current_step]
        self.dateTime = self.dateTimeSteps[self.current_step]
        
        if self.enable_logging:
            total_steps = len(self.secondTimeSteps)
            duration = endTime - startTime
            logger.info(f"Simulation initialized: {startTime} -> {endTime}")
            logger.info(f"Step size: {stepSize}s, Total steps: {total_steps}")
            logger.info(f"Simulation duration: {duration}")
        
    def _do_component_timestep(self, component: System) -> None:
        """Override of the parent class _do_component_timestep method.
        
        This method handles a single timestep for a component, including:
        1. Gathering inputs from connections
        2. Applying any control inputs from the gym environment
        3. Executing the component step
        4. Storing outputs for observation
        
        Args:
            component (System): The component to simulate for one timestep
        """
        # First gather all needed inputs from connections (from parent class)
        for connection_point in component.connectsAt:
            for connection in connection_point.connectsSystemThrough:
                connected_component = connection.connectsSystem
                input_value = connected_component.output[connection.senderPropertyName].get()
                component.input[connection_point.receiverPropertyName].set(input_value)
                
                if self.enable_logging:
                    logger.debug(f"Connection: {connected_component.id}.{connection.senderPropertyName} "
                               f"-> {component.id}.{connection_point.receiverPropertyName} = {input_value}")
        
        # Apply any control inputs for this component if it's being controlled
        if component.id in self.control_inputs:
            for input_name, value in self.control_inputs[component.id].items():
                if input_name in component.input:
                    component.input[input_name].set(value)
                    if self.enable_logging:
                        logger.info(f"Control input applied: {component.id}.{input_name} = {value}")
        
        # Do the component timestep
        component.do_step(secondTime=self.secondTime,
                         dateTime=self.dateTime, 
                         stepSize=self.stepSize)
        
        # Store any outputs we're observing
        if component.id in self.observation_outputs:
            for output_name in self.observation_outputs[component.id]:
                if output_name in component.output:
                    value = component.output[output_name].get()
                    self.observation_outputs[component.id][output_name] = value
                    if self.enable_logging:
                        logger.debug(f"Observation recorded: {component.id}.{output_name} = {value}")

    def _do_system_time_step(self, model: tb.Model) -> None:
        """
        Override of the parent class _do_system_time_step method, to use the local
        _do_component_timestep method.
        Execute a time step for all components in the model.

        This method executes components in the order specified by the model's execution
        order, ensuring proper propagation of information through the system. It:
        1. Executes components in groups based on dependencies
        2. Updates component states after all executions
        3. Handles both FMU and non-FMU components

        Args:
            model (model.Model): The model containing components to simulate.

        Notes:
            - Components are executed sequentially based on their dependencies
            - Results are updated after all components have been stepped
            - Component execution order is determined by the model's execution_order attribute
            - Updates are propagated through the flat_execution_order after main execution
        """
        for component_group in self.model.execution_order:
            for component in component_group:
                self._do_component_timestep(component)

        for component in self.model.flat_execution_order:
            component.update_results()

    def add_control_input(self, component_id: str, input_name: str) -> None:
        """Add a control input to the simulator.
        
        Args:
            component_id: ID of the component to control
            input_name: Name of the input parameter to control
        """
        if component_id not in self.control_inputs:
            self.control_inputs[component_id] = {}
        self.control_inputs[component_id][input_name] = {}  # Initialize with an empty dictionary
        
    def add_observation_output(self, component_id: str, output_name: str) -> None:
        """Add an observation output to monitor.
        
        Args:
            component_id: ID of the component to observe
            output_name: Name of the output parameter to monitor
        """
        if component_id not in self.observation_outputs:
            self.observation_outputs[component_id] = {}
        self.observation_outputs[component_id][output_name] = {}  # Initialize with an empty dictionary
                        
    def populate_actions_and_obs_from_json(self, json_file: str) -> None:
        """Populate actions and observations from a JSON file.
        
        Args:
            json_file: Path to the JSON file containing actions and observations
        """
        with open(json_file, 'r') as f:
            config = json.load(f)

        #assert the format of the json file
        assert 'actions' in config, "The JSON file must contain an 'actions' key"
        assert 'observations' in config, "The JSON file must contain an 'observations' key"
  
        # Add control inputs from actions
        for component_id, action_config in config['actions'].items():
            self.add_control_input(component_id, action_config['signal_key'])
            
        # Add observation outputs from observations
        for component_id, obs_config in config['observations'].items():
            self.add_observation_output(component_id, obs_config['signal_key'])

        #Add the max and min values for the action and observation spaces
        for component_id, action_config in config['actions'].items():
            input_name = action_config['signal_key']
            self.control_inputs[component_id][input_name]['max'] = action_config['max']
            self.control_inputs[component_id][input_name]['min'] = action_config['min']

        for component_id, obs_config in config['observations'].items():
            output_name = obs_config['signal_key']
            self.observation_outputs[component_id][output_name]['max'] = obs_config['max']
            self.observation_outputs[component_id][output_name]['min'] = obs_config['min']
    
    def get_observations(self) -> Dict[str, Dict[str, float]]:
        """Get current observations from monitored outputs.
        
        Returns:
            Dictionary mapping component IDs to their output values
            Format: {component_id: {output_name: value}}
        """
        observations = {}
        for component_id, outputs in self.observation_outputs.items():
            if component_id in self.model.components:
                component = self.model.components[component_id]
                observations[component_id] = {}
                for output_name in outputs:
                    if output_name in component.output:
                        value = component.output[output_name].get()
                        if value is None:
                            value = 0.0
                        observations[component_id] = value
                        self.observation_outputs[component_id][output_name] = value
        return observations
    
    def step_simulation(self, actions: Optional[Dict[str, Dict[str, float]]] = None) -> Tuple[Dict[str, Dict[str, float]], bool]:
        """Perform one simulation step with the given actions.
        
        This method advances the simulation by one timestep, similar to the original
        simulate method but allowing for step-by-step control. It:
        1. Updates control input storage with new actions
        2. Updates simulation time
        3. Executes one system timestep (which applies controls via _do_component_timestep)
        4. Returns current observations and done status
        
        Args:
            actions: Control actions to apply
            
        Returns:
            Tuple containing:
            - Current observations after applying actions
            - Boolean indicating if simulation is complete (reached endTime)
        """
        # Check if we've reached the end of simulation
        if self.current_step >= len(self.secondTimeSteps):
            if self.enable_logging:
                logger.info("Simulation complete")
            return self.get_observations(), True
            
        # Store the new control actions (will be applied during component timesteps)
        if self.enable_logging:
            logger.info(f"Step {self.current_step + 1}/{len(self.secondTimeSteps)}")
            logger.debug(f"Time: {self.dateTimeSteps[self.current_step]}")
        
        if actions is not None:
            for component_id, inputs in actions.items():
                if component_id in self.control_inputs:
                    self.control_inputs[component_id].update(inputs)
        elif self.enable_logging:
            logger.info("No actions provided, using default values")
        
        # Get current timestep values
        self.secondTime = self.secondTimeSteps[self.current_step]
        self.dateTime = self.dateTimeSteps[self.current_step]
        
        # Execute system timestep using the local _do_system_time_step override
        self._do_system_time_step(self.model)
        
        # Increment step counter
        self.current_step += 1
        
        done = self.current_step >= len(self.secondTimeSteps)
        
        return done
    

class T4BGymEnv(gym.Env):
    """
    Gymnasium environment wrapper for Twin4Build models.
    
    This environment provides a standard gym interface for interacting with
    Twin4Build simulation models, allowing for reinforcement learning applications.

    Things to be defined by the user:
    - start_time (conditional on the model data available)
    - end_time (conditional on the model data available)
    - episode_length (can be smaller than the total simulation time)
    - random_start (boolean to define if the start time is random or not (only used if episode_length is smaller than the total simulation time))
    - excluding_periods (list of tuples defining periods of the simulation that should be excluded from the training data)
    - stepSize
    - io_config_file (this is defining action and observation spaces)
    - warmup_period (not implemented yet, but would be a period of the simulation that runs before the training starts)
    - reward_function (implemented by inheriting from the class and overriding the method)

    """
    
    def __init__(self, 
                 model: tb.Model, 
                 io_config_file: str, # Mandatory for now
                 start_time: datetime = None,
                 end_time: datetime = None,
                 episode_length: int = None,
                 random_start = False,
                 excluding_periods: List[Tuple[datetime, datetime]] = None,
                 forecast_horizon: int = 0,
                 step_size: int = 600,
                 warmup_period = 0):
        """Initialize the gym environment.
        
        Args:
            model: Twin4Build model instance
            io_config_file: Path to the JSON file containing actions and observations
            start_time: Start time of the simulation (must have timezone)
            end_time: End time of the simulation (must have timezone)
            episode_length: Length of each episode in steps (must be smaller than total simulation time)
            random_start: Whether to start episodes at random times within the simulation period
            excluding_periods: List of (start, end) datetime tuples defining periods to exclude from training
            step_size: Simulation step size in seconds
            warmup_period: Number of steps to run before starting the episode (not implemented yet)
        """
        super().__init__()
        self.simulator = GymSimulator(model)

        # Set simulation parameters
        self.step_size = step_size

        self.global_start_time = start_time 
        self.global_end_time = end_time
        self.episode_length = episode_length
        self.random_start = random_start
        self.excluding_periods = excluding_periods or []
        self.warmup_period = warmup_period
        self.forecast_horizon = forecast_horizon

        # Set up control inputs and observation outputs if io_config_file is provided
        assert io_config_file is not None, """io_config_file is mandatory. The JSON file should have this format:
                            {
                                "actions": {
                                    "Component_ID": {
                                        "signal_key": "input_name",
                                        "min": float,
                                        "max": float,
                                        "description": "string"
                                    }
                                },
                                "observations": {
                                    "Component_ID": {
                                        "signal_key": "output_name",
                                        "min": float,
                                        "max": float,
                                        "description": "string"
                                    }
                                }
                            }"""
        #Assert that the io_config_file has the correct format
        assert os.path.exists(io_config_file), "The io_config_file does not exist"
        assert os.path.isfile(io_config_file), "The io_config_file is not a file"   
        assert os.path.splitext(io_config_file)[1] == '.json', "The io_config_file must be a JSON file"
        with open(io_config_file) as f:
            io_dict = json.load(f)
        assert 'actions' in io_dict, "The io_config_file must contain an 'actions' key"
        assert 'observations' in io_dict, "The io_config_file must contain an 'observations' key"
        
        self.io_config_dict = io_dict

        if self.episode_length is not None:
            #Assert that the episode length is smaller than the total simulation time, raise a value error if not
            if self.episode_length > (self.global_end_time - self.global_start_time).total_seconds()/self.step_size:
                raise ValueError("Episode length must be smaller than the total simulation time")
        if self.random_start and self.excluding_periods is not None:
            #Assert that there is at least one chunk of available time >= episode_length
            if self.excluding_periods is not None and not self._check_available_time_chunks():
                raise ValueError("Excluding periods fragment the available time into chunks smaller than episode_length")
            for start, end in self.excluding_periods:
                    if start < self.global_start_time or end > self.global_end_time:
                        raise ValueError("Excluding periods must be within the total simulation time")
        
        #Populate the simulator with the actions and observations
        self.simulator.populate_actions_and_obs_from_json(io_config_file)        
        self.create_observation_space()
        self.create_action_space()

        #self.simulator.initialize_simulation(self.start_time, self.end_time, self.step_size)
    
    def _check_available_time_chunks(self):
        """Check if excluding periods fragment the available time into chunks smaller than episode length.
        
        Returns:
            bool: True if there is at least one chunk of available time >= episode_length
        """
        # Sort excluding periods by start time
        sorted_periods = sorted(self.excluding_periods, key=lambda x: x[0])
        
        # Get all time boundaries
        boundaries = [self.global_start_time] + [t for period in sorted_periods for t in period] + [self.global_end_time]
        
        # Find available chunks between excluding periods
        available_chunks = []
        for i in range(0, len(boundaries)-1, 2):
            chunk_start = boundaries[i]
            chunk_end = boundaries[i+1]
            chunk_duration = (chunk_end - chunk_start).total_seconds() / self.step_size
            available_chunks.append(chunk_duration)
        
        # Check if any chunk is large enough for the episode
        return any(chunk >= self.episode_length for chunk in available_chunks)

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility (unused)
            options: Additional options for reset (unused)
            
        Returns:
            tuple: (observations, info)
                observations: Initial state observations
                info: Additional information
        """

        if self.episode_length is not None:
            self.sim_start_time = self.global_start_time
            self.sim_end_time = self.global_start_time + timedelta(seconds=self.episode_length*self.step_size)

        if self.random_start:

            #Generate a random start time
            max_start_time = self.global_end_time - timedelta(seconds=self.episode_length*self.step_size)
            random_start_time = self.global_start_time + timedelta(seconds=random.randint(0, int((max_start_time - self.global_start_time).total_seconds())))
            
            #Check that end time is not beyond the global end time
            episode_end_time = min(random_start_time + timedelta(seconds=self.episode_length*self.step_size), self.global_end_time)
            
            #Check if any part of the episode overlaps with excluding periods
            if self.excluding_periods is not None:
                max_attempts = 1000  # Maximum number of attempts to find a valid time slot
                attempts = 0
                while any((start <= random_start_time < end) or 
                         (start < episode_end_time <= end) or
                         (random_start_time <= start and episode_end_time >= end) 
                         for start, end in self.excluding_periods):
                    random_start_time = self.global_start_time + timedelta(seconds=random.randint(0, int((max_start_time - self.global_start_time).total_seconds())))
                    episode_end_time = min(random_start_time + timedelta(seconds=self.episode_length*self.step_size), self.global_end_time)
                    attempts += 1
                    if attempts >= max_attempts:
                        raise ValueError("Could not find a valid time slot after maximum attempts. The excluding periods may leave no room for the episode length.")

            self.sim_start_time = random_start_time
            self.sim_end_time = episode_end_time
            
        if self.episode_length is None:
            self.sim_start_time = self.global_start_time
            self.sim_end_time = self.global_end_time
        
        # Reset simulator state
        self.simulator.initialize_simulation(self.sim_start_time, self.sim_end_time, self.step_size)
        
        # Get initial observations
        observations = self._get_obs()
        
        return observations, {}
    
    def step(self, action: Dict[str, Dict[str, float]]):
        """Take a step in the environment.
        
        Args:
            action: Dictionary of control actions
                   Format: {component_id: {input_name: value}}
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
                observation: Current state observation
                reward: Reward from the action
                terminated: Whether episode is done
                truncated: Whether episode was artificially terminated
                info: Additional information
        """
        #Assert the format of the action
        assert isinstance(action, dict), "The action must be a dictionary"
        assert isinstance(list(action.values())[0], dict), "The action must contain a dictionary of input names and values"
        assert len(list(action.values())[0]) > 0, "The action must contain at least one input name and value"

        # Apply action and get new observations
        done = self.simulator.step_simulation(action)
        
        observations = self._get_obs()

        # Calculate reward (placeholder - should be implemented based on specific task)
        reward = self.get_reward(observations, action)
        
        # Check if episode is done (placeholder)
        terminated = done
        truncated = False
        
        # Additional info
        info = {}
        
        return observations, reward, terminated, truncated, info
    
    def get_reward(self, observations: Dict[str, Dict[str, float]], action: Dict[str, Dict[str, float]]) -> float:
        """Calculate the reward based on the observations and action.
        
        Args:
            observations: Current state observations
            action: Control actions applied
        Returns:
            float: Reward value
        """
        #Placeholder for the reward function, meant to be implemented by the user
        return 0.0

    def create_action_space(self):
        """Create the action space based on the control inputs defined in the simulator.
        
        Returns:
            spaces.Dict: Action space
        """
        
        #Define the action and observation spaces, if io_config_file is provided, max and min values are defined in the json file
        action_keys = []
        low_bounds = []
        upper_bounds = []

        for component_id, inputs in self.simulator.control_inputs.items():
            for input_name in inputs:
                action_keys.append(input_name)
                low_bounds.append(self.simulator.control_inputs[component_id][input_name]['min'])
                upper_bounds.append(self.simulator.control_inputs[component_id][input_name]['max'])

        #Define the action space
        self.action_space = spaces.Box(low=np.array(low_bounds), high=np.array(upper_bounds), dtype=np.float32)
    
    def create_observation_space(self):
        """Create the observation space based on the observations defined in the simulator and the io_config_file.
        1. Gets the populated component outputs from the simulator
        2. Checks if config file contains "time_embeddings" or "forecasts" keys
        3. If "time_embeddings" key is present, the observation space is extended based on the time embeddings
        4. If "forecasts" key is present, the observation space is extended based on the forecasts
        5. If neither key is present, the observation space is defined based on only the component outputs
        Returns:
            spaces.Dict: Observation space
        """
        #Get the populated component outputs from the simulator
        observations = []
        low_bounds = []
        upper_bounds = []

        for component_id, outputs in self.simulator.observation_outputs.items():
            for output_name in outputs:
                observations.append(outputs[output_name])
                low_bounds.append(self.simulator.observation_outputs[component_id][output_name]['min'])
                upper_bounds.append(self.simulator.observation_outputs[component_id][output_name]['max'])

        #Check if config file contains "time_embeddings" or "forecasts" keys
        if 'time_embeddings' in self.io_config_dict:
            time_embedding_keys = list(self.io_config_dict['time_embeddings'].keys())
            for key in time_embedding_keys:
                #Using sin and cos embeddings for the time of day, day of week, month of year
                #Therefore there are two observations for each time embedding
                observations.append(self.io_config_dict['time_embeddings'][key]["signal_key"] + "_sin")
                observations.append(self.io_config_dict['time_embeddings'][key]["signal_key"] + "_cos")
                low_bounds.append(-1)
                low_bounds.append(-1)
                upper_bounds.append(1)
                upper_bounds.append(1)

        if 'forecasts' in self.io_config_dict:
            forecast_keys = list(self.io_config_dict['forecasts'].keys())
            for key in forecast_keys:
                # For each forecast, we need forecast_horizon + 1 dimensions (current value + future values)
                for _ in range(self.forecast_horizon + 1):
                    observations.append(self.io_config_dict['forecasts'][key]["signal_key"])
                    low_bounds.append(self.io_config_dict['forecasts'][key]['min'])
                    upper_bounds.append(self.io_config_dict['forecasts'][key]['max'])

        #Define the observation space
        self.observation_space = spaces.Box(low=np.array(low_bounds), high=np.array(upper_bounds), dtype=np.float32)

    def _get_forecast(self, df: pd.DataFrame, column_name: str) -> pd.Series:
        """Helper method to get forecast window for a given column.
        
        Args:
            df: DataFrame containing the forecast data
            column_name: Name of the column to get forecast for
            
        Returns:
            pd.Series: Forecast window padded if needed
        """
        start_idx = self.simulator.current_step
        end_idx = start_idx + self.forecast_horizon + 1
        forecast = df[column_name].iloc[start_idx:min(end_idx, len(df))]
        if len(forecast) < self.forecast_horizon + 1:
            # Pad with last value to reach desired length
            last_value = forecast.iloc[-1]
            padding_length = self.forecast_horizon + 1 - len(forecast)
            padding = pd.Series([last_value] * padding_length)
            forecast = pd.concat([forecast, padding])
        return forecast

    def _get_obs(self):
        """Get the current observations from the simulator.
        
        Returns:
            numpy.ndarray: Current observations
        """
        model_obs = self.simulator.get_observations()

        #Check if config file contains "time_embeddings" or "forecasts" keys
        if 'time_embeddings' in self.io_config_dict:
            current_time = self.simulator.dateTime
            time_embedding_keys = list(self.io_config_dict['time_embeddings'].keys())
            if "time_of_day" in time_embedding_keys:
                # Use sin cos embeddings for the time of day
                model_obs["time_of_day_sin"] = np.sin(2 * np.pi * current_time.hour / 24)
                model_obs["time_of_day_cos"] = np.cos(2 * np.pi * current_time.hour / 24)
            if "day_of_week" in time_embedding_keys:
                model_obs["day_of_week_sin"] = np.sin(2 * np.pi * current_time.weekday() / 7)
                model_obs["day_of_week_cos"] = np.cos(2 * np.pi * current_time.weekday() / 7)
            if "month_of_year" in time_embedding_keys:
                model_obs["month_of_year_sin"] = np.sin(2 * np.pi * current_time.month / 12)
                model_obs["month_of_year_cos"] = np.cos(2 * np.pi * current_time.month / 12)
        if 'forecasts' in self.io_config_dict:
            #Find the OutdoorEnvironment component
            outdoor_env_component = None
            for component in self.simulator.model.components:
                #TODO: Make this more robust
                if component == "Outdoor_environment":
                    outdoor_env_component = self.simulator.model.components[component]
                    break
            if outdoor_env_component is None:
                raise ValueError("OutdoorEnvironment component not found in the model")
            #get the df from the OutdoorEnvironment component
            df = outdoor_env_component.df
            forecast_keys = list(self.io_config_dict['forecasts'].keys())
            
            # Map forecast keys to their corresponding column names
            global_forecast_columns = {
                "outdoor_temperature": "outdoorTemperature",
                "global_irradiation": "globalIrradiation",
            }
            
            # Get forecasts for each requested key
            for key in forecast_keys:
                if key in global_forecast_columns:
                    model_obs[key] = self._get_forecast(df, global_forecast_columns[key]).values

            #Occupancy values vary from model to model, they are provided as a list of schedule components
            #Go through all the remaining keys in the forecasts dictionary in the io_config_file
            for key in forecast_keys:
                if key not in global_forecast_columns:
                    #Get the schedule component
                    component = self.simulator.model.components[key]
                    signal_key = self.io_config_dict['forecasts'][key]["signal_key"]
                    if hasattr(component, 'df'):
                        df = component.df
                    else:
                        #Component has a schedule ruleset, not a dataframe, so we need to get the schedule values
                        current_time = self.simulator.dateTime
                        forecast_datetimes = [current_time + timedelta(seconds=i*self.step_size) for i in range(self.forecast_horizon + 1)]
                        #Get the schedule values
                        schedule_values = [component.get_schedule_value(dt) for dt in forecast_datetimes]
                        
                        df = pd.DataFrame({
                            'datetime': forecast_datetimes,
                            signal_key: schedule_values
                        })
                    column_name = component.id + "_" + signal_key
                    if self.forecast_horizon > 0:
                        forecast = self._get_forecast(df, signal_key)
                        model_obs[column_name] = forecast.values
                    else:
                        model_obs[column_name] = df[signal_key].values

        #Return the observations as a numpy array
        obs = []
        for key in model_obs.keys():
            if isinstance(model_obs[key], np.ndarray):
                # If it's a forecast array, append each value
                obs.extend(model_obs[key])
            else:
                # If it's a single value, append it directly
                obs.append(model_obs[key])

        return np.array(obs)

class NormalizedObservationWrapper(gym.ObservationWrapper):
    '''This wrapper normalizes the values of the observation space to lie
    between -1 and 1. Normalization can significantly help with convergence
    speed. 
    
    Notes
    -----
    The concept of wrappers is very powerful, with which we are capable 
    to customize observation, action, step function, etc. of an env. 
    No matter how many wrappers are applied, `env.unwrapped` always gives 
    back the internal original environment object. Typical use:
    `env = BoptestGymEnv()`
    `env = NormalizedObservationWrapper(env)`
    
    '''
    
    def __init__(self, env):
        '''
        Constructor
        
        Parameters
        ----------
        env: gym.Env
            Original gym environment
        
        '''
        
        # Construct from parent class
        super().__init__(env)
        
    def observation(self, observation):
        '''
        This method accepts a single parameter (the 
        observation to be modified) and returns the modified observation.
        
        Parameters
        ----------
        observation: 
            Observation in the original environment observation space format 
            to be modified. 
        
        Returns
        -------
            Modified observation returned by the wrapped environment. 
        
        Notes
        -----
        To better understand what this method needs to do, see how the 
        `gym.ObservationWrapper` parent class is doing in `gym.core`:
        
        '''
        
        # Convert to one number for the wrapped environment
        observation_wrapper = 2*(observation - self.observation_space.low)/\
            (self.observation_space.high-self.observation_space.low)-1
        
        return observation_wrapper
     
class NormalizedActionWrapper(gym.ActionWrapper):
    '''This wrapper normalizes the values of the action space to lie
    between -1 and 1. Normalization can significantly help with convergence
    speed. 
    
    Notes
    -----
    The concept of wrappers is very powerful, with which we are capable 
    to customize observation, action, step function, etc. of an env. 
    No matter how many wrappers are applied, `env.unwrapped` always gives 
    back the internal original environment object. Typical use:
    `env = BoptestGymEnv()`
    `env = NormalizedActionWrapper(env)`
    
    '''
    
    def __init__(self, env):
        '''
        Constructor
        
        Parameters
        ----------
        env: gym.Env
            Original gym environment
        
        '''
        
        # Construct from parent class
        super().__init__(env)
        
        # Assert that original observation space is a Box space
        assert isinstance(self.unwrapped.action_space, spaces.Box), 'This wrapper only works with continuous action space (spaces.Box)'
        
        # Store low and high bounds of action space
        self.low    = self.unwrapped.action_space.low
        self.high   = self.unwrapped.action_space.high
        
        # Redefine action space to lie between [-1,1]
        self.action_space = spaces.Box(low = -1, 
                                       high = 1,
                                       shape=self.unwrapped.action_space.shape, 
                                       dtype= np.float32)        
        
    def action(self, action_wrapper):
        '''This method accepts a single parameter (the modified action
        in the wrapper format) and returns the action to be passed to the 
        original environment. Thus, this method basically rescales the  
        action inside the environment.
        
        Parameters
        ----------
        action_wrapper: 
            Action in the modified environment action space format 
            to be reformulated back to the original environment format.
        
        Returns
        -------
            Action in the original environment format.  
        
        Notes
        -----
        To better understand what this method needs to do, see how the 
        `gym.ActionWrapper` parent class is doing in `gym.core`:
        
        '''
        
        return self.low + (0.5*(action_wrapper+1.0)*(self.high-self.low))