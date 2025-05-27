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
from typing import Dict, List, Tuple, Any, Optional
from twin4build.saref4syst.system import System
from datetime import datetime, timedelta
import pandas as pd
from twin4build.saref.device.sensor.sensor import Sensor
from twin4build.saref.device.meter.meter import Meter
import logging
import json
from dateutil.tz import gettz 

# Set up logging
logger = logging.getLogger(__name__)

class gym_simulator(tb.Simulator):
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
        self.control_inputs[component_id][input_name] = 0.0  # Initialize with default value
        
    def add_observation_output(self, component_id: str, output_name: str) -> None:
        """Add an observation output to monitor.
        
        Args:
            component_id: ID of the component to observe
            output_name: Name of the output parameter to monitor
        """
        if component_id not in self.observation_outputs:
            self.observation_outputs[component_id] = {}
        self.observation_outputs[component_id][output_name] = 0.0  # Initialize with default value
                        
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
            for input_name in action_config['signal_key']:
                self.add_control_input(component_id, input_name)
            
        # Add observation outputs from observations
        for component_id, obs_config in config['observations'].items():
            for output_name in obs_config['signal_key']:
                self.add_observation_output(component_id, output_name)
    
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
                        observations[component_id][output_name] = value
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
        
        # Get observations and check if done
        observations = self.get_observations()
        #TODO: Calculate how to define done based on the defined episode lenght in the gym env
        done = self.current_step >= len(self.secondTimeSteps)
        
        return observations, done
    

class t4b_gym_env(gym.Env):
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
                 model, 
                 io_config_file: str = None,
                 start_time: datetime = None,
                 end_time: datetime = None,
                 episode_length: int = None,
                 random_start: bool = False,
                 excluding_periods: List[Tuple[datetime, datetime]] = None,
                 step_size: int = 600,
                 warmup_period: int = 0):
        """Initialize the gym environment.
        
        Args:
            model: Twin4Build model instance
            io_config_file: Path to the JSON file containing actions and observations
            start_time: Start time of the simulation (must have timezone)
            end_time: End time of the simulation (must have timezone)
            episode_length: Length of each episode in steps (can be smaller than total simulation time)
            random_start: Whether to start episodes at random times within the simulation period
            excluding_periods: List of (start, end) datetime tuples defining periods to exclude from training
            step_size: Simulation step size in seconds
            warmup_period: Number of steps to run before starting the episode (not implemented yet)
        """
        super().__init__()
        self.simulator = gym_simulator(model)

        # Set simulation parameters
        #TODO: Take into account all parameters in the reset and step functions
        self.step_size = step_size
        self.start_time = start_time or datetime.datetime(year=2024, month=1, day=10, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))
        self.end_time = end_time or datetime.datetime(year=2024, month=1, day=12, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))
        self.episode_length = episode_length
        self.random_start = random_start
        self.excluding_periods = excluding_periods or []
        self.warmup_period = warmup_period

        # Set up control inputs and observation outputs if io_config_file is provided
        if io_config_file is not None:
            self.simulator.populate_actions_and_obs_from_json(io_config_file)
    
        self.simulator.initialize_simulation(self.start_time, self.end_time, self.step_size)
    
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

        #TODO: At some point, the simulation time should be defined during training of the RL agent

        # Reset simulator state
        self.simulator.initialize_simulation(self.start_time, self.end_time, self.step_size)
        
        # Get initial observations
        observations = self.simulator.get_observations()
        
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
        observations, done = self.simulator.step_simulation(action)
        
        # Calculate reward (placeholder - should be implemented based on specific task)
        reward = self.calculate_reward(observations, action, self.simulator.model)
        
        # Check if episode is done (placeholder)
        terminated = done
        truncated = False
        
        # Additional info
        info = {}
        
        return observations, reward, terminated, truncated, info
    
    def calculate_reward(self, observations: Dict[str, Dict[str, float]], action: Dict[str, Dict[str, float]]) -> float:
        """Calculate the reward based on the observations and action.
        
        Args:
            observations: Current state observations
            action: Control actions applied
        Returns:
            float: Reward value
        """
        #Placeholder for the reward function, meant to be implemented by the user
        return 0.0
