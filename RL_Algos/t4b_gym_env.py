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

class gym_simulator(tb.Simulator):
    def __init__(self, model):
        """Initialize the gym simulator with a twin4build model.
        
        Args:
            model: A twin4build model instance
        """
        super().__init__(model)
        self.control_inputs: Dict[str, Dict[str, Any]] = {}  # Maps component_id to {input_name: current_value}
        self.observation_outputs: Dict[str, Dict[str, Any]] = {}  # Maps component_id to {output_name: current_value}
        self.current_step = 0
        
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
                component.input[connection_point.receiverPropertyName].set(
                    connected_component.output[connection.senderPropertyName].get()
                )
        
        # Apply any control inputs for this component if it's being controlled
        if component.id in self.control_inputs:
            for input_name, value in self.control_inputs[component.id].items():
                if input_name in component.input:
                    component.input[input_name].set(value)
        
        # Do the component timestep
        component.do_step(secondTime=self.secondTime, #This will be using the timing from the parent class
                         dateTime=self.dateTime, 
                         stepSize=self.stepSize)
        
        # Store any outputs we're observing
        if component.id in self.observation_outputs:
            for output_name in self.observation_outputs[component.id]:
                if output_name in component.output:
                    self.observation_outputs[component.id][output_name] = \
                        component.output[output_name].get()
        
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
        
    def apply_control(self, actions: Dict[str, Dict[str, float]]) -> None:
        """Apply control actions to the simulation.
        
        Args:
            actions: Dictionary mapping component IDs to their input values
                    Format: {component_id: {input_name: value}}
        """
        for component_id, inputs in actions.items():
            if component_id in self.model.components:
                component = self.model.components[component_id]
                for input_name, value in inputs.items():
                    if input_name in component.input:
                        component.input[input_name].set(value)
                        self.control_inputs[component_id][input_name] = value
                        
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
                        observations[component_id][output_name] = value
                        self.observation_outputs[component_id][output_name] = value
        return observations
    
    def step_simulation(self, actions: Dict[str, Dict[str, float]]) -> Tuple[Dict[str, Dict[str, float]], bool]:
        """Perform one simulation step with the given actions.
        
        This method advances the simulation by one timestep, similar to the original
        simulate method but allowing for step-by-step control. It:
        1. Applies control actions
        2. Updates simulation time
        3. Executes one system timestep
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
            return self.get_observations(), True
            
        # Apply control actions
        self.apply_control(actions)
        
        # Get current timestep values
        self.secondTime = self.secondTimeSteps[self.current_step]
        self.dateTime = self.dateTimeSteps[self.current_step]
        
        # Execute system timestep
        self._do_system_time_step(self.model)
        
        # Increment step counter
        self.current_step += 1
        
        # Get observations and check if done
        observations = self.get_observations()
        done = self.current_step >= len(self.secondTimeSteps)
        
        return observations, done
    

class t4b_gym_env(gym.Env):
    """
    Gymnasium environment wrapper for Twin4Build models.
    
    This environment provides a standard gym interface for interacting with
    Twin4Build simulation models, allowing for reinforcement learning applications.
    """
    
    def __init__(self, model, control_inputs: Dict[str, List[str]], observation_outputs: Dict[str, List[str]]):
        """Initialize the gym environment.
        
        Args:
            model: Twin4Build model instance
            control_inputs: Dictionary mapping component IDs to list of input names to control
            observation_outputs: Dictionary mapping component IDs to list of output names to observe
        """
        super().__init__()
        self.simulator = gym_simulator(model)
        
        # Set up control inputs and observation outputs
        for component_id, input_names in control_inputs.items():
            for input_name in input_names:
                self.simulator.add_control_input(component_id, input_name)
                
        for component_id, output_names in observation_outputs.items():
            for output_name in output_names:
                self.simulator.add_observation_output(component_id, output_name)
        
        # Define action and observation spaces
        # This is a placeholder - actual spaces should be defined based on the specific model
        self.action_space = gym.spaces.Dict({
            component_id: gym.spaces.Dict({
                input_name: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
                for input_name in input_names
            }) for component_id, input_names in control_inputs.items()
        })
        
        self.observation_space = gym.spaces.Dict({
            component_id: gym.spaces.Dict({
                output_name: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
                for output_name in output_names
            }) for component_id, output_names in observation_outputs.items()
        })
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset (unused)
            
        Returns:
            tuple: (observations, info)
                observations: Initial state observations
                info: Additional information
        """
        super().reset(seed=seed)
        
        # Reset simulator state
        self.simulator.current_step = 0
        
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
        # Apply action and get new observations
        observations, done = self.simulator.step_simulation(action)
        
        # Calculate reward (placeholder - should be implemented based on specific task)
        reward = 0.0
        
        # Check if episode is done (placeholder)
        terminated = done
        truncated = False
        
        # Additional info
        info = {}
        
        return observations, reward, terminated, truncated, info
    

