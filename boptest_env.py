import numpy as np
import requests

class BoptestEnv:
    def __init__(self):
        # Initialize any necessary variables
        self.base_url = 'http://localhost:5000'
        self.action_space = self.define_action_space()
        self.observation_space = self.define_observation_space()
        self.current_state = None

    def define_action_space(self):
        # Define the action space limits (e.g., fan speeds, temperature setpoints)
        action_low = np.array([0.0, 0.0, 18.0])   # Example lower bounds
        action_high = np.array([1.0, 1.0, 24.0])  # Example upper bounds
        return {'low': action_low, 'high': action_high}

    def define_observation_space(self):
        # Define the observation space limits
        obs_low = np.array([-np.inf]*5)  # Adjust based on actual observations
        obs_high = np.array([np.inf]*5)
        return {'low': obs_low, 'high': obs_high}

    def reset(self):
        # Reset the BOPTEST emulator to the initial state
        response = requests.put(f'{self.base_url}/initialize')
        if response.status_code == 200:
            # Get the initial observation
            self.current_state = self.get_observation()
            return self.current_state
        else:
            raise Exception('Failed to reset the BOPTEST emulator.')

    def step(self, action):
        # Validate action
        action = np.clip(action, self.action_space['low'], self.action_space['high'])
        # Send the action to the BOPTEST emulator
        payload = {'inputs': self.format_action(action)}
        response = requests.post(f'{self.base_url}/advance', json=payload)
        if response.status_code == 200:
            # Get the new state, reward, and done flag
            self.current_state = self.get_observation()
            reward = self.calculate_reward(self.current_state, action)
            done = self.check_done()
            info = {}  # Additional info if needed
            return self.current_state, reward, done, info
        else:
            raise Exception('Failed to advance the BOPTEST emulator.')

    def get_observation(self):
        # Get the current state from the BOPTEST emulator
        response = requests.get(f'{self.base_url}/measurements')
        if response.status_code == 200:
            data = response.json()
            # Extract relevant observations
            observation = np.array([
                data['temperatures']['zone1'],
                data['temperatures']['zone2'],
                data['energy_consumption'],
                # Add other observations as needed
            ])
            return observation
        else:
            raise Exception('Failed to get observations from the BOPTEST emulator.')

    def format_action(self, action):
        # Format the action into the payload expected by BOPTEST
        return {
            'supply_fan_speed': action[0],
            'return_fan_speed': action[1],
            'zone_temperature_setpoint': action[2],
            # Add other actions as needed
        }

    def calculate_reward(self, state, action):
        # Define your reward function
        comfort_penalty = ...  # Calculate based on temperature deviations
        energy_penalty = ...   # Calculate based on energy consumption
        reward = - (comfort_penalty + energy_penalty)
        return reward

    def check_done(self):
        # Determine if the episode is done (e.g., end of simulation)
        response = requests.get(f'{self.base_url}/scenario')
        data = response.json()
        done = data['time_elapsed'] >= data['time_period']
        return done

    def close(self):
        # Close any resources if needed
        pass
