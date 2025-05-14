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
from t4b_gym_env import t4b_gym_env

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
    self.add_connection(supply_water_schedule, self.component_dict["[020B][020B_space_heater]"], "scheduleValue", "supplyWaterTemperature") # Add missing input
    self.component_dict["020B_temperature_sensor"].filename = utils.get_path(["parameter_estimation_example", "temperature_sensor.csv"])
    self.component_dict["020B_co2_sensor"].filename = utils.get_path(["parameter_estimation_example", "co2_sensor.csv"])
    self.component_dict["020B_valve_position_sensor"].filename = utils.get_path(["parameter_estimation_example", "valve_position_sensor.csv"])
    self.component_dict["020B_damper_position_sensor"].filename = utils.get_path(["parameter_estimation_example", "damper_position_sensor.csv"])
    self.component_dict["BTA004"].filename = utils.get_path(["parameter_estimation_example", "supply_air_temperature.csv"])
    self.component_dict["020B_co2_setpoint"].weekDayRulesetDict = {"ruleset_default_value": 900,
                                                                    "ruleset_start_minute": [],
                                                                    "ruleset_end_minute": [],
                                                                    "ruleset_start_hour": [],
                                                                    "ruleset_end_hour": [],
                                                                    "ruleset_value": []}
    self.component_dict["020B_temperature_heating_setpoint"].useFile = True
    self.component_dict["020B_temperature_heating_setpoint"].filename = utils.get_path(["parameter_estimation_example", "temperature_heating_setpoint.csv"])
    self.component_dict["outdoor_environment"].filename = utils.get_path(["parameter_estimation_example", "outdoor_environment.csv"])

def load_test_model():
    model = tb.Model(id="neural_policy_example")
    filename = utils.get_path(["parameter_estimation_example", "one_room_example_model.xlsm"])
    model.load(semantic_model_filename=filename, fcn=fcn, verbose=False)
    return model


class TestT4BGymEnv(unittest.TestCase):
    def setUp(self):
        # Load any T4B model for testing
        self.model = load_test_model()  # You define this based on your needs
        
        # Load I/O configuration
        with open("path/to/io_config.json") as f:
            self.io_config = json.load(f)
            
        # Create environment
        self.env = t4b_gym_env(
            model=self.model,
            io_config=self.io_config,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc) + timedelta(days=1),
            step_size=300  # 5 minutes
        )

    def test_space_creation(self):
        """Test if observation and action spaces are created correctly"""
        # Check observation space matches input config
        for comp_id, config in self.io_config['input'].items():
            self.assertIn(comp_id, self.env.observation_space.spaces)
            space = self.env.observation_space.spaces[comp_id]
            self.assertEqual(space.low[0], config['min'])
            self.assertEqual(space.high[0], config['max'])

        # Check action space matches output config
        for comp_id, config in self.io_config['output'].items():
            self.assertIn(comp_id, self.env.action_space.spaces)
            space = self.env.action_space.spaces[comp_id]
            self.assertEqual(space.low[0], config['min'])
            self.assertEqual(space.high[0], config['max'])

    def test_reset(self):
        """Test environment reset functionality"""
        obs, info = self.env.reset()
        
        # Check observation structure
        self.assertEqual(set(obs.keys()), set(self.io_config['input'].keys()))
        
        # Check observation values are within bounds
        for comp_id, value in obs.items():
            config = self.io_config['input'][comp_id]
            self.assertTrue(np.all(value >= config['min']))
            self.assertTrue(np.all(value <= config['max']))

    def test_step(self):
        """Test environment step functionality"""
        obs, _ = self.env.reset()
        
        # Create valid action
        action = {
            comp_id: np.array([
                (config['max'] + config['min']) / 2  # middle of range
            ])
            for comp_id, config in self.io_config['output'].items()
        }
        
        # Take step
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Check observation structure maintained
        self.assertEqual(set(next_obs.keys()), set(self.io_config['input'].keys()))
        
        # Check values within bounds
        for comp_id, value in next_obs.items():
            config = self.io_config['input'][comp_id]
            self.assertTrue(np.all(value >= config['min']))
            self.assertTrue(np.all(value <= config['max']))
        
        # Check reward is finite
        self.assertTrue(np.isfinite(reward))

    def test_time_progression(self):
        """Test that simulation time progresses correctly"""
        initial_time = self.env.simulator.dateTime
        
        # Take multiple steps
        for _ in range(5):
            action = self.get_dummy_action()
            _, _, _, _, _ = self.env.step(action)
        
        # Check time advanced correctly
        expected_time = initial_time + timedelta(seconds=5 * self.env.simulator.stepSize)
        self.assertEqual(self.env.simulator.dateTime, expected_time)

    def test_simulation_bounds(self):
        """Test simulation respects time bounds"""
        # Run until end
        done = False
        while not done:
            action = self.get_dummy_action()
            _, _, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
        
        # Check we didn't exceed endTime
        self.assertLessEqual(self.env.simulator.dateTime, self.env.simulator.endTime)

    def test_invalid_actions(self):
        """Test environment handles invalid actions appropriately"""
        self.env.reset()
        
        # Test action too high
        action = {
            comp_id: np.array([config['max'] * 2])
            for comp_id, config in self.io_config['output'].items()
        }
        next_obs, _, _, _, _ = self.env.step(action)
        # Should still get valid observations
        self.check_observations_valid(next_obs)

        # Test action too low
        action = {
            comp_id: np.array([config['min'] - 1])
            for comp_id, config in self.io_config['output'].items()
        }
        next_obs, _, _, _, _ = self.env.step(action)
        self.check_observations_valid(next_obs)

    def test_component_failure(self):
        """Test environment handles component failures gracefully"""
        self.env.reset()
        
        # Simulate component failure by setting invalid values
        for component in self.env.simulator.model.components.values():
            if hasattr(component, 'output'):
                for output in component.output.values():
                    output.set(float('nan'))
        
        # Environment should still function
        action = self.get_dummy_action()
        try:
            next_obs, _, _, _, _ = self.env.step(action)
            self.check_observations_valid(next_obs)
        except Exception as e:
            self.fail(f"Environment failed to handle component failure: {e}")

    def test_step_timing(self):
        """Test environment step execution time"""
        import time
        
        self.env.reset()
        action = self.get_dummy_action()
        
        # Measure average step time
        n_steps = 100
        start_time = time.time()
        
        for _ in range(n_steps):
            self.env.step(action)
        
        avg_step_time = (time.time() - start_time) / n_steps
        
        # Assert reasonable performance (adjust threshold as needed)
        self.assertLess(avg_step_time, 0.1)  # 100ms per step

    def test_memory_usage(self):
        """Test environment memory usage"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Run many steps
        self.env.reset()
        for _ in range(1000):
            self.env.step(self.get_dummy_action())
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Check memory growth is reasonable
        self.assertLess(memory_increase / 1024 / 1024, 100)  # Less than 100MB growth

    def get_dummy_action(self):
        """Create a valid dummy action"""
        return {
            comp_id: np.array([(config['max'] + config['min']) / 2])
            for comp_id, config in self.io_config['output'].items()
        }

    def check_observations_valid(self, obs):
        """Check if observations are valid"""
        for comp_id, value in obs.items():
            config = self.io_config['input'][comp_id]
            self.assertTrue(np.all(np.isfinite(value)))
            self.assertTrue(np.all(value >= config['min']))
            self.assertTrue(np.all(value <= config['max']))


if __name__ == "__main__":
    unittest.main()
    
