"""
This script uses the t4b_gym environment with the t4b model of the BOPTEST model multizone_simple_air 
It defines a custom reward function and uses the PPO algorithm to control different setpoints of the model. 
"""

import twin4build as tb
import datetime
import twin4build.examples.utils as utils
from dateutil.tz import gettz 
import sys
import os
import logging
import copy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(MAIN_DIR)
from t4b_gym.t4b_gym_env import T4BGymEnv, NormalizedObservationWrapper, NormalizedActionWrapper
from boptest_model.rooms_and_ahu_model import load_model_and_params
from use_case.model_eval import test_model

log_dir = os.path.join(SCRIPT_DIR, 'logs_finetune_bc_4')
os.makedirs(log_dir, exist_ok=True)


POLICY_CONFIG_PATH = os.path.join(SCRIPT_DIR, "policy_input_output_co2sets.json")
device = 'cpu'
bc_model_path = os.path.join(os.path.dirname(__file__), "ppo_pretrained_bc.zip")
policy_size = 'large'
test_model_flag = True
reload_model_flag = False
total_timesteps = 1000000

def get_custom_env(stepSize, start_time, end_time):
    model = load_model_and_params()
    class T4BGymEnvCustomReward(T4BGymEnv):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.previous_objective = 0.0
        def get_reward(self, action, observation):
            zones = ['core', 'north', 'east', 'south', 'west']
            
            # Get current temperature violations with comfort gradients
            temp_violations = []
            comfort_rewards = []
            
            for zone in zones:
                temp = self.simulator.model.components[f"{zone}_indoor_temp_sensor"].output["measuredValue"]
                heating_setpoint = self.simulator.model.components[f"{zone}_temperature_heating_setpoint"].output["scheduleValue"]
                cooling_setpoint = self.simulator.model.components[f"{zone}_temperature_cooling_setpoint"].output["scheduleValue"]
                
                # Calculate comfort gradient reward with pre-cooling/pre-heating incentives
                comfort_reward = self.get_comfort_gradient_reward(temp, heating_setpoint, cooling_setpoint, zone, observation)
                comfort_rewards.append(comfort_reward)
                
                # Calculate immediate temperature violations (keeping original logic for compatibility)
                heating_violation = max(0, heating_setpoint - temp)
                cooling_violation = max(0, temp - cooling_setpoint)
                zone_violation = (1+heating_violation)**2 + (1+cooling_violation)**2
                temp_violations.append(zone_violation)
            
            # Get forecast-aware temperature violation prediction
            forecast_violation_penalty = self.get_forecast_aware_temp_penalty(observation, zones)
            
            # Calculate energy efficiency bonus for exploiting relaxed setpoints
            energy_efficiency_bonus = self.calculate_energy_efficiency_bonus(observation, zones)
            
            # Original immediate temperature violation penalty
            temp_violation_penalty = 1000 * sum(temp_violations)
            
            # Energy consumption calculations (keeping original logic)
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
            
            # Normalize all reward components to similar ranges (roughly 0-1)
            # Comfort gradient reward (already in range -0.5 to 1.0, normalize to 0-1)
            comfort_reward_total = sum(comfort_rewards) / len(comfort_rewards)  # Average per zone
            comfort_reward_normalized = (comfort_reward_total + 0.5) / 1.5  # Normalize to 0-1
            
            # Temperature violations (typically 0-50 range, normalize to 0-1)
            immediate_temp_penalty = sum(temp_violations) / len(temp_violations)  # Average per zone
            immediate_temp_normalized = min(immediate_temp_penalty * 10.0, 1.0)  # Normalize to 0-1, cap at 1
            
            # Forecast violations (typically 0-100 range, normalize to 0-1)
            forecast_temp_normalized = min(forecast_violation_penalty / 100.0, 1.0)  # Normalize to 0-1, cap at 1
            
            # Energy penalties (typically 0-1000 range, normalize to 0-1)
            total_energy = coils_power_consumption_penalty + ahu_power_consumption_penalty
            energy_normalized = min(total_energy / 100.0, 1.0)  # Normalize to 0-1, cap at 1
            
            # Energy efficiency bonus (typically 0-0.5 range, normalize to 0-1)
            energy_efficiency_normalized = min(energy_efficiency_bonus * 2.0, 1.0)  # Normalize to 0-1, cap at 1
            
            # Combine normalized components with meaningful weights
            reward = (comfort_reward_normalized * 0.2 -      # 20% weight for comfort
                     immediate_temp_normalized * 0.3 -       # 30% weight for immediate violations
                     forecast_temp_normalized * 0.2 -        # 20% weight for forecast violations
                     energy_normalized * 0.25 +              # 25% weight for energy
                     energy_efficiency_normalized * 0.05)    # 5% weight for energy efficiency bonus
                     
            
            if np.isnan(reward):
                raise ValueError("Reward is not a number")
            return -reward
        
        def get_comfort_gradient_reward(self, zone_temp, heating_sp, cooling_sp, zone_name=None, observation=None):
            """Calculate comfort gradient reward with pre-cooling/pre-heating incentives.
            Encourages exploiting relaxed nighttime setpoints for energy efficiency."""
            
            # Get forecast setpoints if available
            forecast_setpoints = None
            if zone_name and observation is not None:
                forecast_setpoints = self.get_forecast_setpoints(zone_name, observation)
            
            # Determine current mode and target
            if zone_temp < heating_sp:
                target_temp = heating_sp
                mode = "heating"
            elif zone_temp > cooling_sp:
                target_temp = cooling_sp
                mode = "cooling"
            else:
                if zone_temp < (heating_sp + cooling_sp) / 2:
                    target_temp = heating_sp
                    mode = "heating"
                else:
                    target_temp = cooling_sp
                    mode = "cooling"
            
            # Calculate base comfort reward
            distance_from_target = abs(zone_temp - target_temp)
            comfort_range = (cooling_sp - heating_sp) / 2
            
            if distance_from_target <= comfort_range:
                base_reward = 1.0 - (distance_from_target / comfort_range) * 0.3
            else:
                excess = distance_from_target - comfort_range
                base_reward = -0.5 - (excess / comfort_range) * 2.0
            
            # Add pre-cooling/pre-heating incentives if forecasts are available
            strategy_bonus = 0.0
            if forecast_setpoints and len(forecast_setpoints['heating']) > 0:
                strategy_bonus = self.calculate_strategy_bonus(
                    zone_temp, heating_sp, cooling_sp, forecast_setpoints, mode
                )
            
            return base_reward + strategy_bonus
        
        def calculate_strategy_bonus(self, zone_temp, heating_sp, cooling_sp, forecast_setpoints, current_mode):
            """Calculate bonus reward for pre-cooling/pre-heating strategies."""
            
            # Look ahead 2-6 hours (4-12 timesteps with 30-min steps) for setpoint changes
            look_ahead_hours = 4  # 4 hours ahead
            look_ahead_steps = min(look_ahead_hours * 2, len(forecast_setpoints['heating']))  # 2 steps per hour
            
            if look_ahead_steps < 2:
                return 0.0
            
            # Find significant setpoint changes in the forecast
            current_comfort_range = cooling_sp - heating_sp
            strategy_opportunities = []
            
            for i in range(1, look_ahead_steps):
                future_heating_sp = forecast_setpoints['heating'][i]
                future_cooling_sp = forecast_setpoints['cooling'][i]
                future_comfort_range = future_cooling_sp - future_heating_sp
                
                # Check if there's a significant change in comfort range
                range_change = future_comfort_range - current_comfort_range
                
                if abs(range_change) > 2.0:  # Significant change (more than 2°C)
                    # Determine if this is a relaxation (nighttime) or tightening (daytime)
                    if range_change > 0:
                        # Relaxation - opportunity for pre-cooling/pre-heating
                        strategy_opportunities.append({
                            'hours_ahead': i / 2,  # Convert steps to hours
                            'type': 'relaxation',
                            'range_change': range_change,
                            'future_heating_sp': future_heating_sp,
                            'future_cooling_sp': future_cooling_sp
                        })
                    else:
                        # Tightening - need to prepare for stricter requirements
                        strategy_opportunities.append({
                            'hours_ahead': i / 2,
                            'type': 'tightening',
                            'range_change': range_change,
                            'future_heating_sp': future_heating_sp,
                            'future_cooling_sp': future_cooling_sp
                        })
            
            if not strategy_opportunities:
                return 0.0
            
            # Calculate strategy bonus based on opportunities
            total_bonus = 0.0
            
            for opportunity in strategy_opportunities:
                if opportunity['type'] == 'relaxation':
                    # Pre-cooling/pre-heating opportunity
                    bonus = self.calculate_preconditioning_bonus(
                        zone_temp, heating_sp, cooling_sp, opportunity, current_mode
                    )
                else:
                    # Preparation for stricter requirements
                    bonus = self.calculate_preparation_bonus(
                        zone_temp, heating_sp, cooling_sp, opportunity, current_mode
                    )
                
                # Weight by time distance (closer opportunities get higher weight)
                time_weight = max(0.1, 1.0 - (opportunity['hours_ahead'] / 6.0))
                total_bonus += bonus * time_weight
            
            return total_bonus * 0.5  # Scale factor for strategy bonus
        
        def calculate_preconditioning_bonus(self, zone_temp, heating_sp, cooling_sp, opportunity, current_mode):
            """Calculate bonus for pre-cooling/pre-heating during relaxation periods."""
            
            future_heating_sp = opportunity['future_heating_sp']
            future_cooling_sp = opportunity['future_cooling_sp']
            range_change = opportunity['range_change']
            
            # Determine optimal preconditioning strategy
            if current_mode == "cooling":
                # Currently cooling - check if we can pre-cool more aggressively
                if zone_temp > future_cooling_sp:
                    # Can pre-cool to future cooling setpoint
                    optimal_temp = future_cooling_sp
                    distance_to_optimal = zone_temp - optimal_temp
                    
                    if distance_to_optimal > 0.5:  # Significant pre-cooling opportunity
                        # Bonus increases with range change and decreases with distance
                        bonus = min(0.3, (range_change / 10.0) * (distance_to_optimal / 5.0))
                        return bonus
            
            elif current_mode == "heating":
                # Currently heating - check if we can pre-heat more aggressively
                if zone_temp < future_heating_sp:
                    # Can pre-heat to future heating setpoint
                    optimal_temp = future_heating_sp
                    distance_to_optimal = optimal_temp - zone_temp
                    
                    if distance_to_optimal > 0.5:  # Significant pre-heating opportunity
                        # Bonus increases with range change and decreases with distance
                        bonus = min(0.3, (range_change / 10.0) * (distance_to_optimal / 5.0))
                        return bonus
            
            return 0.0
        
        def calculate_preparation_bonus(self, zone_temp, heating_sp, cooling_sp, opportunity, current_mode):
            """Calculate bonus for preparing for stricter requirements."""
            
            future_heating_sp = opportunity['future_heating_sp']
            future_cooling_sp = opportunity['future_cooling_sp']
            
            # Check if current temperature is well-positioned for future requirements
            future_midpoint = (future_heating_sp + future_cooling_sp) / 2
            current_midpoint = (heating_sp + cooling_sp) / 2
            
            # Calculate how well-positioned we are for the future
            distance_from_future_midpoint = abs(zone_temp - future_midpoint)
            future_comfort_range = future_cooling_sp - future_heating_sp
            
            # Bonus for being close to future comfort zone center
            if distance_from_future_midpoint < future_comfort_range / 2:
                bonus = 0.2 * (1.0 - distance_from_future_midpoint / (future_comfort_range / 2))
                return bonus
            
            return 0.0
        
        def calculate_energy_efficiency_bonus(self, observation, zones):
            """Calculate bonus for energy-efficient strategies using relaxed setpoints."""
            
            total_bonus = 0.0
            
            for zone in zones:
                # Get current conditions
                current_temp = self.simulator.model.components[f"{zone}_indoor_temp_sensor"].output["measuredValue"]
                current_heating_sp = self.simulator.model.components[f"{zone}_temperature_heating_setpoint"].output["scheduleValue"]
                current_cooling_sp = self.simulator.model.components[f"{zone}_temperature_cooling_setpoint"].output["scheduleValue"]
                current_comfort_range = current_cooling_sp - current_heating_sp
                
                # Get forecast setpoints
                forecast_setpoints = self.get_forecast_setpoints(zone, observation)
                if not forecast_setpoints or len(forecast_setpoints['heating']) == 0:
                    continue
                
                # Look for energy efficiency opportunities
                zone_bonus = 0.0
                
                # Check if we're in a relaxed period (nighttime)
                if current_comfort_range > 8.0:  # Relaxed setpoints (18°C range vs 2°C daytime)
                    # Bonus for exploiting relaxed setpoints for energy savings
                    zone_bonus += self.calculate_relaxed_period_bonus(
                        current_temp, current_heating_sp, current_cooling_sp, forecast_setpoints
                    )
                
                # Check for pre-cooling/pre-heating opportunities
                zone_bonus += self.calculate_preconditioning_efficiency_bonus(
                    current_temp, current_heating_sp, current_cooling_sp, forecast_setpoints
                )
                
                total_bonus += zone_bonus
            
            return total_bonus / len(zones)  # Average across zones
        
        def calculate_relaxed_period_bonus(self, current_temp, heating_sp, cooling_sp, forecast_setpoints):
            """Calculate bonus for energy savings during relaxed setpoint periods."""
            
            comfort_range = cooling_sp - heating_sp
            if comfort_range < 8.0:  # Not a relaxed period
                return 0.0
            
            # Find when setpoints will tighten (return to daytime values)
            tightening_hours = None
            for i in range(1, min(12, len(forecast_setpoints['heating']))):  # Look up to 6 hours ahead
                future_heating_sp = forecast_setpoints['heating'][i]
                future_cooling_sp = forecast_setpoints['cooling'][i]
                future_comfort_range = future_cooling_sp - future_heating_sp
                
                if future_comfort_range < 4.0:  # Setpoints will tighten
                    tightening_hours = i / 2  # Convert steps to hours
                    break
            
            if tightening_hours is None:
                return 0.0
            
            # Calculate energy savings potential
            # During relaxed periods, we can allow more temperature drift
            # Bonus for staying within relaxed bounds while preparing for tightening
            current_midpoint = (heating_sp + cooling_sp) / 2
            distance_from_midpoint = abs(current_temp - current_midpoint)
            
            # Higher bonus for being closer to the relaxed midpoint
            # This encourages energy savings while maintaining comfort
            if distance_from_midpoint < comfort_range / 4:
                bonus = 0.3 * (1.0 - distance_from_midpoint / (comfort_range / 4))
                # Scale by time until tightening (more time = more savings potential)
                time_scale = min(1.0, tightening_hours / 4.0)
                return bonus * time_scale
            
            return 0.0
        
        def calculate_preconditioning_efficiency_bonus(self, current_temp, heating_sp, cooling_sp, forecast_setpoints):
            """Calculate bonus for efficient pre-cooling/pre-heating strategies."""
            
            # Look for opportunities to use relaxed setpoints for energy storage
            total_bonus = 0.0
            
            for i in range(1, min(8, len(forecast_setpoints['heating']))):  # Look up to 4 hours ahead
                future_heating_sp = forecast_setpoints['heating'][i]
                future_cooling_sp = forecast_setpoints['cooling'][i]
                future_comfort_range = future_cooling_sp - future_heating_sp
                current_comfort_range = cooling_sp - heating_sp
                
                # Check if we're transitioning from relaxed to strict setpoints
                if current_comfort_range > 8.0 and future_comfort_range < 4.0:
                    # Opportunity for energy storage during relaxed period
                    hours_ahead = i / 2
                    
                    # Calculate optimal temperature for energy storage
                    future_midpoint = (future_heating_sp + future_cooling_sp) / 2
                    current_midpoint = (heating_sp + cooling_sp) / 2
                    
                    # Bonus for moving toward future optimal temperature
                    distance_to_future_optimal = abs(current_temp - future_midpoint)
                    distance_to_current_optimal = abs(current_temp - current_midpoint)
                    
                    if distance_to_future_optimal < distance_to_current_optimal:
                        # Moving toward future optimal - good energy storage strategy
                        improvement = distance_to_current_optimal - distance_to_future_optimal
                        bonus = 0.2 * improvement * (1.0 - hours_ahead / 4.0)  # Decay with time
                        total_bonus += bonus
            
            return total_bonus
        
        def get_forecast_aware_temp_penalty(self, observation, zones):
            """Calculate forecast-aware temperature violation penalty."""
            if not hasattr(self, '_observations') or self._observations is None:
                # Fallback to immediate violations only if observation names not available
                return 0.0
            
            total_forecast_penalty = 0.0
            
            for zone in zones:
                # Get current temperature
                current_temp = self.simulator.model.components[f"{zone}_indoor_temp_sensor"].output["measuredValue"]
                
                # Get forecast setpoints and weather
                forecast_setpoints = self.get_forecast_setpoints(zone, observation)
                forecast_weather = self.get_forecast_weather(observation)
                
                # Predict future temperature violations
                future_violations = self.predict_future_violations(
                    zone, current_temp, forecast_setpoints, forecast_weather
                )
                
                # Weighted penalty: immediate + predicted future violations
                immediate_violation = self.calculate_temp_violation(zone)
                future_penalty = sum([0.8**i * v for i, v in enumerate(future_violations)])
                
                total_forecast_penalty += immediate_violation + 0.5 * future_penalty
            
            return total_forecast_penalty * 500  # Scale factor for forecast penalties
        
        def get_forecast_setpoints(self, zone, observation):
            """Extract forecast setpoints for a zone from observation array."""
            if not hasattr(self, '_observations') or self._observations is None:
                return None
            
            forecast_horizon = 50  # From environment configuration
            setpoints = {'heating': [], 'cooling': []}
            
            # Find indices for heating and cooling setpoint forecasts
            heating_sp_name = f"{zone}_temperature_heating_setpoint:scheduleValue"
            cooling_sp_name = f"{zone}_temperature_cooling_setpoint:scheduleValue"
            
            try:
                # Find the indices in the observation array
                heating_indices = [i for i, name in enumerate(self._observations) if heating_sp_name in name]
                cooling_indices = [i for i, name in enumerate(self._observations) if cooling_sp_name in name]
                
                if heating_indices and cooling_indices:
                    # Extract forecast values (first forecast_horizon + 1 values)
                    for i in range(min(forecast_horizon + 1, len(heating_indices))):
                        setpoints['heating'].append(observation[heating_indices[i]])
                        setpoints['cooling'].append(observation[cooling_indices[i]])
                else:
                    # Fallback: use current setpoints for all future timesteps
                    current_heating_sp = self.simulator.model.components[f"{zone}_temperature_heating_setpoint"].output["scheduleValue"]
                    current_cooling_sp = self.simulator.model.components[f"{zone}_temperature_cooling_setpoint"].output["scheduleValue"]
                    setpoints['heating'] = [current_heating_sp] * (forecast_horizon + 1)
                    setpoints['cooling'] = [current_cooling_sp] * (forecast_horizon + 1)
                
            except (IndexError, ValueError):
                # Fallback: use current setpoints for all future timesteps
                current_heating_sp = self.simulator.model.components[f"{zone}_temperature_heating_setpoint"].output["scheduleValue"]
                current_cooling_sp = self.simulator.model.components[f"{zone}_temperature_cooling_setpoint"].output["scheduleValue"]
                setpoints['heating'] = [current_heating_sp] * (forecast_horizon + 1)
                setpoints['cooling'] = [current_cooling_sp] * (forecast_horizon + 1)
            
            return setpoints
        
        def get_forecast_weather(self, observation):
            """Extract weather forecasts from observation array."""
            if not hasattr(self, '_observations') or self._observations is None:
                return None
            
            forecast_horizon = 50
            weather = {'outdoor_temp': [], 'solar': []}
            
            try:
                # Find indices for weather forecasts
                outdoor_temp_indices = [i for i, name in enumerate(self._observations) if 'outdoorTemperature' in name]
                solar_indices = [i for i, name in enumerate(self._observations) if 'globalIrradiation' in name]
                
                if outdoor_temp_indices:
                    for i in range(min(forecast_horizon + 1, len(outdoor_temp_indices))):
                        weather['outdoor_temp'].append(observation[outdoor_temp_indices[i]])
                else:
                    # Fallback: use current weather for all future timesteps
                    current_outdoor_temp = self.simulator.model.components["outdoor_environment"].output["outdoorTemperature"]
                    weather['outdoor_temp'] = [current_outdoor_temp] * (forecast_horizon + 1)
                
                if solar_indices:
                    for i in range(min(forecast_horizon + 1, len(solar_indices))):
                        weather['solar'].append(observation[solar_indices[i]])
                else:
                    weather['solar'] = [0] * (forecast_horizon + 1)  # Default solar value
                
            except (IndexError, ValueError):
                # Fallback: use current weather for all future timesteps
                current_outdoor_temp = self.simulator.model.components["outdoor_environment"].output["outdoorTemperature"]
                weather['outdoor_temp'] = [current_outdoor_temp] * (forecast_horizon + 1)
                weather['solar'] = [0] * (forecast_horizon + 1)  # Default solar value
            
            return weather
        
        def predict_future_violations(self, zone, current_temp, forecast_setpoints, forecast_weather):
            """Predict future temperature violations based on forecasts."""
            if forecast_setpoints is None or forecast_weather is None:
                return [0] * 10  # Return zeros if forecasts not available
            
            violations = []
            temp = current_temp
            
            # Zone-specific thermal parameters (simplified)
            thermal_params = {
                'core': {'time_constant': 7200, 'outdoor_influence': 0.2},      # 2 hours, less outdoor influence
                'north': {'time_constant': 5400, 'outdoor_influence': 0.4},     # 1.5 hours
                'south': {'time_constant': 5400, 'outdoor_influence': 0.4},     # 1.5 hours
                'east': {'time_constant': 5400, 'outdoor_influence': 0.4},      # 1.5 hours
                'west': {'time_constant': 5400, 'outdoor_influence': 0.4}       # 1.5 hours
            }
            
            params = thermal_params.get(zone, {'time_constant': 5400, 'outdoor_influence': 0.3})
            thermal_time_constant = params['time_constant']
            outdoor_influence = params['outdoor_influence']
            step_size = self.step_size
            
            for i in range(min(10, len(forecast_setpoints['heating']))):  # Predict next 10 steps
                heating_sp = forecast_setpoints['heating'][i]
                cooling_sp = forecast_setpoints['cooling'][i]
                outdoor_temp = forecast_weather['outdoor_temp'][i] if i < len(forecast_weather['outdoor_temp']) else outdoor_temp
                
                # Improved thermal response model
                # Temperature tends toward a weighted combination of outdoor temp and setpoint
                setpoint_influence = 1.0 - outdoor_influence
                temp_toward = outdoor_temp * outdoor_influence + (heating_sp + cooling_sp) / 2 * setpoint_influence
                
                # Exponential decay toward target temperature
                temp = temp + (temp_toward - temp) * (step_size / thermal_time_constant)
                
                # Calculate violation using the same formula as immediate violations
                heating_violation = max(0, heating_sp - temp)
                cooling_violation = max(0, temp - cooling_sp)
                violation = (1 + heating_violation)**2 + (1 + cooling_violation)**2
                
                violations.append(violation)
            
            return violations
        
        def calculate_temp_violation(self, zone):
            """Calculate current temperature violation for a zone."""
            temp = self.simulator.model.components[f"{zone}_indoor_temp_sensor"].output["measuredValue"]
            heating_setpoint = self.simulator.model.components[f"{zone}_temperature_heating_setpoint"].output["scheduleValue"]
            cooling_setpoint = self.simulator.model.components[f"{zone}_temperature_cooling_setpoint"].output["scheduleValue"]
            
            heating_violation = max(0, heating_setpoint - temp)
            cooling_violation = max(0, temp - cooling_setpoint)
            return (1 + heating_violation)**2 + (1 + cooling_violation)**2
        
        def debug_forecast_extraction(self, observation):
            """Debug method to print forecast extraction information."""
            if hasattr(self, '_observations') and self._observations is not None:
                print(f"Total observations: {len(self._observations)}")
                print(f"Observation names: {self._observations[:10]}...")  # First 10 names
                
                # Check for setpoint forecasts
                setpoint_names = [name for name in self._observations if 'temperature_heating_setpoint' in name or 'temperature_cooling_setpoint' in name]
                print(f"Setpoint forecast names found: {len(setpoint_names)}")
                if setpoint_names:
                    print(f"First few setpoint names: {setpoint_names[:5]}")
                
                # Check for weather forecasts
                weather_names = [name for name in self._observations if 'outdoorTemperature' in name or 'globalIrradiation' in name]
                print(f"Weather forecast names found: {len(weather_names)}")
                if weather_names:
                    print(f"Weather names: {weather_names}")
            else:
                print("No observation names available")
    
    
    # Create base environment first to get observation names
    base_env = T4BGymEnv(
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
    
    # Create custom environment with observation names
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
    
    # Copy observation names from base environment
    env._observations = base_env._observations

    return env

class SingleLineProgressCallback(BaseCallback):
    """
    Custom callback for printing training progress in a single line.
    Displays model's cumulative timesteps against an overall target.
    """
    def __init__(self, overall_target_agent_steps, verbose=0):
        super().__init__(verbose)
        self.overall_target_agent_steps = overall_target_agent_steps
        self.target_reached_printed_newline = False # Flag to ensure newline is printed only once

    def _init_callback(self) -> None:
        super()._init_callback() # Call parent's _init_callback if it exists and is needed
        # self.model is now available, and self.model.num_timesteps is its current cumulative step count.
        if self.model.num_timesteps > 0:
            print(f"\nCallback: Continuing training. Model has {self.model.num_timesteps} of {self.overall_target_agent_steps} timesteps.")
        else:
            print(f"\nCallback: Starting training from 0. Target: {self.overall_target_agent_steps} timesteps.")
        
        # If already at/beyond target when starting, mark newline as (conceptually) done for _on_step
        if self.model.num_timesteps >= self.overall_target_agent_steps:
            print() # Ensure this initial status message is on its own line if target already met
            self.target_reached_printed_newline = True
        else:
            self.target_reached_printed_newline = False

    def _on_step(self) -> bool:
        current_cumulative_steps = self.num_timesteps
        progress_percent = (current_cumulative_steps / self.overall_target_agent_steps) * 100 if self.overall_target_agent_steps > 0 else 0
        print(
            f"\rTraining progress: {current_cumulative_steps}/{self.overall_target_agent_steps} steps "
            f"({progress_percent:.1f}%)",
            end="",
        )
        
        # Add a newline only once when the overall target is first reached or exceeded.
        if current_cumulative_steps >= self.overall_target_agent_steps and not self.target_reached_printed_newline:
            print() # Print the newline
            self.target_reached_printed_newline = True # Set flag to prevent further newlines
        return True


class PeriodicSaveCallback(BaseCallback):
    """
    Callback for saving a model periodically during training.
    Fixed to work properly with vectorized environments.
    """
    def __init__(self, save_freq: int, save_path: str, verbose: int = 0):
        """
        Args:
            save_freq: Save frequency in total timesteps (not callback calls)
            save_path: Path to save the model
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.last_save_step = 0

    def _on_step(self) -> bool:
        current_steps = self.model.num_timesteps
        if current_steps - self.last_save_step >= self.save_freq:
            model_path = os.path.join(self.save_path, f'model_{current_steps}')
            try:
                self.model.save(model_path)
                self.last_save_step = current_steps
                if self.verbose > 0:
                    print(f"\nSaving model checkpoint to {model_path}")
            except Exception as e:
                if self.verbose > 0:
                    print(f"\nWarning: Failed to save model at step {current_steps}: {e}")
        return True


class AdaptKLtoBC(BaseCallback):
    def __init__(
        self,
        target_kl=0.02,
        beta0=1.0,
        beta_final=0.05,          # set to 0.0 if you want to remove KL entirely
        decay_steps=200_000,      # how fast to relax the BC constraint (timesteps)
        up=1.5,
        down=0.7,
        ema=0.9,                  # EMA for KL smoothing
        min_beta=1e-5,
        max_beta=5.0,
        verbose=0,
    ):
        super().__init__(verbose)
        self.target_kl = float(target_kl)
        self.beta0 = float(beta0)
        self.beta_final = float(beta_final)
        self.decay_steps = int(decay_steps)
        self.up, self.down = float(up), float(down)
        self.ema = float(ema)
        self.min_beta, self.max_beta = float(min_beta), float(max_beta)
        self._ema_kl = None

    def _envelope(self, t):
        # Exponential decay: beta_env = beta0 * (beta_final/beta0)^(progress)
        progress = min(1.0, t / max(1, self.decay_steps))
        return self.beta0 * (self.beta_final / self.beta0) ** progress

    def _on_step(self) -> bool:
        # This method is called on every step, but we only want to adapt KL on rollout end
        # So we just return True to continue training
        return True

    def _on_training_start(self) -> None:
        # Initialize model beta to envelope at t=0 if not set
        if not hasattr(self.model, "beta_kl"):
            self.model.beta_kl = self.beta0
        self.model.set_beta_kl(self.model.beta_kl)

    def _on_rollout_end(self) -> bool:
        # Read KL measured during training step - try multiple sources
        kl = None
        
        # First try to get from model instance
        if hasattr(self.model, 'kl_bc_value'):
            kl = self.model.kl_bc_value
            if self.verbose > 0:
                print(f"[KL-BC] Found KL value in model instance: {kl}")
        
        # Fallback to logger
        if kl is None:
            kl = self.model.logger.name_to_value.get("train/kl_bc", None)
            if self.verbose > 0 and kl is not None:
                print(f"[KL-BC] Found KL value in logger: {kl}")
        
        # Debug: Print available keys in logger if still None
        if self.verbose > 0 and kl is None:
            available_keys = list(self.model.logger.name_to_value.keys())
            print(f"[KL-BC] DEBUG: Available logger keys: {available_keys}")
            print(f"[KL-BC] DEBUG: Looking for 'train/kl_bc', found: {kl}")
            print(f"[KL-BC] DEBUG: Model has kl_bc_value attribute: {hasattr(self.model, 'kl_bc_value')}")
        
        if kl is None:
            if self.verbose > 0:
                print(f"[KL-BC] WARNING: No KL value found. Available logger keys: {list(self.model.logger.name_to_value.keys())}")
            return True

        # Clamp extreme KL values to prevent numerical instability
        if kl > 10.0:  # Reduced threshold to match new KL clamp
            print(f"[KL-BC] WARNING: Extreme KL value detected: {kl:.2e}. Clamping to 10.0")
            kl = 10.0
        elif np.isnan(kl) or np.isinf(kl):
            print(f"[KL-BC] WARNING: Invalid KL value detected: {kl}. Using previous value or 1.0")
            kl = self._ema_kl if self._ema_kl is not None else 1.0

        # EMA smoothing
        if self._ema_kl is None:
            self._ema_kl = kl
        else:
            self._ema_kl = self.ema * self._ema_kl + (1 - self.ema) * kl

        beta = float(self.model.beta_kl)

        # Adaptive multiplicative control around target KL
        if self._ema_kl > 2.0 * self.target_kl:
            beta *= self.up
        elif self._ema_kl < 0.5 * self.target_kl:
            beta *= self.down

        # Apply decaying envelope as an upper bound (monotonic relaxation)
        beta_env = self._envelope(self.num_timesteps)
        beta = min(beta, beta_env)

        # Final clamp
        beta = float(np.clip(beta, self.min_beta, self.max_beta))
        self.model.set_beta_kl(beta)

        # Log for monitoring
        self.model.logger.record("train/beta_kl", beta)
        self.model.logger.record("train/beta_env", beta_env)
        self.model.logger.record("train/kl_bc_ema", self._ema_kl)
        if self.verbose:
            print(f"[KL-BC] kl_ema={self._ema_kl:.4f} env={beta_env:.4f} -> beta={beta:.4f}")
        return True


class PolicyDistributionMonitorCallback(BaseCallback):
    """
    Monitor policy distribution parameters to detect numerical instability.
    """
    def __init__(self, check_freq=1000, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        
    def _on_rollout_end(self) -> bool:
        if self.num_timesteps % self.check_freq == 0:
            # Get policy parameters
            policy = self.model.policy
            if hasattr(policy, 'action_net'):
                # For continuous action spaces
                action_mean = policy.action_net.mean
                action_log_std = policy.action_net.log_std
                
                # Check for numerical issues
                mean_norm = torch.norm(action_mean).item()
                log_std_norm = torch.norm(action_log_std).item()
                log_std_min = torch.min(action_log_std).item()
                log_std_max = torch.max(action_log_std).item()
                
                # Log warnings for potential issues
                if log_std_min < -10:  # Very small standard deviations
                    print(f"[PolicyMonitor] WARNING: Very small log_std detected: {log_std_min:.4f}")
                if log_std_max > 10:  # Very large standard deviations
                    print(f"[PolicyMonitor] WARNING: Very large log_std detected: {log_std_max:.4f}")
                if mean_norm > 100:  # Very large means
                    print(f"[PolicyMonitor] WARNING: Large action means detected: {mean_norm:.4f}")
                
                # Log distribution statistics
                self.model.logger.record("policy/log_std_min", log_std_min)
                self.model.logger.record("policy/log_std_max", log_std_max)
                self.model.logger.record("policy/mean_norm", mean_norm)
                
        return True


class ActionBoundsMonitorCallback(BaseCallback):
    """Monitor action bounds during training to catch when actions go out of bounds."""
    
    def __init__(self, check_freq=1000, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.step_count = 0
        
    def _on_step(self) -> bool:
        self.step_count += 1
        
        # Check action bounds periodically
        if self.step_count % self.check_freq == 0:
            # Get a sample observation and test policy output
            obs = self.training_env.observation_space.sample()
            with torch.no_grad():
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Check if actions are within bounds
                if action.min() < -1.1 or action.max() > 1.1:  # Allow small tolerance
                    print(f"\nWARNING: Actions out of bounds at step {self.step_count}!")
                    print(f"Action range: [{action.min():.4f}, {action.max():.4f}]")
                    print(f"Action shape: {action.shape}")
                    
                    # Log to tensorboard if available
                    if hasattr(self.model, 'logger'):
                        self.model.logger.record("train/action_min", action.min())
                        self.model.logger.record("train/action_max", action.max())
        
        return True


class RobustNormalizedActionWrapper(gym.ActionWrapper):
    """Robust version of NormalizedActionWrapper that handles out-of-bounds actions gracefully."""
    
    def __init__(self, env):
        super().__init__(env)
        
        # Assert that original action space is a Box space
        assert isinstance(self.unwrapped.action_space, gym.spaces.Box), 'This wrapper only works with continuous action space (spaces.Box)'
        
        # Store low and high bounds of action space
        self.low = self.unwrapped.action_space.low
        self.high = self.unwrapped.action_space.high
        
        # Check for invalid action space bounds
        if np.any(self.low >= self.high):
            raise ValueError("Action space bounds are invalid: low >= high")
            
        # Check for NaN values in action space bounds
        if np.isnan(self.low).any() or np.isnan(self.high).any():
            raise ValueError("NaN values detected in action space bounds")
        
        # Redefine action space to lie between [-1,1]
        self.action_space = gym.spaces.Box(low=-1, high=1,
                                          shape=self.unwrapped.action_space.shape, 
                                          dtype=np.float32)
        
        print("Applied RobustNormalizedActionWrapper with automatic action clipping")
        
    def action(self, action_wrapper):
        """Convert normalized actions back to original space with automatic clipping."""
        # Check for NaN values in input action
        if np.isnan(action_wrapper).any():
            raise ValueError("NaN values detected in input action")
            
        # Clip actions to [-1, 1] bounds if they're outside
        if not np.all(action_wrapper >= -1) or not np.all(action_wrapper <= 1):
            # Log clipping (but limit frequency)
            if not hasattr(self, '_clip_count'):
                self._clip_count = 0
            self._clip_count += 1
            if self._clip_count % 1000 == 1:  # Print first occurrence and every 1000th after
                print(f"WARNING: Actions clipped from [{action_wrapper.min():.4f}, {action_wrapper.max():.4f}] to [-1.0, 1.0] (occurrence #{self._clip_count})")
            
            # Clip the actions
            action_wrapper = np.clip(action_wrapper, -1.0, 1.0)
            
        # Convert to original action space
        action = self.low + (0.5 * (action_wrapper + 1.0) * (self.high - self.low))
        
        # Check for NaN values in output action
        if np.isnan(action).any():
            raise ValueError("NaN values detected in output action")
            
        # Final safety check - clip to original bounds if needed
        action = np.clip(action, self.low, self.high)
        
        return action


class ActionClippingWrapper(gym.ActionWrapper):
    """Wrapper to clip actions to valid bounds before passing to environment."""
    
    def __init__(self, env):
        super().__init__(env)
        print("Applied ActionClippingWrapper to ensure actions stay within bounds")
        
    def action(self, action):
        """Clip actions to valid bounds."""
        # Clip actions to [-1, 1] range
        clipped_action = np.clip(action, -1.0, 1.0)
        
        # Log if clipping occurred (but limit frequency to avoid spam)
        if not np.array_equal(action, clipped_action):
            # Only print every 1000th clipping event to avoid spam
            if not hasattr(self, '_clip_count'):
                self._clip_count = 0
            self._clip_count += 1
            if self._clip_count % 1000 == 1:  # Print first occurrence and every 1000th after
                print(f"WARNING: Actions clipped from [{action.min():.4f}, {action.max():.4f}] to [{clipped_action.min():.4f}, {clipped_action.max():.4f}] (occurrence #{self._clip_count})")
        
        return clipped_action

def create_large_ppo_policy_kwargs(network_size='large'):
    """Create PPO with customizable MLP policy network size."""
    if network_size == 'small':
        policy_kwargs = {
            "net_arch": {
                "pi": [64, 64],      # Policy network: 2 layers with 64 units each
                "vf": [64, 64]       # Value function network: same architecture
            },
            "activation_fn": torch.nn.ReLU
        }
    elif network_size == 'medium':
        policy_kwargs = {
            "net_arch": {
                "pi": [256, 256],    # Policy network: 2 layers with 256 units each
                "vf": [256, 256]     # Value function network: same architecture
            },
            "activation_fn": torch.nn.ReLU
        }
    elif network_size == 'large':
        policy_kwargs = {
            "net_arch": {
                "pi": [512, 512, 256, 256],  # Policy network: 4 layers
                "vf": [512, 512, 256, 256]   # Value function network: same architecture
            },
            "activation_fn": torch.nn.ReLU
        }
    elif network_size == 'xlarge':
        policy_kwargs = {
            "net_arch": {
                "pi": [1024, 1024, 512, 512, 256],  # Policy network: 5 layers
                "vf": [1024, 1024, 512, 512, 256]   # Value function network: same architecture
            },
            "activation_fn": torch.nn.ReLU
        }
    else:
        raise ValueError(f"Unknown network size: {network_size}")
    
    return policy_kwargs

def create_stable_ppokl4bc_class():
    """
    Create a more numerically stable version of PPOKL4BC with better KL divergence calculation.
    """
    import copy
    import torch
    import torch.nn.functional as F
    import numpy as np
    from gymnasium import spaces
    from stable_baselines3.ppo import PPO
    from stable_baselines3.common.utils import explained_variance
    
    class StablePPOKL4BC(PPO):
        def __init__(self, *args, bc_policy=None, beta_kl=1.0, **kwargs):
            super().__init__(*args, **kwargs)
            self.beta_kl = float(beta_kl)
            # Initialize KL value storage
            self.kl_bc_value = 0.0
            
            # Set up BC policy if provided
            if bc_policy is not None:
                self.bc_policy = copy.deepcopy(bc_policy).eval()
                for p in self.bc_policy.parameters():
                    p.requires_grad_(False)
                # Copy BC weights into the new PPO policy
                self._copy_bc_weights()
            else:
                self.bc_policy = None
            
        def _copy_bc_weights(self):
            """Copy weights from the BC policy to the current PPO policy with small noise."""
            with torch.no_grad():
                # Copy features extractor weights
                if hasattr(self.policy, 'features_extractor') and hasattr(self.bc_policy, 'features_extractor'):
                    self._copy_module_weights(self.policy.features_extractor, self.bc_policy.features_extractor)
                
                # Copy MLP extractor weights (shared or separate)
                if hasattr(self.policy, 'mlp_extractor') and hasattr(self.bc_policy, 'mlp_extractor'):
                    self._copy_module_weights(self.policy.mlp_extractor, self.bc_policy.mlp_extractor)
                
                # Copy action network weights with small noise
                if hasattr(self.policy, 'action_net') and hasattr(self.bc_policy, 'action_net'):
                    self._copy_module_weights_with_noise(self.policy.action_net, self.bc_policy.action_net, noise_std=0.001)
                
                # Copy value network weights
                if hasattr(self.policy, 'value_net') and hasattr(self.bc_policy, 'value_net'):
                    self._copy_module_weights(self.policy.value_net, self.bc_policy.value_net)
                
                # Copy log_std if it exists (for continuous actions) with small noise
                if hasattr(self.policy, 'log_std') and hasattr(self.bc_policy, 'log_std'):
                    self.policy.log_std.data.copy_(self.bc_policy.log_std.data)
                    # Add small noise to log_std to create initial divergence
                    noise = torch.randn_like(self.policy.log_std.data) * 0.001
                    self.policy.log_std.data.add_(noise)
                
                print("Successfully copied BC policy weights to PPO policy with small noise for initial divergence")

        def _copy_module_weights(self, target_module, source_module):
            """Helper method to copy weights from source module to target module."""
            target_state_dict = target_module.state_dict()
            source_state_dict = source_module.state_dict()
            
            # Only copy weights for parameters that exist in both modules
            for key in target_state_dict.keys():
                if key in source_state_dict:
                    if target_state_dict[key].shape == source_state_dict[key].shape:
                        target_state_dict[key].copy_(source_state_dict[key])
                    else:
                        raise ValueError(f"Shape mismatch for {key}: target {target_state_dict[key].shape} vs source {source_state_dict[key].shape}")

        def _copy_module_weights_with_noise(self, target_module, source_module, noise_std=0.01):
            """Helper method to copy weights with small noise for initial divergence."""
            target_state_dict = target_module.state_dict()
            source_state_dict = source_module.state_dict()
            
            # Only copy weights for parameters that exist in both modules
            for key in target_state_dict.keys():
                if key in source_state_dict:
                    if target_state_dict[key].shape == source_state_dict[key].shape:
                        # Copy weights and add small noise
                        target_state_dict[key].copy_(source_state_dict[key])
                        noise = torch.randn_like(target_state_dict[key]) * noise_std
                        target_state_dict[key].add_(noise)
                    else:
                        raise ValueError(f"Shape mismatch for {key}: target {target_state_dict[key].shape} vs source {source_state_dict[key].shape}")

        def set_beta_kl(self, val: float):
            self.beta_kl = float(val)

        @classmethod
        def load(cls, path, env, bc_policy=None, **kwargs):
            """
            Load a saved StablePPOKL4BC model using SB3's built-in loading mechanism.
            """
            # Load the model using SB3's standard loading mechanism
            # This will properly restore all attributes and state
            model = super().load(path, env, **kwargs)
            
            # Set up the BC policy if provided
            if bc_policy is not None:
                model.bc_policy = copy.deepcopy(bc_policy).eval()
                for p in model.bc_policy.parameters():
                    p.requires_grad_(False)
            
            return model

        @torch.no_grad()
        def _bc_log_prob(self, obs_tensor, actions_tensor):
            dist_bc = self.bc_policy.get_distribution(obs_tensor)
            # SB3 already sums over action dims in log_prob for each distribution
            logp_bc = dist_bc.log_prob(actions_tensor)
            if logp_bc.ndim > 1:
                logp_bc = logp_bc.sum(-1)
            return logp_bc

        def _compute_stable_kl_bc(self, log_prob, logp_bc):
            """
            Compute KL divergence to BC policy using a numerically stable method.
            """
            bc_log_ratio = log_prob - logp_bc
            
            # Clip log_ratio to prevent numerical overflow
            bc_log_ratio = torch.clamp(bc_log_ratio, -20.0, 20.0)
            
            # Use more stable KL calculation
            # KL = E[log(p/q)] = E[log(p) - log(q)] = E[log_ratio]
            # But we want KL(p||q) = E_p[log(p/q)] = E_p[log_ratio]
            # The approximation is: KL ≈ E[(exp(log_ratio) - 1) - log_ratio]
            
            # Split into positive and negative parts for numerical stability
            pos_mask = bc_log_ratio > 0
            neg_mask = bc_log_ratio <= 0
            
            kl_pos = torch.zeros_like(bc_log_ratio)
            kl_neg = torch.zeros_like(bc_log_ratio)
            
            if pos_mask.any():
                # For positive log_ratio: KL ≈ exp(log_ratio) - 1 - log_ratio
                kl_pos[pos_mask] = torch.exp(bc_log_ratio[pos_mask]) - 1 - bc_log_ratio[pos_mask]
            
            if neg_mask.any():
                # For negative log_ratio: KL ≈ 1 - exp(log_ratio) + log_ratio
                # This is more stable than the original formula
                kl_neg[neg_mask] = 1 - torch.exp(bc_log_ratio[neg_mask]) + bc_log_ratio[neg_mask]
            
            approx_kl_bc = torch.mean(kl_pos + kl_neg)
            
            # Debug: Print detailed KL calculation info occasionally
            if hasattr(self, '_kl_debug_count'):
                self._kl_debug_count += 1
            else:
                self._kl_debug_count = 0
                
            if self._kl_debug_count % 1000 == 0 and self.verbose >= 1:
                print(f"[KL-CALC] Debug batch {self._kl_debug_count}:")
                print(f"[KL-CALC] Log ratio range: [{bc_log_ratio.min().item():.4f}, {bc_log_ratio.max().item():.4f}]")
                print(f"[KL-CALC] Pos mask count: {pos_mask.sum().item()}, Neg mask count: {neg_mask.sum().item()}")
                print(f"[KL-CALC] KL components - Pos: {kl_pos.sum().item():.6f}, Neg: {kl_neg.sum().item():.6f}")
                print(f"[KL-CALC] Final KL: {approx_kl_bc.item():.6f}")
            
            # Additional safety check
            if torch.isnan(approx_kl_bc) or torch.isinf(approx_kl_bc):
                print(f"WARNING: Invalid KL value detected: {approx_kl_bc.item()}. Using fallback.")
                # Fallback: use simple L2 distance between log probabilities
                approx_kl_bc = torch.mean((log_prob - logp_bc) ** 2)
            
            # Final safety check - clamp to reasonable range
            approx_kl_bc = torch.clamp(approx_kl_bc, 0.0, 100.0)
            
            return approx_kl_bc

        def train(self):
            """Update policy using the currently gathered rollout buffer with KL divergence to BC policy."""
            # Check if BC policy is available
            if self.bc_policy is None:
                raise ValueError("BC policy is not set. Please ensure bc_policy is provided during initialization or loading.")
            
            # Switch to train mode (this affects batch norm / dropout)
            self.policy.set_training_mode(True)
            # Update optimizer learning rate
            self._update_learning_rate(self.policy.optimizer)
            # Compute current clip range
            clip_range = self.clip_range(self._current_progress_remaining)
            # Optional: clip range for the value function
            if self.clip_range_vf is not None:
                clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

            entropy_losses = []
            pg_losses, value_losses = [], []
            clip_fractions = []
            kl_bc_meter = []

            continue_training = True
            # train for n_epochs epochs
            for epoch in range(self.n_epochs):
                approx_kl_divs = []
                # Do a complete pass on the rollout buffer
                for rollout_data in self.rollout_buffer.get(self.batch_size):
                    actions = rollout_data.actions
                    if isinstance(self.action_space, spaces.Discrete):
                        # Convert discrete action from float to long
                        actions = rollout_data.actions.long().flatten()

                    values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                    values = values.flatten()
                    
                    # Clip log probabilities to prevent extreme values
                    log_prob = torch.clamp(log_prob, -20.0, 20.0)
                    
                    # Normalize advantage
                    advantages = rollout_data.advantages
                    # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                    if self.normalize_advantage and len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # ratio between old and new policy, should be one at the first iteration
                    ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                    # clipped surrogate loss
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                    # Logging
                    pg_losses.append(policy_loss.item())
                    clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                    clip_fractions.append(clip_fraction)

                    if self.clip_range_vf is None:
                        # No clipping
                        values_pred = values
                    else:
                        # Clip the difference between old and new value
                        # NOTE: this depends on the reward scaling
                        values_pred = rollout_data.old_values + torch.clamp(
                            values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                        )
                    # Value loss using the TD(gae_lambda) target
                    value_loss = F.mse_loss(rollout_data.returns, values_pred)
                    value_losses.append(value_loss.item())

                    # Entropy loss favor exploration
                    if entropy is None:
                        # Approximate entropy when no analytical form
                        entropy_loss = -torch.mean(-log_prob)
                    else:
                        entropy_loss = -torch.mean(entropy)

                    entropy_losses.append(entropy_loss.item())

                    # === KL to BC (using stable calculation) ===
                    with torch.no_grad():
                        logp_bc = self._bc_log_prob(rollout_data.observations, actions)
                        # Clip BC log probabilities as well
                        logp_bc = torch.clamp(logp_bc, -20.0, 20.0)
                    
                    # Use stable KL calculation
                    approx_kl_bc = self._compute_stable_kl_bc(log_prob, logp_bc)
                    kl_bc_meter.append(approx_kl_bc.detach().cpu().item())
                    
                    # Debug: Print KL values and log probabilities occasionally
                    if len(kl_bc_meter) % 1000 == 0 and self.verbose >= 1:
                        print(f"[KL-DEBUG] Batch {len(kl_bc_meter)}: KL_BC = {approx_kl_bc.item():.6f}")
                        print(f"[KL-DEBUG] Log prob stats - Current: mean={log_prob.mean().item():.4f}, std={log_prob.std().item():.4f}")
                        print(f"[KL-DEBUG] Log prob stats - Expert: mean={logp_bc.mean().item():.4f}, std={logp_bc.std().item():.4f}")
                        print(f"[KL-DEBUG] Log ratio stats - mean={(log_prob - logp_bc).mean().item():.4f}, std={(log_prob - logp_bc).std().item():.4f}")
                    
                    # Total loss including KL divergence to BC
                    loss = policy_loss + self.beta_kl * approx_kl_bc + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                    # Calculate approximate form of reverse KL Divergence for early stopping
                    # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                    # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                    # and Schulman blog: http://joschu.net/blog/kl-approx.html
                    with torch.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        approx_kl_divs.append(approx_kl_div)

                    if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                        continue_training = False
                        if self.verbose >= 1:
                            print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                        break

                    # Optimization step
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    # Clip grad norm
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()

                self._n_updates += 1
                if not continue_training:
                    break

            explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

            # Logs
            if kl_bc_meter:
                kl_bc_mean = np.mean(kl_bc_meter)
                self.logger.record("train/kl_bc", kl_bc_mean)
                # Store KL value directly in model instance for callback access
                self.kl_bc_value = kl_bc_mean
                if self.verbose >= 1:
                    print(f"[TRAIN] Logged KL_BC: {kl_bc_mean:.6f} (from {len(kl_bc_meter)} batches)")
            else:
                # If no KL values were computed, log 0
                self.logger.record("train/kl_bc", 0.0)
                self.kl_bc_value = 0.0
                if self.verbose >= 1:
                    print("[TRAIN] WARNING: No KL values computed, logging 0.0")
                
            self.logger.record("train/entropy_loss", np.mean(entropy_losses))
            self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
            self.logger.record("train/value_loss", np.mean(value_losses))
            self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
            self.logger.record("train/clip_fraction", np.mean(clip_fractions))
            self.logger.record("train/loss", loss.item())
            self.logger.record("train/explained_variance", explained_var)
            if hasattr(self.policy, "log_std"):
                self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/clip_range", clip_range)
            if self.clip_range_vf is not None:
                self.logger.record("train/clip_range_vf", clip_range_vf)
    
    return StablePPOKL4BC

def test_policy_differences(model, bc_model, env, num_samples=10):
    """
    Test if the current and expert policies are producing different outputs.
    """
    print("Testing policy differences...")
    
    current_actions = []
    expert_actions = []
    
    for i in range(num_samples):
        obs = env.observation_space.sample()
        
        # Get actions from both policies
        current_action = model.predict(obs, deterministic=True)[0]
        expert_action = bc_model.predict(obs, deterministic=True)[0]
        
        current_actions.append(current_action)
        expert_actions.append(expert_action)
    
    current_actions = np.array(current_actions)
    expert_actions = np.array(expert_actions)
    
    # Calculate differences
    action_diff = np.abs(current_actions - expert_actions)
    mean_diff = np.mean(action_diff, axis=0)
    max_diff = np.max(action_diff, axis=0)
    
    print(f"Action differences - Mean: {mean_diff}")
    print(f"Action differences - Max: {max_diff}")
    print(f"Overall mean difference: {np.mean(mean_diff):.6f}")
    print(f"Overall max difference: {np.mean(max_diff):.6f}")
    
    # Check if policies are identical
    if np.allclose(current_actions, expert_actions, atol=1e-6):
        print("WARNING: Policies appear to be identical! This explains KL=0.")
        print("The BC weights were copied exactly, so no KL divergence is expected.")
        return False
    else:
        print("Policies are different, KL should be non-zero.")
        return True


def test_kl_calculation():
    """
    Test function to verify KL divergence calculation is working.
    """
    print("Testing KL divergence calculation...")
    
    # Create a simple test case
    import torch
    
    # Create test log probabilities
    log_prob_current = torch.tensor([-1.0, -2.0, -3.0])  # Current policy log probs
    log_prob_expert = torch.tensor([-1.1, -1.9, -3.1])   # Expert policy log probs
    
    # Create the stable PPOKL4BC class
    StablePPOKL4BC = create_stable_ppokl4bc_class()
    
    # Create a dummy model instance just to access the KL calculation method
    class DummyModel:
        def _compute_stable_kl_bc(self, log_prob, logp_bc):
            """
            Compute KL divergence to BC policy using a numerically stable method.
            """
            bc_log_ratio = log_prob - logp_bc
            
            # Clip log_ratio to prevent numerical overflow (reduced from 20.0 to 10.0)
            bc_log_ratio = torch.clamp(bc_log_ratio, -10.0, 10.0)
            
            # Use more stable KL calculation
            pos_mask = bc_log_ratio > 0
            neg_mask = bc_log_ratio <= 0
            
            kl_pos = torch.zeros_like(bc_log_ratio)
            kl_neg = torch.zeros_like(bc_log_ratio)
            
            if pos_mask.any():
                kl_pos[pos_mask] = torch.exp(bc_log_ratio[pos_mask]) - 1 - bc_log_ratio[pos_mask]
            
            if neg_mask.any():
                kl_neg[neg_mask] = 1 - torch.exp(bc_log_ratio[neg_mask]) + bc_log_ratio[neg_mask]
            
            approx_kl_bc = torch.mean(kl_pos + kl_neg)
            
            # Final safety check - clamp to reasonable range (reduced from 100.0 to 10.0)
            approx_kl_bc = torch.clamp(approx_kl_bc, 0.0, 10.0)
            
            return approx_kl_bc
    
    dummy_model = DummyModel()
    
    # Test KL calculation
    kl_value = dummy_model._compute_stable_kl_bc(log_prob_current, log_prob_expert)
    print(f"Test KL calculation result: {kl_value.item():.6f}")
    
    # Test with more extreme values
    log_prob_extreme = torch.tensor([-10.0, -15.0, -20.0])
    log_prob_expert_extreme = torch.tensor([-1.0, -1.0, -1.0])
    
    kl_extreme = dummy_model._compute_stable_kl_bc(log_prob_extreme, log_prob_expert_extreme)
    print(f"Extreme KL calculation result: {kl_extreme.item():.6f}")
    
    print("KL calculation test completed!")
    return kl_value.item(), kl_extreme.item()


def analyze_policy_distributions(bc_model_path, env, num_samples=1000):
    """
    Comprehensive analysis of both expert and current policy distributions.
    """
    print("="*60)
    print("COMPREHENSIVE POLICY DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Load expert policy
    bc_model = PPO.load(bc_model_path, env=env, device='cpu')
    
    # Sample actions from expert policy
    expert_actions = []
    current_actions = []
    
    print("Sampling actions from both policies...")
    for i in range(num_samples):
        obs = env.observation_space.sample()
        
        # Expert policy
        expert_action = bc_model.predict(obs, deterministic=False)[0]
        expert_actions.append(expert_action)
        
        # Current policy (random initialization)
        current_action = env.action_space.sample()
        current_actions.append(current_action)
        
        if (i + 1) % 200 == 0:
            print(f"  Sampled {i + 1}/{num_samples} actions...")
    
    expert_actions = np.array(expert_actions)
    current_actions = np.array(current_actions)
    
    print("\nEXPERT POLICY ANALYSIS:")
    print("-" * 30)
    expert_std = np.std(expert_actions, axis=0)
    expert_mean = np.mean(expert_actions, axis=0)
    expert_min = np.min(expert_actions, axis=0)
    expert_max = np.max(expert_actions, axis=0)
    
    print(f"Action means: {expert_mean}")
    print(f"Action stds:  {expert_std}")
    print(f"Action range: [{expert_min}, {expert_max}]")
    
    print("\nCURRENT POLICY ANALYSIS (Random):")
    print("-" * 40)
    current_std = np.std(current_actions, axis=0)
    current_mean = np.mean(current_actions, axis=0)
    current_min = np.min(current_actions, axis=0)
    current_max = np.max(current_actions, axis=0)
    
    print(f"Action means: {current_mean}")
    print(f"Action stds:  {current_std}")
    print(f"Action range: [{current_min}, {current_max}]")
    
    print("\nDISTRIBUTION COMPARISON:")
    print("-" * 25)
    mean_diff = np.abs(expert_mean - current_mean)
    std_diff = np.abs(expert_std - current_std)
    
    print(f"Mean differences: {mean_diff}")
    print(f"Std differences:  {std_diff}")
    
    # Check for potential KL divergence issues
    print("\nPOTENTIAL ISSUES:")
    print("-" * 18)
    issues = []
    
    # Check for very small expert variances (could cause KL spikes)
    if np.any(expert_std < 1e-2):
        small_std_indices = np.where(expert_std < 1e-2)[0]
        issues.append(f"Expert policy has very small std in actions {small_std_indices}: {expert_std[small_std_indices]}")
    
    # Check for very large action values
    if np.any(np.abs(expert_actions) > 10):
        large_action_count = np.sum(np.abs(expert_actions) > 10)
        issues.append(f"Expert policy has {large_action_count} actions with magnitude > 10")
    
    # Check for large distribution differences
    if np.any(mean_diff > 5):
        large_diff_indices = np.where(mean_diff > 5)[0]
        issues.append(f"Large mean differences in actions {large_diff_indices}: {mean_diff[large_diff_indices]}")
    
    if issues:
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("  No obvious issues detected")
    
    print("\nRECOMMENDATIONS:")
    print("-" * 16)
    if np.any(expert_std < 1e-2):
        print("  • Consider adding minimum std constraint to prevent KL spikes")
        print("  • Use more conservative KL adaptation parameters")
    if np.any(mean_diff > 5):
        print("  • Consider gradual policy initialization")
        print("  • Use smaller learning rate initially")
    
    print("="*60)
    return expert_actions, current_actions


def analyze_expert_policy_distribution(bc_model_path, env, num_samples=1000):
    """
    Analyze the expert policy distribution to detect potential issues.
    """
    print("Analyzing expert policy distribution...")
    
    # Load expert policy
    bc_model = PPO.load(bc_model_path, env=env, device='cpu')
    
    # Sample actions from expert policy
    expert_actions = []
    
    for _ in range(num_samples):
        obs = env.observation_space.sample()
        action = bc_model.predict(obs, deterministic=False)[0]  # predict returns (action, state)
        expert_actions.append(action)
    
    expert_actions = np.array(expert_actions)
    
    # Analyze distribution
    action_std = np.std(expert_actions, axis=0)
    action_mean = np.mean(expert_actions, axis=0)
    
    print(f"Expert action statistics:")
    print(f"  Action means: {action_mean}")
    print(f"  Action stds: {action_std}")
    print(f"  Action range: [{np.min(expert_actions, axis=0)}, {np.max(expert_actions, axis=0)}]")
    
    # Check for potential issues
    issues = []
    if np.any(action_std < 1e-3):
        issues.append("Very low action variance detected")
    if np.any(np.abs(action_mean) > 10):
        issues.append("Large action means detected")
    if np.any(np.abs(expert_actions) > 20):
        issues.append("Very large action values detected")
    
    if issues:
        print("POTENTIAL ISSUES DETECTED:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Expert policy distribution looks reasonable")
    
    return expert_actions


def PPO_BC_finetune_training(test_model_flag=False, reload_model_flag=False,  total_timesteps=100000,
                 load_pretrained_bc=False):
    """
    Train PPO with optional autoencoder support.
    
    Args:
        test_model_flag: If True, test the model instead of training
        reload_model_flag: If True, reload an existing model for continued training
        load_pretrained_bc: If True, load pretrained behavioral cloning model
    """
    
    stepSize = 600 #Seconds
    #Define the range of available data
    start_time = datetime.datetime(year=2024, month=5, day=17, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))
    end_time = datetime.datetime(year=2024, month=5, day=31, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))        
    fine_tune_lr = 5e-4
    StablePPOKL4BC = create_stable_ppokl4bc_class()
    env = get_custom_env(stepSize, start_time, end_time)
    env = NormalizedObservationWrapper(env)
    env = RobustNormalizedActionWrapper(env)  # Use robust wrapper that handles out-of-bounds actions
    env = Monitor(env=env, filename=os.path.join(log_dir,'monitor.csv'))

    policy_kwargs = create_large_ppo_policy_kwargs(network_size=policy_size)

    if test_model_flag:
        model_path = os.path.join(log_dir, "ppo_pretrained_bc.zip")
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            print("Available model files:")
            for file in os.listdir(log_dir):
                if file.endswith('.zip'):
                    print(f"  - {file}")
            return
        
        bc_model = PPO.load(bc_model_path, env=env, device=device)
        model = StablePPOKL4BC.load(model_path, env=env, bc_policy=bc_model.policy, policy_kwargs=policy_kwargs, device=device)
        print(f"Training steps: {model.num_timesteps}")
        test_model(env, model)
        return

    # Load pretrained behavioral cloning model if requested

    
    if os.path.exists(bc_model_path):
        print(f"Loading pretrained behavioral cloning model from {bc_model_path}")
        bc_model = PPO.load(bc_model_path, env=env, device=device)
        print(f"Loaded pretrained model with {bc_model.num_timesteps} timesteps")
        
        # Test KL calculation
        #test_kl_calculation()
        
        # Analyze expert policy distribution
        #analyze_policy_distributions(bc_model_path, env)
        
        
        model = StablePPOKL4BC(
        "MlpPolicy", env, bc_policy=bc_model.policy, policy_kwargs=policy_kwargs, beta_kl=0.5,
        n_steps=4096, batch_size=256, n_epochs=10,
        gamma=0.997, gae_lambda=0.95, learning_rate=fine_tune_lr, clip_range=0.15,  
        ent_coef=0.0, vf_coef=0.5, device=device, verbose=0
        )
        
        # Debug: Check action space bounds
        print(f"Environment action space: {env.action_space}")
        print(f"Environment action space bounds: [{env.action_space.low}, {env.action_space.high}]")
        
        # Debug: Test policy output bounds
        test_obs = env.observation_space.sample()
        with torch.no_grad():
            test_action, _ = model.predict(test_obs, deterministic=True)
            print(f"Test policy output bounds (deterministic): [{test_action.min():.4f}, {test_action.max():.4f}]")
            if test_action.min() < -1 or test_action.max() > 1:
                print(f"WARNING: Policy producing actions outside [-1, 1] bounds!")
                print(f"Action range: [{test_action.min():.4f}, {test_action.max():.4f}]")
            
            # Test stochastic actions (like during training)
            test_action_stoch, _ = model.predict(test_obs, deterministic=False)
            print(f"Test policy output bounds (stochastic): [{test_action_stoch.min():.4f}, {test_action_stoch.max():.4f}]")
            if test_action_stoch.min() < -1 or test_action_stoch.max() > 1:
                print(f"WARNING: Stochastic policy producing actions outside [-1, 1] bounds!")
                print(f"Stochastic action range: [{test_action_stoch.min():.4f}, {test_action_stoch.max():.4f}]")
            
            # Test multiple random observations to get a better sense of action distribution
            print("Testing action bounds with multiple random observations...")
            action_mins = []
            action_maxs = []
            stoch_action_mins = []
            stoch_action_maxs = []
            for i in range(10):
                test_obs = env.observation_space.sample()
                test_action, _ = model.predict(test_obs, deterministic=True)
                test_action_stoch, _ = model.predict(test_obs, deterministic=False)
                action_mins.append(test_action.min())
                action_maxs.append(test_action.max())
                stoch_action_mins.append(test_action_stoch.min())
                stoch_action_maxs.append(test_action_stoch.max())
            
            print(f"Deterministic action range across 10 samples: [{min(action_mins):.4f}, {max(action_maxs):.4f}]")
            print(f"Stochastic action range across 10 samples: [{min(stoch_action_mins):.4f}, {max(stoch_action_maxs):.4f}]")
            print(f"Average deterministic action range: [{np.mean(action_mins):.4f}, {np.mean(action_maxs):.4f}]")
            print(f"Average stochastic action range: [{np.mean(stoch_action_mins):.4f}, {np.mean(stoch_action_maxs):.4f}]")
        
        # Test policy differences after model creation
        test_policy_differences(model, bc_model, env)
        
        
    else:
        raise FileNotFoundError(f"Pretrained behavioral cloning model not found at {bc_model_path}. "
                                f"Please run the pretraining script first to generate the required model file.")


    # Set up callback for BC fine-tuning

    AdaptKLtoBC_callback = AdaptKLtoBC(target_kl=0.01,  # Lower target since we start with similar policies
                                            beta0=0.5,   # Lower initial beta to allow gradual divergence
                                            beta_final=1e-4,  # Higher final beta to maintain some BC constraint
                                            decay_steps=20000,  
                                            verbose=0
                                        )
    progress_callback = SingleLineProgressCallback(total_timesteps, verbose=0)
    periodic_save_callback = PeriodicSaveCallback(save_freq=100000, save_path=log_dir, verbose=0)
    action_bounds_callback = ActionBoundsMonitorCallback(check_freq=1000, verbose=1)
    callback = CallbackList([AdaptKLtoBC_callback, progress_callback, periodic_save_callback, action_bounds_callback])

    # Train the model
    print(f"\n{'='*60}")
    print("STARTING TRAINING")
    print(f"{'='*60}")
    
    print(f"Total timesteps to train: {total_timesteps:,}")
    print(f"Environment: {env.__class__.__name__}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Device: {device}")
    print(f"Log directory: {log_dir}")
    print(f"{'='*60}\n")
    
    if reload_model_flag:
        model_path = os.path.join(log_dir, "model_1000000.zip")
        print(f"Reloading existing model from {model_path}")

        # Load the BC model to get its policy
        bc_model = PPO.load(bc_model_path, env=env, device=device)
        
        model = StablePPOKL4BC.load(model_path, env=env, bc_policy=bc_model.policy, policy_kwargs=policy_kwargs, device=device)

        print(f"Loaded model with {model.num_timesteps} previous timesteps")

        # Set lower learning rate for fine-tuning from pretrained model
        
        model.learning_rate = fine_tune_lr
        print(f"Set learning rate to {fine_tune_lr} for fine-tuning from pretrained model")

        new_logger = configure(log_dir, ['csv'])
        model.set_logger(new_logger)

        total_training_timesteps = total_timesteps - model.num_timesteps

        print("Continuing training with existing model...")
        model.learn(total_timesteps=total_training_timesteps, callback=callback, reset_num_timesteps=False)
         
    else:
        new_logger = configure(log_dir, ['csv'])
        model.set_logger(new_logger)

        print("Starting fresh training...")
        model.learn(total_timesteps=total_timesteps, callback=callback)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Final model timesteps: {model.num_timesteps}")
    print(f"{'='*60}\n")

    # Save the model
    model_save_path = os.path.join(log_dir, "ppo_model")
    print(f"Saving trained model to {model_save_path}")
    model.save(model_save_path)
    print(f"Model saved successfully!")
    print(f"Model file size: {os.path.getsize(model_save_path + '.zip') / (1024*1024):.2f} MB")

    #Test the model
    test_model(env, model)


if __name__ == "__main__":
    PPO_BC_finetune_training(test_model_flag=test_model_flag, reload_model_flag=reload_model_flag, load_pretrained_bc=True, total_timesteps=total_timesteps)


