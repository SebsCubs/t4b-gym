import twin4build as tb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import datetime
from dateutil.tz import gettz
import json
import copy
from RL_Algos.networks import PolicyNetwork
from neural_policy_standalone_sim import fcn
import twin4build.examples.utils as utils

class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        
    def compute_gae(self, next_value, gamma=0.99, lambda_=0.95):
        rewards = torch.tensor(self.rewards)
        values = torch.tensor(self.values + [next_value])
        advantages = []
        gae = 0
        
        for t in reversed(range(len(self.rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + gamma * lambda_ * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
            
        return torch.tensor(advantages)

class PolicyTrainer:
    def __init__(self, model_path, input_output_schema_path):
        self.setup_twin4build_model(model_path, input_output_schema_path)
        self.setup_ppo()
        
    def setup_twin4build_model(self, model_path, schema_path):
        # Load the base model
        self.base_model = tb.Model(id="training_model")
        with open(schema_path) as f:
            self.input_output_schema = json.load(f)
        
        self.input_size = len(self.input_output_schema["input"])
        self.output_size = len(self.input_output_schema["output"])
        
        # Create policy network matching your example architecture
        self.policy = PolicyNetwork(self.input_size, self.output_size, action_bound=1.0)
        self.base_model.load(semantic_model_filename=model_path, fcn=fcn, verbose=False)

    def setup_ppo(self):
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.memory = PPOMemory()
        self.eps_clip = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        
    def compute_reward(self, simulator, space_id):
        # TODO: Get relevant metrics from the simulation
        temperature = simulator.get_result(space_id, 'indoorTemperature')[-1] - 273.15  # Convert to Celsius
        co2 = simulator.get_result(space_id, 'indoorCo2Concentration')[-1]
        energy = simulator.get_result(space_id, 'heatingPower')[-1]
        
        # Penalties for constraint violations
        temp_penalty = max(0, temperature - 21.0) * 10
        co2_penalty = max(0, co2 - 1000) * 10
        
        # Reward for energy efficiency (negative because we want to minimize)
        energy_reward = -energy * 0.001
        
        return energy_reward - temp_penalty - co2_penalty

    def train(self, num_episodes=1000):
        best_reward = float('-inf')
        
        for episode in range(num_episodes):
            # Reset environment (create new simulation instance)
            simulation_model = copy.deepcopy(self.base_model)
            simulator = tb.Simulator()
            
            # Setup simulation timeframe
            startTime = datetime.datetime(year=2023, month=11, day=27, hour=0, 
                                       tzinfo=gettz("Europe/Copenhagen"))
            endTime = datetime.datetime(year=2023, month=11, day=28, hour=0, 
                                     tzinfo=gettz("Europe/Copenhagen"))
            
            # Run simulation with current policy
            episode_reward = self.run_episode(simulation_model, simulator, startTime, endTime)
            
            # Update policy using PPO
            self.update_policy()
            
            # Save best policy
            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save(self.policy.state_dict(), 'best_policy.pth')
                
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Best: {best_reward:.2f}")

    def run_episode(self, model, simulator, start_time, end_time):
        step_size = 600  # 10 minutes
        current_time = start_time
        episode_reward = 0
        
        while current_time < end_time:
            # Get current state
            state = self.get_state(simulator)
            
            # Get action from policy
            with torch.no_grad():
                mean, std = self.policy(torch.FloatTensor(state))
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            # Apply action to model
            self.apply_action(model, action)
            
            # Simulate one step
            simulator.simulate(model, startTime=current_time, 
                            endTime=current_time + datetime.timedelta(seconds=step_size),
                            stepSize=step_size)
            
            # Calculate reward
            reward = self.compute_reward(simulator, '[020B][020B_space_heater]')
            episode_reward += reward
            
            # Store transition
            self.memory.states.append(state)
            self.memory.actions.append(action)
            self.memory.rewards.append(reward)
            self.memory.log_probs.append(log_prob)
            
            current_time += datetime.timedelta(seconds=step_size)
            
        return episode_reward

    def update_policy(self):
        # Compute advantages and returns
        advantages = self.memory.compute_gae(0)  # Assuming final value is 0
        
        # PPO update
        for _ in range(10):  # Multiple epochs
            for idx in range(len(self.memory.states)):
                state = torch.FloatTensor(self.memory.states[idx])
                action = self.memory.actions[idx]
                old_log_prob = self.memory.log_probs[idx]
                advantage = advantages[idx]
                
                # Get current policy distribution
                mean, std = self.policy(state)
                dist = torch.distributions.Normal(mean, std)
                new_log_prob = dist.log_prob(action)
                
                # Compute ratio and clipped loss
                ratio = torch.exp(new_log_prob - old_log_prob)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
                loss = -torch.min(surr1, surr2).mean()
                
                # Update policy
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        #Update the policy network in the model
        self.base_model.component_dict["neural_controller"].policy = self.policy

        # Clear memory after update
        self.memory.clear()

if __name__ == "__main__":
    model_filename = utils.get_path(["parameter_estimation_example", "one_room_example_model.xlsm"])
    input_output_schema_path = utils.get_path(["neural_policy_controller_example", "policy_input_output.json"])
    trainer = PolicyTrainer(
        model_path=model_filename,
        input_output_schema_path=input_output_schema_path
    )
    trainer.train(num_episodes=1000)
