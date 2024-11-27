import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) # Add the grandparent directory to the system path

import twin4build as tb
import torch
from collections import deque
import datetime
from dateutil.tz import gettz
import json
import copy
from RL_Algos.ppo_agent import PPOAgent, Memory
from neural_policy_standalone_sim import fcn
import twin4build.examples.utils as utils
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = os.path.join(uppath(os.path.abspath(__file__), 4), "Twin4Build")
    sys.path.append(file_path)

class PolicyTrainer:
    def __init__(self, model_path, input_output_schema_path):
        self.setup_twin4build_model(model_path, input_output_schema_path)
        self.setup_ppo()
        self.setup_tensorboard()
        
    def setup_twin4build_model(self, model_path, schema_path):
        # Load the base model
        self.base_model = tb.Model(id="training_model")
        with open(schema_path) as f:
            self.input_output_schema = json.load(f)
            
        self.input_size = sum(len(signals) for signals in self.input_output_schema["input"].values())
        self.output_size =  5 + 2 + 1 #TODO: Make this dynamic
        
        # Create policy and value networks
        self.ppo_agent = PPOAgent(
            state_dim=self.input_size,
            action_dim=self.output_size,
            action_bound=1.0
        )
        self.train_policy = self.ppo_agent.policy_old
        self.base_model.load(semantic_model_filename=model_path, fcn=fcn, create_signature_graphs=False, validate_model=True, verbose=False, force_config_update=True)
        # update the policy in the base model
        #The policy in the model must be a PolicyNetwork object, not a nn.Module object
        self.base_model.components["neural_controller"].policy.load_state_dict(self.train_policy.state_dict())
        self.base_model.components["neural_controller"].is_training = True

    def setup_ppo(self):
        self.memory = Memory()  # Using the Memory class from ppo_agent.py
        self.eps_clip = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.training_step = 0  # Add a training step counter here
        # Add state normalization
        self.state_mean = None
        self.state_std = None
        # Add action normalization
        self.action_scale = 1.0
        self.reward_scale = 0.01  # Scale down rewards

    def setup_tensorboard(self):
        """Initialize TensorBoard writer"""
        self.writer = SummaryWriter(f'runs/ppo_training_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
        
    def train(self, num_episodes=350):
        best_reward = float('-inf')
        rewards_history = []
        
        for episode in range(num_episodes):
            # Reset environment (create new simulation instance)
            simulation_model = copy.deepcopy(self.base_model)
            simulator = tb.Simulator() 
            
            # Setup simulation timeframe
            startTime = datetime.datetime(year=2023, month=11, day=27, hour=0, minute=0, second=0, 
                                       tzinfo=gettz("Europe/Copenhagen"))
            endTime = datetime.datetime(year=2023, month=12, day=7, hour=0, minute=0, second=0,
                                     tzinfo=gettz("Europe/Copenhagen"))
            
            # Run simulation with current policy
            episode_reward = self.run_episode(simulation_model, simulator, startTime, endTime)
            rewards_history.append(episode_reward)
            
            # Log metrics to TensorBoard
            self.writer.add_scalar('Reward/episode', episode_reward, episode)
            self.writer.add_scalar('Reward/moving_average', 
                                 np.mean(rewards_history[-100:]), episode)
            
            # Update policy using PPO
            self.update_policy()
            
            # Save best policy
            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save(self.ppo_agent.policy.state_dict(), 'best_policy.pth')
                
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Best: {best_reward:.2f}")

    def run_episode(self, model:tb.Model, simulator:tb.Simulator, start_time, end_time):
        
        
        step_size = 600  # 10 minutes
        # Run the full simulation at once
        simulator.simulate(model, startTime=start_time, endTime=end_time, stepSize=step_size)
        
        # Get state from saved outputs
        states = self.get_state_from_saved(model)
        actions = self.get_action_from_saved(model)
        
        # Get state and normalize it
        if self.state_mean is None:
            self.state_mean = states.mean(0)
            self.state_std = states.std(0) + 1e-8
        else:
            self.state_mean = 0.99 * self.state_mean + 0.01 * states.mean(0)
            self.state_std = 0.99 * self.state_std + 0.01 * states.std(0)
        
        # Normalize states
        normalized_states = (states - self.state_mean) / self.state_std
        
        # Calculate rewards for all timesteps at once
        rewards = self.compute_reward_from_saved(model, '[020B][020B_space_heater]')
        episode_reward = rewards.sum().item()  # Total episode reward
        rewards = rewards * self.reward_scale  # Scale rewards
        
        with torch.no_grad():
            if normalized_states.dim() == 1:
                normalized_states = normalized_states.unsqueeze(0)
            
            action_mean, action_std = self.ppo_agent.policy_old(normalized_states)
            action_mean = torch.clamp(action_mean, -1.0, 1.0)
            action_std = torch.clamp(action_std, 0.1, 0.5)
            
            if actions.dim() == 1:
                actions = actions.unsqueeze(0)
                
            dist = torch.distributions.Normal(action_mean, action_std)
            log_prob = dist.log_prob(actions)
            log_prob = torch.clamp(log_prob, -20.0, 2.0)
            log_prob = log_prob.sum(dim=-1)
        
        # Store episode data
        self.memory.states = normalized_states
        self.memory.actions = actions
        self.memory.rewards = rewards
        self.memory.logprobs = log_prob
        
        model.reset()

        return episode_reward

    def update_policy(self):
        # Add value checks before update
        if torch.isnan(self.memory.states).any() or torch.isinf(self.memory.states).any():
            print("Invalid values in states, skipping update")
            self.memory.clear()
            return
        
        if torch.isnan(self.memory.actions).any() or torch.isinf(self.memory.actions).any():
            print("Invalid values in actions, skipping update")
            self.memory.clear()
            return
        
        # Normalize rewards
        rewards = self.memory.rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        memory_dict = {
            'states': self.memory.states.detach(),
            'actions': self.memory.actions.detach(),
            'rewards': rewards.detach(),
            'logprobs': self.memory.logprobs.detach()
        }
        
        try:
            # Update both policy and value networks using PPO agent
            loss_stats = self.ppo_agent.update(memory_dict)
            
            # Log training metrics
            if loss_stats:
                self.writer.add_scalar('Loss/policy', loss_stats.get('policy_loss', 0), self.training_step)
                self.writer.add_scalar('Loss/value', loss_stats.get('value_loss', 0), self.training_step)
                self.writer.add_scalar('Loss/total', loss_stats.get('total_loss', 0), self.training_step)
                self.writer.add_scalar('Policy/entropy', loss_stats.get('entropy', 0), self.training_step)
                
            self.training_step += 1
            
            # Update the policy network in the model
            self.base_model.components["neural_controller"].policy.load_state_dict(
                self.ppo_agent.policy.state_dict()
            )
        except Exception as e:
            print("Error during policy update:", str(e))
            print("Memory content summary:")
            for key, value in memory_dict.items():
                print(f"{key}:", value.shape, "Range:", value.min().item(), "to", value.max().item())
            raise  # Re-raise the exception after printing debug info
        
        self.memory.clear()

    
    def get_state_from_saved(self, model):
        """
        Extract state variables from model.components saved outputs.
        
        Args:
            model: The model containing component dictionaries with saved outputs.
            
        Returns:
            torch.Tensor: A tensor containing the state variables with shape (timesteps, state_dim)
        """
        neural_controller = model.components["neural_controller"]
        states = []
        
        # First, collect all state variables
        for component_key, signals in neural_controller.input_output_schema["input"].items():
            input_component = model.components[component_key]
            for signal_name in signals.keys():
                controller_input = input_component.savedOutput[signal_name]
                states.append(controller_input)
        
        # Convert to numpy array and transpose
        states = np.array(states)  # Shape: (state_dim, timesteps)
        states = states.T  # Shape: (timesteps, state_dim)
        
        return torch.tensor(states, dtype=torch.float32)


    def get_action_from_saved(self, model):
        """
        Extract action values from the neural controller's saved outputs.

        Args:
            model: The model containing component dictionaries with saved outputs.

        Returns:
            torch.Tensor: A tensor containing the action values with shape (timesteps, action_dim)
        """
        neural_controller = model.components["neural_controller"]
        actions = []
        
        # Collect all action variables
        for key in neural_controller.savedOutput.keys():
            actions.append(neural_controller.savedOutput[key])
        
        # Convert to numpy array and transpose
        actions = np.array(actions)  # Shape: (action_dim, timesteps)
        actions = actions.T  # Shape: (timesteps, action_dim)
        
        return torch.tensor(actions, dtype=torch.float32)

    def compute_reward_from_saved(self, model, space_id):
        """
        Compute rewards for all timesteps at once.
        
        Args:
            model: The model containing component dictionaries with saved outputs.
            space_id: The ID of the space to compute reward for.
        
        Returns:
            torch.Tensor: Tensor of rewards for all timesteps
        """
        # Get all timesteps at once
        temperature = torch.tensor(model.components[space_id].savedOutput['indoorTemperature'])
        co2 = torch.tensor(model.components[space_id].savedOutput['indoorCo2Concentration'])
        room_heating_energy = torch.tensor(model.components[space_id].savedOutput['spaceHeaterPower'])
        fan_energy = torch.tensor(model.components['supply_fan'].savedOutput['Power'])

        # Calculate reward components
        temp_setpoint = 21.0
        co2_setpoint = 900.0
        
        # Temperature penalty (quadratic)
        temp_error = torch.abs(temperature - temp_setpoint)
        temp_penalty = -(temp_error) * 10
        
        # CO2 penalty (quadratic above setpoint)
        co2_error = torch.abs(co2 - co2_setpoint)
        co2_penalty = -(co2_error) * 0.01
        
        # Energy penalty (linear)
        energy_penalty = -room_heating_energy * 0.01 - fan_energy * 0.01
        
        # Combine rewards
        rewards = temp_penalty + co2_penalty + energy_penalty
        
        # Add debugging info for first and last timestep
        print(f"First timestep - Temp: {temperature[0]:.2f}, CO2: {co2[0]:.2f}, Energy: {room_heating_energy[0]:.2f}, Reward: {rewards[0]:.2f}")
        print(f"Last timestep - Temp: {temperature[-1]:.2f}, CO2: {co2[-1]:.2f}, Energy: {room_heating_energy[-1]:.2f}, Reward: {rewards[-1]:.2f}")
        
        return rewards

if __name__ == "__main__":
    model_filename = os.path.join(uppath(os.path.abspath(__file__), 1), "fan_flow_configuration_template_DP37_full_no_cooling.xlsm")
    #Load the input/output dictionary from the file policy_input_output.json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_output_schema_path = os.path.join(script_dir, "policy_input_output.json")

    trainer = PolicyTrainer(
        model_path=model_filename,
        input_output_schema_path=input_output_schema_path
    )
    trainer.train(num_episodes=350)
