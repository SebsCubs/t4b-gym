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
        
        # Create policy and value networks
        self.ppo_agent = PPOAgent(
            state_dim=self.input_size,
            action_dim=self.output_size,
            action_bound=1.0
        )
        self.train_policy = self.ppo_agent.policy_old
        self.base_model.load(semantic_model_filename=model_path, fcn=fcn, verbose=False)
        # update the policy in the base model
        #The policy in the model must be a PolicyNetwork object, not a nn.Module object
        self.base_model.component_dict["neural_controller"].policy.load_state_dict(self.train_policy.state_dict())

    def setup_ppo(self):
        self.memory = Memory()  # Using the Memory class from ppo_agent.py
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
                torch.save(self.ppo_agent.policy.state_dict(), 'best_policy.pth')
                
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Best: {best_reward:.2f}")

    def run_episode(self, model:tb.Model, simulator:tb.Simulator, start_time, end_time):
        step_size = 600  # 10 minutes
        # Run the full simulation at once
        simulator.simulate(model, startTime=start_time, endTime=end_time, stepSize=step_size)
        episode_reward = 0
        # Get state from saved outputs at this timestep
        states = self.get_state_from_saved(model)
        # Get the action that was taken (from saved outputs)
        actions = self.get_action_from_saved(model)

        # Get the log probability of the action using the policy network directly
        with torch.no_grad():
            # Ensure states has shape [batch_size, state_dim]
            if states.dim() == 1:
                states = states.unsqueeze(0)
            
            # Get action distribution parameters for all states
            action_mean, action_std = self.ppo_agent.policy_old(states)
            
            # Ensure actions has same batch dimension as states
            if actions.dim() == 1:
                actions = actions.unsqueeze(0)
                
            # Create distribution and get log probs for all actions
            dist = torch.distributions.Normal(action_mean, action_std)
            log_prob = dist.log_prob(actions).sum(dim=-1)  # Sum across action dimensions, keeping batch dimension
        
        # Calculate reward for this timestep
        reward = self.compute_reward_from_saved(model, '[020B][020B_space_heater]')
        episode_reward += reward
            
        # Store transitions in memory
        self.memory.add(states, actions, reward, log_prob)
        
        return episode_reward

    def update_policy(self):
        # Convert memory to dictionary format expected by PPO agent
        memory_dict = {
            'states': self.memory.states,
            'actions': self.memory.actions,
            'rewards': self.memory.rewards,
            'logprobs': self.memory.logprobs
        }
        
        # Update both policy and value networks using PPO agent
        self.ppo_agent.update(memory_dict)
        
        # Update the policy network in the model
        self.base_model.component_dict["neural_controller"].policy = self.ppo_agent.policy

        # Clear memory after update
        self.memory.clear()

    
    def get_state_from_saved(self, model):
        # Extract state variables from model.component_dict saved outputs
        # Return as tensor/array matching your state format
        neural_controller = model.component_dict["neural_controller"]
        #The tensor will be a 2d array with the first dimension being the timesteps and the second dimension being the state variables
        state = []
        for key in neural_controller.savedInput.keys():
            state.append(neural_controller.savedInput[key])
        return torch.tensor(state, dtype=torch.float32)


    def get_action_from_saved(self, model):
        # Extract actions from model.component_dict['neural_controller'].savedOutput
        # Return as tensor/array matching your action format
        neural_controller = model.component_dict["neural_controller"]
        actions = []
        for key in neural_controller.savedOutput['output'].keys():
            actions.append(neural_controller.savedOutput['output'][key])
        return torch.tensor(actions, dtype=torch.float32)

    def compute_reward_from_saved(self, model, space_id):
        # Extract relevant metrics from saved outputs
        temperature = model.component_dict[space_id].savedOutput['indoorTemperature']
        co2 = model.component_dict[space_id].savedOutput['indoorCo2Concentration']
        energy = model.component_dict[space_id].savedOutput['heatingPower']
        # Calculate reward 
        temp_penalty = max(0, temperature - 21.0) * 10
        co2_penalty = max(0, co2 - 1000) * 10
        energy_reward = -energy * 0.001
        
        return energy_reward - temp_penalty - co2_penalty

if __name__ == "__main__":
    model_filename = utils.get_path(["parameter_estimation_example", "one_room_example_model.xlsm"])
    input_output_schema_path = utils.get_path(["neural_policy_controller_example", "policy_input_output.json"])
    trainer = PolicyTrainer(
        model_path=model_filename,
        input_output_schema_path=input_output_schema_path
    )
    trainer.train(num_episodes=1000)
