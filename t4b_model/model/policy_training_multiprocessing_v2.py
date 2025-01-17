import multiprocessing as mp
from multiprocessing import Pool
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) # Add the grandparent directory to the system path
import twin4build as tb
import torch
import datetime
from dateutil.tz import gettz
import json
import copy
from RL_Algos.ppo_agent import PPOAgent, Memory
from neural_policy_standalone_sim import fcn
import twin4build.examples.utils as utils
from torch.utils.tensorboard import SummaryWriter
import numpy as np

if __name__ == '__main__':
    uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = os.path.join(uppath(os.path.abspath(__file__), 4), "Twin4Build")
    sys.path.append(file_path)


class ExperienceCollector:
    """
    Collects experience from the simulation environment in a multiprocessing manner.
    Saves the experience to a memory object. buffered in a queue.
    Goes inside the PolicyTrainer class.
    Instantiates the model and the memory object, it contains the RL agent policy.
    Starts the simulation in a multiprocessing manner.
    Collects the experience from the simulation and saves it to the memory object through a queue.
    The PolicyTrainer updates the policy every num_processes episodes. -> Needs a lock to update the policy.
    """
    def __init__(self, t4b_model, queue=None):
        self.t4b_model = t4b_model
        self.queue = queue

    @staticmethod
    def _run_episode_worker(t4b_model, start_time, end_time, queue):
        """
        Static worker method that runs in each process.
        
        Args:
            model_path: Path to the Twin4Build model
            schema_path: Path to the I/O schema
            policy_state_dict: State dict of the policy to use
            start_time: Simulation start time
            end_time: Simulation end time
            queue: Multiprocessing queue to store results
        """
        try:
            # Create a new collector instance for this process
            simulation_model = copy.deepcopy(t4b_model)
            collector = ExperienceCollector(simulation_model, queue) 
            # Run the episode
            simulator = tb.Simulator()  # Create simulator instance
            simulator.simulate(simulation_model, start_time, end_time, stepSize=600)
            
            # Get the episode data from the model's saved outputs
            states = collector.get_state_from_saved(simulation_model)
            actions = collector.get_action_from_saved(simulation_model)
            rewards = collector.compute_reward_from_saved(simulation_model, '[020B][020B_space_heater]')

            # Get logprobs for the actions (needed for PPO)
            with torch.no_grad():
                action_mean, action_std = simulation_model.components["neural_controller"].policy(states)
                dist = torch.distributions.Normal(action_mean, action_std)
                logprobs = dist.log_prob(actions).sum(dim=-1)
            
            """
            #Windows only
            # Convert tensors to numpy arrays before putting in queue
            episode_data = {
                'states': states.numpy(),
                'actions': actions.numpy(),
                'rewards': rewards.numpy(),
                'logprobs': logprobs.numpy()
            }
            """
            episode_data = {
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'logprobs': logprobs
            }

            queue.put(episode_data)
            
        except Exception as e:
            print(f"Error in worker process: {str(e)}")
            queue.put(None)
            raise e

    def start_episode(self, t4b_model, start_time, end_time, queue):
        """
        Starts a new episode in a separate process.
        
        Args:
            t4b_model: The Twin4Build model
            start_time: Simulation start time
            end_time: Simulation end time
            queue: Multiprocessing queue to store results
        Returns:
            multiprocessing.Process: The created process object
        """
        # Create and return a new process
        return mp.Process(
            target=self._run_episode_worker,
            args=(
                t4b_model,
                start_time,
                end_time,
                queue
            )
        )

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


class PolicyTrainer:
    """
    Trains the policy using the collected experience.
    Retrieves the experience from the memory object.
    Updates the policy using PPO.
    The PolicyTrainer updates the policy every num_processes episodes. -> Needs a lock to update the policy.
    """
    def __init__(self, model_path, input_output_schema_path):
        self.model_path = model_path  # Store paths for worker processes
        self.schema_path = input_output_schema_path
        self.setup_twin4build_model(self.model_path, self.schema_path)
        self.setup_ppo()
        self.queue = mp.Queue()
        self.memory = Memory()
        self.experience_collector = ExperienceCollector(self.base_model, self.queue)
        self.writer = SummaryWriter(f'runs/ppo_training_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')

    def setup_twin4build_model(self, model_path, schema_path):
        # Load the base model
        self.base_model = tb.Model(id="training_model")
        with open(schema_path) as f:
            self.input_output_schema = json.load(f)
            
        self.input_size = sum(len(signals) for signals in self.input_output_schema["input"].values())
        self.output_size =  5 + 2 + 1 #TODO: Make this dynamic
        self.base_model.load(semantic_model_filename=model_path, fcn=fcn, create_signature_graphs=False, validate_model=True, verbose=False, force_config_update=True)
        # update the policy in the base model
        #The policy in the model must be a PolicyNetwork object, not a nn.Module object
        self.base_model.components["neural_controller"].is_training = True

    def setup_ppo(self):
        self.ppo_agent = PPOAgent(
            state_dim=self.input_size,
            action_dim=self.output_size,
            action_bound=1.0
        )
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
                #Report average reward
                self.writer.add_scalar('Reward/average', np.mean(rewards), self.training_step)
                #Report average moving average of the rewards
                self.writer.add_scalar('Reward/moving_average', np.mean(rewards[-100:]), self.training_step)
                
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

    def run_batch(self, num_processes, start_episode):
        """
        Runs a batch of episodes in parallel and collects their results
        
        Args:
            batch_size (int): Number of parallel processes to run
            start_episode (int): The starting episode number for logging
            
        Returns:
            list: Collected rewards from the batch
        """
        processes = []
        batch_rewards = []
        # Start processes
        for i in range(num_processes):
            startTime = datetime.datetime(year=2023, month=11, day=27, hour=0, minute=0, second=0, 
                                       tzinfo=gettz("Europe/Copenhagen"))
            endTime = datetime.datetime(year=2023, month=12, day=7, hour=0, minute=0, second=0,
                                     tzinfo=gettz("Europe/Copenhagen"))
            p = self.experience_collector.start_episode(self.base_model, startTime, endTime, self.queue)
            processes.append(p)
            p.start()

        # Wait for all processes to complete
        for p in processes:
            p.join()

        # Collect results
        for i in range(num_processes):
            episode_data = self.queue.get()  # Blocking get
            if episode_data is None:
                print("Episode data is None, skipping")
                continue
            episode_reward = episode_data['rewards'].sum()
            batch_rewards.append(episode_reward)
            
            # Update memory with episode data
            self.memory.extend({
                'states': episode_data['states'],
                'actions': episode_data['actions'],
                'rewards': episode_data['rewards'],
                'logprobs': episode_data['logprobs']
            })
            
            # Log metrics
            current_episode = start_episode + i
            #self.writer.add_scalar('Reward/episode', episode_reward, current_episode)
            #write a moving average of the rewards
            #self.writer.add_scalar('Reward/moving_average', np.mean(batch_rewards[-100:]), current_episode)
            print(f"Episode {current_episode}, Reward: {episode_reward:.2f}")

        return batch_rewards

    def train(self, num_episodes=350, num_processes=4):
        """
        Runs the training loop using batches of parallel episodes.
        After each batch, updates the policy and logs results.
        
        Args:
            num_episodes (int): Total number of episodes to run
            batch_size (int): Number of parallel processes per batch
        """
        best_reward = float('-inf')

        
        for batch_start in range(0, num_episodes, num_processes):
            # Run a batch of episodes
            current_batch_size = min(num_processes, num_episodes - batch_start)
            batch_rewards = self.run_batch(current_batch_size, batch_start)

            # Save best policy
            batch_best = max(batch_rewards)
            if batch_best > best_reward:
                best_reward = batch_best
                torch.save(self.ppo_agent.policy.state_dict(), 'best_policy.pth')
                print(f"New best reward: {best_reward:.2f}")
            
            # Update policy using collected experience
            self.update_policy()
            
            # Update the policy in the t4b model
            self.base_model.components["neural_controller"].policy.load_state_dict(self.ppo_agent.policy.state_dict())
            
            # Clear memory after policy update
            self.memory.clear()


if __name__ == "__main__":
    model_path = os.path.join(uppath(os.path.abspath(__file__), 1), "fan_flow_configuration_template_DP37_full_no_cooling.xlsm")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_output_schema_path = os.path.join(script_dir, "policy_input_output.json")

    trainer = PolicyTrainer(model_path, input_output_schema_path)
    trainer.train(num_episodes=350, num_processes=2)
