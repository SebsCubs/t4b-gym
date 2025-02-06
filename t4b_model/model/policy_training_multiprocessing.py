import torch.multiprocessing as mp
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
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
import multiprocessing.resource_tracker as resource_tracker

if __name__ == '__main__':
    uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = os.path.join(uppath(os.path.abspath(__file__), 4), "Twin4Build")
    sys.path.append(file_path)


class ExperienceCollector:
    """
    Collects experience from the simulation environment in a multiprocessing manner.
    Saves the experience to a memory object.
    Goes inside the PolicyTrainer class.
    Instantiates the model and the memory object, it contains the RL agent policy.
    Starts the simulation in a multiprocessing manner.
    The PolicyTrainer updates the policy every num_processes episodes. -> Needs a lock to update the policy.
    """
    def __init__(self, t4b_model):
        self.t4b_model = t4b_model

    @staticmethod
    def setup_logger(process_id, process_logs=False):
        """
        Setup a logger for the current process
        """
        try:
            # Create absolute path for logs directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            log_dir = os.path.join(script_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            
            # Create a logger
            logger = logging.getLogger(process_id)
            logger.setLevel(logging.INFO)
            
            # Clear any existing handlers
            if logger.hasHandlers():
                logger.handlers.clear()
            
            # Create handlers with absolute paths
            handlers = []
            
            # Process-specific log file (only if process_logs is True)
            if process_logs:
                process_handler = RotatingFileHandler(
                    os.path.join(log_dir, f'process_{process_id}.log'),
                    maxBytes=10*1024*1024,  # 10MB
                    backupCount=5
                )
                handlers.append(process_handler)
            
            # Main log file that combines all processes
            main_handler = RotatingFileHandler(
                os.path.join(log_dir, 'main.log'),
                maxBytes=50*1024*1024,  # 50MB
                backupCount=5
            )
            handlers.append(main_handler)
            
            # Create formatter and add it to handlers
            log_format = logging.Formatter(
                '[%(asctime)s][%(name)s][%(processName)s] %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # Add formatters and handlers to logger
            for handler in handlers:
                handler.setFormatter(log_format)
                logger.addHandler(handler)
            
            # Write a test message to verify logger setup
            logger.info(f"Logger initialized for {process_id}")
            logger.info(f"Logging to directory: {log_dir}")
            if process_logs:
                logger.info(f"Process log file: {os.path.join(log_dir, f'process_{process_id}.log')}")
            
            return logger
            
        except Exception as e:
            # Fallback to basic file logging if rotating handler fails
            try:
                # Ensure the logs directory exists
                script_dir = os.path.dirname(os.path.abspath(__file__))
                log_dir = os.path.join(script_dir, "logs")
                os.makedirs(log_dir, exist_ok=True)
                
                # Create a basic file handler
                basic_handler = logging.FileHandler(
                    os.path.join(log_dir, f'process_{process_id}_basic.log')
                )
                
                # Set up basic logging
                logger = logging.getLogger(process_id)
                logger.setLevel(logging.INFO)
                
                if logger.hasHandlers():
                    logger.handlers.clear()
                    
                basic_handler.setFormatter(logging.Formatter(
                    '[%(asctime)s][%(name)s][%(processName)s] %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                ))
                
                logger.addHandler(basic_handler)
                logger.error(f"Failed to set up rotating file handler: {str(e)}")
                logger.info("Falling back to basic file handler")
                
                return logger
                
            except Exception as e2:
                # If all else fails, print to stderr
                print(f"Critical error setting up logging: {str(e2)}", file=sys.stderr)
                raise

    @staticmethod
    def _run_episode_worker(t4b_model, start_time, end_time, process_logs=False):
        """
        Static worker method that runs in each process.
        """
        process_id = mp.current_process().name
        logger = None
        
        try:
            # Only set up logger if process_logs is True
            if process_logs:
                logger = ExperienceCollector.setup_logger(process_id, process_logs)
                logger.info(f"Process {process_id} initialized")
                logger.info("Creating simulation model copy")
            
            simulation_model = copy.deepcopy(t4b_model)
            collector = ExperienceCollector(simulation_model)
            
            if process_logs:
                logger.info("Starting simulation")
            simulator = tb.Simulator()
            simulator.simulate(simulation_model, start_time, end_time, stepSize=600)
            if process_logs:
                logger.info("Simulation completed")
                logger.info("Processing simulation data")
            
            states = collector.get_state_from_saved(simulation_model)
            actions = collector.get_action_from_saved(simulation_model)
            rewards = collector.compute_reward_from_saved(simulation_model, '[035A][035A_space_heater]')

            if process_logs:
                logger.info("Computing logprobs")
            with torch.no_grad():
                action_mean, action_std = simulation_model.components["neural_controller"].policy(states)
                dist = torch.distributions.Normal(action_mean, action_std)
                logprobs = dist.log_prob(actions).sum(dim=-1)
            
            episode_data = {
                'states': states.cpu().detach().numpy(),
                'actions': actions.cpu().detach().numpy(),
                'rewards': rewards.cpu().detach().numpy(),
                'logprobs': logprobs.cpu().detach().numpy()
            }

            if process_logs:
                logger.info("Episode data shapes:")
                for key, value in episode_data.items():
                    logger.info(f"- {key}: {value.shape}, range: [{value.min():.3f}, {value.max():.3f}]")
            
            return episode_data
            
        except Exception as e:
            if process_logs and logger:
                logger.error(f"Error in worker process:", exc_info=True)
            return None

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
        temp_setpoint =  torch.tensor(model.components['035A_temperature_heating_setpoint'].savedOutput['scheduleValue'])
        co2_setpoint =  torch.tensor(model.components['035A_co2_setpoint'].savedOutput['scheduleValue'])
        
        # Temperature penalty
        temp_error = torch.abs(temperature - temp_setpoint)
        temp_penalty = -(temp_error) * 10
        
        # CO2 penalty (above setpoint)
        co2_error = torch.clamp(co2 - co2_setpoint, min=0)
        co2_penalty = -(co2_error) * 0.1
        
        # Energy penalty (linear)
        energy_penalty = -room_heating_energy * 0.01 - fan_energy * 0.001
        
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
    def __init__(self, model_path, input_output_schema_path, num_processes=4):
        try:
            # Create logs directory first
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.log_dir = os.path.join(script_dir, "logs")
            os.makedirs(self.log_dir, exist_ok=True)
            
            # Verify directory exists and is writable
            if not os.path.exists(self.log_dir):
                raise OSError(f"Failed to create log directory: {self.log_dir}")
            
            test_file = os.path.join(self.log_dir, "test_write.txt")
            try:
                with open(test_file, 'w') as f:
                    f.write("Test write access\n")
                os.remove(test_file)
            except Exception as e:
                raise OSError(f"Log directory is not writable: {str(e)}")
            
            # Setup main logger
            self.logger = ExperienceCollector.setup_logger('MainProcess')
            if not self.logger:
                raise RuntimeError("Failed to create logger")
                
            self.logger.info("Initializing PolicyTrainer")
            
            # Log important initialization information
            self.logger.info(f"Main process ID: {os.getpid()}")
            self.logger.info(f"Python version: {sys.version}")
            self.logger.info(f"Operating system: {sys.platform}")
            self.logger.info(f"Working directory: {os.getcwd()}")
            self.logger.info(f"Log directory: {self.log_dir}")
            
            # Initialize multiprocessing components
            try:
                mp.set_start_method('spawn', force=True)
                self.ctx = mp.get_context('spawn')
                self.num_processes = num_processes
                self.pool = None  # Initialize pool as None
                self.logger.info("Multiprocessing components initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize multiprocessing: {str(e)}")
                raise

            # Initialize rest of the trainer
            self.model_path = model_path
            self.schema_path = input_output_schema_path
            self.setup_twin4build_model(self.model_path, self.schema_path)
            self.setup_ppo()
            self.memory = Memory()
            self.experience_collector = ExperienceCollector(self.base_model)
            self.writer = SummaryWriter(f'runs/ppo_training_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')

            # Register cleanup handler for system exit
            import atexit
            atexit.register(self.close)

        except Exception as e:
            print(f"Critical error during PolicyTrainer initialization: {str(e)}", file=sys.stderr)
            raise

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
        #rewards = self.memory.rewards
        #rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        rewards = self.memory.rewards.detach() # One reward per timestep, of all the processes
        memory_dict = {
            'states': self.memory.states.detach(),
            'actions': self.memory.actions.detach(),
            'rewards': self.memory.rewards.detach(),
            'logprobs': self.memory.logprobs.detach()
        }
        

        #sum the episode rewards by dividing by the number of processes
        episode_average_reward = rewards.sum(dim=0) / self.num_processes

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
                self.writer.add_scalar('Reward/average', episode_average_reward, self.training_step)
                #Report average moving average of the rewards
                #self.writer.add_scalar('Reward/moving_average', np.mean(rewards[-100:]), self.training_step)
                
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

    def _ensure_pool(self):
        """Ensure the process pool exists and is active"""
        try:
            # Check if pool needs to be created or recreated
            if self.pool is None:
                self.logger.info("Creating new process pool - no pool exists")
                self.pool = self.ctx.Pool(processes=self.num_processes, maxtasksperchild=5)
                return True

            # Test if pool is still usable by checking one of its public methods
            try:
                self.pool.apply_async(int, (0,)).get(timeout=1)
            except (TimeoutError, AttributeError, ValueError, ConnectionError):
                self.logger.info("Creating new process pool - existing pool is inactive")
                # Clean up old pool if it exists
                try:
                    self.pool.terminate()
                    self.pool.close()
                    self.pool.join()
                except Exception as e:
                    self.logger.warning(f"Error cleaning up old pool: {str(e)}")
                
                # Create new pool
                self.pool = self.ctx.Pool(processes=self.num_processes, maxtasksperchild=5)

            self.logger.info(f"Process pool is active with {self.num_processes} processes")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create/verify process pool: {str(e)}")
            return False

    def run_batch(self, num_processes, start_episode):
        self.logger.info(f"\n=== Starting batch with {num_processes} processes ===")
        
        if not self._ensure_pool():
            self.logger.error("Failed to create process pool!")
            return []
            
        self.logger.info("Process pool is ready")
        
        process_args = []
        for i in range(num_processes):
            startTime = datetime.datetime(year=2023, month=11, day=27, hour=0, minute=0, second=0, 
                                       tzinfo=gettz("Europe/Copenhagen"))
            endTime = datetime.datetime(year=2023, month=12, day=7, hour=0, minute=0, second=0,
                                     tzinfo=gettz("Europe/Copenhagen"))
            process_args.append((self.base_model, startTime, endTime))
            self.logger.info(f"Prepared args for process {i}")

        # Launch processes
        async_results = []
        for i, args in enumerate(process_args):
            try:
                self.logger.info(f"Launching process {i}")
                result = self.pool.apply_async(self.experience_collector._run_episode_worker, args)
                async_results.append(result)
                self.logger.info(f"Successfully launched process {i}")
            except Exception as e:
                self.logger.error(f"Failed to launch process {i}: {str(e)}")
                continue
        
        
        # Collect results
        results = []
        timeout = 120
        for i, async_result in enumerate(async_results):
            try:
                self.logger.info(f"Waiting for result from process {i}")
                result = async_result.get(timeout=timeout)
                results.append(result)
                self.logger.info(f"Received result from process {i}")
            except Exception as e:
                self.logger.error(f"Error getting result from process {i}: {str(e)}")

        # Process results
        self.logger.info(f"\nProcessing {len(results)} results")
        batch_rewards = []
        for i, episode_data in enumerate(results):
            if isinstance(episode_data, dict) and 'error' in episode_data:
                self.logger.error(f"\nProcess {i} failed with error:")
                self.logger.error(episode_data['error'])
                self.logger.error(f"\nTraceback:\n{episode_data['traceback']}")
                continue
                
            if episode_data is None:
                self.logger.warning(f"Episode {start_episode + i}: Data is None, skipping")
                continue
                
            self.logger.info(f"Converting episode {start_episode + i} data to tensors")
            episode_data = {
                'states': torch.FloatTensor(episode_data['states']),
                'actions': torch.FloatTensor(episode_data['actions']),
                'rewards': torch.FloatTensor(episode_data['rewards']),
                'logprobs': torch.FloatTensor(episode_data['logprobs'])
            }
            
            episode_reward = episode_data['rewards'].sum()
            batch_rewards.append(episode_reward)
            
            self.logger.info(f"Episode {start_episode + i} stats:")
            self.logger.info(f"- States shape: {episode_data['states'].shape}")
            self.logger.info(f"- Actions shape: {episode_data['actions'].shape}")
            self.logger.info(f"- Rewards shape: {episode_data['rewards'].shape}")
            self.logger.info(f"- Total reward: {episode_reward:.2f}")
            
            self.memory.extend(episode_data)

        self.logger.info(f"\n=== Batch completed, collected {len(batch_rewards)} episodes ===\n")
        return batch_rewards

    def train(self, num_episodes=350):
        """
        Runs the training loop using batches of parallel episodes.
        After each batch, updates the policy and logs results.
        
        Args:
            num_episodes (int): Total number of episodes to run
            batch_size (int): Number of parallel processes per batch
        """
        best_reward = float('-inf')

        
        for batch_start in range(0, num_episodes, self.num_processes):
            # Run a batch of episodes
            current_batch_size = min(self.num_processes, num_episodes - batch_start)
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

        # Save the final policy with the best reward and a timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        torch.save(self.ppo_agent.policy.state_dict(), f'final_policy_{timestamp}.pth')

    def close(self):
        """Explicit cleanup method"""
        # First close the pool if it exists
        if hasattr(self, 'pool') and self.pool is not None:
            try:
                self.logger.info("Closing process pool") if hasattr(self, 'logger') else print("Closing process pool")
                # Terminate all processes first
                self.pool.terminate()
                # Then close and join
                self.pool.close()
                self.pool.join()
                self.pool = None
            except Exception as e:
                print(f"Error during pool cleanup: {str(e)}")
        
        # Clean up any remaining multiprocessing resources
        try:
            if hasattr(resource_tracker._resource_tracker, '_pid'):
                # Only attempt cleanup if resource tracker exists and has a valid PID
                if resource_tracker._resource_tracker._pid is not None:
                    resource_tracker._resource_tracker._check_alive()
                    resource_tracker._resource_tracker._stop()
        except Exception as e:
            print(f"Error cleaning up multiprocessing resources: {str(e)}")

    def __del__(self):
        """Cleanup method that calls close() only if not already cleaned up"""
        print("Deleting PolicyTrainer object")
        if hasattr(self, 'pool') and self.pool is not None:
            try:
                self.close()
            except:
                pass  # Suppress all errors during deletion


if __name__ == "__main__":
    model_path = os.path.join(uppath(os.path.abspath(__file__), 1), "fan_flow_configuration_template_DP37_full_no_cooling.xlsm")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_output_schema_path = os.path.join(script_dir, "policy_input_output.json")

    trainer = PolicyTrainer(model_path, input_output_schema_path, num_processes=4)
    try:
        trainer.train(num_episodes=350)
    finally:
        trainer.close()  # Ensure cleanup happens
