"""
This script uses the t4b_gym environment with the t4b model of the BOPTEST model multizone_simple_air 
It defines a custom reward function and uses the A2C algorithm to control different setpoints of the model. 

USAGE EXAMPLES:

# Train a new A2C agent:
python use_case/multizone_simple_air_RL_A2C.py --algo a2c

# Continue training a A2C agent from the last checkpoint:
python use_case/multizone_simple_air_RL_A2C.py --algo a2c --continue

# Continue training and then test the A2C agent:
python use_case/multizone_simple_air_RL_A2C.py --algo a2c --continue --test

"""

import twin4build as tb
import datetime
import twin4build.examples.utils as utils
from dateutil.tz import gettz 
import sys
import os
import logging
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import numpy as np
import glob
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(MAIN_DIR)
from t4b_gym.t4b_gym_env import T4BGymEnv, NormalizedObservationWrapper, NormalizedActionWrapper
from boptest_model.rooms_and_ahu_model import load_model_and_params
from use_case.model_eval import test_model

A2C_LOG_DIR = os.path.join(SCRIPT_DIR, 'logs_a2c')
os.makedirs(A2C_LOG_DIR, exist_ok=True)

POLICY_CONFIG_PATH = os.path.join(SCRIPT_DIR, "policy_input_output.json")
device = 'cpu'


def A2C_training(test_model_flag=False, reload_model_flag=False, use_autoencoder=False, total_timesteps=100000,
                 latent_dim=64, network_size='large', load_pretrained_bc=False):
    """
    Train A2C with optional autoencoder support.
    
    Args:
        test_model_flag: If True, test the model instead of training
        reload_model_flag: If True, reload an existing model for continued training
        use_autoencoder: If True, use autoencoder for observation compression
        latent_dim: Latent dimension for autoencoder (default: 64)
        network_size: Autoencoder network size (small/medium/large/xlarge)
        load_pretrained_bc: If True, load pretrained behavioral cloning model
    """
    
    # Create a new model
    model = load_model_and_params()

    stepSize = 600 #Seconds
    #Define the range of available data
    start_time = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))
    end_time = datetime.datetime(year=2024, month=1, day=15, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))        

    class T4BGymEnvCustomReward(T4BGymEnv):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.previous_objective = 0.0
            # Initialize previous_action to zeros - will be properly set after first action
            self.previous_action = None
            # Initialize imitation learning components
            self.expert_actions = None
            self.imitation_weight = 0.0
            self.training_phase = 'regular'
            
        def get_reward(self, observations, action):
            # Temperature violations for all zones
            zones = ['core', 'north', 'east', 'south', 'west']
            temp_violations = []
            
            for zone in zones:
                temp = self.simulator.model.components[f"{zone}_indoor_temp_sensor"].output["measuredValue"]
                heating_setpoint = self.simulator.model.components[f"{zone}_temperature_heating_setpoint"].output["scheduleValue"]
                cooling_setpoint = self.simulator.model.components[f"{zone}_temperature_cooling_setpoint"].output["scheduleValue"]
                
                # Calculate violations with deadband
                heating_violation = max(0, heating_setpoint - temp)
                cooling_violation = max(0, temp - cooling_setpoint)
                
                # Use quadratic penalty instead of exponential for stability
                zone_violation = (1+heating_violation)**2 + (1+cooling_violation)**2
                temp_violations.append(zone_violation)
            
            # Balanced temperature penalty
            temp_violation_penalty = 1000 * sum(temp_violations)  # Reduced from 10000
            
            # Estimated coil power consumption
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
            
            # Moderate coil power consumption penalty
            coils_power_consumption_penalty = 0.01 * sum(coils_power_consumption)
            
            # AHU power consumption
            fan_power = self.simulator.model.components["vent_power_sensor"].output["measuredValue"]
            supply_cooling_coil_power = self.simulator.model.components["supply_cooling_coil"].output["Power"]
            supply_heating_coil_power = self.simulator.model.components["supply_heating_coil"].output["Power"]
            ahu_power_consumption_penalty = 0.01 * (fan_power + supply_cooling_coil_power + supply_heating_coil_power)
            
            # Penalty for closing the dampers below 15%
            damper_position_penalty = 0
            for zone in zones:
                damper_position = self.simulator.model.components[f"{zone}_supply_damper_position_sensor"].output["measuredValue"]
                if damper_position < 0.15:
                    damper_position_penalty += 1000 * (0.15 - damper_position)

            #Penalty for temporal smoothness of the actions
            if self.previous_action is not None:
                action_penalty = 0.01 * np.sum(np.abs(action - self.previous_action))
            else:
                action_penalty = 0.0  # No penalty on first step
            self.previous_action = action.copy()  # Use copy to avoid reference issues

            # Imitation learning component - encourage staying close to expert demonstrations
            imitation_penalty = 0.0
            if self.expert_actions is not None:
                # Get current timestep in episode
                current_step = getattr(self.simulator, 'current_step', 0)
                
                if current_step < len(self.expert_actions):
                    expert_action = self.expert_actions[current_step]
                    # Imitation penalty - encourage staying close to expert
                    imitation_penalty = np.sum((action - expert_action)**2)
                else:
                    # If we've exceeded the expert trajectory length, use the last expert action
                    # This handles cases where episodes are longer than the expert data
                    expert_action = self.expert_actions[-1]
                    imitation_penalty = np.sum((action - expert_action)**2)

            # Weighted combination: start with high imitation weight, gradually reduce
            if self.training_phase == 'bc_fine_tune':
                # During BC fine-tuning, use imitation learning with decay
                imitation_weight = max(0.1, 1.0 - getattr(self.simulator, 'current_step', 0) / 1000)  # Decay over time
                task_weight = 1.0 - imitation_weight
            else:
                # During regular training, no imitation
                imitation_weight = 0.0
                task_weight = 1.0

            # Total objective with imitation learning
            task_reward = temp_violation_penalty + coils_power_consumption_penalty + ahu_power_consumption_penalty + damper_position_penalty + action_penalty
            task_reward = task_reward/1000 #Making the number smaller for readability
            
            reward = task_weight * (-task_reward) + imitation_weight * (-imitation_penalty)
            
     

            if np.isnan(reward):
                raise ValueError("Reward is not a number")
            
            
            return reward
        
        def load_expert_demonstrations(self, expert_data_path=None):
            """
            Load expert demonstrations for imitation learning from expert_trajectories.npz.
            The expert data is unnormalized, so we need to normalize it to match the environment's action space.
            """
            if expert_data_path is None:
                expert_data_path = os.path.join(os.path.dirname(__file__), "expert_trajectories.npz")
            
            if not os.path.exists(expert_data_path):
                print(f"Warning: Expert data file not found at {expert_data_path}")
                print("Using dummy expert actions as fallback.")
                # Create dummy expert actions that are reasonable for HVAC control
                episode_length = int(3600*24*5 / 600)  # 5 days with 600s steps
                action_dim = self.action_space.shape[0]  # Use actual action space dimension
                self.expert_actions = np.random.uniform(0.2, 0.8, (episode_length, action_dim))
                self.training_phase = 'bc_fine_tune'
                return
            
            print(f"Loading expert demonstrations from {expert_data_path}")
            
            # Load expert trajectories
            data = np.load(expert_data_path, allow_pickle=True)
            expert_acts = data['acts']  # Shape: (timesteps, action_dim)
            
            print(f"Loaded expert actions with shape: {expert_acts.shape}")
            print(f"Expert action range: [{expert_acts.min():.3f}, {expert_acts.max():.3f}]")
            
            # Normalize expert actions to match the environment's action space
            # The environment uses NormalizedActionWrapper which maps actions to [-1, 1]
            # We need to apply the same normalization as the wrapper
            
            # Get the original action space bounds (before normalization wrapper)
            # Create a temporary environment without wrappers to get the original bounds
            from boptest_model.rooms_and_ahu_model import load_model_and_params
            from t4b_gym.t4b_gym_env import T4BGymEnv
            
            model = load_model_and_params()
            temp_env = T4BGymEnv(
                model=model, 
                io_config_file=POLICY_CONFIG_PATH,
                start_time=datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen")),
                end_time=datetime.datetime(year=2024, month=1, day=15, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen")),
                episode_length=int(3600*24*5 / 600),
                random_start=True, 
                excluding_periods=None, 
                forecast_horizon=50,
                step_size=600,
                warmup_period=0
            )
            
            # Get original action bounds
            act_low = temp_env.action_space.low
            act_high = temp_env.action_space.high
            
            # Apply the same normalization as NormalizedActionWrapper: 2*(acts - low)/(high - low) - 1
            normalized_expert_acts = 2 * (expert_acts - act_low) / (act_high - act_low) - 1
            
            # Clip to [-1, 1] bounds (same as wrapper)
            normalized_expert_acts = np.clip(normalized_expert_acts, -1, 1)
            
            print(f"Normalized expert actions to range: [{normalized_expert_acts.min():.3f}, {normalized_expert_acts.max():.3f}]")
            
            # Store the normalized expert actions
            self.expert_actions = normalized_expert_acts
            self.training_phase = 'bc_fine_tune'
            
            # Close temporary environment
            temp_env.close()
            
            print(f"Successfully loaded {len(self.expert_actions)} expert action timesteps for imitation learning")

    # Create environment with conditional normalization
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
    
    # Apply normalization wrappers conditionally
    if not use_autoencoder:
        # Apply observation normalization only when NOT using autoencoder
        env = NormalizedObservationWrapper(env)
        print("Applied observation normalization wrapper")
    else:
        print("Skipped observation normalization (autoencoder will handle it)")
    
    # Always apply action normalization (autoencoder doesn't affect actions)
    env = NormalizedActionWrapper(env)
    print("Applied action normalization wrapper")

    # Apply autoencoder if requested
    if use_autoencoder:
        print(f"Setting up autoencoder with latent_dim={latent_dim}, network_size={network_size}")
        
        # Import autoencoder functions from pretrain script
        from pretrain_with_expert import create_autoencoder_env, diagnose_autoencoder_outputs
        
        # Create autoencoder environment
        env, encoder, decoder = create_autoencoder_env(
            env, 
            network_size=network_size, 
            latent_dim=latent_dim
        )
        
        print(f"Observation space compressed to {latent_dim} dimensions")

    # Modify the environment to include the callback
    env = Monitor(env=env, filename=os.path.join(A2C_LOG_DIR,'monitor.csv'))

    if test_model_flag:
        model_path = os.path.join(A2C_LOG_DIR, "a2c_pretrained_bc.zip")
        model = A2C.load(model_path, env=env, device=device)
        print(f"Training steps: {model.num_timesteps}")
        test_model(env, model)
        return

    # Load pretrained behavioral cloning model if requested
    if load_pretrained_bc:
        bc_model_path = os.path.join(os.path.dirname(__file__), "a2c_pretrained_bc.zip")
        if os.path.exists(bc_model_path):
            print(f"Loading pretrained behavioral cloning model from {bc_model_path}")
            model = A2C.load(bc_model_path, env=env, device=device)
            print(f"Loaded pretrained model with {model.num_timesteps} timesteps")
            
            # Set up imitation learning for BC fine-tuning
            print("Setting up imitation learning for BC fine-tuning...")
            env.load_expert_demonstrations()  # Load expert demonstrations
            env.training_phase = 'bc_fine_tune'
            
            # Set lower learning rate for fine-tuning from pretrained model
            fine_tune_lr = 1e-6  # 10x lower than default for fine-tuning
            model.learning_rate = fine_tune_lr
            print(f"Set learning rate to {fine_tune_lr} for fine-tuning from pretrained model")
            
            # Set high verbosity for detailed training output
            model.verbose = 2
            print("Set training verbosity to maximum (2) for detailed progress")
        else:
            raise FileNotFoundError(f"Pretrained behavioral cloning model not found at {bc_model_path}. "
                                  f"Please run the pretraining script first to generate the required model file.")
    else:
        # Create new A2C model with appropriate hyperparameters
        model = A2C('MlpPolicy', env, verbose=2, gamma=0.99,      
            learning_rate=7e-4, n_steps=5, ent_coef=0.01, 
            vf_coef=0.25, max_grad_norm=0.5, tensorboard_log=A2C_LOG_DIR, device=device)
        print("Set training verbosity to maximum (2) for detailed progress")

    # Create the callback
    #Disable evaluation for now
    callback = EvalCallback(env, best_model_save_path=A2C_LOG_DIR, log_path=A2C_LOG_DIR, eval_freq=1000000, n_eval_episodes=5)

    # Train the model
    print(f"\n{'='*60}")
    print("STARTING A2C TRAINING")
    print(f"{'='*60}")
    
    print(f"Total timesteps to train: {total_timesteps:,}")
    print(f"Environment: {env.__class__.__name__}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Device: {device}")
    print(f"Log directory: {A2C_LOG_DIR}")
    print(f"{'='*60}\n")
    
    if reload_model_flag:
        # Find the latest model file
        model_files = glob.glob(os.path.join(A2C_LOG_DIR, "*.zip"))
        if model_files:
            latest_model = max(model_files, key=os.path.getctime)
            print(f"Reloading existing model from {latest_model}")
            model = A2C.load(latest_model, env=env, device=device)
            print(f"Loaded model with {model.num_timesteps} previous timesteps")

            # Set lower learning rate for fine-tuning from pretrained model
            fine_tune_lr = 1e-6  # 10x lower than default for fine-tuning
            model.learning_rate = fine_tune_lr
            print(f"Set learning rate to {fine_tune_lr} for fine-tuning from pretrained model")

            new_logger = configure(A2C_LOG_DIR, ['csv'])
            model.set_logger(new_logger)

            print("Continuing training with existing model...")
            model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=False)
        else:
            print("No existing model found. Starting fresh training...")
            new_logger = configure(A2C_LOG_DIR, ['csv'])
            model.set_logger(new_logger)
            model.learn(total_timesteps=total_timesteps, callback=callback)
    else:
        new_logger = configure(A2C_LOG_DIR, ['csv'])
        model.set_logger(new_logger)

        print("Starting fresh training...")
        model.learn(total_timesteps=total_timesteps, callback=callback)
    
    print(f"\n{'='*60}")
    print("A2C TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Final model timesteps: {model.num_timesteps}")
    print(f"{'='*60}\n")

    # Save the model
    model_save_path = os.path.join(A2C_LOG_DIR, "a2c_model")
    print(f"Saving trained model to {model_save_path}")
    model.save(model_save_path)
    print(f"Model saved successfully!")
    print(f"Model file size: {os.path.getsize(model_save_path + '.zip') / (1024*1024):.2f} MB")

    #Test the model
    test_model(env, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train A2C agent for HVAC control')
    parser.add_argument('--algo', type=str, default='a2c', help='Algorithm to use (default: a2c)')
    parser.add_argument('--continue_flag', action='store_true', default=False, help='Continue training from existing model')
    parser.add_argument('--test', action='store_true', default=False, help='Test the model after training')
    parser.add_argument('--autoencoder', action='store_true', default=False, help='Use autoencoder for observation compression')
    parser.add_argument('--latent_dim', type=int, default=64, help='Latent dimension for autoencoder (default: 64)')
    parser.add_argument('--network_size', type=str, default='large', choices=['small', 'medium', 'large', 'xlarge'], 
                       help='Autoencoder network size (default: large)')
    parser.add_argument('--load_pretrained_bc', action='store_true', help='Load pretrained behavioral cloning model')
    parser.add_argument('--total_timesteps', type=int, default=100000, help='Total timesteps for training (default: 100000)')
    
    args = parser.parse_args()
    
    # Example usage scenarios:
    
    # 1. Standard A2C training
    # A2C_training(test_model_flag=False, reload_model_flag=False)
    
    # 2. A2C training with autoencoder
    # A2C_training(test_model_flag=False, reload_model_flag=False, use_autoencoder=True, latent_dim=64, network_size='large')
    
    # 3. Load pretrained behavioral cloning model and continue RL fine-tuning
    # A2C_training(test_model_flag=False, reload_model_flag=False, use_autoencoder=True, latent_dim=64, network_size='large', load_pretrained_bc=True)
    
    # 4. Test trained model
    # A2C_training(test_model_flag=True, reload_model_flag=False, use_autoencoder=True, latent_dim=64, network_size='large')
    
    # Execute based on command line arguments
    if args.test and not args.continue_flag:
        # Test mode without continuing training
        A2C_training(test_model_flag=True, reload_model_flag=False, 
                    use_autoencoder=args.autoencoder, latent_dim=args.latent_dim, 
                    network_size=args.network_size)
    else:
        # Training mode (with optional continue and test)
        A2C_training(test_model_flag=False, reload_model_flag=args.continue_flag, 
                    use_autoencoder=args.autoencoder, latent_dim=args.latent_dim, 
                    network_size=args.network_size, load_pretrained_bc=args.load_pretrained_bc,
                    total_timesteps=args.total_timesteps)
        
        # If test flag is set, test the model after training
        if args.test:
            print("\n" + "="*60)
            print("TESTING TRAINED MODEL")
            print("="*60)
            A2C_training(test_model_flag=True, reload_model_flag=False, 
                        use_autoencoder=args.autoencoder, latent_dim=args.latent_dim, 
                        network_size=args.network_size)
