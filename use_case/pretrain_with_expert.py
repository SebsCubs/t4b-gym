"""
This script loads expert trajectories, visualizes the observations, and pretrains a PPO or A2C policy using behavioral cloning (imitation learning) for further RL training.

Usage Examples:
==============

1. Pretrain a PPO policy with behavioral cloning (default large network):
   python pretrain_with_expert.py

2. Pretrain an A2C policy with behavioral cloning:
   python pretrain_with_expert.py --algo a2c

3. Pretrain PPO policy with extra-large network:
   python pretrain_with_expert.py --network-size xlarge

4. Pretrain A2C policy with small network:
   python pretrain_with_expert.py --algo a2c --network-size small

5. Pretrain PPO policy and test it after training:
   python pretrain_with_expert.py --test

6. Pretrain A2C policy with medium network and test it:
   python pretrain_with_expert.py --algo a2c --network-size medium --test

7. Pretrain with sanity checks and save plots:
   python pretrain_with_expert.py --save-plots

8. Use autoencoder for observation compression (with unnormalized data):
   python pretrain_with_expert.py --use-autoencoder --latent-dim 32 --use-unnormalized

9. Combine autoencoder with large network:
   python pretrain_with_expert.py --use-autoencoder --latent-dim 64 --network-size large --use-unnormalized

10. Force retrain autoencoder (clears saved models):
    python pretrain_with_expert.py --retrain-autoencoder

Network Sizes:
=============
- small: [64, 64] (2 layers, 64 units each)
- medium: [256, 256] (2 layers, 256 units each)  
- large: [512, 512, 256, 256] (4 layers, default)
- xlarge: [1024, 1024, 512, 512, 256] (5 layers)

Autoencoder Benefits:
===================
- Compresses 890-dimensional observations to 32-128 dimensions
- Learns meaningful representations from expert data
- Reduces policy complexity and training time
- Improves generalization and sample efficiency
- Bounded latent outputs ([-1, 1] range) for stable training
- Models are automatically saved and reused to avoid retraining

Normalization Handling:
======================
✅ NEW: The script now automatically handles normalization based on autoencoder usage:
- With autoencoder: Skips observation normalization (autoencoder handles it)
- Without autoencoder: Uses NormalizedObservationWrapper (automatically sets observation space to [-1, 1])
- Action normalization is always applied via NormalizedActionWrapper
- Expert trajectory actions are pre-normalized to match NormalizedActionWrapper
- Expert trajectory observations are left unnormalized (wrapper handles normalization during training)
- Autoencoder outputs are bounded to [-1, 1] range with Tanh activation
- Built-in diagnosis verifies autoencoder output ranges

Autoencoder Model Management:
===========================
- Models are automatically saved to `autoencoder_models/` directory
- Saved models are reused on subsequent runs (no retraining needed)
- Model filenames include network size and latent dimension for easy identification
- Use `--retrain-autoencoder` flag to force retraining and clear saved models

Recommended Workflow:
====================
1. Record expert trajectories:
   - For autoencoder: python record_expert_trajectories_unnormalized.py --normalize-actions
   - For non-autoencoder: python record_expert_trajectories.py (normalized observations and actions)

2. Pretrain with autoencoder using optimal data:
   python pretrain_with_expert.py --use-autoencoder --latent-dim 64 --use-unnormalized

3. Pretrain without autoencoder (uses NormalizedObservationWrapper):
   python pretrain_with_expert.py --algo ppo

Output Files:
============
- ppo_pretrained_bc.zip: Pretrained PPO policy weights
- a2c_pretrained_bc.zip: Pretrained A2C policy weights
- autoencoder_models/: Directory containing saved autoencoder models

Note: This script requires one of the following expert trajectory files:
- expert_trajectories.npz (normalized observations and actions)
- expert_trajectories_unnormalized.npz (unnormalized observations and actions)  
- expert_trajectories_unnormalized_norm_actions.npz (unnormalized observations + normalized actions, optimal for autoencoders)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(MAIN_DIR)

from t4b_gym.t4b_gym_env import T4BGymEnv
import datetime
from dateutil.tz import gettz
from use_case.model_eval import test_model

# Imitation library imports
from imitation.algorithms.bc import BC
from imitation.data.types import Transitions

EXPERT_PATH = os.path.join(os.path.dirname(__file__), "expert_trajectories.npz")
EXPERT_PATH_UNNORMALIZED = os.path.join(os.path.dirname(__file__), "expert_trajectories_unnormalized.npz")
EXPERT_PATH_UNNORMALIZED_NORM_ACTIONS = os.path.join(os.path.dirname(__file__), "expert_trajectories_unnormalized_norm_actions.npz")
PRETRAINED_MODEL_PATHS = {
    'ppo': os.path.join(os.path.dirname(__file__), "ppo_pretrained_bc.zip"),
    'a2c': os.path.join(os.path.dirname(__file__), "a2c_pretrained_bc.zip"),
}

stepSize = 600
start_time = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))
end_time = datetime.datetime(year=2024, month=1, day=15, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))


class AutoencoderWrapper(gym.ObservationWrapper):
    """Wrapper that compresses observations using a pretrained autoencoder."""
    
    def __init__(self, env, encoder, latent_dim):
        super().__init__(env)
        self.encoder = encoder
        self.latent_dim = latent_dim
        
        # Set encoder to evaluation mode to avoid BatchNorm issues
        self.encoder.eval()
        
        # Update observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(latent_dim,), 
            dtype=np.float32
        )
    
    def observation(self, obs):
        """Compress observation using the encoder."""
        with torch.no_grad():
            # Ensure encoder is in eval mode
            self.encoder.eval()
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            latent = self.encoder(obs_tensor)
            return latent.squeeze(0).numpy()


def diagnose_autoencoder_outputs(encoder, obs_arr, device='cpu'):
    """Diagnose autoencoder output ranges to ensure they're properly bounded."""
    print("\n=== Autoencoder Output Diagnosis ===")
    
    with torch.no_grad():
        obs_tensor = torch.FloatTensor(obs_arr).to(device)
        
        # Test with a batch of observations
        batch_size = min(1000, len(obs_arr))
        test_batch = obs_tensor[:batch_size]
        
        # Get encoder outputs
        latent = encoder(test_batch)
        latent_np = latent.cpu().numpy()
        
        # Analyze output ranges
        min_vals = np.min(latent_np, axis=0)
        max_vals = np.max(latent_np, axis=0)
        mean_vals = np.mean(latent_np, axis=0)
        std_vals = np.std(latent_np, axis=0)
        
        print(f"Latent space shape: {latent_np.shape}")
        print(f"Output range: [{latent_np.min():.3f}, {latent_np.max():.3f}]")
        print(f"Mean range: [{mean_vals.min():.3f}, {mean_vals.max():.3f}]")
        print(f"Std range: [{std_vals.min():.3f}, {std_vals.max():.3f}]")
        
        # Check if outputs are properly bounded
        if latent_np.min() >= -1.1 and latent_np.max() <= 1.1:
            print("[SUCCESS] Autoencoder outputs are properly bounded (within [-1, 1] range)")
        else:
            print("[WARNING] Autoencoder outputs are NOT properly bounded!")
            print(f"   Expected range: [-1, 1], Actual range: [{latent_np.min():.3f}, {latent_np.max():.3f}]")
        
        # Check for any extreme values
        extreme_count = np.sum(np.abs(latent_np) > 5)
        if extreme_count > 0:
            print(f"[WARNING] Found {extreme_count} values with magnitude > 5")
        else:
            print("[SUCCESS] No extreme values detected")
        
        return latent_np


def pretrain_autoencoder(obs_arr, latent_dim=64, autoencoder_layers=[512, 256, 128], 
                        epochs=100, batch_size=64, device='cpu'):
    """Pretrain the autoencoder on expert observations."""
    print(f"Pretraining autoencoder with latent_dim={latent_dim}...")
    
    # Convert to torch tensors
    obs_tensor = torch.FloatTensor(obs_arr).to(device)
    
    # Create autoencoder
    obs_dim = obs_arr.shape[1]
    encoder_layers = []
    prev_dim = obs_dim
    for layer_dim in autoencoder_layers:
        encoder_layers.extend([
            nn.Linear(prev_dim, layer_dim),
            nn.BatchNorm1d(layer_dim),  # ✅ Before ReLU
            nn.ReLU(),
            nn.Dropout(0.1)
        ])
        prev_dim = layer_dim
    
    # Final encoder layer (with Tanh activation for bounded latent space)
    encoder_layers.append(nn.Linear(prev_dim, latent_dim))
    encoder_layers.append(nn.Tanh())  # ✅ Bound outputs to [-1, 1]
    encoder = nn.Sequential(*encoder_layers).to(device)
    
    # Decoder
    decoder_layers = []
    prev_dim = latent_dim
    for layer_dim in reversed(autoencoder_layers):
        decoder_layers.extend([
            nn.Linear(prev_dim, layer_dim),
            nn.BatchNorm1d(layer_dim),  # ✅ Before ReLU
            nn.ReLU(),
            nn.Dropout(0.1)
        ])
        prev_dim = layer_dim
    
    # Final decoder layer (no activation for reconstruction)
    decoder_layers.append(nn.Linear(prev_dim, obs_dim))
    decoder = nn.Sequential(*decoder_layers).to(device)
    
    # Training
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    criterion = nn.MSELoss()
    
    num_batches = len(obs_arr) // batch_size
    
    for epoch in range(epochs):
        total_loss = 0
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_obs = obs_tensor[start_idx:end_idx]
            
            # Forward pass
            latent = encoder(batch_obs)
            reconstructed = decoder(latent)
            loss = criterion(reconstructed, batch_obs)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.6f}")
    
    print("Autoencoder pretraining complete!")
    return encoder, decoder


def list_autoencoder_models():
    """List all available autoencoder models."""
    model_dir = os.path.join(os.path.dirname(__file__), "autoencoder_models")
    
    if not os.path.exists(model_dir):
        print("No autoencoder models directory found.")
        return []
    
    models = []
    for filename in os.listdir(model_dir):
        if filename.endswith('.pth'):
            # Parse filename: encoder_network_size_latentN.pth
            parts = filename.replace('.pth', '').split('_')
            if len(parts) >= 3 and parts[0] in ['encoder', 'decoder']:
                network_size = parts[1]
                latent_dim = int(parts[2].replace('latent', ''))
                models.append((parts[0], network_size, latent_dim))
    
    if models:
        print("Available autoencoder models:")
        encoders = [m for m in models if m[0] == 'encoder']
        decoders = [m for m in models if m[0] == 'decoder']
        
        for encoder, decoder in zip(encoders, decoders):
            print(f"  - {encoder[1]} network, latent_dim={encoder[2]}")
    else:
        print("No autoencoder models found.")
    
    return models


def clear_autoencoder_models(network_size='large', latent_dim=64):
    """Clear saved autoencoder models to force retraining."""
    model_dir = os.path.join(os.path.dirname(__file__), "autoencoder_models")
    encoder_path = os.path.join(model_dir, f"encoder_{network_size}_latent{latent_dim}.pth")
    decoder_path = os.path.join(model_dir, f"decoder_{network_size}_latent{latent_dim}.pth")
    
    cleared = False
    if os.path.exists(encoder_path):
        os.remove(encoder_path)
        print(f"Removed encoder model: {encoder_path}")
        cleared = True
    
    if os.path.exists(decoder_path):
        os.remove(decoder_path)
        print(f"Removed decoder model: {decoder_path}")
        cleared = True
    
    if cleared:
        print("Autoencoder models cleared. Will retrain on next run.")
    else:
        print("No autoencoder models found to clear.")
    
    return cleared


def create_autoencoder_env(env, network_size='large', latent_dim=64, device='cpu', expert_path=None, force_retrain=False):
    """Create environment with autoencoder observation compression."""
    # Define model save paths
    model_dir = os.path.join(os.path.dirname(__file__), "autoencoder_models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Use provided expert_path or default to normalized path
    if expert_path is None:
        expert_path = EXPERT_PATH
    
    encoder_path = os.path.join(model_dir, f"encoder_{network_size}_latent{latent_dim}.pth")
    decoder_path = os.path.join(model_dir, f"decoder_{network_size}_latent{latent_dim}.pth")
    
    # Check if pretrained models exist and force_retrain is False
    if os.path.exists(encoder_path) and os.path.exists(decoder_path) and not force_retrain:
        print(f"Loading pretrained autoencoder from {model_dir}...")
        
        # Get observation data for model architecture setup
        data = np.load(expert_path, allow_pickle=True)
        obs_arr = data['obs']
        obs_dim = obs_arr.shape[1]
        
        # Define autoencoder architecture
        autoencoder_layers = {
            'small': [256, 128],
            'medium': [512, 256, 128],
            'large': [512, 256, 128],
            'xlarge': [1024, 512, 256, 128]
        }
        
        # Create encoder architecture
        encoder_layers = []
        prev_dim = obs_dim
        for layer_dim in autoencoder_layers[network_size]:
            encoder_layers.extend([
                nn.Linear(prev_dim, layer_dim),
                nn.BatchNorm1d(layer_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = layer_dim
        
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        encoder_layers.append(nn.Tanh())
        encoder = nn.Sequential(*encoder_layers).to(device)
        
        # Create decoder architecture
        decoder_layers = []
        prev_dim = latent_dim
        for layer_dim in reversed(autoencoder_layers[network_size]):
            decoder_layers.extend([
                nn.Linear(prev_dim, layer_dim),
                nn.BatchNorm1d(layer_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = layer_dim
        
        decoder_layers.append(nn.Linear(prev_dim, obs_dim))
        decoder = nn.Sequential(*decoder_layers).to(device)
        
        # Load pretrained weights
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        decoder.load_state_dict(torch.load(decoder_path, map_location=device))
        
        print(f"Successfully loaded pretrained autoencoder (latent_dim={latent_dim})")
        
    else:
        if force_retrain:
            print("Force retraining autoencoder (cleared saved models)...")
        else:
            print(f"No pretrained autoencoder found. Training new model...")
        
        # Get observation data for pretraining
        data = np.load(expert_path, allow_pickle=True)
        obs_arr = data['obs']
        
        # Pretrain autoencoder
        autoencoder_layers = {
            'small': [256, 128],
            'medium': [512, 256, 128],
            'large': [512, 256, 128],
            'xlarge': [1024, 512, 256, 128]
        }
        
        encoder, decoder = pretrain_autoencoder(
            obs_arr, 
            latent_dim=latent_dim,
            autoencoder_layers=autoencoder_layers[network_size],
            epochs=150,  
            device=device
        )
        
        # Save the trained models
        print(f"Saving autoencoder models to {model_dir}...")
        torch.save(encoder.state_dict(), encoder_path)
        torch.save(decoder.state_dict(), decoder_path)
        print(f"Autoencoder models saved successfully")
    
    # Diagnose autoencoder outputs
    data = np.load(expert_path, allow_pickle=True)
    obs_arr = data['obs']
    diagnose_autoencoder_outputs(encoder, obs_arr, device)
    
    # Wrap environment with autoencoder
    wrapped_env = AutoencoderWrapper(env, encoder, latent_dim)
    
    print(f"Observation space compressed from {obs_arr.shape[1]} to {latent_dim} dimensions")
    
    return wrapped_env, encoder, decoder


class AutoencoderPolicy(nn.Module):
    """Custom policy with autoencoder for observation compression."""
    
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, activation_fn=nn.ReLU, 
                 latent_dim=64, autoencoder_layers=[512, 256, 128]):
        super().__init__()
        
        self.observation_space = observation_space
        self.action_space = action_space
        self.latent_dim = latent_dim
        
        # Autoencoder architecture
        obs_dim = observation_space.shape[0]
        
        # Encoder
        encoder_layers = []
        prev_dim = obs_dim
        for layer_dim in autoencoder_layers:
            encoder_layers.extend([
                nn.Linear(prev_dim, layer_dim),
                nn.BatchNorm1d(layer_dim),  # ✅ Before activation
                activation_fn(),
                nn.Dropout(0.1)
            ])
            prev_dim = layer_dim
        
        # Final encoding layer (with activation for bounded latent space)
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        encoder_layers.append(nn.Tanh())  # ✅ Bound outputs to [-1, 1]
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder (for reconstruction loss)
        decoder_layers = []
        prev_dim = latent_dim
        for layer_dim in reversed(autoencoder_layers):
            decoder_layers.extend([
                nn.Linear(prev_dim, layer_dim),
                nn.BatchNorm1d(layer_dim),  # ✅ Before activation
                activation_fn(),
                nn.Dropout(0.1)
            ])
            prev_dim = layer_dim
        
        # Final decoding layer (no activation for reconstruction)
        decoder_layers.append(nn.Linear(prev_dim, obs_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Policy head (using compressed representation)
        if net_arch is None:
            net_arch = [64, 64]
        
        policy_layers = []
        prev_dim = latent_dim
        for layer_dim in net_arch:
            policy_layers.extend([
                nn.Linear(prev_dim, layer_dim),
                nn.BatchNorm1d(layer_dim),  # ✅ Before activation
                activation_fn(),
                nn.Dropout(0.1)
            ])
            prev_dim = layer_dim
        
        self.policy_head = nn.Sequential(*policy_layers)
        
        # Action distribution parameters
        if isinstance(action_space, gym.spaces.Box):
            self.action_dim = action_space.shape[0]
            self.action_mean = nn.Linear(prev_dim, self.action_dim)
            self.action_log_std = nn.Parameter(torch.zeros(self.action_dim))
        else:
            raise NotImplementedError("Only continuous action spaces supported")
    
    def forward(self, obs, deterministic=False):
        # Encode observations
        latent = self.encoder(obs)
        
        # Policy head
        policy_features = self.policy_head(latent)
        
        # Action distribution
        action_mean = self.action_mean(policy_features)
        action_log_std = self.action_log_std.expand_as(action_mean)
        
        if deterministic:
            action = action_mean
        else:
            action_std = torch.exp(action_log_std)
            action = torch.normal(action_mean, action_std)
        
        return action, action_mean, action_log_std
    
    def encode(self, obs):
        """Get latent representation of observations."""
        return self.encoder(obs)
    
    def decode(self, latent):
        """Reconstruct observations from latent representation."""
        return self.decoder(latent)
    
    def reconstruction_loss(self, obs):
        """Compute autoencoder reconstruction loss."""
        latent = self.encoder(obs)
        reconstructed = self.decoder(latent)
        return F.mse_loss(reconstructed, obs)


class AutoencoderMlpPolicy(nn.Module):
    """Stable Baselines3 compatible policy with autoencoder."""
    
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, 
                 activation_fn=nn.ReLU, latent_dim=64, autoencoder_layers=[512, 256, 128]):
        super().__init__()
        
        self.observation_space = observation_space
        self.action_space = action_space
        self.latent_dim = latent_dim
        
        # Autoencoder architecture
        obs_dim = observation_space.shape[0]
        
        # Encoder
        encoder_layers = []
        prev_dim = obs_dim
        for layer_dim in autoencoder_layers:
            encoder_layers.extend([
                nn.Linear(prev_dim, layer_dim),
                nn.BatchNorm1d(layer_dim),  # ✅ Before activation
                activation_fn(),
                nn.Dropout(0.1)
            ])
            prev_dim = layer_dim
        
        # Final encoding layer (with activation for bounded latent space)
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        encoder_layers.append(nn.Tanh())  # ✅ Bound outputs to [-1, 1]
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder (for reconstruction loss)
        decoder_layers = []
        prev_dim = latent_dim
        for layer_dim in reversed(autoencoder_layers):
            decoder_layers.extend([
                nn.Linear(prev_dim, layer_dim),
                nn.BatchNorm1d(layer_dim),  # ✅ Before activation
                activation_fn(),
                nn.Dropout(0.1)
            ])
            prev_dim = layer_dim
        
        # Final decoding layer (no activation for reconstruction)
        decoder_layers.append(nn.Linear(prev_dim, obs_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Policy head (using compressed representation)
        if net_arch is None:
            net_arch = [64, 64]
        
        policy_layers = []
        prev_dim = latent_dim
        for layer_dim in net_arch:
            policy_layers.extend([
                nn.Linear(prev_dim, layer_dim),
                nn.BatchNorm1d(layer_dim),  # ✅ Before activation
                activation_fn(),
                nn.Dropout(0.1)
            ])
            prev_dim = layer_dim
        
        self.policy_head = nn.Sequential(*policy_layers)
        
        # Action distribution parameters
        if isinstance(action_space, gym.spaces.Box):
            self.action_dim = action_space.shape[0]
            self.action_mean = nn.Linear(prev_dim, self.action_dim)
            self.action_log_std = nn.Parameter(torch.zeros(self.action_dim))
        else:
            raise NotImplementedError("Only continuous action spaces supported")
    
    def forward(self, obs, deterministic=False):
        # Encode observations
        latent = self.encoder(obs)
        
        # Policy head
        policy_features = self.policy_head(latent)
        
        # Action distribution
        action_mean = self.action_mean(policy_features)
        action_log_std = self.action_log_std.expand_as(action_mean)
        
        if deterministic:
            action = action_mean
        else:
            action_std = torch.exp(action_log_std)
            action = torch.normal(action_mean, action_std)
        
        return action, action_mean, action_log_std
    
    def encode(self, obs):
        """Get latent representation of observations."""
        return self.encoder(obs)
    
    def decode(self, latent):
        """Reconstruct observations from latent representation."""
        return self.decoder(latent)
    
    def reconstruction_loss(self, obs):
        """Compute autoencoder reconstruction loss."""
        latent = self.encoder(obs)
        reconstructed = self.decoder(latent)
        return F.mse_loss(reconstructed, obs)


def create_large_ppo_policy(env, network_size='large', device='cpu'):
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
    
    return PPO(
        'MlpPolicy', 
        env, 
        verbose=1,
        policy_kwargs=policy_kwargs,
        device=device
    )


def create_large_a2c_policy(env, network_size='large', device='cpu'):
    """Create A2C with customizable MLP policy network size."""
    if network_size == 'small':
        policy_kwargs = {
            "net_arch": [64, 64],        # Shared network: 2 layers
            "activation_fn": torch.nn.ReLU
        }
    elif network_size == 'medium':
        policy_kwargs = {
            "net_arch": [256, 256],      # Shared network: 2 layers
            "activation_fn": torch.nn.ReLU
        }
    elif network_size == 'large':
        policy_kwargs = {
            "net_arch": [512, 512, 256, 256],  # Shared network: 4 layers
            "activation_fn": torch.nn.ReLU
        }
    elif network_size == 'xlarge':
        policy_kwargs = {
            "net_arch": [1024, 1024, 512, 512, 256],  # Shared network: 5 layers
            "activation_fn": torch.nn.ReLU
        }
    else:
        raise ValueError(f"Unknown network size: {network_size}")
    
    return A2C(
        'MlpPolicy', 
        env, 
        verbose=1,
        policy_kwargs=policy_kwargs,
        device=device
    )


def visualize_observations(obs_arr, num_dims=5, num_steps=500):
    """Plot a few observation dimensions over time to verify data quality."""
    plt.figure(figsize=(12, 6))
    for i in range(min(num_dims, obs_arr.shape[1])):
        plt.plot(obs_arr[:num_steps, i], label=f"Obs dim {i}")
    plt.xlabel("Timestep")
    plt.ylabel("Observation value")
    plt.title("Sampled Observation Dimensions from Expert Trajectories")
    plt.legend()
    plt.tight_layout()
    plt.show()


def test_pretrained_policy(policy_path, env, algo='ppo', num_episodes=1):
    """Test a pre-trained policy in the environment and plot rewards."""
    print(f"Testing policy from {policy_path}...")
    if algo == 'ppo':
        policy = PPO.load(policy_path, env=env)
    elif algo == 'a2c':
        policy = A2C.load(policy_path, env=env)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    episode_rewards = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        rewards = []
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            total_reward += reward
            done = terminated or truncated
        episode_rewards.append(total_reward)
        plt.plot(rewards, label=f"Episode {ep+1}")
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.title(f"Pretrained {algo.upper()} Policy Reward per Episode")
    plt.legend()
    plt.tight_layout()
    plt.show()
    print(f"Average reward over {num_episodes} episode(s): {np.mean(episode_rewards):.2f}")


def check_time_alignment_and_episode_cuts(obs_arr, next_obs_arr, dones_arr):
    """Check that next_obs[t] == obs[t+1] except where done[t] is True."""
    print("=== Time-alignment & Episode Cuts Check ===")
    
    # Check for misalignments
    misalignments = []
    for t in range(len(obs_arr) - 1):
        if not dones_arr[t]:  # If not done, next_obs[t] should equal obs[t+1]
            if not np.allclose(obs_arr[t+1], next_obs_arr[t], rtol=1e-5, atol=1e-8):
                misalignments.append(t)
    
    if misalignments:
        print(f"[ERROR] Found {len(misalignments)} time-alignment issues!")
        print(f"   First few misalignments at timesteps: {misalignments[:5]}")
        return False
    else:
        print("[SUCCESS] Time-alignment is correct")
        return True


def check_scaling_and_units(obs_arr, acts_arr):
    """Check observation and action scaling and identify potential issues."""
    print("\n=== Scaling & Units Check ===")
    
    # Observation statistics
    obs_mean = np.mean(obs_arr, axis=0)
    obs_std = np.std(obs_arr, axis=0)
    obs_min = np.min(obs_arr, axis=0)
    obs_max = np.max(obs_arr, axis=0)
    
    # Action statistics
    acts_mean = np.mean(acts_arr, axis=0)
    acts_std = np.std(acts_arr, axis=0)
    acts_min = np.min(acts_arr, axis=0)
    acts_max = np.max(acts_arr, axis=0)
    
    print(f"Observations shape: {obs_arr.shape}")
    print(f"Actions shape: {acts_arr.shape}")
    
    # Check for extreme values
    obs_extreme = np.sum(np.abs(obs_arr) > 100, axis=0)
    acts_extreme = np.sum(np.abs(acts_arr) > 10, axis=0)
    
    print(f"\nObservation statistics:")
    print(f"  Mean range: [{obs_mean.min():.3f}, {obs_mean.max():.3f}]")
    print(f"  Std range: [{obs_std.min():.3f}, {obs_std.max():.3f}]")
    print(f"  Value range: [{obs_min.min():.3f}, {obs_max.max():.3f}]")
    print(f"  Extreme values (>100): {obs_extreme.sum()} total")
    
    print(f"\nAction statistics:")
    print(f"  Mean range: [{acts_mean.min():.3f}, {acts_mean.max():.3f}]")
    print(f"  Std range: [{acts_std.min():.3f}, {acts_std.max():.3f}]")
    print(f"  Value range: [{acts_min.min():.3f}, {acts_max.max():.3f}]")
    print(f"  Extreme values (>10): {acts_extreme.sum()} total")
    
    # Check for zero variance (constant features)
    obs_zero_var = np.sum(obs_std < 1e-6)
    acts_zero_var = np.sum(acts_std < 1e-6)
    
    if obs_zero_var > 0:
        print(f"[WARNING] {obs_zero_var} observation features have zero variance")
    if acts_zero_var > 0:
        print(f"[WARNING] {acts_zero_var} action features have zero variance")
    
    return True


def check_coverage_and_balance(obs_arr, acts_arr, save_plots=False, save_dir=None):
    """Check data coverage and balance using histograms and basic statistics."""
    print("\n=== Coverage & Balance Check ===")
    
    num_obs_features = obs_arr.shape[1]
    num_act_features = acts_arr.shape[1]
    
    # Plot histograms for a subset of features
    num_to_plot = min(8, num_obs_features)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(num_to_plot):
        axes[i].hist(obs_arr[:, i], bins=50, alpha=0.7)
        axes[i].set_title(f'Obs Feature {i}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    if save_plots and save_dir:
        plt.savefig(os.path.join(save_dir, 'observation_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # Plot action distributions
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(min(8, num_act_features)):
        axes[i].hist(acts_arr[:, i], bins=50, alpha=0.7)
        axes[i].set_title(f'Action Feature {i}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    if save_plots and save_dir:
        plt.savefig(os.path.join(save_dir, 'action_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # Check for data sparsity
    obs_sparse = np.sum(obs_arr == 0, axis=0) / len(obs_arr)
    acts_sparse = np.sum(acts_arr == 0, axis=0) / len(acts_arr)
    
    print(f"Observation sparsity (fraction of zeros):")
    for i, sparsity in enumerate(obs_sparse):
        if sparsity > 0.5:
            print(f"  Feature {i}: {sparsity:.3f} (high sparsity)")
    
    print(f"Action sparsity (fraction of zeros):")
    for i, sparsity in enumerate(acts_sparse):
        if sparsity > 0.5:
            print(f"  Feature {i}: {sparsity:.3f} (high sparsity)")
    
    return True


def check_data_quality(obs_arr, acts_arr, dones_arr):
    """Comprehensive data quality checks."""
    print("\n=== Data Quality Check ===")
    
    # Check for NaN values
    obs_nans = np.isnan(obs_arr).sum()
    acts_nans = np.isnan(acts_arr).sum()
    
    if obs_nans > 0:
        print(f"[ERROR] Found {obs_nans} NaN values in observations")
        return False
    else:
        print("[SUCCESS] No NaN values in observations")
    
    if acts_nans > 0:
        print(f"[ERROR] Found {acts_nans} NaN values in actions")
        return False
    else:
        print("[SUCCESS] No NaN values in actions")
    
    # Check for infinite values
    obs_inf = np.isinf(obs_arr).sum()
    acts_inf = np.isinf(acts_arr).sum()
    
    if obs_inf > 0:
        print(f"[ERROR] Found {obs_inf} infinite values in observations")
        return False
    else:
        print("[SUCCESS] No infinite values in observations")
    
    if acts_inf > 0:
        print(f"[ERROR] Found {acts_inf} infinite values in actions")
        return False
    else:
        print("[SUCCESS] No infinite values in actions")
    
    # Check episode structure
    num_episodes = np.sum(dones_arr)
    print(f"Number of episodes: {num_episodes}")
    print(f"Total timesteps: {len(obs_arr)}")
    print(f"Average episode length: {len(obs_arr) / max(num_episodes, 1):.1f} steps")
    
    return True


def run_sanity_checks(obs_arr, acts_arr, next_obs_arr, dones_arr, save_plots=False, save_dir=None):
    """Run all sanity checks on the expert trajectory data."""
    print("Running sanity checks on expert trajectory data...")
    print("=" * 60)
    
    checks_passed = True
    
    # Run all checks
    checks_passed &= check_data_quality(obs_arr, acts_arr, dones_arr)
    checks_passed &= check_time_alignment_and_episode_cuts(obs_arr, next_obs_arr, dones_arr)
    checks_passed &= check_scaling_and_units(obs_arr, acts_arr)
    checks_passed &= check_coverage_and_balance(obs_arr, acts_arr, save_plots, save_dir)
    
    print("\n" + "=" * 60)
    if checks_passed:
        print("[SUCCESS] All sanity checks passed! Data looks good for training.")
    else:
        print("[ERROR] Some sanity checks failed. Please review the data before training.")
    
    return checks_passed


def get_env(stepSize, start_time, end_time, use_autoencoder=False):
    """Create environment with conditional normalization based on autoencoder usage.
    
    Args:
        stepSize: Simulation step size in seconds
        start_time: Simulation start time
        end_time: Simulation end time
        use_autoencoder: If True, skip observation normalization (autoencoder handles it)
    
    Returns:
        Environment with appropriate wrappers applied
    """
    from use_case.multizone_simple_air_RL_control import load_model_and_params, POLICY_CONFIG_PATH
    from t4b_gym.t4b_gym_env import NormalizedObservationWrapper, NormalizedActionWrapper
    
    model = load_model_and_params()
    
    class T4BGymEnvCustomReward(T4BGymEnv):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.previous_objective = 0.0
            
        def get_reward(self, action, observation):
            zones = ['core', 'north', 'east', 'south', 'west']
            temp_violations = []
            for zone in zones:
                temp = self.simulator.model.components[f"{zone}_indoor_temp_sensor"].output["measuredValue"]
                heating_setpoint = self.simulator.model.components[f"{zone}_temperature_heating_setpoint"].output["scheduleValue"]
                cooling_setpoint = self.simulator.model.components[f"{zone}_temperature_cooling_setpoint"].output["scheduleValue"]
                heating_violation = max(0, heating_setpoint - temp)
                cooling_violation = max(0, temp - cooling_setpoint)
                zone_violation = (1+heating_violation)**2 + (1+cooling_violation)**2
                temp_violations.append(zone_violation)
            temp_violation_penalty = 1000 * sum(temp_violations)
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
            reward = temp_violation_penalty + coils_power_consumption_penalty + ahu_power_consumption_penalty
            reward = reward/1000
            if np.isnan(reward):
                raise ValueError("Reward is not a number")
            return -reward
    
    # Create base environment
    env = T4BGymEnvCustomReward(
        model=model, 
        io_config_file=POLICY_CONFIG_PATH,
        start_time=start_time,
        end_time=end_time,
        episode_length=int(3600*24*5 / stepSize),  # 5 days
        random_start=True, 
        excluding_periods=None, 
        forecast_horizon=50,
        step_size=stepSize,
        warmup_period=0
    )
    
    # Apply wrappers conditionally
    if not use_autoencoder:
        # Apply observation normalization only when NOT using autoencoder
        env = NormalizedObservationWrapper(env)
        print("Applied observation normalization wrapper")
    else:
        print("Skipped observation normalization (autoencoder will handle it)")
    
    # Always apply action normalization (autoencoder doesn't affect actions)
    env = NormalizedActionWrapper(env)
    print("Applied action normalization wrapper")
    
    return env


def main():
    parser = argparse.ArgumentParser(description="Pretrain RL policy with behavioral cloning using expert trajectories.")
    parser.add_argument('--algo', choices=['ppo', 'a2c'], default='ppo', help='RL algorithm to use (default: ppo)')
    parser.add_argument('--test', action='store_true', help='Test the pretrained policy after training')
    parser.add_argument('--skip-checks', action='store_true', help='Skip sanity checks (not recommended)')
    parser.add_argument('--save-plots', action='store_true', default=True, help='Save sanity check plots to plots_sanity_checks folder')
    parser.add_argument('--network-size', choices=['small', 'medium', 'large', 'xlarge'], default='large', 
                       help='Network size for the policy (default: large)')
    parser.add_argument('--use-autoencoder', action='store_true', default=False, help='Use autoencoder for observation compression')
    parser.add_argument('--latent-dim', type=int, default=64, help='Latent dimension for autoencoder (default: 64)')
    parser.add_argument('--use-unnormalized', action='store_true', default=False, help='Use unnormalized expert trajectories (recommended for autoencoders)')
    parser.add_argument('--retrain-autoencoder', action='store_true', help='Force retrain autoencoder and clear saved models')
    args = parser.parse_args()
    algo = args.algo
    model_path = PRETRAINED_MODEL_PATHS[algo]

    # Choose expert trajectory source
    if args.use_autoencoder and args.use_unnormalized:
        # For autoencoders: use unnormalized observations + normalized actions
        expert_path = EXPERT_PATH_UNNORMALIZED_NORM_ACTIONS
        print("Using UNNORMALIZED observations + NORMALIZED actions (optimal for autoencoders)")
    elif args.use_unnormalized:
        expert_path = EXPERT_PATH_UNNORMALIZED
        print("Using UNNORMALIZED expert trajectories")
    else:
        expert_path = EXPERT_PATH
        print("Using NORMALIZED expert trajectories")
        if args.use_autoencoder:
            print("[WARNING] Using normalized trajectories with autoencoder may cause double normalization issues!")
            print("   Consider using --use-unnormalized flag instead.")

    # Create save directory for plots if needed
    save_dir = None
    if args.save_plots:
        save_dir = os.path.join(os.path.dirname(__file__), "plots_sanity_checks")
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving sanity check plots to: {save_dir}")

    # Load expert trajectories
    data = np.load(expert_path, allow_pickle=True)
    obs_arr = data['obs']
    acts_arr = data['acts']
    next_obs_arr = data['next_obs']
    dones_arr = data['dones']
    
    # For BC training without autoencoder, we need to normalize expert trajectories
    # to match what the environment will provide (after NormalizedObservationWrapper)
    if not args.use_autoencoder:
        print("Normalizing expert trajectory observations and actions to match environment normalization...")
        
        # Get the original observation space bounds (before normalization wrapper)
        # We need to create a temporary environment without wrappers to get the original bounds
        from use_case.multizone_simple_air_RL_control import load_model_and_params, POLICY_CONFIG_PATH
        from t4b_gym.t4b_gym_env import T4BGymEnv
        
        # Create base environment without wrappers
        model = load_model_and_params()
        temp_env = T4BGymEnv(
            model=model, 
            io_config_file=POLICY_CONFIG_PATH,
            start_time=start_time,
            end_time=end_time,
            episode_length=int(3600*24*5 / stepSize),
            random_start=True, 
            excluding_periods=None, 
            forecast_horizon=50,
            step_size=stepSize,
            warmup_period=0
        )
        
        # Get original bounds
        obs_low = temp_env.observation_space.low
        obs_high = temp_env.observation_space.high
        act_low = temp_env.action_space.low
        act_high = temp_env.action_space.high
        
        # Apply the same normalization as NormalizedObservationWrapper: 2*(obs - low)/(high - low) - 1
        obs_arr = 2 * (obs_arr - obs_low) / (obs_high - obs_low) - 1
        next_obs_arr = 2 * (next_obs_arr - obs_low) / (obs_high - obs_low) - 1
        
        # Apply the same normalization as NormalizedActionWrapper: 2*(acts - low)/(high - low) - 1
        acts_arr = 2 * (acts_arr - act_low) / (act_high - act_low) - 1
        
        # Clip to [-1, 1] bounds (same as wrappers)
        obs_arr = np.clip(obs_arr, -1, 1)
        next_obs_arr = np.clip(next_obs_arr, -1, 1)
        acts_arr = np.clip(acts_arr, -1, 1)
        
        print(f"Expert trajectory observations normalized to range [{obs_arr.min():.3f}, {obs_arr.max():.3f}]")
        print(f"Expert trajectory actions normalized to range [{acts_arr.min():.3f}, {acts_arr.max():.3f}]")
        
        # Close temporary environment
        temp_env.close()
    
    # Run sanity checks
    if not args.skip_checks:
        checks_passed = run_sanity_checks(obs_arr, acts_arr, next_obs_arr, dones_arr, 
                                        save_plots=args.save_plots, save_dir=save_dir)
        if not checks_passed:
            print("Sanity checks failed. Consider fixing the data or use --skip-checks to proceed anyway.")
            return
    else:
        print("Skipping sanity checks (not recommended)")

    # Visualize observations
    #visualize_observations(obs_arr)

    # Prepare environment (for policy and BC trainer)
    # Use the same environment setup as regular RL training for consistency
    env = get_env(stepSize, start_time, end_time, args.use_autoencoder)
    
    # Store original observation space for later restoration
    original_obs_space = env.observation_space
    
    # Apply autoencoder if requested
    if args.use_autoencoder:
        print(f"Using autoencoder with latent_dim={args.latent_dim}")
        
        # Show available models
        list_autoencoder_models()
        
        # Clear models if retrain flag is set
        if args.retrain_autoencoder:
            clear_autoencoder_models(args.network_size, args.latent_dim)
        
        env, encoder, decoder = create_autoencoder_env(
            env, 
            network_size=args.network_size, 
            latent_dim=args.latent_dim,
            expert_path=expert_path,
            force_retrain=args.retrain_autoencoder
        )
        
        # Compress expert trajectory observations to match autoencoder output
        print("Compressing expert trajectory observations...")
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs_arr)
            next_obs_tensor = torch.FloatTensor(next_obs_arr)
            
            # Compress in batches to avoid memory issues
            batch_size = 128
            compressed_obs = []
            compressed_next_obs = []
            
            for i in range(0, len(obs_arr), batch_size):
                end_idx = min(i + batch_size, len(obs_arr))
                batch_obs = obs_tensor[i:end_idx]
                batch_next_obs = next_obs_tensor[i:end_idx]
                
                compressed_batch_obs = encoder(batch_obs).numpy()
                compressed_batch_next_obs = encoder(batch_next_obs).numpy()
                
                compressed_obs.append(compressed_batch_obs)
                compressed_next_obs.append(compressed_batch_next_obs)
            
            obs_arr = np.vstack(compressed_obs)
            next_obs_arr = np.vstack(compressed_next_obs)
            
            print(f"Expert trajectories compressed from {data['obs'].shape[1]} to {obs_arr.shape[1]} dimensions")
    
    # Create an RL policy (untrained) with specified network size
    print(f"Creating {algo.upper()} policy with {args.network_size} network size...")
    if algo == 'ppo':
        policy = create_large_ppo_policy(env, network_size=args.network_size)
    elif algo == 'a2c':
        policy = create_large_a2c_policy(env, network_size=args.network_size)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    # Prepare transitions for BC
    infos_arr = [{} for _ in range(len(obs_arr))]
    transitions = Transitions(
        obs=obs_arr,
        acts=acts_arr,
        infos=infos_arr,
        next_obs=next_obs_arr,
        dones=dones_arr
    )

    # Pretrain with Behavioral Cloning
    rng = np.random.default_rng(0)
    bc_trainer = BC( 
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        policy=policy.policy,  # Use the RL policy's network
        device='cpu',
        rng=rng
    )
    print(f"Pretraining {algo.upper()} policy with behavioral cloning...")
    bc_trainer.train(n_epochs=150)
    print("Pretraining complete.")

    # Save the pretrained policy weights for later RL training
    policy.save(model_path)
    print(f"Pretrained {algo.upper()} policy saved to {model_path}")

    # Test the pretrained policy
    if args.test:
        test_pretrained_policy(model_path, env, algo=algo, num_episodes=1)
    else:
        # Use the same environment for testing (consistent with training)
        test_model(env, policy)

if __name__ == "__main__":
    main() 