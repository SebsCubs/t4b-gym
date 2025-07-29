"""
Complete workflow for PPO training with autoencoder and behavioral cloning.

This script demonstrates the full pipeline:
1. Record expert trajectories with normalized actions
2. Pretrain behavioral cloning model with autoencoder
3. Fine-tune with PPO using the pretrained model

Usage:
    python run_ppo_with_autoencoder.py
"""

import os
import sys
import subprocess
import argparse

def run_command(command, description):
    """Run a command and print its output."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("[SUCCESS] Command completed successfully")
        if result.stdout:
            print("Output:")
            print(result.stdout)
    else:
        print("[ERROR] Command failed")
        if result.stderr:
            print("Error:")
            print(result.stderr)
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Complete PPO training workflow with autoencoder")
    parser.add_argument('--skip-recording', action='store_true', 
                       help='Skip recording expert trajectories (use existing ones)')
    parser.add_argument('--skip-pretraining', action='store_true', 
                       help='Skip behavioral cloning pretraining (use existing model)')
    parser.add_argument('--latent-dim', type=int, default=64, 
                       help='Autoencoder latent dimension (default: 64)')
    parser.add_argument('--network-size', choices=['small', 'medium', 'large', 'xlarge'], 
                       default='large', help='Autoencoder network size (default: large)')
    parser.add_argument('--test-only', action='store_true', 
                       help='Only test the final model (skip training)')
    args = parser.parse_args()
    
    print("[START] Starting complete PPO training workflow with autoencoder")
    print(f"Configuration:")
    print(f"  Latent dimension: {args.latent_dim}")
    print(f"  Network size: {args.network_size}")
    print(f"  Test only: {args.test_only}")
    
    # Step 1: Record expert trajectories with normalized actions
    if not args.skip_recording:
        print("\n[STEP1] Recording expert trajectories with normalized actions")
        success = run_command(
            "python record_expert_trajectories_unnormalized.py --normalize-actions",
            "Recording expert trajectories with normalized actions"
        )
        if not success:
            print("[ERROR] Failed to record expert trajectories")
            return
    else:
        print("[SKIP] Skipping expert trajectory recording")
    
    # Step 2: Pretrain behavioral cloning model with autoencoder
    if not args.skip_pretraining:
        print("\n[STEP2] Pretraining behavioral cloning model with autoencoder")
        success = run_command(
            f"python pretrain_with_expert.py --use-autoencoder --latent-dim {args.latent_dim} "
            f"--network-size {args.network_size} --use-unnormalized --algo ppo",
            "Pretraining behavioral cloning model with autoencoder"
        )
        if not success:
            print("[ERROR] Failed to pretrain behavioral cloning model")
            return
    else:
        print("[SKIP] Skipping behavioral cloning pretraining")
    
    # Step 3: Fine-tune with PPO using pretrained model
    if not args.test_only:
        print("\n[STEP3] Fine-tuning PPO with pretrained model")
        
        # Import and run PPO training with autoencoder and pretrained model
        from multizone_simple_air_RL_control import PPO_training
        
        success = PPO_training(
            test_model_flag=False,
            reload_model_flag=False,
            use_autoencoder=True,
            latent_dim=args.latent_dim,
            network_size=args.network_size,
            load_pretrained_bc=True
        )
        
        if success is False:
            print("[ERROR] Failed to fine-tune PPO model")
            return
    else:
        print("[SKIP] Skipping PPO fine-tuning")
    
    # Step 4: Test the final model
    print("\n[STEP4] Testing the final model")
    
    from multizone_simple_air_RL_control import PPO_training
    
    PPO_training(
        test_model_flag=True,
        reload_model_flag=False,
        use_autoencoder=True,
        latent_dim=args.latent_dim,
        network_size=args.network_size
    )
    
    print("\n[SUCCESS] Complete workflow finished successfully!")
    print("\n[FILES] Generated files:")
    print("  - expert_trajectories_unnormalized_norm_actions.npz (expert data)")
    print("  - ppo_pretrained_bc.zip (pretrained behavioral cloning model)")
    print("  - logs/ppo_model.zip (final fine-tuned PPO model)")

if __name__ == "__main__":
    main() 