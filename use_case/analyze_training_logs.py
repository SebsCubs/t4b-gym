#!/usr/bin/env python3
"""
Script to analyze and plot training logs from T4B Gym PPO training.

This script reads the monitor.csv and progress.csv files generated during training
and creates comprehensive plots showing:
1. Mean reward over time
2. Training metrics (losses, KL divergence, etc.)
3. Episode statistics
4. Performance trends

Usage:
    python analyze_training_logs.py [log_directory]
    
Example:
    python analyze_training_logs.py use_case/logs
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from typing import Optional, Tuple

# Set style for better-looking plots
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

class TrainingLogAnalyzer:
    """Analyzer for training logs from T4B Gym PPO training."""
    
    def __init__(self, log_dir: str):
        """
        Initialize the analyzer with a log directory.
        
        Args:
            log_dir: Path to the training log directory
        """
        self.log_dir = Path(log_dir)
        self.monitor_file = self.log_dir / "monitor.csv"
        self.progress_file = self.log_dir / "progress.csv"
        
        # Check if files exist
        if not self.monitor_file.exists():
            raise FileNotFoundError(f"Monitor file not found: {self.monitor_file}")
        if not self.progress_file.exists():
            raise FileNotFoundError(f"Progress file not found: {self.progress_file}")
        
        # Load data
        self.monitor_data = self._load_monitor_data()
        self.progress_data = self._load_progress_data()
        
        print(f"Loaded training data from {log_dir}")
        print(f"Monitor data: {len(self.monitor_data)} episodes")
        print(f"Progress data: {len(self.progress_data)} training steps")
    
    def _load_monitor_data(self) -> pd.DataFrame:
        """Load and parse monitor.csv data."""
        # Read the file, skipping the JSON header
        with open(self.monitor_file, 'r') as f:
            lines = f.readlines()
        
        # Find the start of actual data (after JSON header)
        data_start = 0
        for i, line in enumerate(lines):
            if line.startswith('r,l,t'):
                data_start = i + 1
                break
        
        # Read the data
        data = pd.read_csv(self.monitor_file, skiprows=data_start, names=['reward', 'length', 'time'])
        
        # Add episode number
        data['episode'] = range(1, len(data) + 1)
        
        # Convert time to hours from start
        data['time_hours'] = (data['time'] - data['time'].iloc[0]) / 3600
        
        return data
    
    def _load_progress_data(self) -> pd.DataFrame:
        """Load and parse progress.csv data."""
        data = pd.read_csv(self.progress_file)
        
        # Convert timesteps to millions for better readability
        if 'time/total_timesteps' in data.columns:
            data['timesteps_millions'] = data['time/total_timesteps'] / 1e6
        else:
            # Fallback if column name is different
            data['timesteps_millions'] = range(len(data))
        
        return data
    
    def plot_mean_reward_trend(self, window_size: int = 50, save_path: Optional[str] = None):
        """
        Plot mean reward over training episodes with rolling average.
        
        Args:
            window_size: Size of rolling window for smoothing
            save_path: Optional path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Raw rewards and rolling mean
        ax1.plot(self.monitor_data['episode'], self.monitor_data['reward'], 
                alpha=0.3, color='lightblue', label='Individual Episodes')
        
        rolling_mean = self.monitor_data['reward'].rolling(window=window_size, center=True).mean()
        ax1.plot(self.monitor_data['episode'], rolling_mean, 
                color='red', linewidth=2, label=f'Rolling Mean ({window_size} episodes)')
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Reward Over Episodes')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Reward over time
        ax2.plot(self.monitor_data['time_hours'], self.monitor_data['reward'], 
                alpha=0.3, color='lightblue', label='Individual Episodes')
        
        time_rolling_mean = self.monitor_data['reward'].rolling(window=window_size, center=True).mean()
        ax2.plot(self.monitor_data['time_hours'], time_rolling_mean, 
                color='red', linewidth=2, label=f'Rolling Mean ({window_size} episodes)')
        
        ax2.set_xlabel('Training Time (hours)')
        ax2.set_ylabel('Reward')
        ax2.set_title('Training Reward Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved mean reward plot to {save_path}")
        
        # Only show if not saving (interactive mode)
        if not save_path:
            plt.show()
        else:
            plt.close()  # Close figure to free memory
    
    def plot_training_metrics(self, save_path: Optional[str] = None):
        """
        Plot key training metrics from progress.csv.
        
        Args:
            save_path: Optional path to save the plot
        """
        # Determine how many plots we can make based on available columns
        available_metrics = []
        metric_configs = [
            ('rollout/ep_rew_mean', 'Mean Episode Reward', 'blue'),
            ('rollout/ep_len_mean', 'Mean Episode Length', 'green'),
            ('train/policy_gradient_loss', 'Policy Gradient Loss', 'red'),
            ('train/value_loss', 'Value Loss', 'orange'),
            ('train/approx_kl', 'Approximate KL Divergence', 'purple'),
            ('train/clip_fraction', 'Policy Update Clip Fraction', 'brown'),
            ('train/entropy_loss', 'Entropy Loss', 'cyan'),
            ('train/explained_variance', 'Explained Variance', 'magenta')
        ]
        
        for col, title, color in metric_configs:
            if col in self.progress_data.columns:
                available_metrics.append((col, title, color))
        
        if not available_metrics:
            print("No training metrics found in progress.csv")
            return
        
        # Create subplots
        n_metrics = len(available_metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, (col, title, color) in enumerate(available_metrics):
            axes[i].plot(self.progress_data['timesteps_millions'], 
                        self.progress_data[col], 
                        color=color, linewidth=2)
            axes[i].set_xlabel('Training Timesteps (millions)')
            axes[i].set_ylabel(title)
            axes[i].set_title(title)
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved training metrics plot to {save_path}")
        
        # Only show if not saving (interactive mode)
        if not save_path:
            plt.show()
        else:
            plt.close()  # Close figure to free memory
    
    def plot_reward_distribution(self, save_path: Optional[str] = None):
        """
        Plot reward distribution and statistics.
        
        Args:
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Histogram of rewards
        axes[0, 0].hist(self.monitor_data['reward'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(self.monitor_data['reward'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {self.monitor_data["reward"].mean():.2f}')
        axes[0, 0].set_xlabel('Reward')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Reward Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Box plot of rewards
        axes[0, 1].boxplot(self.monitor_data['reward'])
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].set_title('Reward Box Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Reward statistics over time (split into quarters)
        quarter_size = len(self.monitor_data) // 4
        quarters = []
        quarter_means = []
        
        for i in range(4):
            start_idx = i * quarter_size
            end_idx = (i + 1) * quarter_size if i < 3 else len(self.monitor_data)
            quarter_data = self.monitor_data['reward'].iloc[start_idx:end_idx]
            quarters.append(f'Q{i+1}')
            quarter_means.append(quarter_data.mean())
        
        axes[1, 0].bar(quarters, quarter_means, color=['lightcoral', 'lightblue', 'lightgreen', 'lightyellow'])
        axes[1, 0].set_ylabel('Mean Reward')
        axes[1, 0].set_title('Mean Reward by Training Quarter')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Cumulative reward
        cumulative_reward = self.monitor_data['reward'].cumsum()
        axes[1, 1].plot(self.monitor_data['episode'], cumulative_reward, color='darkblue', linewidth=2)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Cumulative Reward')
        axes[1, 1].set_title('Cumulative Reward')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved reward distribution plot to {save_path}")
        
        # Only show if not saving (interactive mode)
        if not save_path:
            plt.show()
        else:
            plt.close()  # Close figure to free memory
    
    def plot_training_speed(self, save_path: Optional[str] = None):
        """
        Plot training speed and efficiency metrics.
        
        Args:
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: FPS over time
        if 'time/fps' in self.progress_data.columns:
            axes[0, 0].plot(self.progress_data['timesteps_millions'], 
                           self.progress_data['time/fps'], 
                           color='blue', linewidth=2)
            axes[0, 0].set_xlabel('Training Timesteps (millions)')
            axes[0, 0].set_ylabel('Frames Per Second')
            axes[0, 0].set_title('Training Speed (FPS)')
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'FPS data not available', 
                           ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Training Speed (FPS)')
        
        # Plot 2: Time elapsed
        if 'time/time_elapsed' in self.progress_data.columns:
            time_hours = self.progress_data['time/time_elapsed'] / 3600
            axes[0, 1].plot(self.progress_data['timesteps_millions'], 
                           time_hours, 
                           color='green', linewidth=2)
            axes[0, 1].set_xlabel('Training Timesteps (millions)')
            axes[0, 1].set_ylabel('Time Elapsed (hours)')
            axes[0, 1].set_title('Training Time')
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'Time elapsed data not available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Training Time')
        
        # Plot 3: Episodes per hour
        if len(self.monitor_data) > 1:
            total_time_hours = self.monitor_data['time_hours'].iloc[-1]
            episodes_per_hour = len(self.monitor_data) / total_time_hours
            
            # Calculate episodes per hour over time windows
            window_size = max(1, len(self.monitor_data) // 20)
            episode_rates = []
            time_points = []
            
            for i in range(0, len(self.monitor_data) - window_size, window_size):
                window_data = self.monitor_data.iloc[i:i+window_size]
                time_span = window_data['time_hours'].iloc[-1] - window_data['time_hours'].iloc[0]
                if time_span > 0:
                    rate = len(window_data) / time_span
                    episode_rates.append(rate)
                    time_points.append(window_data['time_hours'].iloc[-1])
            
            axes[1, 0].plot(time_points, episode_rates, color='red', linewidth=2)
            axes[1, 0].set_xlabel('Training Time (hours)')
            axes[1, 0].set_ylabel('Episodes per Hour')
            axes[1, 0].set_title('Training Efficiency')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Insufficient data for efficiency plot', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Training Efficiency')
        
        # Plot 4: Learning rate schedule
        if 'train/learning_rate' in self.progress_data.columns:
            axes[1, 1].plot(self.progress_data['timesteps_millions'], 
                           self.progress_data['train/learning_rate'], 
                           color='purple', linewidth=2)
            axes[1, 1].set_xlabel('Training Timesteps (millions)')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Learning rate data not available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Learning Rate Schedule')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved training speed plot to {save_path}")
        
        # Only show if not saving (interactive mode)
        if not save_path:
            plt.show()
        else:
            plt.close()  # Close figure to free memory
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report of the training."""
        report = []
        report.append("=" * 60)
        report.append("T4B GYM PPO TRAINING ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Basic statistics
        report.append("BASIC STATISTICS:")
        report.append(f"  Total episodes: {len(self.monitor_data)}")
        report.append(f"  Total training time: {self.monitor_data['time_hours'].iloc[-1]:.2f} hours")
        if 'time/total_timesteps' in self.progress_data.columns:
            report.append(f"  Total timesteps: {self.progress_data['time/total_timesteps'].iloc[-1]:,}")
        report.append("")
        
        # Reward statistics
        report.append("REWARD STATISTICS:")
        report.append(f"  Mean reward: {self.monitor_data['reward'].mean():.2f}")
        report.append(f"  Std reward: {self.monitor_data['reward'].std():.2f}")
        report.append(f"  Min reward: {self.monitor_data['reward'].min():.2f}")
        report.append(f"  Max reward: {self.monitor_data['reward'].max():.2f}")
        report.append("")
        
        # Training progress
        if len(self.monitor_data) >= 50:
            initial_reward = self.monitor_data['reward'].iloc[:50].mean()
            final_reward = self.monitor_data['reward'].iloc[-50:].mean()
            improvement = final_reward - initial_reward
            
            report.append("TRAINING PROGRESS:")
            report.append(f"  Initial mean reward (first 50 episodes): {initial_reward:.2f}")
            report.append(f"  Final mean reward (last 50 episodes): {final_reward:.2f}")
            report.append(f"  Improvement: {improvement:.2f}")
            if abs(initial_reward) > 0:
                report.append(f"  Improvement percentage: {(improvement/abs(initial_reward)*100):.1f}%")
            report.append("")
        
        # Episode statistics
        report.append("EPISODE STATISTICS:")
        report.append(f"  Mean episode length: {self.monitor_data['length'].mean():.1f} timesteps")
        report.append(f"  Std episode length: {self.monitor_data['length'].std():.1f} timesteps")
        report.append("")
        
        # Training efficiency
        if self.monitor_data['time_hours'].iloc[-1] > 0:
            episodes_per_hour = len(self.monitor_data) / self.monitor_data['time_hours'].iloc[-1]
            report.append("TRAINING EFFICIENCY:")
            report.append(f"  Episodes per hour: {episodes_per_hour:.1f}")
            if 'time/total_timesteps' in self.progress_data.columns:
                timesteps_per_hour = self.progress_data['time/total_timesteps'].iloc[-1] / self.monitor_data['time_hours'].iloc[-1]
                report.append(f"  Timesteps per hour: {timesteps_per_hour:,.0f}")
            report.append("")
        
        # Final training metrics
        if len(self.progress_data) > 0:
            final_metrics = self.progress_data.iloc[-1]
            report.append("FINAL TRAINING METRICS:")
            if 'rollout/ep_rew_mean' in final_metrics:
                report.append(f"  Final mean episode reward: {final_metrics['rollout/ep_rew_mean']:.2f}")
            if 'train/value_loss' in final_metrics:
                report.append(f"  Final value loss: {final_metrics['train/value_loss']:.2f}")
            if 'train/approx_kl' in final_metrics:
                report.append(f"  Final KL divergence: {final_metrics['train/approx_kl']:.4f}")
            if 'train/clip_fraction' in final_metrics:
                report.append(f"  Final clip fraction: {final_metrics['train/clip_fraction']:.3f}")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def save_all_plots(self, output_dir: str):
        """
        Generate and save all plots to the specified directory.
        
        Args:
            output_dir: Directory to save the plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate all plots
        self.plot_mean_reward_trend(save_path=output_path / "mean_reward_trend.png")
        self.plot_training_metrics(save_path=output_path / "training_metrics.png")
        self.plot_reward_distribution(save_path=output_path / "reward_distribution.png")
        self.plot_training_speed(save_path=output_path / "training_speed.png")
        
        # Save summary report
        report = self.generate_summary_report()
        with open(output_path / "training_summary.txt", 'w') as f:
            f.write(report)
        
        print(f"All plots and summary saved to {output_path}")
        print(report)

def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description='Analyze T4B Gym training logs')
    parser.add_argument('log_dir', help='Path to training log directory')
    parser.add_argument('--output', '-o', help='Output directory for plots', default=None)
    parser.add_argument('--no-display', action='store_true', help='Don\'t display plots (only save)')
    
    args = parser.parse_args()
    
    # Set matplotlib to non-interactive if --no-display is used
    if args.no_display:
        plt.ioff()
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
    
    try:
        # Create analyzer
        analyzer = TrainingLogAnalyzer(args.log_dir)
        
        # Generate plots
        if args.output:
            analyzer.save_all_plots(args.output)
        else:
            # Display plots interactively (only if not --no-display)
            if not args.no_display:
                analyzer.plot_mean_reward_trend()
                analyzer.plot_training_metrics()
                analyzer.plot_reward_distribution()
                analyzer.plot_training_speed()
            else:
                # Save plots to default location when --no-display is used
                default_output = Path(args.log_dir) / "analysis_plots"
                analyzer.save_all_plots(str(default_output))
            
            # Print summary
            print(analyzer.generate_summary_report())
    
    except Exception as e:
        print(f"Error analyzing logs: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 