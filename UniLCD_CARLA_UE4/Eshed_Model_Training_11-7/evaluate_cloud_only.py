#!/usr/bin/env python3
"""
Cloud-Only Model Evaluation Script
Evaluates ONLY the cloud model without the routing policy.
Matches the methodology from UniLCD paper Table 2.
"""

import unilcd_env
import gymnasium as gym
import json
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, '/home/shival/UniLCD')

def evaluate_cloud_only():
    """
    Evaluate cloud model across 150 episodes (5 routes × 30 episodes).
    This matches the paper's baseline evaluation methodology.
    """

    # Load the evaluation config
    config_path = '/home/shival/UniLCD/unilcd_emb_eval_config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Update config for cloud-only evaluation
    config['env_mode'] = 'eval'  # Enable video recording
    config['rollout_video'] = '/home/shival/UniLCD/Eshed_Model_Training_11-7/rollout_video_cloud_only.mp4'
    config['minimap_video'] = '/home/shival/UniLCD/Eshed_Model_Training_11-7/minimap_video_cloud_only.mp4'
    config['task_metrics_path'] = '/home/shival/UniLCD/Eshed_Model_Training_11-7/task_metrics_cloud_only.csv'
    config['steps_per_episode'] = 1500

    # Use existing pre-trained cloud model
    config['cloud_model_checkpoint'] = '/home/shival/UniLCD/cloud_model.pth'
    config['local_model_checkpoint'] = '/home/shival/UniLCD/local_model.pth'  # Still needed for env initialization

    # Fix path to be absolute
    config['path'] = '/home/shival/UniLCD/unilcd_env/envs/path_points/path_points_t10_32_95.npy'
    config['log_dir'] = '/home/shival/UniLCD/Eshed_Model_Training_11-7/logs_dir/'

    print("=" * 70)
    print("CLOUD-ONLY MODEL EVALUATION")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Episodes: 150 (matching paper methodology)")
    print(f"  Max steps per episode: {config['steps_per_episode']}")
    print(f"  Cloud model: {config['cloud_model_checkpoint']}")
    print(f"  Videos will be saved to: {os.path.dirname(config['rollout_video'])}/")
    print(f"  Metrics will be saved to: {config['task_metrics_path']}")
    print("=" * 70)
    print()

    # Create environment
    print("Creating environment...")
    env = gym.make(**config)
    print("✅ Environment created")
    print()

    # Evaluation parameters
    n_episodes = 150
    episode_rewards = []
    episode_lengths = []

    print(f"Starting evaluation of {n_episodes} episodes...")
    print("(Cloud model will be used for ALL decisions)")
    print()

    try:
        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            episode_length = 0

            while not (done or truncated):
                # ALWAYS use cloud model (action=1)
                # This bypasses the routing policy entirely
                action = 1  # Cloud mode

                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # Print progress every 10 episodes
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_length = np.mean(episode_lengths[-10:])
                print(f"Episode {episode + 1}/{n_episodes} | "
                      f"Avg Reward (last 10): {avg_reward:.2f} | "
                      f"Avg Length (last 10): {avg_length:.0f} steps")

        # Final statistics
        print()
        print("=" * 70)
        print("EVALUATION COMPLETE")
        print("=" * 70)
        print(f"Episodes completed: {n_episodes}")
        print(f"Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Mean episode length: {np.mean(episode_lengths):.0f} ± {np.std(episode_lengths):.0f} steps")
        print()
        print("Detailed metrics (NS, SR, RC, Infraction Rate, Energy, FPS) saved to:")
        print(f"  {config['task_metrics_path']}")
        print()
        print("Videos saved to:")
        print(f"  {config['rollout_video']}")
        print(f"  {config['minimap_video']}")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
        print(f"Completed {len(episode_rewards)} episodes")
    except Exception as e:
        print(f"\n\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nClosing environment...")
        try:
            env.close()
            print("✅ Environment closed")
        except:
            print("⚠️  Warning: Error closing environment (may be already closed)")

if __name__ == "__main__":
    evaluate_cloud_only()
