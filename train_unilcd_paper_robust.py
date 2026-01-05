#!/usr/bin/env python3
"""
Train UniLCD routing policy with PPO - Paper Parameters (Robust Version)
Following exact specifications from arXiv:2409.11403
- 1,000 episodes
- 1,500 max steps per episode
- Policy network: MLP with hidden size 16
- Value network: MLP with hidden size 256
- Discount factor γ = 0.99

Robustness features:
- Checkpoint saving every 10 episodes
- Automatic resume from last checkpoint
"""

import unilcd_env
import gymnasium as gym
import json
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import time
import os
import glob
import pickle

class RobustEpisodeCallback(BaseCallback):
    """
    Callback with checkpoint saving for crash recovery
    """
    def __init__(self, max_episodes=1000, checkpoint_freq=10,
                 checkpoint_dir="./checkpoints_ppo", verbose=1):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_dir = checkpoint_dir
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1

        # Check if episode ended
        if self.locals['dones'][0]:
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)

            if self.verbose >= 1:
                print(f"Episode {self.episode_count}/{self.max_episodes} | "
                      f"Reward: {self.current_episode_reward:.2f} | "
                      f"Length: {self.current_episode_length} steps")

            # Save checkpoint every N episodes
            if self.episode_count % self.checkpoint_freq == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_ep{self.episode_count}")
                self.model.save(checkpoint_path)

                # Save callback state (rewards, lengths, episode count)
                state_path = os.path.join(self.checkpoint_dir, f"callback_state_ep{self.episode_count}.pkl")
                state = {
                    'episode_count': self.episode_count,
                    'episode_rewards': self.episode_rewards,
                    'episode_lengths': self.episode_lengths
                }
                with open(state_path, 'wb') as f:
                    pickle.dump(state, f)

                print(f"  ✓ Checkpoint saved: {checkpoint_path}.zip")

            # Reset counters
            self.current_episode_reward = 0
            self.current_episode_length = 0

            # Stop if we've reached max episodes
            if self.episode_count >= self.max_episodes:
                print(f"\n✅ Reached {self.max_episodes} episodes. Stopping training.")
                return False

        return True

    def load_state(self, state_path):
        """Load callback state from checkpoint"""
        with open(state_path, 'rb') as f:
            state = pickle.load(f)
        self.episode_count = state['episode_count']
        self.episode_rewards = state['episode_rewards']
        self.episode_lengths = state['episode_lengths']
        print(f"  ✓ Loaded callback state: {self.episode_count} episodes completed")

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint file"""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_ep*.zip"))
    if not checkpoints:
        return None, 0

    # Extract episode numbers and find max
    episode_nums = []
    for cp in checkpoints:
        try:
            ep_num = int(cp.split('checkpoint_ep')[1].split('.zip')[0])
            episode_nums.append((ep_num, cp))
        except:
            continue

    if not episode_nums:
        return None, 0

    episode_nums.sort(reverse=True)
    latest_ep, latest_path = episode_nums[0]

    # Find corresponding callback state
    state_path = os.path.join(checkpoint_dir, f"callback_state_ep{latest_ep}.pkl")

    return latest_path, state_path if os.path.exists(state_path) else None

def train_unilcd_paper():
    """Train UniLCD routing policy using paper specifications with robustness features"""

    checkpoint_dir = "./checkpoints_ppo"

    print("=" * 70)
    print("UniLCD Routing Policy Training - Paper Reproduction (Robust)")
    print("=" * 70)
    print("\nTraining Parameters (from arXiv:2409.11403):")
    print("  - Episodes: 1,000")
    print("  - Max steps per episode: 1,500")
    print("  - Policy network hidden size: 16")
    print("  - Value network hidden size: 256")
    print("  - Discount factor γ: 0.99")
    print("  - Algorithm: PPO (MlpPolicy)")
    print("\nRobustness Features:")
    print("  - Checkpoint saving: Every 10 episodes")
    print("  - Auto-resume: From latest checkpoint")
    print("=" * 70)

    # Load config
    config = json.load(open('unilcd_emb_eval_config.json'))

    # Verify config parameters
    assert config['steps_per_episode'] == 1500, "Config must have steps_per_episode=1500"
    assert config['cloud_model_checkpoint'] == './cloud_model.pth', "Cloud model path incorrect"
    assert config['local_model_checkpoint'] == './local_model.pth', "Local model path incorrect"

    print(f"\nConfig verified:")
    print(f"  - Steps per episode: {config['steps_per_episode']}")
    print(f"  - Cloud model: {config['cloud_model_checkpoint']}")
    print(f"  - Local model: {config['local_model_checkpoint']}")
    print(f"  - CARLA port: {config['port']}")

    # Check for existing checkpoints
    print(f"\nChecking for existing checkpoints in {checkpoint_dir}/...")
    latest_checkpoint, state_path = find_latest_checkpoint(checkpoint_dir)

    if latest_checkpoint:
        print(f"  ✓ Found checkpoint: {latest_checkpoint}")
        resume = True
    else:
        print(f"  • No checkpoint found - starting fresh")
        resume = False
    print()

    # Create environment
    print("Creating gymnasium environment...")
    env = gym.make(**config)
    env.reset()
    print("✅ Environment created successfully")

    # Define policy kwargs to match paper specifications
    policy_kwargs = dict(
        net_arch=dict(pi=[16], vf=[256])  # pi=policy, vf=value function
    )

    if resume:
        print(f"\nResuming from checkpoint...")
        print(f"  Loading model: {latest_checkpoint}")
        model = PPO.load(latest_checkpoint, env=env, tensorboard_log="./logs_dir/tensorboard/")
        print("  ✓ Model loaded successfully")

        # Create callback and load state
        episode_callback = RobustEpisodeCallback(
            max_episodes=1000,
            checkpoint_freq=10,
            checkpoint_dir=checkpoint_dir,
            verbose=1
        )

        if state_path:
            episode_callback.load_state(state_path)
            start_episode = episode_callback.episode_count
        else:
            print("  ⚠ No callback state found - starting episode count from checkpoint name")
            start_episode = int(latest_checkpoint.split('checkpoint_ep')[1].split('.zip')[0])
            episode_callback.episode_count = start_episode

        remaining_episodes = 1000 - start_episode
        remaining_timesteps = remaining_episodes * 1500

        print(f"\nResuming from episode {start_episode}")
        print(f"Remaining: {remaining_episodes} episodes (~{remaining_timesteps:,} timesteps)")

    else:
        print("\nInitializing PPO with paper-specified architecture...")
        print("  - Policy network: MLP [obs_dim → 16 → action_dim]")
        print("  - Value network: MLP [obs_dim → 256 → 1]")

        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            gamma=0.99,  # Discount factor from paper
            verbose=1,
            tensorboard_log="./logs_dir/tensorboard/"
        )
        print("✅ PPO model initialized")

        episode_callback = RobustEpisodeCallback(
            max_episodes=1000,
            checkpoint_freq=10,
            checkpoint_dir=checkpoint_dir,
            verbose=1
        )

        remaining_timesteps = 1_500_000

    # Start training
    print("\n" + "=" * 70)
    print(f"Starting training for {1000 - episode_callback.episode_count} episodes...")
    print("=" * 70)
    start_time = time.time()

    try:
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=episode_callback,
            progress_bar=True,
            reset_num_timesteps=False  # Don't reset timestep counter when resuming
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\nTraining crashed with error: {e}")
        print(f"Progress saved at episode {episode_callback.episode_count}")
        raise

    training_time = time.time() - start_time

    # Save final model
    model_path = "unilcd_ppo_paper_1000ep"
    print(f"\n✅ Saving final model to: {model_path}.zip")
    model.save(model_path)

    # Print summary
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Episodes completed: {episode_callback.episode_count}")
    print(f"Total training time (this session): {training_time/3600:.2f} hours")
    if episode_callback.episode_rewards:
        print(f"Average reward: {np.mean(episode_callback.episode_rewards):.2f}")
        print(f"Average episode length: {np.mean(episode_callback.episode_lengths):.1f} steps")
    print(f"Final model saved to: {model_path}.zip")
    print(f"Checkpoints saved in: {checkpoint_dir}/")
    print("=" * 70)

    env.close()

if __name__ == "__main__":
    train_unilcd_paper()
