#!/usr/bin/env python3
"""
Train UniLCD routing policy with SAC (Soft Actor-Critic)
Upgraded from PPO to SAC for better sample efficiency and continuous action support.
- 1,000 episodes
- 1,500 max steps per episode
- Continuous action space [0, 1] (threshold at 0.5 for local/cloud decision)
- Discount factor γ = 0.99
"""

import unilcd_env
import gymnasium as gym
import json
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import time

class EpisodeCounterCallback(BaseCallback):
    """
    Callback to count episodes and stop after reaching target
    """
    def __init__(self, max_episodes=1000, verbose=1):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

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

            # Reset counters
            self.current_episode_reward = 0
            self.current_episode_length = 0

            # Stop if we've reached max episodes
            if self.episode_count >= self.max_episodes:
                print(f"\nReached {self.max_episodes} episodes. Stopping training.")
                return False

        return True

def train_unilcd_sac():
    """Train UniLCD routing policy using SAC"""

    print("=" * 70)
    print("UniLCD Routing Policy Training - SAC (Soft Actor-Critic)")
    print("=" * 70)
    print("\nTraining Parameters:")
    print("  - Episodes: 1,000")
    print("  - Max steps per episode: 1,500")
    print("  - Action space: Continuous [0, 1]")
    print("  - Discount factor gamma: 0.99")
    print("  - Algorithm: SAC (MlpPolicy)")
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
    print()

    # Create environment
    print("Creating gymnasium environment...")
    env = gym.make(**config)
    env.reset()
    print("Environment created successfully")

    # SAC policy kwargs - SAC uses different architecture than PPO
    # net_arch specifies shared layers for actor and critic
    policy_kwargs = dict(
        net_arch=[256, 256]  # Two hidden layers with 256 units each
    )

    print("\nInitializing SAC with architecture...")
    print("  - Actor network: MLP [obs_dim -> 256 -> 256 -> action_dim]")
    print("  - Critic network: MLP [obs_dim + action_dim -> 256 -> 256 -> 1]")

    # Create SAC model
    model = SAC(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        gamma=0.99,  # Discount factor
        learning_rate=3e-4,
        buffer_size=100000,  # Replay buffer size
        batch_size=256,
        tau=0.005,  # Soft update coefficient
        ent_coef='auto',  # Automatic entropy tuning
        verbose=1,
        tensorboard_log="./logs_dir/tensorboard/"
    )

    print("SAC model initialized")

    # Create callback to count episodes
    episode_callback = EpisodeCounterCallback(max_episodes=1000, verbose=1)

    # Start training
    print("\n" + "=" * 70)
    print("Starting training for 1,000 episodes...")
    print("=" * 70)
    start_time = time.time()

    try:
        # Train for max possible timesteps, but callback will stop after 1000 episodes
        # Max: 1000 episodes * 1500 steps = 1,500,000 timesteps
        model.learn(
            total_timesteps=1_500_000,
            callback=episode_callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")

    training_time = time.time() - start_time

    # Save model
    model_path = "unilcd_sac_1000ep"
    print(f"\nSaving model to: {model_path}.zip")
    model.save(model_path)

    # Print summary
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Episodes completed: {episode_callback.episode_count}")
    print(f"Total training time: {training_time/3600:.2f} hours")
    if episode_callback.episode_rewards:
        print(f"Average reward: {np.mean(episode_callback.episode_rewards):.2f}")
        print(f"Average episode length: {np.mean(episode_callback.episode_lengths):.1f} steps")
    print(f"Model saved to: {model_path}.zip")
    print("=" * 70)

    env.close()

if __name__ == "__main__":
    train_unilcd_sac()
