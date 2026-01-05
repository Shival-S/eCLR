"""
UniLCD training for exactly 100 episodes with robust error handling
Fixes all issues from 375k training:
- Episode-based training (not timesteps)
- Proper progress tracking across restarts
- Checkpoint resume with correct sorting
- Episode counting that persists across crashes
"""
import unilcd_env
import gymnasium as gym
import json
import os
import pickle
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

class EpisodeCounterCallback(BaseCallback):
    """Track episode count across training sessions"""
    def __init__(self, max_episodes=100, counter_file="episode_counter.pkl", verbose=1):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.counter_file = counter_file
        self.n_episodes = 0
        self.episode_rewards = []
        
        # Load previous episode count if exists
        if os.path.exists(self.counter_file):
            with open(self.counter_file, 'rb') as f:
                data = pickle.load(f)
                self.n_episodes = data.get('n_episodes', 0)
                self.episode_rewards = data.get('episode_rewards', [])
                if self.verbose > 0:
                    print(f"Resuming from episode {self.n_episodes}/{self.max_episodes}")
    
    def _on_step(self):
        # Check if episode ended
        if self.locals.get('dones')[0]:
            self.n_episodes += 1
            
            # Get episode info if available
            infos = self.locals.get('infos', [{}])
            if 'episode' in infos[0]:
                ep_reward = infos[0]['episode']['r']
                self.episode_rewards.append(ep_reward)
            
            # Save progress
            with open(self.counter_file, 'wb') as f:
                pickle.dump({
                    'n_episodes': self.n_episodes,
                    'episode_rewards': self.episode_rewards
                }, f)
            
            if self.verbose > 0:
                avg_reward = sum(self.episode_rewards[-10:]) / min(10, len(self.episode_rewards)) if self.episode_rewards else 0
                print(f"Episode {self.n_episodes}/{self.max_episodes} completed | "
                      f"Avg reward (last 10): {avg_reward:.2f}")
            
            # Stop if reached target
            if self.n_episodes >= self.max_episodes:
                print(f"\n{'='*60}")
                print(f"TARGET REACHED: {self.max_episodes} episodes completed!")
                print(f"Total episodes: {self.n_episodes}")
                print(f"Mean reward (all): {sum(self.episode_rewards)/len(self.episode_rewards):.2f}")
                print(f"Mean reward (last 20): {sum(self.episode_rewards[-20:])/min(20, len(self.episode_rewards)):.2f}")
                print(f"{'='*60}\n")
                return False  # Stop training
        
        return True
    
    def get_progress(self):
        return self.n_episodes, self.max_episodes

class ProgressTrackerCallback(BaseCallback):
    """Track cumulative timesteps across restarts"""
    def __init__(self, initial_timesteps=0, verbose=1):
        super().__init__(verbose)
        self.cumulative_timesteps = initial_timesteps
        self.last_logged = 0
        
    def _on_step(self):
        self.cumulative_timesteps += 1
        
        # Log every 1000 steps
        if self.cumulative_timesteps - self.last_logged >= 1000:
            if self.verbose > 0:
                print(f"Cumulative timesteps: {self.cumulative_timesteps}")
            self.last_logged = self.cumulative_timesteps
        
        return True

def main():
    # Configuration
    TARGET_EPISODES = 100
    MAX_EPISODE_STEPS = 1500
    CHECKPOINT_FREQ = 5000
    COUNTER_FILE = "episode_counter_100ep.pkl"
    
    print("="*60)
    print("UniLCD Training - 100 Episodes")
    print("="*60)
    
    # Load config
    config = json.load(open('unilcd_emb_eval_config.json'))
    config['max_episode_steps'] = MAX_EPISODE_STEPS
    
    # Create environment
    env = gym.make(**config)
    env.reset()
    
    # Create checkpoint directory
    os.makedirs("./checkpoints_100ep", exist_ok=True)
    
    # Check for existing progress
    episode_counter = EpisodeCounterCallback(
        max_episodes=TARGET_EPISODES,
        counter_file=COUNTER_FILE,
        verbose=1
    )
    
    # Check if we already completed training
    if episode_counter.n_episodes >= TARGET_EPISODES:
        print(f"Training already complete! ({episode_counter.n_episodes} episodes)")
        return
    
    # Try to load latest checkpoint
    checkpoint_files = [f for f in os.listdir("./checkpoints_100ep") 
                       if f.startswith("unilcd_100ep_") and f.endswith("_steps.zip")]
    
    initial_timesteps = 0
    if checkpoint_files:
        # Sort numerically by timestep count
        checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.split("_")[-2]))
        latest_checkpoint = os.path.join("./checkpoints_100ep", checkpoint_files[-1])
        
        print(f"\nResuming from checkpoint: {latest_checkpoint}")
        model = PPO.load(latest_checkpoint, env=env)
        
        # Extract timesteps from filename
        initial_timesteps = int(checkpoint_files[-1].split("_")[-2])
        print(f"Starting from timestep: {initial_timesteps}")
        print(f"Episodes completed: {episode_counter.n_episodes}/{TARGET_EPISODES}")
    else:
        print("\nStarting fresh training...")
        model = PPO("MlpPolicy", env, verbose=1)
        initial_timesteps = 0
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path="./checkpoints_100ep/",
        name_prefix="unilcd_100ep"
    )
    
    progress_tracker = ProgressTrackerCallback(
        initial_timesteps=initial_timesteps,
        verbose=1
    )
    
    callback_list = CallbackList([
        episode_counter,
        checkpoint_callback,
        progress_tracker
    ])
    
    # Calculate maximum timesteps needed (safety margin)
    # Worst case: 100 episodes * 1500 max steps = 150,000
    # Add margin for safety
    max_timesteps_needed = (TARGET_EPISODES - episode_counter.n_episodes) * MAX_EPISODE_STEPS
    
    print(f"\nStarting training...")
    print(f"Target: {TARGET_EPISODES} episodes")
    print(f"Max timesteps budget: {max_timesteps_needed:,}")
    print(f"Checkpoint frequency: every {CHECKPOINT_FREQ} timesteps")
    print("="*60 + "\n")
    
    try:
        model.learn(
            total_timesteps=max_timesteps_needed,
            progress_bar=True,
            callback=callback_list,
            reset_num_timesteps=False
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nTraining interrupted by error: {e}")
    
    # Save final model
    final_model_name = f"unilcd_ppo_100ep_final_{episode_counter.n_episodes}episodes"
    model.save(final_model_name)
    print(f"\nModel saved as: {final_model_name}.zip")
    
    # Print final statistics
    if episode_counter.episode_rewards:
        print(f"\nFinal Statistics:")
        print(f"  Episodes completed: {episode_counter.n_episodes}")
        print(f"  Mean reward: {sum(episode_counter.episode_rewards)/len(episode_counter.episode_rewards):.2f}")
        print(f"  Min reward: {min(episode_counter.episode_rewards):.2f}")
        print(f"  Max reward: {max(episode_counter.episode_rewards):.2f}")
    
    env.close()
    print("\nTraining session ended.")

if __name__ == "__main__":
    main()
