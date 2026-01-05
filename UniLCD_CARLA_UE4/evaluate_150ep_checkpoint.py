"""
Checkpoint-based evaluation script for UniLCD
Can resume from last completed episode if interrupted
Restarts CARLA every 10 episodes to prevent resource leaks
"""

import unilcd_env
import gymnasium as gym
import json
import time
import subprocess
import pickle
import os
from stable_baselines3 import PPO
import numpy as np
from datetime import datetime

# Configuration
CHECKPOINT_FILE = "episode_counter_eval.pkl"
RESULTS_FILE = "evaluation_results_checkpoint.pkl"
TOTAL_EPISODES = 150
EPISODES_PER_RESTART = 10

def save_checkpoint(episode_num, results):
    """Save current progress to checkpoint file"""
    checkpoint_data = {
        'episode': episode_num,
        'results': results,
        'timestamp': datetime.now().isoformat()
    }
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"[CHECKPOINT] Saved at episode {episode_num}")

def load_checkpoint():
    """Load checkpoint if exists"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'rb') as f:
            checkpoint_data = pickle.load(f)
        print(f"[CHECKPOINT] Loaded from episode {checkpoint_data['episode']}")
        print(f"[CHECKPOINT] Last saved: {checkpoint_data['timestamp']}")
        return checkpoint_data['episode'], checkpoint_data['results']
    return 0, {'rewards': [], 'lengths': [], 'episode_details': []}

def kill_carla():
    """Kill all CARLA processes"""
    print("\n[CARLA] Stopping all CARLA processes...")
    subprocess.run(['pkill', '-9', '-f', 'CarlaUE4'], check=False)
    time.sleep(3)
    print("[CARLA] Stopped")

def start_carla():
    """Start CARLA server"""
    print("\n[CARLA] Starting CARLA server...")
    carla_process = subprocess.Popen([
        '/home/shival/CarlaUE4.sh',
        '-opengl',
        '-world-port=2000',
        '-RenderOffScreen',
        '-quality-level=Epic',
        '-nosound'
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("[CARLA] Warming up for 60 seconds...")
    time.sleep(60)
    print("[CARLA] Ready")
    return carla_process

def evaluate_episode(env, model, episode_num, total_episodes):
    """Evaluate a single episode"""
    print(f"\n{'='*70}")
    print(f"[EPISODE {episode_num}/{total_episodes}] Starting...")
    print(f"{'='*70}")

    obs, info = env.reset()
    done = False
    truncated = False
    episode_reward = 0
    episode_length = 0

    episode_start = time.time()

    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        episode_length += 1

        # Verbose progress every 100 steps
        if episode_length % 100 == 0:
            print(f"  [Step {episode_length}] Reward so far: {episode_reward:.2f}")

    episode_time = time.time() - episode_start

    termination_reason = "Done" if done else "Truncated"
    print(f"\n[EPISODE {episode_num}] COMPLETE")
    print(f"  Termination: {termination_reason}")
    print(f"  Reward: {episode_reward:.4f}")
    print(f"  Length: {episode_length} steps")
    print(f"  Time: {episode_time:.1f}s")
    print(f"{'='*70}\n")

    return {
        'episode': episode_num,
        'reward': episode_reward,
        'length': episode_length,
        'time': episode_time,
        'termination': termination_reason
    }

def print_statistics(results, start_episode, current_episode):
    """Print current statistics"""
    rewards = results['rewards']
    lengths = results['lengths']

    if len(rewards) > 0:
        print(f"\n{'#'*70}")
        print(f"# STATISTICS (Episodes {start_episode+1} to {current_episode})")
        print(f"{'#'*70}")
        print(f"Episodes Completed: {len(rewards)}/{TOTAL_EPISODES}")
        print(f"Mean Reward: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")
        print(f"Mean Length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")
        print(f"Min/Max Reward: {np.min(rewards):.4f} / {np.max(rewards):.4f}")
        print(f"{'#'*70}\n")

def main():
    print("\n" + "="*70)
    print("CHECKPOINT-BASED EVALUATION FOR UniLCD")
    print("="*70)

    # Load checkpoint
    start_episode, results = load_checkpoint()

    if start_episode > 0:
        print(f"\n[RESUME] Continuing from episode {start_episode + 1}")
        print(f"[RESUME] Already completed: {len(results['rewards'])} episodes")
    else:
        print("\n[START] Beginning fresh evaluation")

    print(f"\nConfiguration:")
    print(f"  Total Episodes: {TOTAL_EPISODES}")
    print(f"  CARLA Restart Every: {EPISODES_PER_RESTART} episodes")
    print(f"  Checkpoint File: {CHECKPOINT_FILE}")

    # Load configuration
    config = json.load(open('unilcd_emb_eval_config.json'))
    config['max_episode_steps'] = 1500

    current_episode = start_episode
    carla_process = None
    env = None
    model = None
    episodes_since_restart = 0

    try:
        while current_episode < TOTAL_EPISODES:
            # Restart CARLA if needed
            if episodes_since_restart == 0 or episodes_since_restart >= EPISODES_PER_RESTART:
                print(f"\n{'#'*70}")
                print(f"# CARLA RESTART CHECKPOINT")
                print(f"# Episodes {current_episode + 1} to {min(current_episode + EPISODES_PER_RESTART, TOTAL_EPISODES)}")
                print(f"{'#'*70}\n")

                # Clean up existing environment
                if env is not None:
                    try:
                        env.close()
                    except:
                        pass

                # Restart CARLA
                kill_carla()
                carla_process = start_carla()

                # Create environment
                print("[ENV] Creating environment...")
                env = gym.make(**config)

                # Load model (only once at start)
                if model is None:
                    print("[MODEL] Loading PPO model...")
                    model = PPO.load("unilcd_ppo_paper_1000ep.zip", env=env)
                    print("[MODEL] Loaded successfully")

                episodes_since_restart = 0

            # Evaluate episode
            try:
                episode_data = evaluate_episode(env, model, current_episode + 1, TOTAL_EPISODES)

                # Store results
                results['rewards'].append(episode_data['reward'])
                results['lengths'].append(episode_data['length'])
                results['episode_details'].append(episode_data)

                # Save checkpoint after each episode
                current_episode += 1
                episodes_since_restart += 1
                save_checkpoint(current_episode, results)

                # Print statistics every 10 episodes
                if current_episode % 10 == 0:
                    print_statistics(results, start_episode, current_episode)

            except Exception as e:
                print(f"\n[ERROR] Exception during episode {current_episode + 1}: {e}")
                print(f"[ERROR] Checkpoint saved at episode {current_episode}")
                print(f"[ERROR] You can restart the script to resume from episode {current_episode + 1}")
                raise

        # Final cleanup
        if env is not None:
            env.close()
        kill_carla()

        # Print final statistics
        print("\n" + "#"*70)
        print("# EVALUATION COMPLETE!")
        print("#"*70)
        print_statistics(results, 0, TOTAL_EPISODES)

        # Save final results
        print(f"\n[SAVE] Writing final results to evaluation_results.txt")
        with open("evaluation_results.txt", "w") as f:
            f.write("UniLCD Evaluation Results (Checkpoint-based)\n")
            f.write("="*70 + "\n\n")
            f.write(f"Total Episodes: {TOTAL_EPISODES}\n")
            f.write(f"Episodes Completed: {len(results['rewards'])}\n\n")
            f.write(f"Mean Reward: {np.mean(results['rewards']):.4f}\n")
            f.write(f"Std Reward: {np.std(results['rewards']):.4f}\n")
            f.write(f"Mean Length: {np.mean(results['lengths']):.2f}\n")
            f.write(f"Std Length: {np.std(results['lengths']):.2f}\n\n")
            f.write("Per-Episode Details:\n")
            f.write("-"*70 + "\n")
            for ep in results['episode_details']:
                f.write(f"Episode {ep['episode']:3d}: "
                       f"Reward={ep['reward']:7.4f}, "
                       f"Length={ep['length']:4d}, "
                       f"Time={ep['time']:6.1f}s, "
                       f"Term={ep['termination']}\n")

        print("[SAVE] Results saved successfully")
        print("\n[CLEANUP] Removing checkpoint file")
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)

        print("\n✓ Evaluation complete!")

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Evaluation interrupted by user")
        print(f"[CHECKPOINT] Progress saved at episode {current_episode}")
        print(f"[CHECKPOINT] Run the script again to resume from episode {current_episode + 1}")
        if env is not None:
            env.close()
        kill_carla()
    except Exception as e:
        print(f"\n\n[FATAL ERROR] {e}")
        print(f"[CHECKPOINT] Progress saved at episode {current_episode}")
        print(f"[CHECKPOINT] Run the script again to resume from episode {current_episode + 1}")
        if env is not None:
            try:
                env.close()
            except:
                pass
        kill_carla()
        raise

if __name__ == "__main__":
    main()
