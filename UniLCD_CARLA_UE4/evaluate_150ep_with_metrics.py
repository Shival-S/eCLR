"""
Checkpoint-based evaluation script for UniLCD with detailed metrics CSV
Matches Kathakoli's evaluation methodology from the paper
- 150 episodes (5 routes × 30 episodes per route)
- 1500 max steps per episode
- Generates detailed CSV with ENS, NS, SR, RC, Infraction Rate, Energy, FPS
- Can resume from last completed episode if interrupted
- Restarts CARLA every 10 episodes to prevent resource leaks
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
CHECKPOINT_FILE = "episode_counter_paper_eval.pkl"
TOTAL_EPISODES = 150
EPISODES_PER_RESTART = 3  # Restart CARLA every 3 episodes to prevent crashes

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
    """Kill native CARLA processes (not Docker)"""
    print("\n[CARLA] Stopping native CARLA server...")
    # Use specific process name to only kill native CARLA, not Docker
    subprocess.run(['pkill', '-f', 'CarlaUE4-Linux-Shipping'], check=False)
    time.sleep(2)
    # Force kill if still running
    subprocess.run(['pkill', '-9', '-f', 'CarlaUE4-Linux-Shipping'], check=False)
    time.sleep(1)
    print("[CARLA] Native CARLA stopped")

def start_carla():
    """Start native CARLA server (not Docker)"""
    print("\n[CARLA] Starting NATIVE CARLA server...")
    print("[CARLA] Using: /home/shival/CarlaUE4.sh (native binary)")

    carla_log = '/home/shival/UniLCD/carla_paper_eval.log'
    print(f"[CARLA] Log file: {carla_log}")

    with open(carla_log, 'a') as log_file:  # Append mode to preserve logs across restarts
        log_file.write(f"\n\n{'='*70}\n")
        log_file.write(f"CARLA Start Time: {datetime.now().isoformat()}\n")
        log_file.write(f"{'='*70}\n\n")
        log_file.flush()

        carla_process = subprocess.Popen([
            '/home/shival/CarlaUE4.sh',
            '-opengl',
            '-world-port=2000',
            '-RenderOffScreen',
            '-quality-level=Epic',
            '-nosound'
        ], stdout=log_file, stderr=log_file)

    print("[CARLA] Warming up for 90 seconds (increased from 60s)...")
    time.sleep(90)  # Increased warmup time
    print("[CARLA] ✓ Native CARLA ready")
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
    print("UNILCD PAPER EVALUATION (WITH DETAILED METRICS)")
    print("="*70)
    print("Matches Kathakoli's evaluation methodology:")
    print("  - 150 episodes")
    print("  - 1500 max steps per episode")
    print("  - Generates CSV with ENS, NS, SR, RC, Infraction, Energy, FPS")
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
    print(f"  CARLA Restart Every: {EPISODES_PER_RESTART} episodes (very frequent restarts for stability)")
    print(f"  CARLA Warmup Time: 90 seconds per restart")
    print(f"  Checkpoint File: {CHECKPOINT_FILE}")
    print(f"  Expected restarts: {TOTAL_EPISODES // EPISODES_PER_RESTART}")

    # Load and configure evaluation settings
    config_path = '/home/shival/UniLCD/unilcd_emb_eval_config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Configure for paper evaluation with detailed metrics
    config['env_mode'] = 'eval'  # CRITICAL: Enable eval mode for detailed CSV generation
    config['steps_per_episode'] = 1500
    config['rollout_video'] = '/home/shival/UniLCD/rollout_video_paper_eval.mp4'
    config['minimap_video'] = '/home/shival/UniLCD/minimap_video_paper_eval.mp4'
    config['task_metrics_path'] = '/home/shival/UniLCD/task_metrics_paper_eval.csv'

    # Use absolute paths for models
    config['cloud_model_checkpoint'] = '/home/shival/UniLCD/cloud_model.pth'
    config['local_model_checkpoint'] = '/home/shival/UniLCD/local_model.pth'
    config['path'] = '/home/shival/UniLCD/unilcd_env/envs/path_points/path_points_t10_32_95.npy'
    config['log_dir'] = '/home/shival/UniLCD/logs_dir/'

    print(f"\nOutput Files:")
    print(f"  CSV Metrics: {config['task_metrics_path']}")
    print(f"  Rollout Video: {config['rollout_video']}")
    print(f"  Minimap Video: {config['minimap_video']}")
    print(f"\nModel Files:")
    print(f"  Cloud Model: {config['cloud_model_checkpoint']}")
    print(f"  Local Model: {config['local_model_checkpoint']}")
    print(f"  PPO Policy: unilcd_ppo_paper_1000ep.zip")

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

                # Create environment with eval mode enabled
                print("[ENV] Creating environment with eval mode...")
                env = gym.make(**config)
                print("[ENV] ✓ Environment created (eval mode active)")

                # Load model (only once at start)
                if model is None:
                    print("[MODEL] Loading PPO routing policy...")
                    model = PPO.load("unilcd_ppo_paper_1000ep.zip", env=env)
                    print("[MODEL] ✓ PPO model loaded")

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
            print("\n[ENV] Closing environment (this will finalize CSV and videos)...")
            env.close()
            print("[ENV] ✓ Environment closed")

        kill_carla()

        # Print final statistics
        print("\n" + "#"*70)
        print("# EVALUATION COMPLETE!")
        print("#"*70)
        print_statistics(results, 0, TOTAL_EPISODES)

        # Save final results summary
        print(f"\n[SAVE] Writing summary to evaluation_paper_results.txt")
        with open("evaluation_paper_results.txt", "w") as f:
            f.write("UniLCD Paper Evaluation Results\n")
            f.write("="*70 + "\n\n")
            f.write("Methodology: Matches Kathakoli's paper evaluation\n")
            f.write(f"Total Episodes: {TOTAL_EPISODES}\n")
            f.write(f"Max Steps per Episode: 1500\n")
            f.write(f"Episodes Completed: {len(results['rewards'])}\n\n")
            f.write("Episode Rewards & Lengths:\n")
            f.write(f"Mean Reward: {np.mean(results['rewards']):.4f}\n")
            f.write(f"Std Reward: {np.std(results['rewards']):.4f}\n")
            f.write(f"Mean Length: {np.mean(results['lengths']):.2f}\n")
            f.write(f"Std Length: {np.std(results['lengths']):.2f}\n\n")
            f.write("Detailed Metrics (ENS, NS, SR, RC, Infraction, Energy, FPS):\n")
            f.write(f"See CSV file: {config['task_metrics_path']}\n\n")
            f.write("Per-Episode Details:\n")
            f.write("-"*70 + "\n")
            for ep in results['episode_details']:
                f.write(f"Episode {ep['episode']:3d}: "
                       f"Reward={ep['reward']:7.4f}, "
                       f"Length={ep['length']:4d}, "
                       f"Time={ep['time']:6.1f}s, "
                       f"Term={ep['termination']}\n")

        print("[SAVE] ✓ Summary saved")
        print("\n[CLEANUP] Removing checkpoint file")
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)

        print("\n" + "="*70)
        print("✓ EVALUATION COMPLETE!")
        print("="*70)
        print(f"\nDetailed metrics CSV: {config['task_metrics_path']}")
        print("Compare these results with Table 2 from the UniLCD paper")
        print("="*70)

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Evaluation interrupted by user")
        print(f"[CHECKPOINT] Progress saved at episode {current_episode}")
        print(f"[CHECKPOINT] Run the script again to resume from episode {current_episode + 1}")
        if env is not None:
            try:
                env.close()
            except:
                pass
        kill_carla()
    except Exception as e:
        print(f"\n\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        print(f"\n[CHECKPOINT] Progress saved at episode {current_episode}")
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
