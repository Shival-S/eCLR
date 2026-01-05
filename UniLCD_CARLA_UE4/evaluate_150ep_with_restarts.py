"""
Evaluation script that restarts CARLA every 10 episodes to prevent resource leaks
This addresses the sensor stream resource accumulation issue that causes CARLA to become unresponsive
"""

import unilcd_env
import gymnasium as gym
import json
import time
import subprocess
import signal
from stable_baselines3 import PPO
import numpy as np

def kill_carla():
    """Kill all CARLA processes"""
    print("Stopping CARLA...")
    subprocess.run(['pkill', '-9', '-f', 'CarlaUE4'], check=False)
    time.sleep(3)
    print("CARLA stopped")

def start_carla():
    """Start CARLA server"""
    print("Starting CARLA...")
    carla_process = subprocess.Popen([
        '/home/shival/CarlaUE4.sh',
        '-opengl',
        '-world-port=2000',
        '-RenderOffScreen',
        '-quality-level=Epic',
        '-nosound'
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Wait for CARLA to warm up
    print("Waiting 60 seconds for CARLA to warm up...")
    time.sleep(60)
    print("CARLA ready")
    return carla_process

def evaluate_batch(model, config, n_episodes, batch_num):
    """Evaluate model for n_episodes"""
    print(f"\n{'='*60}")
    print(f"BATCH {batch_num}: Evaluating {n_episodes} episodes")
    print(f"{'='*60}\n")

    env = gym.make(**config)

    episode_rewards = []
    episode_lengths = []

    try:
        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            episode_length = 0

            while not (done or truncated):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            print(f"  Episode {episode+1}/{n_episodes} - Reward: {episode_reward:.2f}, Length: {episode_length}")

        env.close()

    except Exception as e:
        print(f"Error during batch evaluation: {e}")
        try:
            env.close()
        except:
            pass
        raise

    return episode_rewards, episode_lengths

def main():
    # Configuration
    config = json.load(open('unilcd_emb_eval_config.json'))
    config['max_episode_steps'] = 1500

    EPISODES_PER_BATCH = 10
    TOTAL_EPISODES = 150
    NUM_BATCHES = TOTAL_EPISODES // EPISODES_PER_BATCH

    all_rewards = []
    all_lengths = []

    print("\n" + "="*60)
    print(f"EVALUATION PLAN: {TOTAL_EPISODES} episodes in {NUM_BATCHES} batches")
    print(f"CARLA will restart every {EPISODES_PER_BATCH} episodes")
    print("="*60 + "\n")

    # Load model once (outside the loop)
    print("Loading model...")
    # We'll load the model after starting CARLA the first time

    for batch in range(NUM_BATCHES):
        print(f"\n{'#'*60}")
        print(f"# STARTING BATCH {batch+1}/{NUM_BATCHES}")
        print(f"# Episodes {batch*EPISODES_PER_BATCH + 1} to {(batch+1)*EPISODES_PER_BATCH}")
        print(f"{'#'*60}\n")

        # Kill and restart CARLA
        kill_carla()
        carla_process = start_carla()

        try:
            # Load model for this batch (needs CARLA to be running for env creation)
            if batch == 0:
                print("Loading PPO model...")
                # Create a temporary env just to load the model
                temp_env = gym.make(**config)
                model = PPO.load("unilcd_ppo_paper_1000ep.zip", env=temp_env)
                temp_env.close()
                print("Model loaded successfully")

            # Evaluate this batch
            batch_rewards, batch_lengths = evaluate_batch(
                model, config, EPISODES_PER_BATCH, batch+1
            )

            all_rewards.extend(batch_rewards)
            all_lengths.extend(batch_lengths)

            # Print batch summary
            print(f"\n{'='*60}")
            print(f"BATCH {batch+1} COMPLETE")
            print(f"  Batch Mean Reward: {np.mean(batch_rewards):.2f} ± {np.std(batch_rewards):.2f}")
            print(f"  Batch Mean Length: {np.mean(batch_lengths):.1f} ± {np.std(batch_lengths):.1f}")
            print(f"  Cumulative Episodes: {len(all_rewards)}/{TOTAL_EPISODES}")
            print(f"{'='*60}\n")

        except Exception as e:
            print(f"ERROR in batch {batch+1}: {e}")
            print("Attempting to continue with next batch...")
            continue

        finally:
            # Give a short break between batches
            if batch < NUM_BATCHES - 1:
                print("Pausing 5 seconds before next batch...")
                time.sleep(5)

    # Final cleanup
    kill_carla()

    # Print final results
    print("\n" + "#"*60)
    print("# EVALUATION COMPLETE")
    print("#"*60)
    print(f"\nTotal Episodes: {len(all_rewards)}/{TOTAL_EPISODES}")
    print(f"Mean Reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"Mean Length: {np.mean(all_lengths):.1f} ± {np.std(all_lengths):.1f}")
    print(f"\nResults saved to: eval_results.txt")

    # Save results
    with open("eval_results.txt", "w") as f:
        f.write(f"UniLCD Evaluation Results\n")
        f.write(f"="*60 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Total Episodes: {TOTAL_EPISODES}\n")
        f.write(f"  Episodes per batch: {EPISODES_PER_BATCH}\n")
        f.write(f"  Number of batches: {NUM_BATCHES}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Episodes Completed: {len(all_rewards)}\n")
        f.write(f"  Mean Reward: {np.mean(all_rewards):.4f}\n")
        f.write(f"  Std Reward: {np.std(all_rewards):.4f}\n")
        f.write(f"  Mean Episode Length: {np.mean(all_lengths):.2f}\n")
        f.write(f"  Std Episode Length: {np.std(all_lengths):.2f}\n\n")
        f.write(f"Per-Episode Results:\n")
        for i, (reward, length) in enumerate(zip(all_rewards, all_lengths)):
            f.write(f"  Episode {i+1}: Reward={reward:.4f}, Length={length}\n")

if __name__ == "__main__":
    main()
