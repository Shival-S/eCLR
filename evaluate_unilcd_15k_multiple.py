# Multiple evaluation runs for UniLCD 15K model to reduce uncertainty
import unilcd_env
import gymnasium as gym
import json
import os
import sys
import gc
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

def log_and_print(message):
    """Print and log message"""
    print(message)
    with open("multiple_evaluation_log.txt", "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} - {message}\n")

def run_single_evaluation(run_number):
    """Run a single evaluation and return results"""
    env = None
    model = None
    
    try:
        log_and_print(f"\n🚗 Starting Evaluation Run #{run_number}...")
        
        # Load configuration
        config = json.load(open('unilcd_emb_eval_config.json'))
        
        # Create environment
        env = gym.make(**config)
        obs, info = env.reset()
        
        # Load the 15K trained model
        load_path = "unilcd_ppo_model_15k.zip"
        model = PPO.load(load_path, env=env)
        
        # Evaluate the model
        log_and_print(f"   🧪 Running 3 episodes for Run #{run_number}...")
        
        mean_reward, std_reward = evaluate_policy(
            model, 
            env, 
            n_eval_episodes=3,
            deterministic=True
        )
        
        log_and_print(f"   ✅ Run #{run_number}: Mean={mean_reward:.2f}, Std={std_reward:.2f}")
        
        return True, mean_reward, std_reward
        
    except Exception as e:
        log_and_print(f"   💥 Run #{run_number} FAILED: {e}")
        return False, None, None
        
    finally:
        # Cleanup
        try:
            if env is not None:
                env.close()
                del env
            if model is not None:
                del model
            gc.collect()
        except:
            pass

def evaluate_multiple_runs(num_runs=5):
    """Run multiple evaluations and compute statistics"""
    
    # Clear previous log
    with open("multiple_evaluation_log.txt", "w") as f:
        f.write(f"=== UniLCD 15K Multiple Evaluation Started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    
    log_and_print(f"🎯 Starting {num_runs} evaluation runs to reduce uncertainty...")
    log_and_print("=" * 60)
    
    successful_runs = []
    mean_rewards = []
    std_rewards = []
    
    for run in range(1, num_runs + 1):
        success, mean_reward, std_reward = run_single_evaluation(run)
        
        if success:
            successful_runs.append(run)
            mean_rewards.append(mean_reward)
            std_rewards.append(std_reward)
        
        # Small delay between runs to let CARLA stabilize
        if run < num_runs:
            log_and_print(f"   ⏱️  Waiting 10 seconds before next run...")
            time.sleep(10)
    
    # Calculate statistics
    if len(mean_rewards) > 0:
        log_and_print("\n" + "=" * 60)
        log_and_print("📊 FINAL STATISTICAL ANALYSIS:")
        log_and_print("=" * 60)
        
        overall_mean = np.mean(mean_rewards)
        overall_std = np.std(mean_rewards, ddof=1) if len(mean_rewards) > 1 else 0
        overall_min = np.min(mean_rewards)
        overall_max = np.max(mean_rewards)
        
        log_and_print(f"✅ Successful runs: {len(successful_runs)}/{num_runs}")
        log_and_print(f"📈 Overall Mean Reward: {overall_mean:.2f}")
        log_and_print(f"📊 Standard Deviation: {overall_std:.2f}")
        log_and_print(f"📉 Min Reward: {overall_min:.2f}")
        log_and_print(f"📈 Max Reward: {overall_max:.2f}")
        log_and_print(f"📏 Range: {overall_max - overall_min:.2f}")
        
        if len(mean_rewards) > 1:
            # 95% confidence interval (approximate)
            margin_error = 1.96 * (overall_std / np.sqrt(len(mean_rewards)))
            ci_lower = overall_mean - margin_error
            ci_upper = overall_mean + margin_error
            log_and_print(f"🎯 95% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}]")
        
        log_and_print("\n📋 Individual Run Results:")
        for i, (run, reward) in enumerate(zip(successful_runs, mean_rewards)):
            log_and_print(f"   Run {run}: {reward:.2f}")
        
        return True, overall_mean, overall_std, mean_rewards
    
    else:
        log_and_print("❌ No successful runs completed!")
        return False, None, None, None

if __name__ == "__main__":
    print("🔬 Running Multiple Evaluation Analysis...")
    print("This will help determine the true performance range of your 15K model")
    
    # Run 5 evaluations by default (you can change this)
    success, mean, std, all_rewards = evaluate_multiple_runs(num_runs=5)
    
    if success:
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"Your 15K model's true performance: {mean:.2f} ± {std:.2f}")
        print("\n📄 Check multiple_evaluation_log.txt for full details")
        sys.exit(0)
    else:
        print(f"\n❌ ANALYSIS FAILED!")
        sys.exit(1)
