# Single evaluation run that we can repeat manually
import unilcd_env
import gymnasium as gym
import json
import os
import sys
import gc
import time
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

def run_single_eval():
    env = None
    model = None
    
    try:
        print("🚗 Starting Single UniLCD 15K Evaluation...")
        
        config = json.load(open('unilcd_emb_eval_config.json'))
        env = gym.make(**config)
        obs, info = env.reset()
        
        load_path = "unilcd_ppo_model_15k.zip"
        model = PPO.load(load_path, env=env)
        
        print("🧪 Running 3 episodes...")
        mean_reward, std_reward = evaluate_policy(
            model, 
            env, 
            n_eval_episodes=3,
            deterministic=True
        )
        
        print("=" * 50)
        print(f"🎯 RESULT: Mean={mean_reward:.2f}, Std={std_reward:.2f}")
        print("=" * 50)
        
        # Log result to file for collection
        with open("eval_results.txt", "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} - Mean: {mean_reward:.2f}, Std: {std_reward:.2f}\n")
        
        return True, mean_reward, std_reward
        
    except Exception as e:
        print(f"💥 ERROR: {e}")
        return False, None, None
        
    finally:
        try:
            if env is not None:
                env.close()
                del env
            if model is not None:
                del model
            gc.collect()
        except:
            pass
        print("🧹 Cleanup completed")

if __name__ == "__main__":
    success, mean, std = run_single_eval()
    if success:
        print(f"✅ SUCCESS: {mean:.2f} ± {std:.2f}")
        sys.exit(0)
    else:
        print("❌ FAILED")
        sys.exit(1)
