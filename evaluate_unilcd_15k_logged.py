# Logged evaluation script for UniLCD - captures all output
import unilcd_env
import gymnasium as gym
import json
import os
import sys
import gc
import signal
import time
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

def log_and_print(message):
    """Print and log message"""
    print(message)
    with open("evaluation_log.txt", "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} - {message}\n")

def evaluate_unilcd_logged():
    env = None
    model = None
    
    # Clear previous log
    with open("evaluation_log.txt", "w") as f:
        f.write(f"=== UniLCD 15K Evaluation Started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    
    try:
        log_and_print("🚗 Starting Logged UniLCD 15K Model Evaluation...")
        
        # Load configuration
        config = json.load(open('unilcd_emb_eval_config.json'))
        log_and_print("✅ Configuration loaded")
        
        # Create environment
        log_and_print("🌍 Creating environment...")
        env = gym.make(**config)
        log_and_print("✅ Environment created")
        
        # Reset environment to initialize properly
        log_and_print("🔄 Resetting environment...")
        obs, info = env.reset()
        log_and_print("✅ Environment reset completed")
        
        # Load the 15K trained model
        load_path = "unilcd_ppo_model_15k.zip"
        log_and_print(f"📂 Loading model from: {load_path}")
        
        model = PPO.load(load_path, env=env)
        log_and_print("✅ Model loaded successfully!")
        
        # Evaluate the model
        log_and_print("🧪 Starting evaluation with 3 episodes...")
        
        mean_reward, std_reward = evaluate_policy(
            model, 
            env, 
            n_eval_episodes=3,
            deterministic=True
        )
        
        log_and_print("\n" + "="*50)
        log_and_print("🎯 EVALUATION RESULTS:")
        log_and_print(f"   Mean reward: {mean_reward:.2f}")
        log_and_print(f"   Standard deviation: {std_reward:.2f}")
        log_and_print("="*50)
        
        return True, mean_reward, std_reward
        
    except Exception as e:
        log_and_print(f"💥 ERROR during evaluation: {e}")
        import traceback
        error_details = traceback.format_exc()
        log_and_print(f"Full traceback: {error_details}")
        return False, None, None
        
    finally:
        log_and_print("\n🧹 Starting cleanup...")
        
        try:
            if env is not None:
                log_and_print("   - Closing environment...")
                env.close()
                log_and_print("   ✅ Environment closed")
                del env
                
        except Exception as e:
            log_and_print(f"   ⚠️  Environment cleanup warning: {e}")
        
        try:
            if model is not None:
                log_and_print("   - Clearing model...")
                del model
                log_and_print("   ✅ Model cleared")
        except Exception as e:
            log_and_print(f"   ⚠️  Model cleanup warning: {e}")
        
        gc.collect()
        log_and_print("🏁 Cleanup completed!")

if __name__ == "__main__":
    log_and_print("🛡️  Running LOGGED evaluation...")
    success, mean_reward, std_reward = evaluate_unilcd_logged()
    
    if success:
        log_and_print(f"\n🎉 SUCCESS: Logged evaluation completed!")
        log_and_print(f"   Final Results: Mean={mean_reward:.2f}, Std={std_reward:.2f}")
        print("\n📄 Check evaluation_log.txt for full details")
        sys.exit(0)
    else:
        log_and_print(f"\n❌ FAILED: Logged evaluation encountered errors!")
        print("\n📄 Check evaluation_log.txt for error details")
        sys.exit(1)
