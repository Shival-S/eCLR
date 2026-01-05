# Robust evaluation script for UniLCD - 15K timesteps model with proper cleanup
import unilcd_env
import gymnasium as gym
import json
import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

def evaluate_unilcd_robust():
    env = None
    model = None
    
    try:
        print("🚗 Starting UniLCD 15K Model Evaluation...")
        
        # Load configuration
        config = json.load(open('unilcd_emb_eval_config.json'))
        print("✅ Configuration loaded")
        
        # Create environment
        env = gym.make(**config)
        print("✅ Environment created")
        
        # Load the 15K trained model
        load_path = "unilcd_ppo_model_15k.zip"
        print(f"📂 Loading model from: {load_path}")
        
        model = PPO.load(load_path, env=env)
        print("✅ Model loaded successfully!")
        
        # Evaluate the model
        print("🧪 Starting evaluation with 3 episodes...")
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=3)
        
        print("\n" + "="*50)
        print("🎯 EVALUATION RESULTS:")
        print(f"   Mean reward: {mean_reward:.2f}")
        print(f"   Standard deviation: {std_reward:.2f}")
        print("="*50)
        
        return True, mean_reward, std_reward
        
    except Exception as e:
        print(f"💥 ERROR during evaluation: {e}")
        return False, None, None
        
    finally:
        # Explicit cleanup to prevent sensor errors
        print("\n🧹 Cleaning up resources...")
        
        try:
            if env is not None:
                print("   - Closing environment...")
                env.close()
                print("   ✅ Environment closed")
        except Exception as e:
            print(f"   ⚠️  Environment cleanup warning: {e}")
        
        try:
            if model is not None:
                print("   - Clearing model...")
                del model
                print("   ✅ Model cleared")
        except Exception as e:
            print(f"   ⚠️  Model cleanup warning: {e}")
        
        print("🏁 Cleanup completed - evaluation finished safely!")

if __name__ == "__main__":
    success, mean_reward, std_reward = evaluate_unilcd_robust()
    
    if success:
        print(f"\n🎉 SUCCESS: Model evaluation completed!")
        print(f"   Final Results: Mean={mean_reward:.2f}, Std={std_reward:.2f}")
        sys.exit(0)
    else:
        print(f"\n❌ FAILED: Model evaluation encountered errors!")
        sys.exit(1)
