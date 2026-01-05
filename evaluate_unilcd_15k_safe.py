# Safe evaluation script for UniLCD - addresses sensor cleanup issues
import unilcd_env
import gymnasium as gym
import json
import os
import sys
import gc
import signal
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

def signal_handler(signum, frame):
    """Handle interruption signals gracefully"""
    print("\n🛑 Received interrupt signal - cleaning up...")
    sys.exit(0)

# Set up signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def evaluate_unilcd_safe():
    env = None
    model = None
    
    try:
        print("🚗 Starting Safe UniLCD 15K Model Evaluation...")
        
        # Load configuration
        config = json.load(open('unilcd_emb_eval_config.json'))
        print("✅ Configuration loaded")
        
        # Create environment
        env = gym.make(**config)
        print("✅ Environment created")
        
        # Reset environment to initialize properly
        obs, info = env.reset()
        print("✅ Environment reset completed")
        
        # Load the 15K trained model
        load_path = "unilcd_ppo_model_15k.zip"
        print(f"📂 Loading model from: {load_path}")
        
        model = PPO.load(load_path, env=env)
        print("✅ Model loaded successfully!")
        
        # Evaluate the model with shorter episodes to reduce sensor load
        print("🧪 Starting evaluation with 3 episodes...")
        print("   - Using safer evaluation parameters...")
        
        mean_reward, std_reward = evaluate_policy(
            model, 
            env, 
            n_eval_episodes=3,
            deterministic=True,  # More consistent behavior
            return_episode_rewards=False
        )
        
        print("\n" + "="*50)
        print("🎯 EVALUATION RESULTS:")
        print(f"   Mean reward: {mean_reward:.2f}")
        print(f"   Standard deviation: {std_reward:.2f}")
        print("="*50)
        
        return True, mean_reward, std_reward
        
    except Exception as e:
        print(f"💥 ERROR during evaluation: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        return False, None, None
        
    finally:
        # Comprehensive cleanup to prevent sensor errors
        print("\n🧹 Performing comprehensive cleanup...")
        
        try:
            # Force garbage collection before cleanup
            gc.collect()
            
            if env is not None:
                print("   - Closing environment...")
                
                # Try to access the underlying environment for explicit sensor cleanup
                if hasattr(env, 'env'):
                    underlying_env = env.env
                    while hasattr(underlying_env, 'env'):
                        underlying_env = underlying_env.env
                    
                    # If we can access the sensors, try to clean them up
                    if hasattr(underlying_env, 'sync_mode'):
                        print("   - Attempting sensor cleanup...")
                        try:
                            underlying_env.sync_mode.__exit__(None, None, None)
                        except:
                            pass
                
                env.close()
                print("   ✅ Environment closed")
                del env
                
        except Exception as e:
            print(f"   ⚠️  Environment cleanup warning: {e}")
        
        try:
            if model is not None:
                print("   - Clearing model...")
                del model
                print("   ✅ Model cleared")
        except Exception as e:
            print(f"   ⚠️  Model cleanup warning: {e}")
        
        # Final garbage collection
        gc.collect()
        print("🏁 Safe cleanup completed!")

if __name__ == "__main__":
    print("🛡️  Running SAFE evaluation with enhanced cleanup...")
    success, mean_reward, std_reward = evaluate_unilcd_safe()
    
    if success:
        print(f"\n🎉 SUCCESS: Safe model evaluation completed!")
        print(f"   Final Results: Mean={mean_reward:.2f}, Std={std_reward:.2f}")
        # Exit cleanly
        os._exit(0)
    else:
        print(f"\n❌ FAILED: Safe model evaluation encountered errors!")
        # Exit cleanly even on failure
        os._exit(1)
