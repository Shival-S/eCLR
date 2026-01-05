# Robust UniLCD training with error handling for CARLA synchronization issues
import unilcd_env
import gymnasium as gym
import json
from stable_baselines3 import PPO
import time

config = json.load(open('unilcd_emb_eval_config.json'))

# Add retry logic for environment initialization
max_retries = 3
for attempt in range(max_retries):
    try:
        print(f"Attempting to initialize environment (attempt {attempt + 1}/{max_retries})...")
        env = gym.make(**config)
        print("Environment created successfully!")
        break
    except Exception as e:
        print(f"Failed to create environment: {e}")
        if attempt < max_retries - 1:
            print("Retrying in 10 seconds...")
            time.sleep(10)
        else:
            print("All attempts failed. Exiting.")
            exit(1)

# Add retry logic for environment reset
max_reset_retries = 3
for attempt in range(max_reset_retries):
    try:
        print(f"Attempting to reset environment (attempt {attempt + 1}/{max_reset_retries})...")
        env.reset()
        print("Environment reset successful!")
        break
    except Exception as e:
        print(f"Failed to reset environment: {e}")
        if attempt < max_reset_retries - 1:
            print("Retrying in 5 seconds...")
            time.sleep(5)
        else:
            print("Environment reset failed. Continuing anyway...")
            break

# Start training with robust error handling
try:
    model = PPO("MlpPolicy", env, verbose=1)
    print("Starting robust UniLCD training with 100,000 timesteps...")
    model.learn(total_timesteps=100000, progress_bar=True)
    print("Training completed successfully!")
    model.save("unilcd_ppo_model_robust")
except Exception as e:
    print(f"Training failed with error: {e}")
    print("Attempting to save partial model...")
    try:
        model.save("unilcd_ppo_model_partial")
        print("Partial model saved successfully.")
    except:
        print("Could not save partial model.")
