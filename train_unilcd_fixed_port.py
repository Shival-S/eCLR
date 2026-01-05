# UniLCD training with dynamic port allocation to avoid conflicts
import unilcd_env
import gymnasium as gym
import json
from stable_baselines3 import PPO
import time
import random

# Load base config and modify traffic manager port to avoid conflicts
config = json.load(open('unilcd_emb_eval_config.json'))

# Use a random port between 6001-6999 to avoid conflicts
tm_port = random.randint(6001, 6999)
config['tm_port'] = tm_port
print(f"Using traffic manager port: {tm_port}")

# Add retry logic for environment initialization with different ports
max_retries = 5
for attempt in range(max_retries):
    try:
        print(f"Attempting to initialize environment (attempt {attempt + 1}/{max_retries}) with TM port {config['tm_port']}...")
        env = gym.make(**config)
        print("Environment created successfully!")
        break
    except Exception as e:
        print(f"Failed to create environment: {e}")
        if "bind error" in str(e) and attempt < max_retries - 1:
            # Try a different port if bind error
            config['tm_port'] = random.randint(6001, 6999)
            print(f"Trying different traffic manager port: {config['tm_port']}")
            time.sleep(2)
        elif attempt < max_retries - 1:
            print("Retrying in 5 seconds...")
            time.sleep(5)
        else:
            print("All attempts failed. Exiting.")
            exit(1)

# Reset environment
try:
    print("Attempting to reset environment...")
    env.reset()
    print("Environment reset successful!")
except Exception as e:
    print(f"Environment reset failed: {e}")
    print("Continuing with training anyway...")

# Start training
try:
    model = PPO("MlpPolicy", env, verbose=1)
    print("Starting UniLCD training with 100,000 timesteps...")
    model.learn(total_timesteps=100000, progress_bar=True)
    print("Training completed successfully!")
    model.save("unilcd_ppo_model_final")
except Exception as e:
    print(f"Training failed with error: {e}")
    print("Attempting to save partial model...")
    try:
        model.save("unilcd_ppo_model_partial")
        print("Partial model saved successfully.")
    except:
        print("Could not save partial model.")
