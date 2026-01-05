# UniLCD training with proper sensor cleanup
import unilcd_env
import gymnasium as gym
import json
from stable_baselines3 import PPO
import atexit

config = json.load(open('unilcd_emb_eval_config.json'))
env = gym.make(**config)
env.reset()

# Register cleanup function to run on exit
def cleanup_environment():
    try:
        print("Cleaning up environment...")
        env.close()
        print("Environment cleaned up successfully.")
    except:
        pass

atexit.register(cleanup_environment)

# Training
model = PPO("MlpPolicy", env, verbose=1)
print("Starting UniLCD training with 25,000 timesteps...")
model.learn(total_timesteps=25000, progress_bar=True)
print("Training completed successfully!")
model.save("unilcd_ppo_model_25k")

# Explicit cleanup
cleanup_environment()
