# Starter code for training UniLCD
import unilcd_env
import gymnasium as gym
import json
from stable_baselines3 import PPO

config = json.load(open('unilcd_emb_eval_config.json'))
env = gym.make(**config)
env.reset()
# Select your favorite RL algorithm and train. We recommend Stable Baselines3 for its integration with Gymnasium
model = PPO("MlpPolicy", env, verbose=1)
print("Starting UniLCD training with 100,000 timesteps...")
model.learn(total_timesteps=100000, progress_bar=False)
print("Training completed successfully!")
model.save("unilcd_ppo_model")
