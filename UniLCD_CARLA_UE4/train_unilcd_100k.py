# 100K timestep training with reduced walker load
import unilcd_env
import gymnasium as gym
import json
from stable_baselines3 import PPO

config = json.load(open('unilcd_emb_eval_config_100k.json'))
env = gym.make(**config)
env.reset()
model = PPO("MlpPolicy", env, verbose=1)
print("Starting UniLCD training with 100,000 timesteps...")
print("Using reduced walker configuration to prevent CARLA overwhelm")
model.learn(total_timesteps=100000, progress_bar=True)
print("Training completed successfully!")
model.save("unilcd_ppo_model_100k")
