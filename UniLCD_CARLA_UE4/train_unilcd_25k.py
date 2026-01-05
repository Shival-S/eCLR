# 25K timestep training with full walker configuration (30 walkers)
import unilcd_env
import gymnasium as gym
import json
from stable_baselines3 import PPO

config = json.load(open('unilcd_emb_eval_config.json'))
env = gym.make(**config)
env.reset()
model = PPO("MlpPolicy", env, verbose=1)
print("Starting UniLCD training with 25,000 timesteps...")
print("Using full configuration: 30 walkers, 200 vehicles")
model.learn(total_timesteps=25000, progress_bar=True)
print("Training completed successfully!")
model.save("unilcd_ppo_model_25k")
