# Simple UniLCD training - back to basics
import unilcd_env
import gymnasium as gym
import json
from stable_baselines3 import PPO

config = json.load(open('unilcd_emb_eval_config.json'))
config['max_episode_steps'] = 1500
env = gym.make(**config)
env.reset()

# Full training: 250 episodes × 1500 steps = 375,000 timesteps (~6-7 hours)
model = PPO("MlpPolicy", env, verbose=1)
print("Starting UniLCD training with 375,000 timesteps (250 episodes × 1500 steps)...")
print("Estimated time: ~6-7 hours")
model.learn(total_timesteps=375000, progress_bar=True)
print("Training completed successfully!")
model.save("unilcd_ppo_model_250x1500")
