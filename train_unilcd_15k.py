# UniLCD training with 15k timesteps for video generation
import unilcd_env
import gymnasium as gym
import json
from stable_baselines3 import PPO

config = json.load(open('unilcd_emb_eval_config.json'))
config['max_episode_steps'] = 1500

env = gym.make(**config)
env.reset()

print("Starting UniLCD training with 15,000 timesteps...")
print("Estimated time: ~25-30 minutes")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=15000, progress_bar=True)

print("Training completed!")
model.save("unilcd_ppo_model_15k_new")
print("Model saved as unilcd_ppo_model_15k_new.zip")

env.close()
