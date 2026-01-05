# High-performance UniLCD training with enhanced CARLA configuration
import unilcd_env
import gymnasium as gym
import json
from stable_baselines3 import PPO

# Use high-performance configuration for powerful hardware
config = json.load(open('unilcd_emb_eval_config_optimized.json'))
print("Loading high-performance UniLCD environment...")
print("Configuration: 64GB RAM, 8 CPUs, High quality, 200 vehicles, 30 walkers")
env = gym.make(**config)
env.reset()

# Full training run with enhanced resources
model = PPO("MlpPolicy", env, verbose=1)
print("Starting high-performance UniLCD training with 100,000 timesteps...")
model.learn(total_timesteps=100000, progress_bar=False)
print("Training completed successfully!")
model.save("unilcd_ppo_model_high_performance")
