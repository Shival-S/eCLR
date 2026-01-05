# Starter code for evaluating UniLCD - 15K timesteps model
import unilcd_env
import gymnasium as gym
import json
import os
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

config = json.load(open('unilcd_emb_eval_config.json'))
env = gym.make(**config)

# Load the 15K trained model
load_path = "unilcd_ppo_model_15k.zip"
print(f"Loading model from: {load_path}")

model = PPO.load(load_path, env=env)
print("Model loaded successfully!")

# Evaluate the model
print("Starting evaluation with 3 episodes...")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=3)
print(f"Evaluation Results:")
print(f"Mean reward: {mean_reward}")
print(f"Standard deviation: {std_reward}")
