# Evaluation code for UniLCD
import unilcd_env
import gymnasium as gym
import json
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Load config and set max episode steps to 1500
config = json.load(open('unilcd_emb_eval_config.json'))
config['max_episode_steps'] = 1500

# Create environment
env = gym.make(**config)

# Load the trained model
model_path = "unilcd_ppo_model_250x1500.zip"
model = PPO.load(model_path, env=env)

print(f"Evaluating model: {model_path}")
print(f"Running 3 evaluation episodes with max 1500 steps each...")

# Evaluate
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=3, deterministic=True)

print(f"\nEvaluation Results:")
print(f"Mean Reward: {mean_reward:.3f}")
print(f"Std Reward: {std_reward:.3f}")

env.close()
