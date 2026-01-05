# Evaluation code for UniLCD - Simple 150 episode version
# Based on evaluate_unilcd.py from the repo
import unilcd_env
import gymnasium as gym
import json
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Load config and set max episode steps to 1500
config = json.load(open('unilcd_emb_eval_config.json'))
config['max_episode_steps'] = 1500

# Create environment
print("Creating environment...")
env = gym.make(**config)
print("✅ Environment created")

# Load the trained model
model_path = "unilcd_ppo_paper_1000ep.zip"
print(f"\nLoading model: {model_path}")
model = PPO.load(model_path, env=env)
print("✅ Model loaded")

print(f"\nEvaluating model: {model_path}")
print(f"Running 150 evaluation episodes with max 1500 steps each...")
print("(Using single route, matching training stability pattern)")
print()

# Evaluate with 150 episodes
try:
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=150,
        deterministic=True
    )

    print(f"\n" + "=" * 70)
    print(f"Evaluation Results (150 episodes)")
    print(f"=" * 70)
    print(f"Mean Reward: {mean_reward:.2f}")
    print(f"Std Reward: {std_reward:.2f}")
    print(f"=" * 70)
except Exception as e:
    print(f"\nError during evaluation: {e}")
finally:
    # Try to close env, ignore cleanup errors
    try:
        env.close()
    except:
        pass
