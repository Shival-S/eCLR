import unilcd_env
import gymnasium as gym
import json
from stable_baselines3 import PPO

print("Loading config...")
config = json.load(open('unilcd_emb_eval_config.json'))
print(f"Creating environment on port {config.get('port', 2000)}...")
env = gym.make(**config)
print("Resetting environment...")
env.reset()

print("Creating PPO model...")
model = PPO("MlpPolicy", env, verbose=1)
print("Starting training for 500 timesteps...")
model.learn(total_timesteps=500, progress_bar=True)
print("Training completed successfully!")
model.save("test_single_model")
env.close()