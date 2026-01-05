# UniLCD training with checkpoint support for resuming
import unilcd_env
import gymnasium as gym
import json
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

config = json.load(open('unilcd_emb_eval_config.json'))
config['max_episode_steps'] = 1500
env = gym.make(**config)
env.reset()

os.makedirs("./checkpoints", exist_ok=True)

# Try to load checkpoint if it exists
checkpoint_files = [f for f in os.listdir("./checkpoints") if f.startswith("unilcd_rl_model_") and f.endswith("_steps.zip")]
if checkpoint_files:
    checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.split("_")[-2]))
    latest_checkpoint = os.path.join("./checkpoints", checkpoint_files[-1])
    print(f"Resuming from checkpoint: {latest_checkpoint}")
    model = PPO.load(latest_checkpoint, env=env)
    # Extract timesteps from filename
    completed_steps = int(checkpoint_files[-1].split("_")[-2])
    remaining_steps = 375000 - completed_steps
    print(f"Completed: {completed_steps}, Remaining: {remaining_steps}")
else:
    model = PPO("MlpPolicy", env, verbose=1)
    remaining_steps = 375000
    print("Starting UniLCD training with 375,000 timesteps (250 episodes × 1500 steps)...")
    print("Estimated time: ~10-11 hours")

checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path="./checkpoints/",
    name_prefix="unilcd_rl_model"
)

model.learn(total_timesteps=remaining_steps, progress_bar=True, callback=checkpoint_callback, reset_num_timesteps=False)
print("Training completed successfully!")
model.save("unilcd_ppo_model_250x1500")