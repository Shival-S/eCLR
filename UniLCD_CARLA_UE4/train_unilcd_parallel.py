import unilcd_env
import gymnasium as gym
import json
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import os

N_ENVS = 4
BASE_PORT = 2000
TOTAL_EPISODES = 250
MAX_STEPS_PER_EPISODE = 1500
TOTAL_TIMESTEPS = TOTAL_EPISODES * MAX_STEPS_PER_EPISODE

config_template = json.load(open('unilcd_emb_eval_config.json'))
config_template['max_episode_steps'] = MAX_STEPS_PER_EPISODE

def make_env(rank):
    def _init():
        config = config_template.copy()
        config['port'] = BASE_PORT + rank * 2
        config['tm_port'] = 6000 + rank * 2
        env = gym.make(**config)
        return env
    return _init

if __name__ == '__main__':
    print(f"Training UniLCD with {N_ENVS} parallel environments")
    print(f"Target: {TOTAL_EPISODES} episodes × {MAX_STEPS_PER_EPISODE} steps = {TOTAL_TIMESTEPS:,} timesteps")

    os.makedirs("./checkpoints", exist_ok=True)

    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])

    checkpoint_callback = CheckpointCallback(
        save_freq=10000 // N_ENVS,
        save_path="./checkpoints/",
        name_prefix="unilcd_ppo"
    )

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=2048,
        batch_size=512,
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )

    print("Starting training...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=True,
        callback=checkpoint_callback
    )

    print("Training completed!")
    model.save("unilcd_ppo_model_375k_parallel")
    env.close()