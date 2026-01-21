# eCLR training with SAC (Soft Actor-Critic)
# Uses InternVL3 cloud model and RegNetY local model
import sys
sys.path.insert(0, '/mnt/data5/shival/eCLR/unilcd_env/build/lib')
from unilcd_env.envs.unilcd_emb_env import UniLCDEmbEnv
import json
from stable_baselines3 import SAC

config = json.load(open('eclr_sac_config.json'))

print("=" * 60)
print("eCLR SAC Router Training")
print("=" * 60)
print(f"Cloud model: {config['cloud_model_checkpoint']}")
print(f"Local model: {config['local_model_checkpoint']}")
print(f"Cloud model type: {config.get('cloud_model_type', 'regnet')}")
print()

# Directly instantiate the environment
env = UniLCDEmbEnv(**config)
env.reset()

print("Starting SAC training with 50,000 timesteps...")
print("Estimated time: ~2-3 hours")
print()

model = SAC("MlpPolicy", env, verbose=1, learning_rate=3e-4, buffer_size=50000)
model.learn(total_timesteps=50000, progress_bar=True)

print()
print("Training completed!")
model.save("eclr_sac_router_50k")
print("Model saved as eclr_sac_router_50k.zip")

env.close()
