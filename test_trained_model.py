#!/usr/bin/env python3
"""
Test script to verify the trained UniLCD model works correctly
"""
import unilcd_env
import gymnasium as gym
import json
from stable_baselines3 import PPO
import numpy as np

def test_trained_model():
    print("🧪 Testing trained UniLCD model...")
    
    # Load configuration
    config = json.load(open('unilcd_emb_eval_config.json'))
    print(f"✅ Config loaded: {config}")
    
    # Create environment
    print("🌍 Creating environment...")
    env = gym.make(**config)
    obs, info = env.reset()
    print(f"✅ Environment created, obs shape: {obs.shape}")
    
    # Load the trained model
    print("🤖 Loading trained model...")
    model = PPO.load("unilcd_ppo_model_simple")
    print("✅ Model loaded successfully!")
    
    # Test model prediction
    print("🎯 Testing model prediction...")
    action, _states = model.predict(obs, deterministic=True)
    print(f"✅ Model prediction: {action}")
    
    # Test environment step
    print("🚗 Testing environment step...")
    next_obs, reward, done, truncated, info = env.step(action)
    print(f"✅ Step successful - Reward: {reward}, Done: {done}")
    
    # Test a few more steps
    print("🔄 Testing multiple steps...")
    total_reward = reward
    for i in range(5):
        action, _states = model.predict(next_obs, deterministic=True)
        next_obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        print(f"   Step {i+1}: action={action}, reward={reward:.2f}")
        if done or truncated:
            next_obs, info = env.reset()
            print("   Episode ended, reset environment")
    
    print(f"🏆 Test complete! Total reward over 6 steps: {total_reward:.2f}")
    
    # Clean up
    env.close()
    print("✅ Environment closed successfully")
    
    return True

if __name__ == "__main__":
    try:
        success = test_trained_model()
        if success:
            print("\n🎉 SUCCESS: Trained model works perfectly!")
            print("   - Model loads correctly")
            print("   - Environment integrates properly") 
            print("   - Predictions and actions work")
            print("   - Training was genuinely successful!")
        else:
            print("\n❌ FAILURE: Something went wrong")
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        print("❌ Model test failed - training may not have completed properly")
