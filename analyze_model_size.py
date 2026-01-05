#!/usr/bin/env python3
"""
Analyze the trained PPO model to understand its architecture and size
"""
from stable_baselines3 import PPO
import torch
import numpy as np

def analyze_ppo_model():
    print("🔍 Analyzing UniLCD PPO Model Architecture...")
    
    # Load the model
    model = PPO.load("unilcd_ppo_model_simple")
    
    # Get policy network
    policy = model.policy
    
    print(f"\n📊 Model Information:")
    print(f"   - Policy type: {type(policy).__name__}")
    print(f"   - Observation space: {model.observation_space}")
    print(f"   - Action space: {model.action_space}")
    print(f"   - Device: {model.device}")
    
    print(f"\n🧠 Policy Network Architecture:")
    print(policy)
    
    # Count parameters
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    
    print(f"\n📈 Parameter Count:")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    
    # Estimate expected size
    # Each float32 parameter = 4 bytes
    expected_size_bytes = total_params * 4
    expected_size_kb = expected_size_bytes / 1024
    expected_size_mb = expected_size_kb / 1024
    
    print(f"\n💾 Expected Model Size (parameters only):")
    print(f"   - {expected_size_bytes:,} bytes")
    print(f"   - {expected_size_kb:.1f} KB") 
    print(f"   - {expected_size_mb:.2f} MB")
    
    # Check actual file size
    import os
    actual_size = os.path.getsize("unilcd_ppo_model_simple.zip")
    print(f"\n📁 Actual File Size:")
    print(f"   - {actual_size:,} bytes ({actual_size/1024:.1f} KB)")
    
    # Size breakdown
    print(f"\n🎯 Size Analysis:")
    ratio = actual_size / expected_size_bytes
    print(f"   - Compression ratio: {ratio:.2f}x")
    print(f"   - Additional overhead: {actual_size - expected_size_bytes:,} bytes")
    
    if ratio < 2.0:
        print("✅ Model size is reasonable for a simple MLP policy")
    else:
        print("⚠️  Model might be larger than expected")
        
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("🚗 UniLCD PPO Model Size Analysis")  
    print("=" * 60)
    analyze_ppo_model()
    print("=" * 60)
