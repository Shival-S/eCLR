#!/usr/bin/env python3
"""
Quick test to verify model loading works without CARLA
"""
import sys
sys.path.append('/home/shival/UniLCD/unilcd_venv/lib/python3.8/site-packages')

import torch
from unilcd_env.envs.il_models.cloud_model import CloudModel
from unilcd_env.envs.il_models.local_model import LocalModel

print("Testing model loading...")

# Test cloud model
try:
    cloud_model = CloudModel()
    cloud_model.load_state_dict(torch.load('./cloud_model.pth'))
    cloud_model = cloud_model.eval()
    print("✅ Cloud model loaded successfully!")
    print(f"   Cloud model parameters: {sum(p.numel() for p in cloud_model.parameters()):,}")
except Exception as e:
    print(f"❌ Cloud model failed: {e}")

# Test local model  
try:
    local_model = LocalModel()
    local_model.load_state_dict(torch.load('./local_model.pth'))
    local_model = local_model.eval()
    print("✅ Local model loaded successfully!")
    print(f"   Local model parameters: {sum(p.numel() for p in local_model.parameters()):,}")
except Exception as e:
    print(f"❌ Local model failed: {e}")

print("\n🎉 Model loading test completed!")
