# eCLR Training Concessions Log

This documents all compromises made for faster iteration that could be improved for better performance.

## Cloud Model (InternVL3-2B) Training

1. **LoRA instead of full fine-tuning**
   - Current: Training only 1.1% of parameters (3.4M of 300M)
   - Better: Full fine-tuning of vision encoder
   - Impact: Potentially 5-10% worse feature extraction
   - Time tradeoff: ~40 hours vs ~200+ hours

2. **No custom CARLA UE5 weather/lighting variations**
   - Current: Default Town10 lighting only
   - Better: Rain, fog, night, dawn/dusk variations
   - Impact: Model may not generalize to different conditions
   - Requires: Custom CARLA UE5 build with weather API

3. **Single town (Town10) only**
   - Current: All 10,000 frames from Town10
   - Better: Mix of Town01-Town15 for diverse road layouts
   - Impact: May overfit to Town10 road geometry

4. **Limited dataset size (10,000 frames)**
   - Current: 10,000 frames
   - Better: 50,000-100,000 frames for better coverage
   - Impact: Less diversity in driving scenarios

5. **No data augmentation**
   - Current: Raw images only
   - Better: Random brightness, contrast, horizontal flip
   - Impact: Less robust to visual variations

6. **Using InternVL3 instead of NVIDIA Alpamayo**
   - Current: InternVL3-2B (general-purpose VLM adapted for driving)
   - Better: NVIDIA Alpamayo 1 (10B VLA model purpose-built for autonomous driving)
   - Impact: Alpamayo is specifically trained on 1,700+ hours of driving data, outputs trajectories natively, and includes reasoning capabilities
   - Why not used:
     - Released Jan 5, 2026 (brand new)
     - 10B parameters may not fit on 4x 11GB GPUs
     - Requires gated access on Hugging Face
     - Outputs trajectory waypoints, not steering/speed (needs conversion)
     - Expects multi-camera input (we use single camera)
     - CARLA integration undefined
   - Future consideration: Test after InternVL3 baseline is working
   - References: https://developer.nvidia.com/blog/building-autonomous-vehicles-that-reason-with-nvidia-alpamayo/

## Local Model Training

1. **No depth preprocessing with Depth Anything v2**
   - Current: Raw RGB images fed directly to local model
   - Better: Preprocess images with Depth Anything v2 to extract depth maps
   - Impact: Depth information could help local model perceive distance/obstacles better
   - Why not used: Additional complexity, need to verify benefit first
   - Future consideration: Test depth-augmented inputs after baseline is working
   - Reference: https://github.com/DepthAnything/Depth-Anything-V2

## SAC Router Training
(To be filled when training starts)

## Environment/Infrastructure

1. **FlashAttention2 not installed**
   - Current: Standard attention
   - Better: FlashAttention2 for faster training
   - Impact: ~2x slower attention computation

2. **No mixed precision optimization**
   - Current: bfloat16 (good)
   - Could add: torch.compile() for additional speedup

---
Last updated: 2026-01-06
