#!/bin/bash
# Local-Only Model Evaluation Script
# Evaluates ONLY the local model (bypasses routing policy)
# Matches methodology from UniLCD paper Table 2

echo "=========================================="
echo "Local-Only Model Evaluation"
echo "=========================================="
echo "Episodes: 150 (5 routes × 30 episodes)"
echo "Model: /home/shival/UniLCD/local_model.pth"
echo "Videos: rollout_video_local_only.mp4, minimap_video_local_only.mp4"
echo "Metrics: task_metrics_local_only.csv"
echo "=========================================="
echo ""
echo "IMPORTANT: Make sure CARLA server is running on port 2000"
echo "           (run run_carla_docker.sh in another terminal)"
echo ""
echo "Press Ctrl+C to stop evaluation"
echo ""

cd /home/shival/UniLCD/Eshed_Model_Training_11-7

/home/shival/UniLCD/unilcd_venv/bin/python evaluate_local_only.py \
    2>&1 | tee /home/shival/UniLCD/Eshed_Model_Training_11-7/local_only_eval.log
