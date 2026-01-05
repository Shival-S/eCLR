#!/bin/bash
# Cloud-Only Model Evaluation Script
# Evaluates ONLY the cloud model (bypasses routing policy)
# Matches methodology from UniLCD paper Table 2

echo "=========================================="
echo "Cloud-Only Model Evaluation"
echo "=========================================="
echo "Episodes: 150 (5 routes × 30 episodes)"
echo "Model: /home/shival/UniLCD/cloud_model.pth"
echo "Videos: rollout_video_cloud_only.mp4, minimap_video_cloud_only.mp4"
echo "Metrics: task_metrics_cloud_only.csv"
echo "=========================================="
echo ""
echo "IMPORTANT: Make sure CARLA server is running on port 2000"
echo "           (run run_carla_docker.sh in another terminal)"
echo ""
echo "Press Ctrl+C to stop evaluation"
echo ""

cd /home/shival/UniLCD/Eshed_Model_Training_11-7

/home/shival/UniLCD/unilcd_venv/bin/python evaluate_cloud_only.py \
    2>&1 | tee /home/shival/UniLCD/Eshed_Model_Training_11-7/cloud_only_eval.log
