#!/bin/bash
# UniLCD Paper Evaluation Script
# Matches Kathakoli's methodology with detailed CSV metrics

echo "=========================================="
echo "UniLCD Paper Evaluation (150 episodes)"
echo "=========================================="
echo "Methodology: Matches Table 2 from paper"
echo "Episodes: 150 (5 routes × 30 episodes)"
echo "Max Steps: 1500 per episode"
echo "Model: PPO routing policy + cloud/local models"
echo "CARLA: NATIVE (not Docker)"
echo ""
echo "Output Files:"
echo "  CSV: task_metrics_paper_eval.csv"
echo "  Videos: rollout_video_paper_eval.mp4, minimap_video_paper_eval.mp4"
echo "  Summary: evaluation_paper_results.txt"
echo "  CARLA Log: carla_paper_eval.log"
echo "=========================================="
echo ""
echo "IMPORTANT: Make sure CARLA server is NOT running"
echo "           (script will start/restart NATIVE CARLA automatically)"
echo ""
echo "Press Ctrl+C to stop evaluation"
echo ""

cd /home/shival/UniLCD

# Kill any existing native CARLA instances (not Docker)
echo "Stopping any existing native CARLA instances..."
pkill -f CarlaUE4-Linux-Shipping
sleep 1
pkill -9 -f CarlaUE4-Linux-Shipping
sleep 2

# Run evaluation
/home/shival/UniLCD/unilcd_venv/bin/python evaluate_150ep_with_metrics.py \
    2>&1 | tee /home/shival/UniLCD/evaluation_paper_with_metrics.log
