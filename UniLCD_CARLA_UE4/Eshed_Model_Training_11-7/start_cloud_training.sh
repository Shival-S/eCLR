#!/bin/bash
# Cloud Model Training Script
# Uses the UniLCD virtual environment

cd /home/shival/UniLCD/unilcd_env/unilcd_env/envs/il_models

echo "=========================================="
echo "Starting Cloud Model Training"
echo "=========================================="
echo "Data: /home/shival/UniLCD/data/"
echo "Save: /home/shival/UniLCD/Eshed_Model_Training_11-7/cloud_model.pth"
echo "Epochs: 200 | Batch size: 16"
echo "=========================================="
echo ""

/home/shival/UniLCD/unilcd_venv/bin/python cloud_train.py \
    -d /home/shival/UniLCD/data/ \
    -s /home/shival/UniLCD/Eshed_Model_Training_11-7/cloud_model.pth \
    2>&1 | tee /home/shival/UniLCD/Eshed_Model_Training_11-7/cloud_train.log
