#!/bin/bash
# Local Model Training Script
# Uses the UniLCD virtual environment

cd /home/shival/UniLCD/unilcd_env/unilcd_env/envs/il_models

echo "=========================================="
echo "Starting Local Model Training"
echo "=========================================="
echo "Data: /home/shival/UniLCD/data/"
echo "Save: /home/shival/UniLCD/Eshed_Model_Training_11-7/local_model.pth"
echo "Epochs: 200 | Batch size: 32"
echo "=========================================="
echo ""

/home/shival/UniLCD/unilcd_venv/bin/python local_train.py \
    -d /home/shival/UniLCD/data/ \
    -s /home/shival/UniLCD/Eshed_Model_Training_11-7/local_model.pth \
    2>&1 | tee /home/shival/UniLCD/Eshed_Model_Training_11-7/local_train.log
