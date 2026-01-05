#!/bin/bash

# Auto-restart training script for UniLCD
# Automatically restarts training when CARLA crashes

MAX_RETRIES=100
RETRY_COUNT=0
LOG_FILE="training_250x1500.log"
CARLA_WAIT_TIME=90

echo "=== UniLCD Auto-Restart Training ===" | tee -a $LOG_FILE
echo "Target: 375,000 timesteps (250 episodes × 1500 steps)" | tee -a $LOG_FILE
echo "Max retries: $MAX_RETRIES" | tee -a $LOG_FILE
echo "Started at: $(date)" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Function to check if training is complete
check_completion() {
    if [ -f "unilcd_ppo_model_250x1500.zip" ]; then
        return 0
    fi

    # Check if we've reached 375k timesteps
    LAST_TIMESTEP=$(grep "total_timesteps" $LOG_FILE | tail -1 | awk '{print $4}')
    if [ ! -z "$LAST_TIMESTEP" ] && [ "$LAST_TIMESTEP" -ge 375000 ]; then
        return 0
    fi

    return 1
}

# Function to restart CARLA
restart_carla() {
    echo "[$(date)] Restarting CARLA server..." | tee -a $LOG_FILE
    ./run_carla_docker.sh stop
    sleep 3
    docker rm unilcd_carla_server 2>/dev/null
    ./run_carla_docker.sh start
    echo "[$(date)] Waiting ${CARLA_WAIT_TIME}s for CARLA to initialize..." | tee -a $LOG_FILE
    sleep $CARLA_WAIT_TIME
}

# Main training loop
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    # Check if already complete
    if check_completion; then
        echo "" | tee -a $LOG_FILE
        echo "=== TRAINING COMPLETED SUCCESSFULLY ===" | tee -a $LOG_FILE
        echo "Finished at: $(date)" | tee -a $LOG_FILE
        exit 0
    fi

    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "" | tee -a $LOG_FILE
    echo "[$(date)] === Training attempt #$RETRY_COUNT ===" | tee -a $LOG_FILE

    # Get current progress
    CURRENT_TIMESTEP=$(grep "total_timesteps" $LOG_FILE 2>/dev/null | tail -1 | awk '{print $4}' | grep -E '^[0-9]+$')
    if [ ! -z "$CURRENT_TIMESTEP" ]; then
        PROGRESS=$(awk "BEGIN {printf \"%.1f\", ($CURRENT_TIMESTEP/375000)*100}")
        echo "[$(date)] Current progress: $CURRENT_TIMESTEP / 375,000 timesteps ($PROGRESS%)" | tee -a $LOG_FILE
    else
        echo "[$(date)] Starting fresh training..." | tee -a $LOG_FILE
    fi

    # Restart CARLA before each attempt
    restart_carla

    # Check if CARLA is running
    if ! docker ps | grep -q unilcd_carla_server; then
        echo "[$(date)] ERROR: CARLA failed to start. Retrying in 10s..." | tee -a $LOG_FILE
        sleep 10
        continue
    fi

    # Run training
    echo "[$(date)] Starting training..." | tee -a $LOG_FILE
    source unilcd_env_py38/bin/activate
    python train_unilcd_250x1500.py >> $LOG_FILE 2>&1
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "" | tee -a $LOG_FILE
        echo "=== TRAINING COMPLETED SUCCESSFULLY ===" | tee -a $LOG_FILE
        echo "Finished at: $(date)" | tee -a $LOG_FILE
        exit 0
    else
        echo "[$(date)] Training crashed with exit code $EXIT_CODE. Will restart..." | tee -a $LOG_FILE
        sleep 5
    fi
done

echo "" | tee -a $LOG_FILE
echo "=== MAX RETRIES REACHED ===" | tee -a $LOG_FILE
echo "Attempted $MAX_RETRIES times. Please investigate the issue." | tee -a $LOG_FILE
exit 1