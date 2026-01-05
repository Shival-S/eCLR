#!/bin/bash

# Auto-restart wrapper for 100-episode UniLCD training
# Handles CARLA crashes and automatic recovery

MAX_RETRIES=50
CARLA_WAIT_TIME=90
LOG_FILE="training_100ep.log"
COUNTER_FILE="episode_counter_100ep.pkl"

echo "============================================="
echo "UniLCD Auto-Training - 100 Episodes"
echo "============================================="
echo "Started at: $(date)"
echo ""

# Function to restart CARLA
restart_carla() {
    echo "[$(date)] Restarting CARLA server..."
    ./run_carla_docker.sh stop 2>/dev/null
    sleep 3
    docker rm unilcd_carla_server 2>/dev/null
    ./run_carla_docker.sh start
    sleep $CARLA_WAIT_TIME
    echo "[$(date)] CARLA server restarted"
}

# Function to check if training is complete
check_completion() {
    if [ -f "$COUNTER_FILE" ]; then
        # Use Python to check episode count
        EPISODES=$(python3 -c "
import pickle
try:
    with open('$COUNTER_FILE', 'rb') as f:
        data = pickle.load(f)
        print(data.get('n_episodes', 0))
except:
    print(0)
" 2>/dev/null)
        
        if [ "$EPISODES" -ge 100 ]; then
            return 0  # Complete
        fi
    fi
    return 1  # Not complete
}

# Initialize CARLA
restart_carla

# Training loop with auto-restart
ATTEMPT=1
while [ $ATTEMPT -le $MAX_RETRIES ]; do
    # Check if already complete
    if check_completion; then
        echo ""
        echo "============================================="
        echo "TRAINING COMPLETED!"
        echo "100 episodes finished"
        echo "Completed at: $(date)"
        echo "============================================="
        exit 0
    fi
    
    echo ""
    echo "============================================="
    echo "Training Attempt #$ATTEMPT"
    echo "Started at: $(date)"
    echo "============================================="
    
    # Run training
    source unilcd_env_py38/bin/activate
    python3 train_unilcd_100ep.py >> $LOG_FILE 2>&1
    EXIT_CODE=$?
    
    echo "[$(date)] Training exited with code $EXIT_CODE"
    
    # Check if completed successfully
    if check_completion; then
        echo ""
        echo "============================================="
        echo "TRAINING COMPLETED SUCCESSFULLY!"
        echo "Completed at: $(date)"
        echo "Total attempts: $ATTEMPT"
        echo "============================================="
        exit 0
    fi
    
    # If not complete and not last attempt, restart
    if [ $ATTEMPT -lt $MAX_RETRIES ]; then
        echo "[$(date)] Training incomplete. Will restart..."
        restart_carla
        ATTEMPT=$((ATTEMPT + 1))
    else
        echo "[$(date)] Max retries reached. Stopping."
        exit 1
    fi
done

echo "Training ended after $MAX_RETRIES attempts"
