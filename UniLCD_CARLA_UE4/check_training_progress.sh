#!/bin/bash
# Monitor UniLCD training progress

echo "========================================================================"
echo "UniLCD Training Progress Monitor"
echo "========================================================================"
echo ""

# Check if training is running
if ps aux | grep -v grep | grep "train_unilcd_paper.py" > /dev/null; then
    echo "✅ Training Status: RUNNING"
    TRAINING_PID=$(ps aux | grep -v grep | grep "train_unilcd_paper.py" | awk '{print $2}')
    CPU_USAGE=$(ps aux | grep -v grep | grep "train_unilcd_paper.py" | awk '{print $3}')
    MEM_USAGE=$(ps aux | grep -v grep | grep "train_unilcd_paper.py" | awk '{print $4}')
    echo "   PID: $TRAINING_PID"
    echo "   CPU: ${CPU_USAGE}%"
    echo "   Memory: ${MEM_USAGE}%"
else
    echo "❌ Training Status: NOT RUNNING"
fi

echo ""

# Check CARLA status
if ps aux | grep -v grep | grep "CarlaUE4-Linux-Shipping" > /dev/null; then
    echo "✅ CARLA Status: RUNNING"
else
    echo "❌ CARLA Status: NOT RUNNING"
fi

echo ""
echo "------------------------------------------------------------------------"

# Count episodes from log
TOTAL_EPISODES=1000
LATEST_LOG=$(tail -100 /home/shival/UniLCD/training.log 2>/dev/null | grep "Episode " | tail -1)

if [ ! -z "$LATEST_LOG" ]; then
    # Extract episode number (format: "Episode X/1000 | Reward: Y | Length: Z steps")
    CURRENT_EP=$(echo "$LATEST_LOG" | grep -oP 'Episode \K[0-9]+' | head -1)

    if [ ! -z "$CURRENT_EP" ]; then
        EPISODES_LEFT=$((TOTAL_EPISODES - CURRENT_EP))
        PERCENT_COMPLETE=$(echo "scale=1; ($CURRENT_EP / $TOTAL_EPISODES) * 100" | bc)

        echo "📊 Training Progress:"
        echo "   Episodes completed: $CURRENT_EP / $TOTAL_EPISODES"
        echo "   Episodes remaining: $EPISODES_LEFT"
        echo "   Progress: ${PERCENT_COMPLETE}%"
        echo ""

        # Calculate time estimates based on previous run (76 sec/episode average)
        AVG_TIME_PER_EP=76
        TIME_REMAINING_SEC=$((EPISODES_LEFT * AVG_TIME_PER_EP))
        TIME_REMAINING_HOURS=$(echo "scale=1; $TIME_REMAINING_SEC / 3600" | bc)

        echo "⏱️  Time Estimate (based on ~76 sec/episode):"
        echo "   Estimated time remaining: ${TIME_REMAINING_HOURS} hours"

        # Show latest episode info
        echo ""
        echo "📝 Latest Episode:"
        echo "   $LATEST_LOG"
    else
        echo "⏳ Training starting... (no episodes completed yet)"
    fi
else
    echo "⏳ Waiting for training to begin..."
fi

echo ""
echo "------------------------------------------------------------------------"

# Count checkpoints
CHECKPOINT_COUNT=$(ls /home/shival/UniLCD/checkpoints/*.zip 2>/dev/null | wc -l)
if [ $CHECKPOINT_COUNT -gt 0 ]; then
    LATEST_CHECKPOINT=$(ls -t /home/shival/UniLCD/checkpoints/*.zip 2>/dev/null | head -1 | xargs basename)
    echo "💾 Checkpoints saved: $CHECKPOINT_COUNT"
    echo "   Latest: $LATEST_CHECKPOINT"
else
    echo "💾 Checkpoints: None yet (saved every 10 episodes)"
fi

echo ""
echo "========================================================================"
echo ""
echo "Commands:"
echo "  Watch live: tail -f /home/shival/UniLCD/training.log"
echo "  Stop training: pkill -f train_unilcd_paper.py"
echo "  View this again: bash /home/shival/UniLCD/check_training_progress.sh"
echo ""
