#!/bin/bash
# Script to manage native CARLA 0.9.13 server
# Replaces Docker version for better performance

CARLA_PATH="/home/shival/CarlaUE4.sh"
CARLA_PORT=2000
QUALITY="Epic"
LOG_FILE="/home/shival/UniLCD/carla_server.log"

# Check if CARLA is already running
check_carla() {
    pgrep -f "CarlaUE4-Linux-Shipping" > /dev/null
    return $?
}

start_carla() {
    if check_carla; then
        echo "CARLA server is already running"
        ps aux | grep CarlaUE4-Linux-Shipping | grep -v grep
        return 0
    fi

    echo "Starting native CARLA server..."
    echo "  Port: $CARLA_PORT"
    echo "  Quality: $QUALITY"
    echo "  Log: $LOG_FILE"

    # Start CARLA in background with specified settings
    nohup $CARLA_PATH \
        -opengl \
        -world-port=$CARLA_PORT \
        -RenderOffScreen \
        -quality-level=$QUALITY \
        -nosound \
        > "$LOG_FILE" 2>&1 &

    CARLA_PID=$!
    echo "CARLA server started with PID: $CARLA_PID"
    echo "Waiting for server to initialize..."

    # Wait for server to be ready (check for ~10 seconds)
    for i in {1..10}; do
        sleep 1
        if check_carla; then
            echo "✓ CARLA server is running"
            return 0
        fi
    done

    echo "Warning: CARLA may still be initializing. Check log: $LOG_FILE"
}

stop_carla() {
    if ! check_carla; then
        echo "CARLA server is not running"
        return 0
    fi

    echo "Stopping CARLA server..."
    pkill -f "CarlaUE4-Linux-Shipping"

    # Wait for shutdown
    for i in {1..5}; do
        sleep 1
        if ! check_carla; then
            echo "✓ CARLA server stopped"
            return 0
        fi
    done

    # Force kill if still running
    echo "Force killing CARLA server..."
    pkill -9 -f "CarlaUE4-Linux-Shipping"
    echo "✓ CARLA server killed"
}

status_carla() {
    if check_carla; then
        echo "CARLA server is running:"
        ps aux | grep CarlaUE4-Linux-Shipping | grep -v grep
    else
        echo "CARLA server is not running"
    fi
}

case "$1" in
    start)
        start_carla
        ;;
    stop)
        stop_carla
        ;;
    restart)
        stop_carla
        sleep 2
        start_carla
        ;;
    status)
        status_carla
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        echo ""
        echo "Commands:"
        echo "  start   - Start CARLA server"
        echo "  stop    - Stop CARLA server"
        echo "  restart - Restart CARLA server"
        echo "  status  - Check CARLA server status"
        exit 1
        ;;
esac
