#!/bin/bash

NUM_INSTANCES=${1:-8}
NUM_GPUS=4
CARLA_SCRIPT=./CarlaUE4.sh
DOCKER_NAME=carlasim/carla:0.9.13
BASE_PORT=2000
QUALITY_LEVEL=${3:-Epic}

if [ "$2" == "start" ]; then
    echo "Starting $NUM_INSTANCES CARLA instances across $NUM_GPUS GPUs with quality=$QUALITY_LEVEL..."
    for i in $(seq 0 $((NUM_INSTANCES-1))); do
        PORT=$((BASE_PORT + i*2))
        GPUID=$((i % NUM_GPUS))
        NAME="unilcd_carla_server_$PORT"
        echo "Starting CARLA on port $PORT, GPU $GPUID (container: $NAME)"
        docker run --rm -d --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$GPUID \
            --net=host -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
            --name $NAME $DOCKER_NAME \
            /bin/bash $CARLA_SCRIPT -world-port=$PORT -RenderOffScreen -quality-level=$QUALITY_LEVEL -nosound
        sleep 3
    done
    echo "All CARLA instances started on ports $BASE_PORT to $((BASE_PORT + (NUM_INSTANCES-1)*2))"
    echo "GPU distribution: 2 instances per GPU across GPUs 0-3"
elif [ "$2" == "stop" ]; then
    echo "Stopping all CARLA instances..."
    for i in $(seq 0 $((NUM_INSTANCES-1))); do
        PORT=$((BASE_PORT + i*2))
        NAME="unilcd_carla_server_$PORT"
        docker stop $NAME 2>/dev/null && echo "Stopped $NAME" || echo "$NAME not running"
    done
elif [ "$2" == "status" ]; then
    echo "CARLA instance status:"
    docker ps --filter "name=unilcd_carla_server" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
else
    echo "Usage: $0 [num_instances] {start|stop|status} [quality_level]"
    echo "Example: $0 8 start Epic"
    echo "Quality levels: Low, Epic (default: Epic)"
    exit 1
fi