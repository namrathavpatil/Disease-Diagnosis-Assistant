#!/bin/bash

# Medical RAG Docker Shell Access Script
# Script to access the container shell for debugging

set -e

echo "🐚 Medical RAG Container Shell Access"
echo "====================================="

CONTAINER_NAME="medical-rag-app"

# Check if container is running
if [ ! "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "❌ Container '$CONTAINER_NAME' is not running."
    echo "   Start it first with: ./scripts/docker-setup.sh start"
    exit 1
fi

echo "🔗 Connecting to container shell..."
echo "   Container: $CONTAINER_NAME"
echo "   Use 'exit' to leave the shell"
echo ""

# Connect to container shell
docker exec -it $CONTAINER_NAME bash 