#!/bin/bash

# Medical RAG Docker Logs Script
# Simple script to view application logs

set -e

echo "📋 Medical RAG Application Logs"
echo "==============================="

# Check if docker-compose is available
if command -v docker-compose > /dev/null 2>&1; then
    COMPOSE_CMD="docker-compose"
elif docker compose version > /dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
else
    echo "❌ Neither docker-compose nor 'docker compose' is available."
    exit 1
fi

# Show logs with follow option
echo "🔍 Showing application logs (press Ctrl+C to exit)..."
echo ""

$COMPOSE_CMD logs -f medical-rag 