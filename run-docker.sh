#!/bin/bash

# Medical RAG Docker Runner Script
set -e

echo "🐳 Building Medical RAG Docker container..."

# Build the Docker image
docker build -t medical-rag:latest .

echo "✅ Docker image built successfully!"

# Check if container is already running
if [ "$(docker ps -q -f name=medical-rag-app)" ]; then
    echo "🔄 Stopping existing container..."
    docker stop medical-rag-app
    docker rm medical-rag-app
fi

echo "🚀 Starting Medical RAG application..."

# Run the container
docker-compose up -d

echo "⏳ Waiting for application to start..."
sleep 10

# Check if the application is running
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Medical RAG application is running successfully!"
    echo "🌐 Access the application at: http://localhost:8000"
    echo "📚 API documentation at: http://localhost:8000/docs"
    echo "🏥 Health check at: http://localhost:8000/health"
else
    echo "❌ Application failed to start. Checking logs..."
    docker-compose logs medical-rag
    exit 1
fi

echo ""
echo "📋 Useful commands:"
echo "  View logs: docker-compose logs -f medical-rag"
echo "  Stop app:  docker-compose down"
echo "  Restart:   docker-compose restart"
echo "  Shell:     docker exec -it medical-rag-app bash" 