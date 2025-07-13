#!/bin/bash

# Medical RAG Docker Management Script
# This script provides comprehensive Docker management for the Medical RAG application

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="medical-rag"
CONTAINER_NAME="medical-rag-app"
COMPOSE_FILE="docker-compose.yml"
ENV_FILE="docker.env"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    print_success "Docker is running"
}

# Function to check if docker-compose is available
check_docker_compose() {
    if command -v docker-compose > /dev/null 2>&1; then
        COMPOSE_CMD="docker-compose"
    elif docker compose version > /dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    else
        print_error "Neither docker-compose nor 'docker compose' is available."
        exit 1
    fi
    print_success "Using $COMPOSE_CMD"
}

# Function to build the Docker image
build_image() {
    print_status "Building Docker image..."
    docker build -t $IMAGE_NAME:latest .
    print_success "Docker image built successfully"
}

# Function to start the application
start_app() {
    print_status "Starting Medical RAG application..."
    
    # Check if container is already running
    if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
        print_warning "Container is already running. Stopping it first..."
        stop_app
    fi
    
    # Start with docker-compose
    $COMPOSE_CMD up -d
    
    print_status "Waiting for application to start..."
    sleep 10
    
    # Check if application is healthy
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_success "Medical RAG application is running successfully!"
        echo ""
        echo "🌐 Access your application at:"
        echo "   - Web Interface: http://localhost:8000"
        echo "   - API Documentation: http://localhost:8000/docs"
        echo "   - Health Check: http://localhost:8000/health"
        echo ""
    else
        print_warning "Application may still be starting. Check logs with: ./scripts/docker-logs.sh"
    fi
}

# Function to stop the application
stop_app() {
    print_status "Stopping Medical RAG application..."
    $COMPOSE_CMD down
    print_success "Application stopped"
}

# Function to restart the application
restart_app() {
    print_status "Restarting Medical RAG application..."
    stop_app
    start_app
}

# Function to show logs
show_logs() {
    print_status "Showing application logs..."
    $COMPOSE_CMD logs -f medical-rag
}

# Function to show status
show_status() {
    print_status "Application status:"
    echo ""
    docker ps --filter name=$CONTAINER_NAME
    echo ""
    
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_success "Application is healthy and responding"
        curl -s http://localhost:8000/health | python3 -m json.tool 2>/dev/null || curl -s http://localhost:8000/health
    else
        print_warning "Application is not responding to health checks"
    fi
}

# Function to clean up
cleanup() {
    print_status "Cleaning up Docker resources..."
    
    # Stop and remove containers
    $COMPOSE_CMD down --volumes --remove-orphans
    
    # Remove images
    docker rmi $IMAGE_NAME:latest 2>/dev/null || true
    
    # Clean up unused resources
    docker system prune -f
    
    print_success "Cleanup completed"
}

# Function to show help
show_help() {
    echo "Medical RAG Docker Management Script"
    echo "==================================="
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build     Build the Docker image"
    echo "  start     Start the application"
    echo "  stop      Stop the application"
    echo "  restart   Restart the application"
    echo "  logs      Show application logs"
    echo "  status    Show application status"
    echo "  cleanup   Clean up Docker resources"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build    # Build the Docker image"
    echo "  $0 start    # Start the application"
    echo "  $0 logs     # View logs"
    echo ""
}

# Main script logic
main() {
    # Check prerequisites
    check_docker
    check_docker_compose
    
    # Parse command line arguments
    case "${1:-help}" in
        build)
            build_image
            ;;
        start)
            start_app
            ;;
        stop)
            stop_app
            ;;
        restart)
            restart_app
            ;;
        logs)
            show_logs
            ;;
        status)
            show_status
            ;;
        cleanup)
            cleanup
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@" 