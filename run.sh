#!/bin/bash

# Medical RAG Main Runner Script
# This is the main entry point for managing the Medical RAG application

set -e

# Colors for output
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🏥 Medical RAG Application Manager${NC}"
echo "======================================"
echo ""

# Function to show menu
show_menu() {
    echo "Available commands:"
    echo ""
    echo -e "${GREEN}📦 Build & Deploy:${NC}"
    echo "  1) build     - Build Docker image"
    echo "  2) start     - Start application"
    echo "  3) stop      - Stop application"
    echo "  4) restart   - Restart application"
    echo ""
    echo -e "${YELLOW}🔍 Monitoring:${NC}"
    echo "  5) status    - Show application status"
    echo "  6) logs      - View application logs"
    echo "  7) test      - Test all endpoints"
    echo ""
    echo -e "${BLUE}🛠️  Development:${NC}"
    echo "  8) shell     - Access container shell"
    echo "  9) cleanup   - Clean up Docker resources"
    echo ""
    echo -e "${RED}❓ Help:${NC}"
    echo "  10) help     - Show detailed help"
    echo "  0) exit      - Exit"
    echo ""
}

# Function to handle user input
handle_input() {
    # Extract just the command name (remove ./run.sh if present)
    local cmd=$(echo "$1" | sed 's|^\./run\.sh\s*||' | sed 's|^run\.sh\s*||')
    
    case $cmd in
        1|build)
            echo -e "${BLUE}🔨 Building Docker image...${NC}"
            ./scripts/docker-setup.sh build
            ;;
        2|start)
            echo -e "${BLUE}🚀 Starting application...${NC}"
            ./scripts/docker-setup.sh start
            ;;
        3|stop)
            echo -e "${BLUE}🛑 Stopping application...${NC}"
            ./scripts/docker-setup.sh stop
            ;;
        4|restart)
            echo -e "${BLUE}🔄 Restarting application...${NC}"
            ./scripts/docker-setup.sh restart
            ;;
        5|status)
            echo -e "${BLUE}📊 Checking status...${NC}"
            ./scripts/docker-setup.sh status
            ;;
        6|logs)
            echo -e "${BLUE}📋 Showing logs...${NC}"
            ./scripts/docker-logs.sh
            ;;
        7|test)
            echo -e "${BLUE}🧪 Running tests...${NC}"
            ./scripts/docker-test.sh
            ;;
        8|shell)
            echo -e "${BLUE}🐚 Accessing shell...${NC}"
            ./scripts/docker-shell.sh
            ;;
        9|cleanup)
            echo -e "${BLUE}🧹 Cleaning up...${NC}"
            ./scripts/docker-setup.sh cleanup
            ;;
        10|help)
            echo -e "${BLUE}📖 Showing help...${NC}"
            ./scripts/docker-setup.sh help
            ;;
        0|exit)
            echo -e "${GREEN}👋 Goodbye!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Invalid option: $1${NC}"
            echo "Use 'help' for available commands"
            ;;
    esac
}

# Main script logic
main() {
    # If arguments are provided, execute directly
    if [ $# -gt 0 ]; then
        handle_input "$1"
        exit 0
    fi
    
    # Interactive mode
    while true; do
        show_menu
        echo -n "Enter your choice (0-10, command name, or 'help'): "
        read -r choice
        
        if [ -z "$choice" ]; then
            echo -e "${YELLOW}⚠️  Please enter a valid choice${NC}"
            continue
        fi
        
        handle_input "$choice"
        
        echo ""
        echo -e "${GREEN}✅ Command completed!${NC}"
        echo ""
        echo -n "Press Enter to continue or 'exit' to quit: "
        read -r continue_choice
        
        if [ "$continue_choice" = "exit" ]; then
            echo -e "${GREEN}👋 Goodbye!${NC}"
            exit 0
        fi
        
        echo ""
    done
}

# Run main function with all arguments
main "$@" 