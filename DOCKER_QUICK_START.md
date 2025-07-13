# Medical RAG Docker Quick Start Guide

## 🚀 Quick Start

### Option 1: Interactive Menu
```bash
./run.sh
```
This opens an interactive menu where you can select commands.

### Option 2: Direct Commands
```bash
# Build and start the application
./run.sh start

# View logs
./run.sh logs

# Test endpoints
./run.sh test

# Stop application
./run.sh stop
```

## 📋 Available Commands

### Main Runner (`./run.sh`)
- `./run.sh` - Interactive menu
- `./run.sh start` - Start application
- `./run.sh stop` - Stop application
- `./run.sh restart` - Restart application
- `./run.sh status` - Show status
- `./run.sh logs` - View logs
- `./run.sh test` - Test endpoints
- `./run.sh shell` - Access container shell
- `./run.sh cleanup` - Clean up resources

### Individual Scripts
- `./scripts/docker-setup.sh build` - Build Docker image
- `./scripts/docker-setup.sh start` - Start application
- `./scripts/docker-setup.sh stop` - Stop application
- `./scripts/docker-setup.sh status` - Show status
- `./scripts/docker-setup.sh cleanup` - Clean up resources
- `./scripts/docker-logs.sh` - View logs
- `./scripts/docker-shell.sh` - Access container shell
- `./scripts/docker-test.sh` - Test endpoints

## 🌐 Access Your Application

Once running, access your application at:
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🔧 Configuration

Edit `docker.env` to configure:
- API keys (optional)
- Model settings
- Application parameters

## 📊 Monitoring

### Check Status
```bash
./run.sh status
```

### View Logs
```bash
./run.sh logs
```

### Test Endpoints
```bash
./run.sh test
```

## 🛠️ Development

### Access Container Shell
```bash
./run.sh shell
```

### Clean Up Resources
```bash
./run.sh cleanup
```

## 🚨 Troubleshooting

### Application Won't Start
1. Check if Docker is running
2. Check logs: `./run.sh logs`
3. Check status: `./run.sh status`

### Port Already in Use
```bash
# Stop any existing containers
./run.sh stop

# Or kill processes on port 8000
lsof -ti:8000 | xargs kill -9
```

### Build Issues
```bash
# Clean up and rebuild
./run.sh cleanup
./run.sh start
```

## 📁 File Structure

```
RAG/
├── run.sh                    # Main runner script
├── scripts/
│   ├── docker-setup.sh      # Docker management
│   ├── docker-logs.sh       # Log viewing
│   ├── docker-shell.sh      # Shell access
│   └── docker-test.sh       # Endpoint testing
├── docker-compose.yml       # Docker Compose config
├── Dockerfile              # Docker image definition
├── docker.env              # Environment variables
└── requirements.txt        # Python dependencies
```

## 🎯 Common Workflows

### First Time Setup
```bash
./run.sh start
```

### Daily Development
```bash
./run.sh start    # Start application
./run.sh test     # Test endpoints
./run.sh logs     # Monitor logs
./run.sh stop     # Stop when done
```

### Debugging
```bash
./run.sh shell    # Access container
./run.sh logs     # View logs
./run.sh status   # Check health
```

### Production Deployment
```bash
./run.sh start    # Start application
./run.sh status   # Verify health
./run.sh test     # Test all endpoints
``` 