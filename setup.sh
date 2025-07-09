#!/bin/bash

# AI Safety System Setup Script
# This script sets up the complete local AI safety system

set -e

echo "🚀 AI Safety System Setup"
echo "========================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
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

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check Python
    if ! command_exists python3; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    print_success "Python $PYTHON_VERSION found"
    
    # Check pip
    if ! command_exists pip3; then
        print_error "pip3 is required but not installed"
        exit 1
    fi
    
    # Check Node.js
    if ! command_exists node; then
        print_error "Node.js is required but not installed"
        print_status "Please install Node.js from https://nodejs.org/"
        exit 1
    fi
    
    NODE_VERSION=$(node --version)
    print_success "Node.js $NODE_VERSION found"
    
    # Check npm
    if ! command_exists npm; then
        print_error "npm is required but not installed"
        exit 1
    fi
    
    NPM_VERSION=$(npm --version)
    print_success "npm $NPM_VERSION found"
    
    print_success "All requirements satisfied"
}

# Setup Python virtual environment
setup_python_env() {
    print_status "Setting up Python environment..."
    
    cd backend
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_status "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip
    
    # Install requirements
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt
    
    # Download spaCy model
    print_status "Downloading spaCy English model..."
    python -m spacy download en_core_web_sm
    
    cd ..
    print_success "Python environment setup complete"
}

# Setup Node.js environment
setup_node_env() {
    print_status "Setting up Node.js environment..."
    
    cd frontend
    
    # Install dependencies
    print_status "Installing Node.js dependencies..."
    npm install
    
    # Build the frontend
    print_status "Building frontend..."
    npm run build
    
    cd ..
    print_success "Node.js environment setup complete"
}

# Setup Ollama
setup_ollama() {
    print_status "Setting up Ollama..."
    
    # Check if Ollama is installed
    if ! command_exists ollama; then
        print_status "Installing Ollama..."
        
        # Detect OS and install accordingly
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            if command_exists brew; then
                brew install ollama
            else
                print_warning "Homebrew not found. Please install Ollama manually from https://ollama.com"
                return
            fi
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            # Linux
            curl -fsSL https://ollama.com/install.sh | sh
        else
            print_warning "Unsupported OS. Please install Ollama manually from https://ollama.com"
            return
        fi
    else
        print_success "Ollama already installed"
    fi
    
    # Start Ollama service
    print_status "Starting Ollama service..."
    ollama serve &
    OLLAMA_PID=$!
    
    # Wait for Ollama to start
    sleep 5
    
    # Pull Llama3 model
    print_status "Downloading Llama3 model (this may take a while)..."
    ollama pull llama3
    
    print_success "Ollama setup complete"
}

# Create configuration directories
setup_configs() {
    print_status "Setting up configuration directories..."
    
    # Create logs directory
    mkdir -p logs
    
    # Create data directory
    mkdir -p data
    
    # Set permissions
    chmod 755 logs data
    
    print_success "Configuration directories created"
}

# Create startup scripts
create_startup_scripts() {
    print_status "Creating startup scripts..."
    
    # Backend startup script
    cat > start_backend.sh << 'EOF'
#!/bin/bash
cd backend
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
EOF
    
    # Frontend startup script
    cat > start_frontend.sh << 'EOF'
#!/bin/bash
cd frontend
npm start
EOF
    
    # Combined startup script
    cat > start_system.sh << 'EOF'
#!/bin/bash

echo "🚀 Starting AI Safety System..."

# Start Ollama if not running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama..."
    ollama serve &
    sleep 3
fi

# Start backend
echo "Starting backend..."
./start_backend.sh &
BACKEND_PID=$!

# Wait for backend to start
sleep 5

# Start frontend
echo "Starting frontend..."
./start_frontend.sh &
FRONTEND_PID=$!

echo "✅ System started!"
echo "📱 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap 'echo "Stopping services..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit' INT
wait
EOF
    
    # Make scripts executable
    chmod +x start_backend.sh start_frontend.sh start_system.sh
    
    print_success "Startup scripts created"
}

# Run tests
run_tests() {
    print_status "Running tests..."
    
    # Test backend
    cd backend
    source venv/bin/activate
    
    # Basic import test
    python -c "
import sys
sys.path.append('.')
try:
    from app.main import app
    print('✅ Backend imports successful')
except Exception as e:
    print(f'❌ Backend import failed: {e}')
    sys.exit(1)
"
    
    cd ..
    
    # Test frontend build
    cd frontend
    if [ -d "build" ]; then
        print_success "Frontend build exists"
    else
        print_warning "Frontend not built, running build..."
        npm run build
    fi
    cd ..
    
    print_success "Tests completed"
}

# Create README
create_documentation() {
    print_status "Creating documentation..."
    
    cat > README.md << 'EOF'
# AI Safety System

A comprehensive local AI safety system with NeMo Guardrails, multiple detection models, and a React frontend.

## Quick Start

1. **Setup**: Run the setup script
   ```bash
   ./setup.sh
   ```

2. **Start System**: Launch all services
   ```bash
   ./start_system.sh
   ```

3. **Access**: Open your browser to http://localhost:3000

## Services

- **Frontend**: React app (port 3000)
- **Backend**: FastAPI server (port 8000)
- **Ollama**: LLM server (port 11434)

## Individual Service Control

### Backend Only
```bash
./start_backend.sh
```

### Frontend Only
```bash
./start_frontend.sh
```

### Ollama Only
```bash
ollama serve
```

## Configuration

- Backend config: `backend/configs/app_config.yml`
- Guardrails config: `backend/configs/guardrails/`
- Frontend environment: `frontend/.env`

## API Documentation

- Interactive docs: http://localhost:8000/docs
- OpenAPI spec: http://localhost:8000/openapi.json

## Detectors

1. **Toxicity**: Detects harmful content
2. **PII**: Identifies personal information
3. **Prompt Injection**: Prevents manipulation attempts
4. **Topic Classification**: Filters restricted topics
5. **Fact Checking**: Identifies questionable claims
6. **Spam Detection**: Filters low-quality content

## Troubleshooting

### Common Issues

1. **Port conflicts**: Change ports in config files
2. **Model download fails**: Check internet connection
3. **Permission errors**: Run with appropriate permissions

### Logs

- Backend logs: `logs/backend.log`
- Frontend logs: Browser console
- Ollama logs: System logs

### Reset System

```bash
# Stop all services
pkill -f "uvicorn\|npm\|ollama"

# Clean and restart
rm -rf backend/venv frontend/node_modules
./setup.sh
```

## Development

### Backend Development
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload
```

### Frontend Development
```bash
cd frontend
npm start
```

### Adding New Detectors

1. Create detector class in `backend/app/models/detector_models.py`
2. Add to model manager in `backend/app/models/model_manager.py`
3. Create custom action in `backend/app/guardrails/custom_actions.py`
4. Update frontend detector config

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend│    │  FastAPI Backend│    │     Ollama      │
│   (Port 3000)   │◄───┤   (Port 8000)   │◄───┤   (Port 11434)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  NeMo Guardrails│
                       │   + Detectors   │
                       └─────────────────┘
```

## License

This project is for educational and research purposes.
EOF
    
    print_success "Documentation created"
}

# Main setup function
main() {
    echo "Starting setup process..."
    echo ""
    
    check_requirements
    echo ""
    
    setup_python_env
    echo ""
    
    setup_node_env
    echo ""
    
    setup_ollama
    echo ""
    
    setup_configs
    echo ""
    
    create_startup_scripts
    echo ""
    
    run_tests
    echo ""
    
    create_documentation
    echo ""
    
    print_success "🎉 Setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Start the system: ./start_system.sh"
    echo "2. Open browser: http://localhost:3000"
    echo "3. Check API docs: http://localhost:8000/docs"
    echo ""
    echo "For troubleshooting, see README.md"
}

# Run main function
main "$@"