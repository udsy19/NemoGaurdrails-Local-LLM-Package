#!/bin/bash
echo "🚀 Starting AI Safety Backend..."

# Create logs directory if it doesn't exist
mkdir -p logs

# Start backend with logging
cd backend
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level info 2>&1 | tee -a ../logs/backend.log