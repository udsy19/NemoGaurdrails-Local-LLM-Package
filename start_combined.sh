#!/bin/bash
echo "🚀 Starting Combined AI Safety System (Single Port)..."

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to cleanup processes on exit
cleanup() {
    echo "Stopping services..."
    pkill -f "uvicorn"
    pkill -f "streamlit"
    exit 0
}

# Trap exit signals
trap cleanup EXIT INT TERM

# Start backend
echo "Starting backend..."
cd backend
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Start backend in background
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Wait for backend to start
echo "Waiting for backend to initialize..."
sleep 10

# Start Streamlit on port 8501 but proxy through backend
echo "Starting Streamlit interface..."
streamlit run app/streamlit_app.py --server.port 8501 --server.address localhost --browser.gatherUsageStats false 2>&1 | tee -a ../logs/streamlit.log &
STREAMLIT_PID=$!

echo ""
echo "✅ Combined System started!"
echo "🌐 Main Interface: http://localhost:8000"
echo "📱 Streamlit Direct: http://localhost:8501"
echo "🔧 Backend API: http://localhost:8000/api"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for both processes
wait $BACKEND_PID $STREAMLIT_PID