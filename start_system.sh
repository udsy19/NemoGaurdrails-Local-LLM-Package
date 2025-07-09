#!/bin/bash

echo "🚀 Starting AI Safety System..."
echo "=============================="

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
echo "Waiting for backend to initialize..."
sleep 10

# Start Streamlit interface
echo "Starting Streamlit interface..."
cd backend && python -m streamlit run app/streamlit_app.py --server.port 8501 --server.headless true &
STREAMLIT_PID=$!

echo ""
echo "✅ System started!"
echo "🌐 Streamlit Interface: http://localhost:8501"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap 'echo "Stopping services..."; kill $BACKEND_PID $STREAMLIT_PID 2>/dev/null; exit' INT
wait