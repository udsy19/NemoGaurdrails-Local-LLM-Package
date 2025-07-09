#!/bin/bash
echo "🎨 Starting AI Safety Streamlit Frontend..."

# Create logs directory if it doesn't exist
mkdir -p logs

# Start Streamlit with logging
streamlit run backend/app/streamlit_app.py --server.port 8501 --server.address localhost --browser.gatherUsageStats false 2>&1 | tee -a logs/streamlit.log