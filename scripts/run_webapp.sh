#!/usr/bin/env bash
# run_webapp.sh — Start the welding simulation web app
cd "$(dirname "$0")/.."
echo "Starting Welding Simulation Web App..."
echo "Open http://localhost:8000 in your browser"
echo "Press Ctrl+C to stop."
mkdir -p logs
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="logs/webapp_${timestamp}.log"
echo "Logging all server output to ${log_file}"
echo ""
PYTHONPATH=src .venv/bin/uvicorn welding_simulator.api.main:app --host 0.0.0.0 --port 8000 --reload 2>&1 | tee "${log_file}"
