#!/bin/bash

# Satellite Risk Assessment Dashboard Launcher
# This script starts a local web server and opens the dashboard

echo "=========================================="
echo "Satellite Risk Dashboard Launcher"
echo "=========================================="
echo ""

# Check if port 8000 is already in use
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "✓ Server already running on port 8000"
else
    echo "Starting local web server on port 8000..."
    cd "$(dirname "$0")"
    python3 -m http.server 8000 > /dev/null 2>&1 &
    SERVER_PID=$!
    echo "✓ Server started (PID: $SERVER_PID)"
    sleep 2
fi

echo ""
echo "Dashboard URL: http://localhost:8000/risk_dashboard.html"
echo ""
echo "Opening dashboard in your default browser..."
echo ""

# Open in default browser
open http://localhost:8000/risk_dashboard.html

echo "=========================================="
echo "✅ Dashboard launched successfully!"
echo "=========================================="
echo ""
echo "The dashboard should open in your browser."
echo "If it doesn't, manually visit:"
echo "  http://localhost:8000/risk_dashboard.html"
echo ""
echo "To stop the server later, run:"
echo "  kill \$(lsof -t -i:8000)"
echo ""
echo "Press Ctrl+C to exit this script (server will keep running)"
echo "=========================================="
