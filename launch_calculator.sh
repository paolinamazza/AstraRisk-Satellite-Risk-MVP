#!/bin/bash

echo "=========================================="
echo "🛰️  Satellite Risk Calculator Launcher"
echo "=========================================="
echo ""

cd "$(dirname "$0")"

# Kill any existing server on port 5555
if lsof -Pi :5555 -sTCP:LISTEN -t >/dev/null ; then
    echo "Stopping existing server on port 5555..."
    kill $(lsof -t -i:5555) 2>/dev/null
    sleep 2
fi

echo "Starting risk calculator backend..."
python3 risk_calculator_backend.py > /dev/null 2>&1 &
BACKEND_PID=$!

sleep 3

echo "✓ Backend started (PID: $BACKEND_PID)"
echo ""
echo "Opening calculator in browser..."
open http://localhost:5555/

echo ""
echo "=========================================="
echo "✅ Calculator launched successfully!"
echo "=========================================="
echo ""
echo "Calculator URL: http://localhost:5555/"
echo ""
echo "To stop the server:"
echo "  kill $BACKEND_PID"
echo "  or: kill \$(lsof -t -i:5555)"
echo ""
echo "Press Ctrl+C to exit (server will keep running)"
echo "=========================================="
