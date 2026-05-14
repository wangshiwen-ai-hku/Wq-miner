#!/usr/bin/env bash

# This script forcefully kills any running instance of the Alpha Miner agent.
# Use this if Ctrl+C is stuck or suspended.

echo "🛑 Stopping Evolutionary Alpha Miner..."
# Use pkill to find and kill the python process running run_agent.py
pkill -9 -f "python run_agent.py"

if [ $? -eq 0 ]; then
    echo "✅ Agent successfully terminated."
else
    echo "⚠️ No running agent found."
fi
