#!/bin/bash

# Find the most recent log file in the logs/ directory
LATEST_LOG=$(ls -t logs/*.log 2>/dev/null | head -n 1)

if [ -z "$LATEST_LOG" ]; then
    echo "No log files found in logs/ directory!"
    exit 1
fi

echo "Monitoring latest log file: $LATEST_LOG"
echo "Press Ctrl+C to stop."
echo "-----------------------------------"

# Follow the file
tail -f "$LATEST_LOG"
