#!/bin/bash

# Simple script to run the gRPC Plugin Demo

BUILD_DIR="build/debug-sanitized"
BETA_SERVER="./$BUILD_DIR/src/beta/beta_server"
ALPHA_REMOTE="./$BUILD_DIR/src/alpha/alpha_remote"
ALPHA_LOCAL="./$BUILD_DIR/src/alpha/alpha_local"

echo "------------------------------------------------"
echo "Starting Remote Demo (Alpha -> Libtwo -> Beta)"
echo "------------------------------------------------"

# Start the server in the background
$BETA_SERVER &
SERVER_PID=$!

# Give the server a moment to start
sleep 2

# Run the remote client
$ALPHA_REMOTE

# Kill the server
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null

echo ""
echo "------------------------------------------------"
echo "Starting Local Demo (Alpha -> Libone)"
echo "------------------------------------------------"

# Run the local client
$ALPHA_LOCAL

echo ""
echo "Demo Complete."
