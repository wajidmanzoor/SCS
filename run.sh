#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

datasets=('fat200' 'GSE1730_q' 'GSE10158_q' 'orani' 'ego-facebook' 'grqc_q')
fixed_args="1000 1000 0.5 1 10 100"
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="./logs/log_${timestamp}.txt"

# Create log file if it doesn't exist
mkdir -p ./logs
: > "$log_file"  # This creates/empties the log file

# Function to log messages
log_message() {
    echo "$(date +"%Y-%m-%d %H:%M:%S") - $1" >> "$log_file"
}

# Ensure we're in the correct directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

# Export any necessary environment variables
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$script_dir/lib"

server_path="./SCS"
client_path="./client/batch"
client_prefix="./client/query/maxdeg/"

for dataset in "${datasets[@]}"; do
    server_arg="../../data/edgeList/$dataset $fixed_args"
    client_args="$client_prefix$dataset"

    echo "Processing dataset: $dataset"
    log_message "Processing dataset: $dataset"

    {
        log_message "Starting server"
        $server_path $server_arg &
        SERVER_PID=$!

        # Wait for the server to initialize
        sleep 10

        # Check if server is still running
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            log_message "Error: Server process died unexpectedly"
            exit 1
        fi

        log_message "Starting client"
        timeout 300s $client_path $client_args
        CLIENT_EXIT_CODE=$?

        if [ $CLIENT_EXIT_CODE -eq 124 ]; then
            log_message "Error: Client process timed out after 5 minutes"
            kill $SERVER_PID
            exit 1
        elif [ $CLIENT_EXIT_CODE -ne 0 ]; then
            log_message "Error: Client process failed with exit code $CLIENT_EXIT_CODE"
            kill $SERVER_PID
            exit 1
        fi

        log_message "Client process completed successfully"
        log_message "Waiting for server to finish"
        wait $SERVER_PID
        SERVER_EXIT_CODE=$?

        if [ $SERVER_EXIT_CODE -ne 0 ]; then
            log_message "Error: Server process exited with code $SERVER_EXIT_CODE"
            exit 1
        fi

        log_message "Server process completed successfully"

    } >> "$log_file" 2>&1

    log_message "Finished processing dataset: $dataset"
done

#New version

echo "All datasets processed successfully"
log_message "All datasets processed successfully"