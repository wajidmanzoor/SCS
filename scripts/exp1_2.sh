#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status
set -u  # Treat unset variables as errors

# Change to the required directory first
cd /data/user/kefan/Wajid/finalSCS/totalTime/SCS || {
    echo "Failed to change to directory /data/user/kefan/Wajid/finalSCS/totalTime/SCS"
    exit 1
}


# Redirect all output to log file
timestamp=$(date +"%Y%m%d_%H%M%S")
log_dir="./logs/exp1_2"
log_file="${log_dir}/log_${timestamp}.txt"
exec 1>>"$log_file" 2>&1



# Configuration
datasets=( 'ego-facebook')

fixed_args="300000 100000 0.5 1 8 100 1 1 1 1 1"
results_dir="./results/exp1_2"
data_dir="../../../data/edgeList"


# Function to log messages with timestamps
log_message() {
    echo "$(date +"%Y-%m-%d %H:%M:%S") - $1"
}

# Function to cleanup on script exit
cleanup() {
    local exit_code=$?
    if [ -n "${SERVER_PID:-}" ]; then
        kill -0 $SERVER_PID 2>/dev/null && kill $SERVER_PID
    fi
    exit $exit_code
}

trap cleanup EXIT

# Validate executables exist
if [ ! -x "./SCS" ]; then
    log_message "Error: Server executable './SCS' not found or not executable"
    exit 1
fi

for dataset in "${datasets[@]}"; do
    
    log_message "Processing dataset: $dataset"
        
    if [ ! -f "${data_dir}/${dataset}" ]; then
        log_message "Error: Dataset file '${data_dir}/${dataset}' not found"
        continue
    fi
    
    query_dir="./client/query/exp1_2/${dataset}"
    
    
    server_args="${data_dir}/${dataset} $fixed_args ${results_dir}/ .txt $query_dir"

    log_message "Starting server with arguments: ${server_args[*]}"
    ./SCS $server_args &
    SERVER_PID=$!

    wait $SERVER_PID
    server_exit=$?
    
    if [ $server_exit -ne 0 ]; then
        log_message "Error: Server exited with code $server_exit"
        continue
    fi
    
    log_message "Successfully completed processing for $dataset"

done

log_message "Script completed successfully"