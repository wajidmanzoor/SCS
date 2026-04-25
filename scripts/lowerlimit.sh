#!/bin/bash
set -e
set -u

cd /data/user/kefan/Wajid/finalSCS/totalTime/SCS || {
    echo "Failed to change to directory /data/user/kefan/Wajid/finalSCS/totalTime/SCS"
    exit 1
}

timestamp=$(date +"%Y%m%d_%H%M%S")
log_dir="./logs/exp3_l"
mkdir -p "$log_dir"
log_file="${log_dir}/log_${timestamp}.txt"
exec 1>>"$log_file" 2>&1

datasets=('ego-facebook')

# Lower bound l experiment:
# fixed h = 21, vary l from 3 to 18
ls=("3" "6" "9" "12" "15" "18")
h="21"

fixed_args="300000 100000 0.5 1"
results_dir="./results/exp3_l"
data_dir="../../../data/edgeList"

log_message() {
    echo "$(date +"%Y-%m-%d %H:%M:%S") - $1"
}

cleanup() {
    local exit_code=$?
    if [ -n "${SERVER_PID:-}" ]; then
        kill -0 "$SERVER_PID" 2>/dev/null && kill "$SERVER_PID"
    fi
    exit $exit_code
}

trap cleanup EXIT

if [ ! -x "./SCS" ]; then
    log_message "Error: Server executable './SCS' not found or not executable"
    exit 1
fi

mkdir -p "$results_dir"

for dataset in "${datasets[@]}"; do
    for l in "${ls[@]}"; do
        log_message "Processing dataset: $dataset with h = $h and l = $l"

        if [ ! -f "${data_dir}/${dataset}" ]; then
            log_message "Error: Dataset file '${data_dir}/${dataset}' not found"
            continue
        fi

        query_dir="./client/query/exp3/${dataset}"

        server_args=(
            "${data_dir}/${dataset}"
            $fixed_args
            "$h"
            "$l" "1" "1" "1" "1" "1"
            "${results_dir}/"
            "/l_${l}.txt"
            "$query_dir"
        )

        log_message "Starting server with arguments: ${server_args[*]}"

        ./SCS "${server_args[@]}" &
        SERVER_PID=$!

        wait "$SERVER_PID"
        server_exit=$?

        if [ "$server_exit" -ne 0 ]; then
            log_message "Error: Server exited with code $server_exit"
            continue
        fi

        log_message "Successfully completed processing for $dataset with h = $h and l = $l"
    done
done

log_message "Lower bound l experiment completed successfully"