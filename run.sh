#!/bin/bash
datasets=('fat200' 'GSE1730_q' 'GSE10158_q')


#Change these values
fixed_args="200000 1000000 0.5 1 10 100"

timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="log_${timestamp}.txt"

# Create log file if it doesn't exist
: > "$log_file"  # This creates/empties the log file

# Function to log messages
log_message() {
    echo "$1" >> "$log_file"  # Append message to the log file
}

server_path = "./SCS"
client_path = "./client/batchClient.cpp"
client_args = "/client/query/"


for dataset_index in "${!datasets[@]}"; do
    dataset="${datasets[dataset_index]}"
    server_arg="../../data/edgeList/$dataset $fixed_args"
    client_args = "$client_args ${datasets[dataset_index]}"
    echo "$server_arg"
    echo "Started $datasets"
    log_message ">>>>>>>>>"
        log_message "Dataset: $dataset"
        log_message "Full argument: $server_arg"
        
        {
            date
            # Redirect the output of the time command to the log file
            { time $server_path $server_arg; } 2>> "$log_file"  # Append stderr to log file
            $SERVER_PID = $!
            sleep 5
            $client_path $client_args

            wait $!

            wait $SERVER_PID

            date
        } >> "$log_file"  # Append stdout to log file
        
        log_message "<<<<<<<<<<"
    done
done

log_message "Done!"
echo "Yay :) done!"