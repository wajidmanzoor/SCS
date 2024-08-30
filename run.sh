#!/bin/bash

args=(
    "../../../data/100.txt 3 6 2 10000000 1 2 1000000 0.5"
    "../../../data/100.txt 6 9 2 10000000 1 2 1000000 0.5"
    "../../../data/100.txt 9 12 2 10000000 1 2 1000000 0.5"
    "../../../data/100.txt 12 15 2 10000000 1 2 1000000 0.5"
    "../../../data/100.txt 15 18 2 10000000 1 2 1000000 0.5"
    "../../../data/100.txt 18 21 2 10000000 1 2 1000000 0.5"
    "../../../data/100.txt 21 23 2 10000000 1 2 1000000 0.5"
    "../../../data/hepPH_SCS 3 6 2 10000000 1 2 1000000 0.5"
    "../../../data/hepPH_SCS 6 9 2 10000000 1 2 1000000 0.5"
    "../../../data/hepPH_SCS 9 12 2 10000000 1 2 1000000 0.5"
    "../../../data/hepPH_SCS 12 15 2 10000000 1 2 1000000 0.5"
    "../../../data/hepPH_SCS 15 18 2 10000000 1 2 1000000 0.5"
    "../../../data/hepPH_SCS 18 21 2 10000000 1 2 1000000 0.5"
    "../../../data/hepPH_SCS 21 23 2 10000000 1 2 1000000 0.5"



)

output_file="results.txt"

if [ ! -e "$output_file" ]; then
    touch "$output_file"
fi

total=${#args[@]}
current=0

show_progress() {
    local progress=$((current * 100 / total))
    local done=$((progress / 2))
    local left=$((50 - done))
    printf "\rProgress: ["
    printf "%${done}s" | tr ' ' '='
    printf "%${left}s" | tr ' ' ' '
    printf "] %d%%" $progress
}

for arg in "${args[@]}"
do
    ./SCS $arg >> $output_file 2>&1
    current=$((current + 1))
    show_progress
done

printf "\nDone!\n"

