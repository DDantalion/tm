#!/usr/bin/env bash

# Interval between samples (seconds)
interval=1

# Previous totals (empty at first sample)
prev_tx=
prev_rx=

while true; do
    timestamp=$(date +%T)

    # Grab raw nvlink output for GPU 0
    output=$(nvidia-smi nvlink -gt d -i 0)

    # Sum Data Tx and Data Rx over all links
    sum_tx=$(echo "$output" | awk '/Data Tx/ { sum += $5 } END { print sum }')
    sum_rx=$(echo "$output" | awk '/Data Rx/ { sum += $5 } END { print sum }')

    if [[ -n $prev_tx ]]; then
        # Compute deltas (KiB transferred in the last interval)
        delta_tx=$(( sum_tx - prev_tx ))
        delta_rx=$(( sum_rx - prev_rx ))

        # Convert to MiB/s (divide KiB by 1024)
        mb_tx=$(awk -v v="$delta_tx" 'BEGIN { printf "%.2f", v/1024 }')
        mb_rx=$(awk -v v="$delta_rx" 'BEGIN { printf "%.2f", v/1024 }')

        echo "$timestamp  Tx: $delta_tx KiB/s  ($mb_tx MiB/s)  |  Rx: $delta_rx KiB/s  ($mb_rx MiB/s)"
    else
        # On first iteration just print the baseline totals
        echo "$timestamp  Initial totals: Tx: ${sum_tx} KiB  |  Rx: ${sum_rx} KiB"
    fi

    # Store for next iteration
    prev_tx=$sum_tx
    prev_rx=$sum_rx

    sleep "$interval"
done
