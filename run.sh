#!/bin/bash

# Compile both programs
nvcc -o prog_a rdtscp_probe.cu -lpthread
nvcc -o prog_b bulk_transfer.cu

# Run both programs concurrently
./prog_a > prog_a.log &
PID_A=$!

./prog_b > prog_b.log &
PID_B=$!

# Wait for both to finish
wait $PID_A
wait $PID_B

echo "Experiment completed. Results are in prog_a.log and prog_b.log"
