#!/bin/bash

# Compile both programs
nvcc -o prog_a rdtscp_probe.cu -lpthread
nvcc -o prog_b bulkn.cu


SIZES=(65536 131072 262144 524288 1048576)  #  64KB 128KB 256KB 512KB 1MB

mkdir logs
# Run both programs concurrently
for size in "${SIZES[@]}"; do
./prog_a > ./logs/s${size}.log &
PID_A=$!

./prog_b > /dev/null 2>&1 &
PID_B=$!

# Wait for both to finish
wait $PID_A
wait $PID_B
echo "Completed size=$size"
done


echo "Experiment completed. Results are in prog_a.log and prog_b.log"

mkdir plots
python3 plot.py ./logs  ./plots

cd ..
tar -czvf bleed.tar.gz ./bleed