#!/bin/bash

# Compile both programs
nvcc -o prog_a rdtscp_probe.cu -lpthread
nvcc -o prog_b bulkn.cu


SIZES=(65536)  #  64KB

mkdir 2logs
# Run both programs concurrently
./prog_a > ./2logs/s${size}.log &
PID_A=$!

./prog_b --size $size > /dev/null 2>&1 &
PID_B=$!

# Wait for both to finish
wait $PID_A
wait $PID_B
echo "Completed size=$size"
mkdir 2plots
python3 plot.py ./2logs  ./2plots

mkdir 3logs
# Run both programs concurrently
./prog_a > ./3logs/s${size}.log &
PID_A=$!

./prog_b --size $size > /dev/null 2>&1 &
PID_B=$!

./prog_b --size $size --local 2 --remote 1 > /dev/null 2>&1 &
PID_B1=$!

./prog_b --size $size --local 0 --remote 2 > /dev/null 2>&1 &
PID_B1=$!

# Wait for both to finish
wait $PID_A
wait $PID_B
wait $PID_B1
echo "Completed size=$size"
mkdir 3plots
python3 plot.py ./3logs  ./3plots

mkdir 4logs
# Run both programs concurrently
./prog_a > ./4logs/s${size}.log &
PID_A=$!

./prog_b --size $size > /dev/null 2>&1 &
PID_B=$!

./prog_b --size $size --local 2 --remote 1 > /dev/null 2>&1 &
PID_B1=$!

./prog_b --size $size --local 3 --remote 2 > /dev/null 2>&1 &
PID_B1=$!

./prog_b --size $size --local 0 --remote 3 > /dev/null 2>&1 &
PID_B2=$!

# Wait for both to finish
wait $PID_A
wait $PID_B
wait $PID_B1
wait $PID_B2
echo "Completed size=$size"
mkdir 4plots
python3 plot.py ./4logs  ./4plots

echo "Experiment completed."

cd ..
tar -czvf bleed.tar.gz ./bleed