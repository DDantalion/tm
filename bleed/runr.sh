#!/bin/bash

# Compile both programs
nvcc -o prog_a rdtscp_probe.cu -lpthread
nvcc -o prog_b bulkn.cu


SIZES=(65536 131072 262144 524288 1048576)  #  64KB 128KB 256KB 512KB 1MB

mkdir 2logs
# Run both programs concurrently
for size in "${SIZES[@]}"; do
./prog_b --size $size --local 1 --remote 0 > ./2logs/0a1${size}.log &
PID_A=$!

./prog_b --size $size --local 0 --remote 1  > ./2logs/1a0${size}.log  &
PID_B=$!

# Wait for both to finish
wait $PID_A
wait $PID_B
echo "Completed size=$size"
done
mkdir 2plots
python3 plot.py ./2logs  ./2plots

mkdir 3logs
# Run both programs concurrently
for size in "${SIZES[@]}"; do
./prog_b --size $size --local 1 --remote 0 > ./3logs/0a1${size}.log &
PID_A=$!

./prog_b --size $size --local 2 --remote 1 > ./3logs/1a2${size}.log &
PID_B=$!

./prog_b --size $size --local 0 --remote 2 > ./3logs/2a0${size}.log &
PID_B1=$!


# Wait for both to finish
wait $PID_A
wait $PID_B
wait $PID_B1
echo "Completed size=$size"
done
mkdir 3plots
python3 plot.py ./3logs  ./3plots

mkdir 4logs
# Run both programs concurrently
for size in "${SIZES[@]}"; do
./prog_b --size $size --local 1 --remote 0 > ./4logs/0a1${size}.log  &
PID_A=$!

./prog_b --size $size --local 2 --remote 1 > ./4logs/1a2${size}.log  &
PID_B=$!

./prog_b --size $size --local 3 --remote 2 > ./4logs/2a3${size}.log  &
PID_B1=$!

./prog_b --size $size --local 0 --remote 3 > ./4logs/3a0${size}.log  &
PID_B2=$!


# Wait for both to finish
wait $PID_A
wait $PID_B
wait $PID_B1
wait $PID_B2
echo "Completed size=$size"
done
mkdir 4plots
python3 plot.py ./4logs  ./4plots

mkdir 5logs
# Run both programs concurrently
for size in "${SIZES[@]}"; do
./prog_b --size $size --local 1 --remote 0 > ./5logs/0a1${size}.log  &
PID_A=$!

./prog_b --size $size --local 2 --remote 1 > ./5logs/1a2${size}.log  &
PID_B=$!

./prog_b --size $size --local 3 --remote 2 > ./5logs/2a3${size}.log  &
PID_B1=$!

./prog_b --size $size --local 4 --remote 3 > ./5logs/3a4${size}.log  &
PID_B2=$!

./prog_b --size $size --local 0 --remote 4 > ./5logs/4a0${size}.log  &
PID_B3=$!

# Wait for both to finish
wait $PID_A
wait $PID_B
wait $PID_B1
wait $PID_B2
wait $PID_B3
echo "Completed size=$size"
done
mkdir 5plots
python3 plot.py ./5logs  ./5plots

mkdir 6logs
# Run both programs concurrently
for size in "${SIZES[@]}"; do
./prog_b --size $size --local 1 --remote 0 > ./6logs/0a1${size}.log  &
PID_A=$!

./prog_b --size $size --local 2 --remote 1 > ./6logs/1a2${size}.log  &
PID_B=$!

./prog_b --size $size --local 3 --remote 2 > ./6logs/2a3${size}.log  &
PID_B1=$!

./prog_b --size $size --local 4 --remote 3 > ./6logs/3a4${size}.log  &
PID_B2=$!

./prog_b --size $size --local 5 --remote 4 > ./6logs/4a5${size}.log  &
PID_B3=$!

./prog_b --size $size --local 0 --remote 5 > ./6logs/5a0${size}.log  &
PID_B4=$!

# Wait for both to finish
wait $PID_A
wait $PID_B
wait $PID_B1
wait $PID_B2
wait $PID_B3
wait $PID_B4
echo "Completed size=$size"
done
mkdir 6plots
python3 plot.py ./6logs  ./6plots

mkdir 7logs
# Run both programs concurrently
for size in "${SIZES[@]}"; do
./prog_b --size $size --local 1 --remote 0 > ./7logs/0a1${size}.log  &
PID_A=$!

./prog_b --size $size --local 2 --remote 1 > ./7logs/1a2${size}.log  &
PID_B=$!

./prog_b --size $size --local 3 --remote 2 > ./7logs/2a3${size}.log  &
PID_B1=$!

./prog_b --size $size --local 4 --remote 3 > ./7logs/3a4${size}.log  &
PID_B2=$!

./prog_b --size $size --local 5 --remote 4 > ./7logs/4a5${size}.log  &
PID_B3=$!

./prog_b --size $size --local 6 --remote 5 > ./7logs/5a6${size}.log  &
PID_B4=$!

./prog_b --size $size --local 0 --remote 6 > ./7logs/6a0${size}.log  &
PID_B5=$!

# Wait for both to finish
wait $PID_A
wait $PID_B
wait $PID_B1
wait $PID_B2
wait $PID_B3
wait $PID_B4
wait $PID_B5
echo "Completed size=$size"
done
mkdir 7plots
python3 plot.py ./7logs  ./7plots

mkdir 8logs
# Run both programs concurrently
for size in "${SIZES[@]}"; do
./prog_b --size $size --local 1 --remote 0 > ./8logs/0a1${size}.log  &
PID_A=$!

./prog_b --size $size --local 2 --remote 1 > ./8logs/1a2${size}.log  &
PID_B=$!

./prog_b --size $size --local 3 --remote 2 > ./8logs/2a3${size}.log  &
PID_B1=$!

./prog_b --size $size --local 4 --remote 3 > ./8logs/3a4${size}.log  &
PID_B2=$!

./prog_b --size $size --local 5 --remote 4 > ./8logs/4a5${size}.log  &
PID_B3=$!

./prog_b --size $size --local 6 --remote 5 > ./8logs/5a6${size}.log  &
PID_B4=$!

./prog_b --size $size --local 7 --remote 6 > ./8logs/6a7${size}.log  &
PID_B5=$!

./prog_b --size $size --local 0 --remote 7 > ./8logs/7a0${size}.log  &
PID_B6=$!

# Wait for both to finish
wait $PID_A
wait $PID_B
wait $PID_B1
wait $PID_B2
wait $PID_B3
wait $PID_B4
wait $PID_B5
wait $PID_B6
echo "Completed size=$size"
done
mkdir 8plots
python3 plot.py ./8logs  ./8plots

mkdir blogs
# Run both programs concurrently
for size in "${SIZES[@]}"; do
./prog_b --size $size --local 1 --remote 0 > ./blogs/0a1${size}.log  

./prog_b --size $size --local 2 --remote 1 > ./blogs/1a2${size}.log  

./prog_b --size $size --local 3 --remote 2 > ./blogs/2a3${size}.log  

./prog_b --size $size --local 4 --remote 3 > ./blogs/3a4${size}.log  

./prog_b --size $size --local 5 --remote 4 > ./blogs/4a5${size}.log  

./prog_b --size $size --local 6 --remote 5 > ./blogs/5a6${size}.log  

./prog_b --size $size --local 7 --remote 6 > ./blogs/6a7${size}.log  

./prog_b --size $size --local 0 --remote 7 > ./blogs/7a0${size}.log  

echo "Completed size=$size"
done
mkdir baselineplots
python3 plot.py ./blogs  ./baselineplots



echo "Experiment completed."

cd ..
tar -czvf bleed.tar.gz ./bleed