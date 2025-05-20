#!/bin/bash

# Compile both programs
nvcc -o prog_a rdtscp_probe.cu -lpthread
nvcc -o prog_b bulk_transfer.cu
#!/bin/bash

FREQS=(100)
SIZES=(65536 131072 262144 524288 1048576)  #  64KB 128KB 256KB 512KB 1MB

for freq in "${FREQS[@]}"; do
    for size in "${SIZES[@]}"; do
        echo "Testing freq=$freq size=$size"
        
        ./prog_a --freq $freq --size $size --local 0 --remote 1  > gpu1a0${freq}_s${size}.log &
        pid_a=$!
        
        ./prog_a --freq $freq --size $size --local 1 --remote 0 > gpu0a1${freq}_s${size}.log &
        pid_a1=$! 
        
        wait $pid_a 
        wait $pid_a1 
        echo "Completed freq=$freq size=$size"
    done
done

for freq in "${FREQS[@]}"; do
    for size in "${SIZES[@]}"; do
        echo "Testing freq=$freq size=$size"
        
        ./prog_a --freq $freq --size $size --local 0 --remote 1  > baseline_gpu1a0${freq}_s${size}.log
        ./prog_a --freq $freq --size $size --local 1 --remote 0  > baseline_gpu0a1${freq}_s${size}.log
        echo "Completed freq=$freq size=$size"
    done
done

echo "All experiments done."