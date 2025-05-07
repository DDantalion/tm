#!/bin/bash

# Compile both programs
nvcc -o prog_a rdtscp_probe.cu -lpthread
nvcc -o prog_b bulk_transfer.cu
#!/bin/bash

FREQS=(50 100 200)
SIZES=(65536 131072 262144 524288 1048576)  #  64KB 128KB 256KB 512KB 1MB

for freq in "${FREQS[@]}"; do
    for size in "${SIZES[@]}"; do
        echo "Testing freq=$freq size=$size"
        
        ./prog_a --freq $freq --size $size > a_f${freq}_s${size}.log &
        pid_a=$!
        
        ./prog_b --freq $freq --size $size > b_f${freq}_s${size}.log &
        pid_b=$!
        
        wait $pid_a
        wait $pid_b
        echo "Completed freq=$freq size=$size"
    done
done

for freq in "${FREQS[@]}"; do
    for size in "${SIZES[@]}"; do
        echo "Testing freq=$freq size=$size"
        
        ./prog_a --freq $freq --size $size > baseline_a_f${freq}_s${size}.log
        echo "Completed freq=$freq size=$size"
    done
done

echo "All experiments done."
