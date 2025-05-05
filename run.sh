#!/bin/bash

# Compile both programs
nvcc -o prog_a rdtscp_probe.cu -lpthread
nvcc -o prog_b bulk_transfer.cu
#!/bin/bash

FREQS=(10 50 100)
SIZES=(16777216 33554432 67108864)  # 16MB, 32MB, 64MB

for freq in "${FREQS[@]}"; do
    for size in "${SIZES[@]}"; do
        echo "Testing freq=$freq size=$size"
        
        ./program_a --freq $freq --size $size > a_f${freq}_s${size}.log &
        pid_a=$!
        
        ./program_b --freq $freq --size $size > b_f${freq}_s${size}.log &
        pid_b=$!
        
        wait $pid_a
        wait $pid_b
        echo "Completed freq=$freq size=$size"
    done
done

echo "All experiments done."
