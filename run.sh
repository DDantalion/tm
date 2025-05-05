#!/bin/bash

# Compile both programs
nvcc -o prog_a rdtscp_probe.cu -lpthread
nvcc -o prog_b bulk_transfer.cu
#!/bin/bash

FREQS=(10 100 1000)
SIZES=(67108864 134217728 268435456)  # 64MB, 128MB, 256MB

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
