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
        
        ./prog_a --freq $freq --size $size --local 0 --remote 1  > 2gpugpu1a0${freq}_s${size}.log &
        pid_a=$!
        
        ./prog_a --freq $freq --size $size --local 1 --remote 0 > 2gpugpu0a1${freq}_s${size}.log &
        pid_a1=$! 
        
        wait $pid_a 
        wait $pid_a1 
        echo "Completed freq=$freq size=$size"
    done
done

for freq in "${FREQS[@]}"; do
    for size in "${SIZES[@]}"; do
        echo "Testing freq=$freq size=$size"
        
        ./prog_a --freq $freq --size $size --local 0 --remote 1  > 3gpugpu1a0${freq}_s${size}.log &
        pid_a=$!
        
        ./prog_a --freq $freq --size $size --local 0 --remote 2 > 3gpugpu0a1${freq}_s${size}.log &
        pid_a1=$! 
        
        wait $pid_a 
        wait $pid_a1 
        echo "Completed freq=$freq size=$size"
    done
done

for freq in "${FREQS[@]}"; do
    for size in "${SIZES[@]}"; do
        echo "Testing freq=$freq size=$size"
        
        ./prog_a --freq $freq --size $size --local 0 --remote 1  > gpu1a0${freq}_s${size}.log &
        pid_a=$!
        
        ./prog_a --freq $freq --size $size --local 1 --remote 2 > gpu2a1${freq}_s${size}.log &
        pid_a1=$! 

        ./prog_a --freq $freq --size $size --local 2 --remote 3 > gpu3a2${freq}_s${size}.log &
        pid_a2=$!

        ./prog_a --freq $freq --size $size --local 3 --remote 0 > gpu0a3${freq}_s${size}.log &
        pid_a3=$!
        
        wait $pid_a 
        wait $pid_a1 
        wait $pid_a2
        wait $pid_a3
        echo "Completed freq=$freq size=$size"
    done
done

for freq in "${FREQS[@]}"; do
    for size in "${SIZES[@]}"; do
        echo "Testing freq=$freq size=$size"
        
        ./prog_a --freq $freq --size $size --local 0 --remote 1  > baseline_gpu1a0${freq}_s${size}.log
        ./prog_a --freq $freq --size $size --local 1 --remote 2  > baseline_gpu2a1${freq}_s${size}.log
        ./prog_a --freq $freq --size $size --local 2 --remote 3  > baseline_gpu3a2${freq}_s${size}.log
        ./prog_a --freq $freq --size $size --local 3 --remote 0  > baseline_gpu0a3${freq}_s${size}.log
        echo "Completed freq=$freq size=$size"
    done
done

        ./prog_a --freq $freq --size $size --local 0 --remote 1 --count 1 > rdma_gpu1a0${freq}_s${size}.log
        ./prog_a --freq $freq --size $size --local 0 --remote 2 --count 1 > rdma_gpu2a0${freq}_s${size}.log
        ./prog_a --freq $freq --size $size --local 0 --remote 3 --count 1 > rdma_gpu3a0${freq}_s${size}.log
        ./prog_a --freq $freq --size $size --local 0 --remote 3 --count 1 > rdma_gpu4a0${freq}_s${size}.log
        ./prog_a --freq $freq --size $size --local 0 --remote 4 --count 1 > rdma_gpu5a0${freq}_s${size}.log
        ./prog_a --freq $freq --size $size --local 0 --remote 5 --count 1 > rdma_gpu6a0${freq}_s${size}.log
        ./prog_a --freq $freq --size $size --local 0 --remote 6 --count 1 > rdma_gpu7a0${freq}_s${size}.log
        ./prog_a --freq $freq --size $size --local 0 --remote 7 --count 1 > rdma_gpu8a0${freq}_s${size}.log



echo "All experiments done."