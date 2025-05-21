#!/bin/bash

# Compile both programs
nvcc -o prog_a rdtscp_probem.cu -lpthread
nvcc -o prog_b bulk_transfer.cu
#!/bin/bash

FREQS=(200)
SIZES=(65536 131072 262144 524288 1048576)  #  64KB 128KB 256KB 512KB 1MB

mkdir 2gpu
for freq in "${FREQS[@]}"; do
    for size in "${SIZES[@]}"; do
        echo "Testing freq=$freq size=$size"
        
        ./prog_a --freq $freq --size $size --local 1 --remote 0  > ./2gpu/2gpugpu0a1${freq}_s${size}.log &
        pid_a=$!
        
        ./prog_a --freq $freq --size $size --local 0 --remote 1 > ./2gpu/2gpugpu1a0${freq}_s${size}.log &
        pid_a1=$! 
        
        wait $pid_a 
        wait $pid_a1 
        echo "Completed freq=$freq size=$size"
    done
done

mkdir 2gpuplots
python3 plot.py ./2gpu  ./2gpuplots

mkdir 3gpu
for freq in "${FREQS[@]}"; do
    for size in "${SIZES[@]}"; do
        echo "Testing freq=$freq size=$size"
        
        ./prog_a --freq $freq --size $size --local 1 --remote 0  > ./3gpu/3gpugpu0a1${freq}_s${size}.log &
        pid_a=$!
        
        ./prog_a --freq $freq --size $size --local 0 --remote 1 > ./3gpu/3gpugpu1a0${freq}_s${size}.log &
        pid_a1=$! 

        ./prog_a --freq $freq --size $size --local 0 --remote 2 > ./3gpu/3gpugpu2a0${freq}_s${size}.log &
        pid_a2=$! 
        
        wait $pid_a 
        wait $pid_a1 
        wait $pid_a2 
        echo "Completed freq=$freq size=$size"
    done
done

mkdir 3gpuplots
python3 plot.py ./3gpu  ./3gpuplots

mkdir 4gpu
for freq in "${FREQS[@]}"; do
    for size in "${SIZES[@]}"; do
        echo "Testing freq=$freq size=$size"
        
        ./prog_a --freq $freq --size $size --local 1 --remote 0  > ./4gpu/gpu0a1${freq}_s${size}.log &
        pid_a=$!
        
        ./prog_a --freq $freq --size $size --local 0 --remote 1 > ./4gpu/gpu1a0${freq}_s${size}.log &
        pid_a1=$! 

        ./prog_a --freq $freq --size $size --local 0 --remote 2 > ./4gpu/gpu2a0${freq}_s${size}.log &
        pid_a2=$!

        ./prog_a --freq $freq --size $size --local 0 --remote 3 > ./4gpu/gpu3a0${freq}_s${size}.log &
        pid_a3=$!
        
        wait $pid_a 
        wait $pid_a1 
        wait $pid_a2
        wait $pid_a3
        echo "Completed freq=$freq size=$size"
    done
done

mkdir 4gpuplots
python3 plot.py ./4gpu  ./4gpuplots

mkdir 5gpu
for freq in "${FREQS[@]}"; do
    for size in "${SIZES[@]}"; do
        echo "Testing freq=$freq size=$size"
        
        ./prog_a --freq $freq --size $size --local 1 --remote 0  > ./5gpu/gpu0a1${freq}_s${size}.log &
        pid_a=$!
        
        ./prog_a --freq $freq --size $size --local 0 --remote 1 > ./5gpu/gpu1a0${freq}_s${size}.log &
        pid_a1=$! 

        ./prog_a --freq $freq --size $size --local 0 --remote 2 > ./5gpu/gpu2a0${freq}_s${size}.log &
        pid_a2=$!

        ./prog_a --freq $freq --size $size --local 0 --remote 3 > ./5gpu/gpu3a0${freq}_s${size}.log &
        pid_a3=$!

        ./prog_a --freq $freq --size $size --local 0 --remote 4 > ./5gpu/gpu4a0${freq}_s${size}.log &
        pid_a4=$!
        
        wait $pid_a 
        wait $pid_a1 
        wait $pid_a2
        wait $pid_a3
        wait $pid_a4
        echo "Completed freq=$freq size=$size"
    done
done

mkdir 5gpuplots
python3 plot.py ./5gpu  ./5gpuplots

mkdir 6gpu
for freq in "${FREQS[@]}"; do
    for size in "${SIZES[@]}"; do
        echo "Testing freq=$freq size=$size"
        
        ./prog_a --freq $freq --size $size --local 1 --remote 0  > ./6gpu/gpu0a1${freq}_s${size}.log &
        pid_a=$!
        
        ./prog_a --freq $freq --size $size --local 0 --remote 1 > ./6gpu/gpu1a0${freq}_s${size}.log &
        pid_a1=$! 

        ./prog_a --freq $freq --size $size --local 0 --remote 2 > ./6gpu/gpu2a0${freq}_s${size}.log &
        pid_a2=$!

        ./prog_a --freq $freq --size $size --local 0 --remote 3 > ./6gpu/gpu3a0${freq}_s${size}.log &
        pid_a3=$!

        ./prog_a --freq $freq --size $size --local 0 --remote 4 > ./6gpu/gpu4a0${freq}_s${size}.log &
        pid_a4=$!

        ./prog_a --freq $freq --size $size --local 0 --remote 5 > ./6gpu/gpu5a0${freq}_s${size}.log &
        pid_a5=$!

        wait $pid_a 
        wait $pid_a1 
        wait $pid_a2
        wait $pid_a3
        wait $pid_a4
        wait $pid_a5
        echo "Completed freq=$freq size=$size"
    done
done

mkdir 6gpuplots
python3 plot.py ./6gpu  ./6gpuplots

mkdir 7gpu
for freq in "${FREQS[@]}"; do
    for size in "${SIZES[@]}"; do
        echo "Testing freq=$freq size=$size"
        
        ./prog_a --freq $freq --size $size --local 1 --remote 0  > ./7gpu/gpu0a1${freq}_s${size}.log &
        pid_a=$!
        
        ./prog_a --freq $freq --size $size --local 0 --remote 1 > ./7gpu/gpu1a0${freq}_s${size}.log &
        pid_a1=$! 

        ./prog_a --freq $freq --size $size --local 0 --remote 2 > ./7gpu/gpu2a0${freq}_s${size}.log &
        pid_a2=$!

        ./prog_a --freq $freq --size $size --local 0 --remote 3 > ./7gpu/gpu3a0${freq}_s${size}.log &
        pid_a3=$!

        ./prog_a --freq $freq --size $size --local 0 --remote 4 > ./7gpu/gpu4a0${freq}_s${size}.log &
        pid_a4=$!

        ./prog_a --freq $freq --size $size --local 0 --remote 5 > ./7gpu/gpu5a0${freq}_s${size}.log &
        pid_a5=$!

        ./prog_a --freq $freq --size $size --local 0 --remote 6 > ./7gpu/gpu6a0${freq}_s${size}.log &
        pid_a6=$!

        wait $pid_a 
        wait $pid_a1 
        wait $pid_a2
        wait $pid_a3
        wait $pid_a4
        wait $pid_a5
        wait $pid_a6
        echo "Completed freq=$freq size=$size"
    done
done

mkdir 7gpuplots
python3 plot.py ./7gpu  ./7gpuplots

mkdir 8gpu
for freq in "${FREQS[@]}"; do
    for size in "${SIZES[@]}"; do
        echo "Testing freq=$freq size=$size"
        
        ./prog_a --freq $freq --size $size --local 1 --remote 0  > ./8gpu/gpu0a1${freq}_s${size}.log &
        pid_a=$!
        
        ./prog_a --freq $freq --size $size --local 0 --remote 1 > ./8gpu/gpu1a0${freq}_s${size}.log &
        pid_a1=$! 

        ./prog_a --freq $freq --size $size --local 0 --remote 2 > ./8gpu/gpu2a0${freq}_s${size}.log &
        pid_a2=$!

        ./prog_a --freq $freq --size $size --local 0 --remote 3 > ./8gpu/gpu3a0${freq}_s${size}.log &
        pid_a3=$!

        ./prog_a --freq $freq --size $size --local 0 --remote 4 > ./8gpu/gpu4a0${freq}_s${size}.log &
        pid_a4=$!

        ./prog_a --freq $freq --size $size --local 0 --remote 5 > ./8gpu/gpu5a0${freq}_s${size}.log &
        pid_a5=$!

        ./prog_a --freq $freq --size $size --local 0 --remote 6 > ./8gpu/gpu6a0${freq}_s${size}.log &
        pid_a6=$!

        ./prog_a --freq $freq --size $size --local 0 --remote 7 > ./8gpu/gpu7a0${freq}_s${size}.log &
        pid_a7=$!

        wait $pid_a 
        wait $pid_a1 
        wait $pid_a2
        wait $pid_a3
        wait $pid_a4
        wait $pid_a5
        wait $pid_a6
        wait $pid_a7
        echo "Completed freq=$freq size=$size"
    done
done

mkdir 8gpuplots
python3 plot.py ./8gpu  ./8gpuplots

mkdir baseline
for freq in "${FREQS[@]}"; do
    for size in "${SIZES[@]}"; do
        echo "Testing freq=$freq size=$size"
        ./prog_a --freq $freq --size $size --local 1 --remote 0  > ./baseline/baseline_gpu0a1${freq}_s${size}.log
        ./prog_a --freq $freq --size $size --local 0 --remote 1  > ./baseline/baseline_gpu1a0${freq}_s${size}.log
        ./prog_a --freq $freq --size $size --local 0 --remote 2  > ./baseline/baseline_gpu2a0${freq}_s${size}.log
        ./prog_a --freq $freq --size $size --local 0 --remote 3  > ./baseline/baseline_gpu3a0${freq}_s${size}.log
        ./prog_a --freq $freq --size $size --local 0 --remote 4  > ./baseline/baseline_gpu4a0${freq}_s${size}.log
        ./prog_a --freq $freq --size $size --local 0 --remote 5  > ./baseline/baseline_gpu5a0${freq}_s${size}.log
        ./prog_a --freq $freq --size $size --local 0 --remote 6  > ./baseline/baseline_gpu6a0${freq}_s${size}.log
        ./prog_a --freq $freq --size $size --local 0 --remote 7  > ./baseline/baseline_gpu7a0${freq}_s${size}.log
        echo "Completed freq=$freq size=$size"
    done
done

mkdir baselinegpuplots
python3 plot.py ./baseline  ./baselinegpuplots



cd ..
tar -czvf schedulec.tar.gz ./tm
echo "All experiments done."