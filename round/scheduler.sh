#!/bin/bash

# Compile both programs
nvcc -o prog_a rdtscp_probe.cu -lpthread
nvcc -o prog_b bulk_transfer.cu
#!/bin/bash

FREQS=(200)
SIZES=(65536 131072 262144 524288 1048576)  #  64KB 128KB 256KB 512KB 1MB

mkdir 2gpu
for freq in "${FREQS[@]}"; do
    for size in "${SIZES[@]}"; do
        echo "Testing freq=$freq size=$size"
        
        ./prog_a --freq $freq --size $size --local 0 --remote 1 --count 1  > ./2gpu/2gpugpu1a0${freq}_s${size}.log &
        pid_a=$!
        
        ./prog_a --freq $freq --size $size --local 1 --remote 0 --count 1 > ./2gpu/2gpugpu0a1${freq}_s${size}.log &
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
        
        ./prog_a --freq $freq --size $size --local 0 --remote 1 --count 1  > ./3gpu/3gpugpu1a0${freq}_s${size}.log &
        pid_a=$!
        
        ./prog_a --freq $freq --size $size --local 1 --remote 2 --count 1 > ./3gpu/3gpugpu2a1${freq}_s${size}.log &
        pid_a1=$! 

        ./prog_a --freq $freq --size $size --local 2 --remote 0 --count 1 > ./3gpu/3gpugpu0a2${freq}_s${size}.log &
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
        
        ./prog_a --freq $freq --size $size --local 0 --remote 1 --count 1  > ./4gpu/gpu1a0${freq}_s${size}.log &
        pid_a=$!
        
        ./prog_a --freq $freq --size $size --local 1 --remote 2 --count 1 > ./4gpu/gpu2a1${freq}_s${size}.log &
        pid_a1=$! 

        ./prog_a --freq $freq --size $size --local 2 --remote 3 --count 1 > ./4gpu/gpu3a2${freq}_s${size}.log &
        pid_a2=$!

        ./prog_a --freq $freq --size $size --local 3 --remote 0 --count 1 > ./4gpu/gpu0a3${freq}_s${size}.log &
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
        
        ./prog_a --freq $freq --size $size --local 0 --remote 1 --count 1  > ./5gpu/gpu1a0${freq}_s${size}.log &
        pid_a=$!
        
        ./prog_a --freq $freq --size $size --local 1 --remote 2 --count 1 > ./5gpu/gpu2a1${freq}_s${size}.log &
        pid_a1=$! 

        ./prog_a --freq $freq --size $size --local 2 --remote 3 --count 1 > ./5gpu/gpu3a2${freq}_s${size}.log &
        pid_a2=$!

        ./prog_a --freq $freq --size $size --local 3 --remote 4 --count 1 > ./5gpu/gpu4a3${freq}_s${size}.log &
        pid_a3=$!

        ./prog_a --freq $freq --size $size --local 4 --remote 0 --count 1 > ./5gpu/gpu0a4${freq}_s${size}.log &
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
        
        ./prog_a --freq $freq --size $size --local 0 --remote 1 --count 1  > ./6gpu/gpu1a0${freq}_s${size}.log &
        pid_a=$!
        
        ./prog_a --freq $freq --size $size --local 1 --remote 2 --count 1 > ./6gpu/gpu2a1${freq}_s${size}.log &
        pid_a1=$! 

        ./prog_a --freq $freq --size $size --local 2 --remote 3 --count 1 > ./6gpu/gpu3a2${freq}_s${size}.log &
        pid_a2=$!

        ./prog_a --freq $freq --size $size --local 3 --remote 4 --count 1 > ./6gpu/gpu4a3${freq}_s${size}.log &
        pid_a3=$!

        ./prog_a --freq $freq --size $size --local 4 --remote 5 --count 1 > ./6gpu/gpu5a4${freq}_s${size}.log &
        pid_a4=$!

        ./prog_a --freq $freq --size $size --local 5 --remote 0 --count 1 > ./6gpu/gpu0a5${freq}_s${size}.log &
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
        
        ./prog_a --freq $freq --size $size --local 0 --remote 1 --count 1  > ./7gpu/gpu1a0${freq}_s${size}.log &
        pid_a=$!
        
        ./prog_a --freq $freq --size $size --local 1 --remote 2 --count 1 > ./7gpu/gpu2a1${freq}_s${size}.log &
        pid_a1=$! 

        ./prog_a --freq $freq --size $size --local 2 --remote 3 --count 1 > ./7gpu/gpu3a2${freq}_s${size}.log &
        pid_a2=$!

        ./prog_a --freq $freq --size $size --local 3 --remote 4 --count 1 > ./7gpu/gpu4a3${freq}_s${size}.log &
        pid_a3=$!

        ./prog_a --freq $freq --size $size --local 4 --remote 5 --count 1 > ./7gpu/gpu5a4${freq}_s${size}.log &
        pid_a4=$!

        ./prog_a --freq $freq --size $size --local 5 --remote 6 --count 1 > ./7gpu/gpu6a5${freq}_s${size}.log &
        pid_a5=$!

        ./prog_a --freq $freq --size $size --local 6 --remote 0 --count 1 > ./7gpu/gpu0a6${freq}_s${size}.log &
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
        
        ./prog_a --freq $freq --size $size --local 0 --remote 1 --count 1  > ./8gpu/gpu1a0${freq}_s${size}.log &
        pid_a=$!
        
        ./prog_a --freq $freq --size $size --local 1 --remote 2 --count 1 > ./8gpu/gpu2a1${freq}_s${size}.log &
        pid_a1=$! 

        ./prog_a --freq $freq --size $size --local 2 --remote 3 --count 1 > ./8gpu/gpu3a2${freq}_s${size}.log &
        pid_a2=$!

        ./prog_a --freq $freq --size $size --local 3 --remote 4 --count 1 > ./8gpu/gpu4a3${freq}_s${size}.log &
        pid_a3=$!

        ./prog_a --freq $freq --size $size --local 4 --remote 5 --count 1 > ./8gpu/gpu5a4${freq}_s${size}.log &
        pid_a4=$!

        ./prog_a --freq $freq --size $size --local 5 --remote 6 --count 1 > ./8gpu/gpu6a5${freq}_s${size}.log &
        pid_a5=$!

        ./prog_a --freq $freq --size $size --local 6 --remote 7 --count 1 > ./8gpu/gpu7a6${freq}_s${size}.log &
        pid_a6=$!

        ./prog_a --freq $freq --size $size --local 7 --remote 0 --count 1 > ./8gpu/gpu0a7${freq}_s${size}.log &
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


cd ..
tar -czvf v100r.tar.gz ./tm
echo "All experiments done."