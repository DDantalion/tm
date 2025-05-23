#!/bin/bash

# Compile both programs
nvcc -o prog_a testmigration.cu -lpthread
#!/bin/bash

FREQS=(200)
SIZES=(65536 131072 262144 524288 1048576)  #  64KB 128KB 256KB 512KB 1MB

mkdir 2gpu
for freq in "${FREQS[@]}"; do
    for size in "${SIZES[@]}"; do
        echo "Testing freq=$freq size=$size"
        
        ./prog_a --freq $freq --size $size --local 1 --remote 0 --count 1 --mode 0 > ./2gpu/2gpugpu0a1mode0${freq}_s${size}.log
        ./prog_a --freq $freq --size $size --local 1 --remote 0 --count 1 --mode 1 > ./2gpu/2gpugpu0a1mode1${freq}_s${size}.log 
        ./prog_a --freq $freq --size $size --local 1 --remote 0 --count 1 --mode 2 > ./2gpu/2gpugpu0a1mode2${freq}_s${size}.log  

        echo "Completed freq=$freq size=$size"
    done
done

mkdir 2gpuplots
python3 plot.py ./2gpu  ./2gpuplots



cd ..
tar -czvf migration.tar.gz ./round
echo "All experiments done."