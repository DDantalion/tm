#!/bin/bash

# Compile both programs
nvcc -o prog_a multiplek.cu -lpthread
#!/bin/bash

FREQS=(100)
NUMBERS=(2 3 4 5 6 7 8)
SIZES=(65536 131072 262144 524288 1048576)  #  64KB 128KB 256KB 512KB 1MB

mkdir final_m
for freq in "${FREQS[@]}"; do
    for size in "${SIZES[@]}"; do
        for number in "${NUMBERS[@]}"; do
            echo "Testing freq=$freq size=$size number=$number"

            ./prog_a --freq $freq --size $size --number $number > ./final_m/${freq}_s${size}_n${number}.log

            echo "Completed freq=$freq size=$size number=$number"
        done
    done
done

# mkdir 2gpuplots
# python3 plot.py ./2gpu  ./2gpuplots



cd ..
tar -czvf finalmulti.tar.gz ./final
echo "All experiments done."