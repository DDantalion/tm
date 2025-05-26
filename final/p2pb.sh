#!/bin/bash

# Compile both programs
nvcc -o prog_a p2pb.cu -lpthread
#!/bin/bash

FREQS=(100)
SIZES=(65536 1048576 33554432 1073741824 34359738368)  #  64KB 1MB 32MB 1024MB 32GB
NUMBERS=(2)

mkdir p2p
for freq in "${FREQS[@]}"; do
    for size in "${SIZES[@]}"; do
        for number in "${NUMBERS[@]}"; do
            echo "Testing freq=$freq size=$size number=$number"

            ./prog_a --freq $freq --size $size --number $number > ./p2p/${freq}_s${size}_n${number}.log

            echo "Completed freq=$freq size=$size number=$number"
        done
    done
done

mkdir plots
python3 batchhis.py ./p2p  ./plots



cd ..
tar -czvf p2pb.tar.gz ./final
echo "All experiments done."