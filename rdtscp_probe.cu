#include <cuda_runtime.h>
#include <iostream>
#include <x86intrin.h>

#define TRANSFER_SIZE 256  // bytes
#define ITERATIONS 1000000
#define CHECK(call) \
    if ((call) != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(call) << std::endl; \
        exit(EXIT_FAILURE); \
    }

int main() {
    int dev0 = 0, dev1 = 1;
    CHECK(cudaSetDevice(dev0));

    char *src, *dst;
    CHECK(cudaMalloc(&src, TRANSFER_SIZE));
    CHECK(cudaSetDevice(dev1));
    CHECK(cudaMalloc(&dst, TRANSFER_SIZE));

    CHECK(cudaSetDevice(dev0));

    for (int i = 0; i < ITERATIONS; ++i) {
        unsigned int aux;
        uint64_t start = __rdtscp(&aux);
        CHECK(cudaMemcpyPeer(dst, dev1, src, dev0, TRANSFER_SIZE));
        CHECK(cudaDeviceSynchronize());
        uint64_t end = __rdtscp(&aux);
        std::cout << "Cycle: " << (end - start) << std::endl;
    }

    return 0;
}
