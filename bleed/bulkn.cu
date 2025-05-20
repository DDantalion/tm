#include <cuda_runtime.h>
#include <iostream>
#include <x86intrin.h>

#define CHECK(call) \
    if ((call) != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(call) << std::endl; \
        exit(EXIT_FAILURE); \
    }

int main(int argc, char **argv) {
    int dev0 = 1, dev1 = 0;
    size_t TRANSFER_SIZE = 256;
    size_t count = 200;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) TRANSFER_SIZE = atol(argv[++i]);
        if (strcmp(argv[i], "--count") == 0 && i + 1 < argc) count = atol(argv[++i]);
        if (!strcmp(argv[i], "--local" ) && i + 1 < argc) dev0 = atoi(argv[++i]);
        if (!strcmp(argv[i], "--remote") && i + 1 < argc) dev1 = atoi(argv[++i]);
    }
    CHECK(cudaSetDevice(dev0));

    char *src, *dst;
    CHECK(cudaMalloc(&src, TRANSFER_SIZE));
    CHECK(cudaSetDevice(dev1));
    CHECK(cudaMalloc(&dst, TRANSFER_SIZE));

    CHECK(cudaSetDevice(dev0));
    for (int i = 0; i < count; ++i) {
        unsigned int aux;
        uint64_t start = __rdtscp(&aux);
        CHECK(cudaMemcpyPeer(dst, dev1, src, dev0, TRANSFER_SIZE));
        CHECK(cudaDeviceSynchronize());
        uint64_t end = __rdtscp(&aux);
        std::cout << "Cycle: " << (end - start) << std::endl;
    }

    return 0;
}
