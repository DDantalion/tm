#include <cuda_runtime.h>
#include <iostream>
#include <x86intrin.h>
#include <cstring>

#define CHECK(call) \
    if ((call) != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(call) << std::endl; \
        exit(EXIT_FAILURE); \
    }

__global__ void migrate_kernel(char *buf, size_t size) {
    for (size_t j = 0; j < 600; ++j) {
        for (size_t i = 0; i < size; i += size / sizeof(char)) {
            buf[i] += 1;
            if (buf[i] > 100) buf[i] -= 5;
        }
    }
}


int main(int argc, char** argv) {
    size_t size = 256 * 1024 * 1024;
    int freq = 10000;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--freq") == 0 && i + 1 < argc) freq = atoi(argv[++i]);
        if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) size = atol(argv[++i]);
    }

    int local_gpu = 0, remote_gpu = 1;
    CHECK(cudaSetDevice(local_gpu));

    char *buf;
    CHECK(cudaMallocManaged(&buf, size));
    CHECK(cudaMemAdvise(buf, size, cudaMemAdviseSetPreferredLocation, remote_gpu));
    CHECK(cudaMemAdvise(buf, size, cudaMemAdviseSetAccessedBy, local_gpu));
    CHECK(cudaMemPrefetchAsync(buf, size, remote_gpu));
    CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < freq; ++i) {
        unsigned aux;
        uint64_t start = __rdtscp(&aux);
        migrate_kernel<<<1, 1>>>(buf, size);
        CHECK(cudaDeviceSynchronize());
        uint64_t end = __rdtscp(&aux);
        std::cout << "Cycle: " << (end - start) << std::endl;
    }

    CHECK(cudaFree(buf));
    return 0;
}
