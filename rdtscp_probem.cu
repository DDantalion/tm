#include <cuda_runtime.h>
#include <iostream>
#include <x86intrin.h>
#include <cstring>

#define CHECK(call) \
    if ((call) != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(call) << std::endl; \
        exit(EXIT_FAILURE); \
    }

__global__ void migrate_kernel(char *buf, size_t size, size_t count) {
    for (size_t j = 0; j < count; ++j) {
        for (size_t i = 0; i < size; i += size / sizeof(char)) {
            buf[i] += 1;
            if (buf[i] > 100) buf[i] -= 5;
        }
    }
}

void migrate(size_t size, size_t local, size_t remote, size_t count){
    int local_gpu = local, remote_gpu = remote;
    CHECK(cudaSetDevice(local_gpu));

    char *buf;
    CHECK(cudaMallocManaged(&buf, size));
    //CHECK(cudaMemAdvise(buf, size, cudaMemAdviseSetPreferredLocation, remote_gpu));
    //CHECK(cudaMemAdvise(buf, size, cudaMemAdviseSetAccessedBy, local_gpu));
    CHECK(cudaSetDevice(remote_gpu)); 
    CHECK(cudaDeviceSynchronize());
    //migrate_kernel<<<1, 1>>>(buf, size, count);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaFree(buf));
}

int main(int argc, char** argv) {
    size_t size = 64 * 1024 * 1024;
    size_t count = 300;
    int freq = 100;
    int local_gpu, remote_gpu;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--freq") == 0 && i + 1 < argc) freq = atoi(argv[++i]);
        if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) size = atol(argv[++i]);
        if (strcmp(argv[i], "--local") == 0 && i + 1 < argc) local_gpu = atoi(argv[++i]);
        if (strcmp(argv[i], "--remote") == 0 && i + 1 < argc) remote_gpu = atol(argv[++i]);
        if (strcmp(argv[i], "--count") == 0 && i + 1 < argc) count = atol(argv[++i]);
    }

    for (int i = 0; i < freq; ++i) {
        unsigned aux;
        uint64_t start = __rdtscp(&aux);
        migrate(size, local_gpu, remote_gpu, count);
        CHECK(cudaDeviceSynchronize());
        uint64_t end = __rdtscp(&aux);
        std::cout << "Cycle: " << (end - start) << std::endl;
    }
    return 0;
}
