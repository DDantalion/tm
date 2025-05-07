#include <cuda_runtime.h>
#include <iostream>
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

void migrate(size_t size){
    int local_gpu = 1, remote_gpu = 0;
    CHECK(cudaSetDevice(local_gpu));

    char *buf;
    CHECK(cudaMallocManaged(&buf, size));
    //CHECK(cudaMemAdvise(buf, size, cudaMemAdviseSetPreferredLocation, remote_gpu));
    //CHECK(cudaMemAdvise(buf, size, cudaMemAdviseSetAccessedBy, local_gpu));
    CHECK(cudaSetDevice(remote_gpu)); 
    CHECK(cudaDeviceSynchronize());
    migrate_kernel<<<1, 1>>>(buf, size);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaFree(buf));
}

int main(int argc, char** argv) {
    size_t size = 64 * 1024 * 1024;
    int freq = 100;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--freq") == 0 && i + 1 < argc) freq = atoi(argv[++i]);
        if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) size = atol(argv[++i]);
    }
    for (int i = 0; i < freq; ++i) {
        migrate(size);
    }
    std::cout << "Program B: Done " << freq << " transfers of size " << size << " bytes.\n";
    return 0;
}
