#include <cuda_runtime.h>
#include <iostream>

#define BUFFER_SIZE (64 * 1024 * 1024)
#define CHECK(call) \
    if ((call) != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(call) << std::endl; \
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

int main() {
    int local_gpu = 1;
    int remote_gpu = 0;
    CHECK(cudaSetDevice(local_gpu));

    char *buf = nullptr;
    CHECK(cudaMallocManaged(&buf, BUFFER_SIZE));

    CHECK(cudaMemAdvise(buf, BUFFER_SIZE, cudaMemAdviseSetPreferredLocation, remote_gpu));
    CHECK(cudaMemAdvise(buf, BUFFER_SIZE, cudaMemAdviseSetAccessedBy, local_gpu));
    CHECK(cudaMemPrefetchAsync(buf, BUFFER_SIZE, remote_gpu));
    CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < 100; ++i) {
        migrate_kernel<<<1, 1>>>(buf, BUFFER_SIZE);
        CHECK(cudaDeviceSynchronize());
    }

    std::cout << "Program B completed remote page migrations.\n";
    CHECK(cudaFree(buf));
    return 0;
}
