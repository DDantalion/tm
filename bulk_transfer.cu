#include <cuda_runtime.h>
#include <iostream>

#define BUFFER_SIZE (64 * 1024 * 1024)  // 64 MB
#define CHECK(call) \
    if ((call) != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(call) << std::endl; \
        exit(EXIT_FAILURE); \
    }

__global__ void migrate_kernel(char *buf, size_t size) {
    for (size_t j = 0; j < 600; j++) {
        for (size_t i = 0; i < size; i += size / sizeof(char)) {
            buf[i] += 1;
            if (buf[i] > 100) buf[i] -= 5;
        }
    }
}

int main() {
    int dev0 = 1;  // Could be the reverse for stress testing
    CHECK(cudaSetDevice(dev0));

    char *buf;
    CHECK(cudaMallocManaged(&buf, BUFFER_SIZE));  // Unified memory for peer access

    for (int i = 0; i < 100; ++i) {
        migrate_kernel<<<1, 1>>>(buf, BUFFER_SIZE);
        CHECK(cudaDeviceSynchronize());
    }

    std::cout << "Program B: Completed 100 kernel-based page migrations." << std::endl;
    CHECK(cudaFree(buf));
    return 0;
}
