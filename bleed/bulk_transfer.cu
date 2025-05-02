#include <cuda_runtime.h>
#include <iostream>

#define TRANSFER_SIZE (16 * 1024 * 1024)
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
    for (int i = 0; i < 100; ++i) {
        CHECK(cudaMemcpyPeer(dst, dev1, src, dev0, TRANSFER_SIZE));
        CHECK(cudaDeviceSynchronize());
    }

    std::cout << "Program B: Completed 100 transfers of 16MB." << std::endl;
    return 0;
}
