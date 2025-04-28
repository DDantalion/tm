// File: prefetch_latency_test.cu
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#define CHECK_CUDA(call) \
    if ((call) != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(call) << std::endl; \
        exit(EXIT_FAILURE); \
    }

const size_t PAGE_SIZE = 4096;  // 4 KB
const size_t NUM_PAGES = 1024;  // total 4MB

__global__ void touch_page(char *buf, int stride) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < NUM_PAGES) {
        volatile char val = buf[idx * stride];
    }
}

void test_prefetch_latency(int src_gpu, int dst_gpu) {
    CHECK_CUDA(cudaSetDevice(src_gpu));

    char *managed_buf = nullptr;
    CHECK_CUDA(cudaMallocManaged(&managed_buf, NUM_PAGES * PAGE_SIZE));

    // Touch the pages first on src_gpu to populate
    touch_page<<<(NUM_PAGES + 255) / 256, 256>>>(managed_buf, PAGE_SIZE);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Now switch to dst_gpu to test latency
    CHECK_CUDA(cudaSetDevice(dst_gpu));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Warm up
    volatile char dummy = managed_buf[0];
    CHECK_CUDA(cudaDeviceSynchronize());

    // Measure access latency
    auto start = std::chrono::high_resolution_clock::now();
    touch_page<<<(NUM_PAGES + 255) / 256, 256>>>(managed_buf, PAGE_SIZE);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    double duration_us = std::chrono::duration<double, std::micro>(end - start).count();
    std::cout << "Latency accessing neighbor pages from GPU " << dst_gpu
              << ": " << duration_us / NUM_PAGES << " us per page" << std::endl;

    CHECK_CUDA(cudaFree(managed_buf));
}

int main() {
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count < 2) {
        std::cerr << "Need at least 2 GPUs!" << std::endl;
        return -1;
    }

    test_prefetch_latency(0, 1); // Access from GPU1 pages originally owned by GPU0
    test_prefetch_latency(1, 0); // Access from GPU0 pages originally owned by GPU1
    return 0;
}
