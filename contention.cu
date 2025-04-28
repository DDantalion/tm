// File: nvlink_contention_test.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

#define CHECK_CUDA(call) \
    if ((call) != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(call) << std::endl; \
        exit(EXIT_FAILURE); \
    }

const size_t BIG_BUF_SIZE = 1L << 26; // 64MB per GPU

void migrate_and_time(int gpu_id) {
    CHECK_CUDA(cudaSetDevice(gpu_id));

    char *buf;
    CHECK_CUDA(cudaMallocManaged(&buf, BIG_BUF_SIZE));

    // Initialize data
    for (size_t i = 0; i < BIG_BUF_SIZE; i += 4096) {
        buf[i] = 42;
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    int other_gpu = 1 - gpu_id;
    CHECK_CUDA(cudaSetDevice(other_gpu));

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < BIG_BUF_SIZE; i += 4096) {
        buf[i]++;
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Migration time from GPU " << gpu_id << " to GPU " << other_gpu
              << ": " << time_ms << " ms" << std::endl;

    CHECK_CUDA(cudaFree(buf));
}

int main() {
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count < 2) {
        std::cerr << "Need at least 2 GPUs!" << std::endl;
        return -1;
    }

    std::vector<std::thread> threads;
    threads.emplace_back(migrate_and_time, 0);
    threads.emplace_back(migrate_and_time, 1);

    for (auto &t : threads) t.join();

    return 0;
}
