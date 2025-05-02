// File: host_migration_parallelism_test.cu
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

const size_t MEDIUM_BUF_SIZE = 1L << 24; // 16MB

void migrate_one(int gpu_id, bool warmup) {
    CHECK_CUDA(cudaSetDevice(gpu_id));

    char *buf;
    CHECK_CUDA(cudaMallocManaged(&buf, MEDIUM_BUF_SIZE));

    for (size_t i = 0; i < MEDIUM_BUF_SIZE; i += 4096) {
        buf[i] = 1;
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    if (warmup) {
        volatile char temp = buf[0];
    }

    int other_gpu = 1 - gpu_id;
    CHECK_CUDA(cudaSetDevice(other_gpu));
    for (size_t i = 0; i < MEDIUM_BUF_SIZE; i += 4096) {
        buf[i]++;
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaFree(buf));
}

int main() {
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count < 2) {
        std::cerr << "Need at least 2 GPUs!" << std::endl;
        return -1;
    }

    // SERIAL migration
    auto start_serial = std::chrono::high_resolution_clock::now();
    migrate_one(0, true);
    migrate_one(1, true);
    auto end_serial = std::chrono::high_resolution_clock::now();
    double serial_time = std::chrono::duration<double, std::milli>(end_serial - start_serial).count();

    // PARALLEL migration
    auto start_parallel = std::chrono::high_resolution_clock::now();
    std::thread t1(migrate_one, 0, true);
    std::thread t2(migrate_one, 1, true);
    t1.join();
    t2.join();
    auto end_parallel = std::chrono::high_resolution_clock::now();
    double parallel_time = std::chrono::duration<double, std::milli>(end_parallel - start_parallel).count();

    std::cout << "Serial migration time: " << serial_time << " ms" << std::endl;
    std::cout << "Parallel migration time: " << parallel_time << " ms" << std::endl;

    return 0;
}
