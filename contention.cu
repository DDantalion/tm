// File: nvlink_contention_corrected_test.cu
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

const size_t BUFFER_SIZE = 1L << 26;  // 64MB buffer

void migrate_and_measure(int src_gpu, int dst_gpu, double &elapsed_ms) {
    CHECK_CUDA(cudaSetDevice(src_gpu));

    char *buf = nullptr;
    CHECK_CUDA(cudaMallocManaged(&buf, BUFFER_SIZE));

    // Initialize memory
    for (size_t i = 0; i < BUFFER_SIZE; i += 1) {
        buf[i] = 1;
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaSetDevice(dst_gpu));

    // Time the migration by accessing pages
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < BUFFER_SIZE; i += 1) {
        buf[i]++;
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    CHECK_CUDA(cudaFree(buf));
}

void concurrent_migration(int gpu0_src, int gpu1_src, double &gpu0_time, double &gpu1_time) {
    double t0 = 0, t1 = 0;

    std::thread thread0([&]() {
        migrate_and_measure(gpu0_src, gpu1_src, t0);
    });

    std::thread thread1([&]() {
        migrate_and_measure(gpu1_src, gpu0_src, t1);
    });

    thread0.join();
    thread1.join();

    gpu0_time = t0;
    gpu1_time = t1;
}

int main() {
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count < 2) {
        std::cerr << "Need at least 2 GPUs!" << std::endl;
        return -1;
    }

    double single_migration_time_0 = 0, single_migration_time_1 = 0;
    double concurrent_time_0 = 0, concurrent_time_1 = 0;

    std::cout << "=== Single GPU Migration ===" << std::endl;
    migrate_and_measure(0, 1, single_migration_time_0);
    std::cout << "GPU 0 migrate to GPU 1: " << single_migration_time_0 << " ms" << std::endl;

    migrate_and_measure(1, 0, single_migration_time_1);
    std::cout << "GPU 1 migrate to GPU 0: " << single_migration_time_1 << " ms" << std::endl;

    std::cout << "\n=== Concurrent Migration ===" << std::endl;
    concurrent_migration(0, 1, concurrent_time_0, concurrent_time_1);

    std::cout << "Concurrent migration GPU 0 to 1: " << concurrent_time_0 << " ms" << std::endl;
    std::cout << "Concurrent migration GPU 1 to 0: " << concurrent_time_1 << " ms" << std::endl;

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Single Migration Avg: " << (single_migration_time_0 + single_migration_time_1) / 2 << " ms" << std::endl;
    std::cout << "Concurrent Migration Avg: " << (concurrent_time_0 + concurrent_time_1) / 2 << " ms" << std::endl;

    return 0;
}
