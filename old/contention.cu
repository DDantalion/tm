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

const size_t BUFFER_SIZE = 1L << 28;  // 256MB buffer

__global__ void migrate_kernel(char *buf, size_t size, uint64_t *elapsed_cycles) {
    uint64_t start = clock64();
    for (size_t i = 0; i < size; ++i) {
        buf[i] += 1;
        if (buf[i] > 100) buf[i] -=  5;  // just some work
    }
    uint64_t end = clock64();
    *elapsed_cycles = end - start;
}

void migrate_and_measure(int src_gpu, int dst_gpu, uint64_t &elapsed_ms) {
    CHECK_CUDA(cudaSetDevice(src_gpu));

    // 1) allocate managed buffer
    char *buf = nullptr;
    CHECK_CUDA(cudaMallocManaged(&buf, BUFFER_SIZE));

    // 2) allocate managed counter
    uint64_t *d_elapsed = nullptr;
    CHECK_CUDA(cudaMallocManaged(&d_elapsed, sizeof(uint64_t)));

    // init
    for (size_t i = 0; i < BUFFER_SIZE; ++i)
        buf[i] = 1;
    CHECK_CUDA(cudaDeviceSynchronize());

    // switch and launch
    CHECK_CUDA(cudaSetDevice(dst_gpu));
    migrate_kernel<<<1,1>>>(buf, BUFFER_SIZE, d_elapsed);

    // catch any launch error
    CHECK_CUDA(cudaGetLastError());

    // 3) sync so kernel finishes and writes d_elapsed
    CHECK_CUDA(cudaDeviceSynchronize());

    // pull back timing
    elapsed_ms = *d_elapsed;

    // clean up
    CHECK_CUDA(cudaFree(d_elapsed));
    CHECK_CUDA(cudaFree(buf));
}

void concurrent_migration(int gpu0_src, int gpu1_src, uint64_t &gpu0_time, uint64_t &gpu1_time) {
    uint64_t t0 = 0, t1 = 0;

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

    uint64_t single_migration_time_0 = 0, single_migration_time_1 = 0;
    uint64_t concurrent_time_0 = 0, concurrent_time_1 = 0;

    std::cout << "=== Single GPU Migration ===" << std::endl;
    migrate_and_measure(0, 1, single_migration_time_0);
    std::cout << "GPU 0 migrate to GPU 1: " << single_migration_time_0 << " cycles" << std::endl;

    migrate_and_measure(1, 0, single_migration_time_1);
    std::cout << "GPU 1 migrate to GPU 0: " << single_migration_time_1 << " cycles" << std::endl;

    std::cout << "\n=== Concurrent Migration ===" << std::endl;
    concurrent_migration(0, 1, concurrent_time_0, concurrent_time_1);

    std::cout << "Concurrent migration GPU 0 to 1: " << concurrent_time_0 << " cycles" << std::endl;
    std::cout << "Concurrent migration GPU 1 to 0: " << concurrent_time_1 << " cycles" << std::endl;

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Single Migration Avg: " << (single_migration_time_0 + single_migration_time_1) / 2 << " cycles" << std::endl;
    std::cout << "Concurrent Migration Avg: " << (concurrent_time_0 + concurrent_time_1) / 2 << " cycles" << std::endl;

    return 0;
}
