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

void migrate_and_measure(int src_gpu, int dst_gpu, uint64_t &elapsed_cycles) {
    // select source GPU and allocate managed buffer + counter
    CHECK_CUDA(cudaSetDevice(src_gpu));
    char *buf = nullptr;
    uint64_t *d_elapsed = nullptr;
    CHECK_CUDA(cudaMallocManaged(&buf, BUFFER_SIZE));
    CHECK_CUDA(cudaMallocManaged(&d_elapsed, sizeof(uint64_t)));

    // initialize
    for (size_t i = 0; i < BUFFER_SIZE; ++i) buf[i] = 1;
    CHECK_CUDA(cudaDeviceSynchronize());

    // switch to destination GPU and launch
    CHECK_CUDA(cudaSetDevice(dst_gpu));
    migrate_kernel<<<1,1>>>(buf, BUFFER_SIZE, d_elapsed);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // fetch timing
    elapsed_cycles = *d_elapsed;

    // clean up
    CHECK_CUDA(cudaFree(d_elapsed));
    CHECK_CUDA(cudaFree(buf));
}

int main() {
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));

    if (device_count < 4) {
        std::cerr << "Need at least 4 GPUs for the 4-page concurrent test!" << std::endl;
        return -1;
    }

    // --- Single and two-GPU baseline tests ---
    uint64_t t0 = 0, t1 = 0;
    std::cout << "=== Single GPU Migration ===\n";
    migrate_and_measure(0, 1, t0);
    std::cout << "GPU 0 → 1: " << t0 << " cycles\n";
    migrate_and_measure(1, 0, t1);
    std::cout << "GPU 1 → 0: " << t1 << " cycles\n";

    // std::cout << "\n=== Concurrent 2-Page Migration ===\n";
    // t0 = t1 = 0;
    // std::thread th0([&]{ migrate_and_measure(0, 1, t0); });
    // std::thread th1([&]{ migrate_and_measure(1, 0, t1); });
    // th0.join();
    // th1.join();
    // std::cout << "GPU 0 → 1: " << t0 << " cycles\n";
    // std::cout << "GPU 1 → 0: " << t1 << " cycles\n";
    // std::cout << "Avg: " << ((t0 + t1) / 2) << " cycles\n";

    // --- New: Concurrent 4-Page (ring) migration over 4 GPUs ---
    std::cout << "\n=== Concurrent 4-Page Migration (0→1→2→3→0) ===\n";
    uint64_t ring_times[4] = {0,0,0,0};
    std::vector<std::thread> ring_threads;
    ring_threads.reserve(4);

    for (int i = 0; i < 4; ++i) {
        ring_threads.emplace_back(
            [i, &ring_times]() {
                int src = i;
                int dst = (i + 1) % 4;
                migrate_and_measure(src, dst, ring_times[i]);
            }
        );
    }
    for (auto &th : ring_threads) th.join();

    uint64_t sum = 0;
    for (int i = 0; i < 4; ++i) {
        int dst = (i + 1) % 4;
        std::cout << "GPU " << i << " → " << dst << ": "
                  << ring_times[i] << " cycles\n";
        sum += ring_times[i];
    }
    std::cout << "Avg: " << (sum / 4) << " cycles\n";

    return 0;
}