#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cstring>
#include <x86intrin.h>

#define CHECK(call)                                                         \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(_e) << '\n'; \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

float compute_bandwidth(size_t data_bytes, unsigned long long clock_cycles, int device_id = 0) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    // clockRate is in kHz, convert to Hz
    float clock_freq_hz = prop.clockRate * 1000.0f;

    // Bandwidth in bytes per second
    float bandwidth_bps = (data_bytes * clock_freq_hz) / clock_cycles;

    // Convert to GB/s
    float bandwidth_gbps = bandwidth_bps / (1024.0f * 1024.0f * 1024.0f);

    return bandwidth_gbps;
}


int main(int argc, char **argv) {
    int dev_src = 0, dev_dst = 1, number = 2;
    size_t size = 64ULL * 1024 * 1024;  // 64 MiB
    int freq = 100;

    // Parse command line
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--freq") && i + 1 < argc) freq = atoi(argv[++i]);
        if (!strcmp(argv[i], "--size") && i + 1 < argc) size = atol(argv[++i]);
        if (!strcmp(argv[i], "--src") && i + 1 < argc) dev_src = atoi(argv[++i]);
        if (!strcmp(argv[i], "--dst") && i + 1 < argc) dev_dst = atoi(argv[++i]);
        if (!strcmp(argv[i], "--number") && i + 1 < argc) number = atoi(argv[++i]);
    }
for (int i = 0; i < freq; ++i) {
    for (int j =1; j < number; j++){
    dev_dst = j;
    // Enable peer access
    CHECK(cudaSetDevice(dev_src));
    CHECK(cudaDeviceEnablePeerAccess(dev_dst, 0));

    CHECK(cudaSetDevice(dev_dst));
    CHECK(cudaDeviceEnablePeerAccess(dev_src, 0));
    // Allocate buffers
    CHECK(cudaSetDevice(dev_src));
    char *src_buf;
    CHECK(cudaMalloc(&src_buf, size));

    CHECK(cudaSetDevice(dev_dst));
    char *dst_buf;
    CHECK(cudaMalloc(&dst_buf, size));
        unsigned int aux;
        uint64_t start = __rdtscp(&aux);
        CHECK(cudaMemcpyPeer(dst_buf, dev_dst, src_buf, dev_src, size));
        CHECK(cudaDeviceSynchronize());
        uint64_t end = __rdtscp(&aux);
        std::cout << "GPU" << j << "Cycle: " << (end - start) << '\t';
        float gbps = compute_bandwidth(size, (end - start));
        std::cout << "Estimated bandwidth: " << gbps << " GB/s" << std::endl;
    // Cleanup
    CHECK(cudaFree(src_buf));
    CHECK(cudaFree(dst_buf));
    }
}
    return 0;
}
