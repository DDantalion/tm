// File: prefetch_neighbor_latency_test.cu
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#define CHECK_CUDA(call) \
    if ((call) != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(call) << std::endl; \
        exit(EXIT_FAILURE); \
    }

const size_t PAGE_SIZE = 4096;  // 4KB per page
const int NUM_NEIGHBORS = 16;   // Number of neighboring pages to measure

__global__ void access_one_page(char *buf, size_t page_idx) {
    volatile char val = buf[page_idx * PAGE_SIZE];
}

__global__ void measure_neighbor_pages(char *buf, uint64_t *latencies, size_t base_idx, int num_neighbors) {
    int tid = threadIdx.x;
    if (tid < num_neighbors) {
        uint64_t start = clock64();
        volatile char val = buf[(base_idx + tid) * PAGE_SIZE];
        uint64_t end = clock64();
        latencies[tid] = end - start;
    }
}

void test_remote_page_and_neighbors(int owner_gpu, int remote_gpu) {
    CHECK_CUDA(cudaSetDevice(owner_gpu));

    // Allocate large enough managed memory
    char *managed_buf = nullptr;
    size_t alloc_size = PAGE_SIZE * (NUM_NEIGHBORS + 64); // allocate more to be safe
    CHECK_CUDA(cudaMallocManaged(&managed_buf, alloc_size));

    // Owner GPU initializes all pages
    for (size_t i = 0; i < alloc_size; i += PAGE_SIZE) {
        managed_buf[i] = 0;
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Switch to remote GPU
    CHECK_CUDA(cudaSetDevice(remote_gpu));

    // 1. Access a single remote page (trigger migration)
    access_one_page<<<1, 1>>>(managed_buf, 0);  // only page 0
    CHECK_CUDA(cudaDeviceSynchronize());

    // 2. Measure latency of neighbor pages
    uint64_t *d_latencies, *h_latencies;
    CHECK_CUDA(cudaMalloc(&d_latencies, sizeof(uint64_t) * NUM_NEIGHBORS));
    h_latencies = new uint64_t[NUM_NEIGHBORS];

    measure_neighbor_pages<<<1, NUM_NEIGHBORS>>>(managed_buf, d_latencies, 0, NUM_NEIGHBORS);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_latencies, d_latencies, sizeof(uint64_t) * NUM_NEIGHBORS, cudaMemcpyDeviceToHost));

    std::cout << "Neighbor Page Latencies (cycles) after remote access:\n";
    for (int i = 0; i < NUM_NEIGHBORS; ++i) {
        std::cout << "Page " << i << ": " << h_latencies[i] << " cycles" << std::endl;
    }

    // Cleanup
    CHECK_CUDA(cudaFree(managed_buf));
    CHECK_CUDA(cudaFree(d_latencies));
    delete[] h_latencies;
}

int main() {
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count < 2) {
        std::cerr << "Need at least 2 GPUs!" << std::endl;
        return -1;
    }

    test_remote_page_and_neighbors(0, 1); // GPU 0 owns memory, GPU 1 accesses
    test_remote_page_and_neighbors(1, 0); // GPU 1 owns memory, GPU 0 accesses

    return 0;
}
