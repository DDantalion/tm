#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

#define CHECK_CUDA(call) \
    if ((call) != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(call) << std::endl; \
        exit(EXIT_FAILURE); \
    }

__global__ void access_latency_test(uint64_t* latencies, int* array, int page_size, int iterations) {
    int tid = threadIdx.x;

    // Assume thread 0 does the measurement
    if (tid == 0) {
        int cacheline_size = 128 / sizeof(int);  // Assume 128-byte cacheline
        int elements_per_page = page_size / sizeof(int);

        // Pointers to remote page and neighbor pages
        int* remote_page = array + elements_per_page;         // base + 1 page
        int* left_neighbor_page = array;                      // base page
        int* right_neighbor_page = array + 2 * elements_per_page; // base + 2 pages

        uint64_t start, end;
        int dummy = 0;

        // 1. Access remote cacheline 260 times (warming)
        int* target_cacheline = remote_page;
        for (int i = 0; i < 260; ++i) {
            dummy += target_cacheline[0];
        }
        __syncthreads();

        // 2. Access the same cacheline 100 times, record latency
        for (int i = 0; i < 100; ++i) {
            __syncthreads();
            start = clock64();
            dummy += target_cacheline[0];
            end = clock64();
            latencies[i] = end - start;
        }
        __syncthreads();

        // 3. Access left and right neighbor cachelines 100 times, record latency
        for (int i = 0; i < 100; ++i) {
            __syncthreads();
            start = clock64();
            dummy += left_neighbor_page[0];
            dummy += right_neighbor_page[0];
            end = clock64();
            latencies[100 + i] = end - start;
        }
    }
}

void run_latency_test() {
    const size_t page_size = 4096;  // 4KB page
    const int num_pages = 3;        // left, remote, right
    const int total_elements = (page_size * num_pages) / sizeof(int);

    int* d_array;
    CHECK_CUDA(cudaMalloc(&d_array, page_size * num_pages));
    CHECK_CUDA(cudaMemset(d_array, 0, page_size * num_pages));

    uint64_t* d_latencies;
    CHECK_CUDA(cudaMalloc(&d_latencies, sizeof(uint64_t) * 200));  // 100 + 100 samples

    access_latency_test<<<1, 32>>>(d_latencies, d_array, page_size, 100);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy back and analyze
    uint64_t latencies[200];
    CHECK_CUDA(cudaMemcpy(latencies, d_latencies, sizeof(latencies), cudaMemcpyDeviceToHost));

    // Compute and print average latencies
    uint64_t sum_remote = 0, sum_neighbors = 0;
    for (int i = 0; i < 100; ++i) {
        sum_remote += latencies[i];
    }
    for (int i = 100; i < 200; ++i) {
        sum_neighbors += latencies[i];
    }

    std::cout << "Average latency for remote page cacheline access: " 
              << (double)sum_remote / 100 << " cycles" << std::endl;
    std::cout << "Average latency for neighbor pages cacheline access: "
              << (double)sum_neighbors / 100 << " cycles" << std::endl;

    CHECK_CUDA(cudaFree(d_array));
    CHECK_CUDA(cudaFree(d_latencies));
}

int main() {
    run_latency_test();
    return 0;
}
