// File: cache_latency_test.cu
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

#define CHECK_CUDA(call) \
    if ((call) != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(call) << std::endl; \
        exit(EXIT_FAILURE); \
    }

__global__ void measure_latency_kernel(uint64_t *latencies, int *array, int stride, int iterations) {
    int tid = threadIdx.x;
    uint64_t start, end;

    __shared__ uint64_t s_latency;

    int index = tid * stride;

    for (int i = 0; i < iterations; ++i) {
        __syncthreads();
        start = clock64();
        array[index] += 1;  // Access memory
        end = clock64();
        latencies[i] = (end - start);
        __syncthreads();
    }
}

void run_test(size_t array_size_bytes, int stride_elements, int iterations, const char* label) {
    std::cout << "=== " << label << " ===" << std::endl;
    std::cout << "Array size: " << array_size_bytes / 1024 << " KB, stride: " << stride_elements << " elements" << std::endl;

    int *d_array;
    uint64_t *d_latencies, *h_latencies;

    CHECK_CUDA(cudaMalloc(&d_array, array_size_bytes));
    CHECK_CUDA(cudaMemset(d_array, 0, array_size_bytes));

    CHECK_CUDA(cudaMalloc(&d_latencies, iterations * sizeof(uint64_t)));
    h_latencies = new uint64_t[iterations];

    measure_latency_kernel<<<1, 1>>>(d_latencies, d_array, stride_elements, iterations);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_latencies, d_latencies, iterations * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    double total_latency = 0;
    for (int i = 0; i < iterations; ++i) {
        total_latency += h_latencies[i];
    }

    std::cout << "Average latency: " << (total_latency / iterations) << " cycles" << std::endl;
    std::cout << std::endl;

    cudaFree(d_array);
    cudaFree(d_latencies);
    delete[] h_latencies;
}

int main() {
    CHECK_CUDA(cudaSetDevice(0));

    const int iterations = 1000;

    // L1 Cache test (<10MB, should fully fit in SM L1 caches)
    run_test(32 * 1024, 1, iterations, "L1 Cache Test"); // 32 KB

    // L2 Cache test (should fit within 60MB L2)
    run_test(48 * 1024 * 1024, 32, iterations, "L2 Cache Test"); // 48 MB

    // DRAM test (force beyond 60MB cache into DRAM)
    run_test(256 * 1024 * 1024, 256, iterations, "DRAM Test"); // 256 MB

    return 0;
}