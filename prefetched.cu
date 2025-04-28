#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA(call) \
    if ((call) != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(call) << std::endl; \
        exit(EXIT_FAILURE); \
    }

__global__ void remote_access_latency(uint64_t* latencies, int* remote_array, int page_size) {
    int tid = threadIdx.x;

    if (tid == 0) {
        int cacheline_size = 128 / sizeof(int);
        int elements_per_page = page_size / sizeof(int);

        int* remote_page = remote_array + elements_per_page;         // +1 page
        int* left_neighbor_page = remote_array;                      // page 0
        int* right_neighbor_page = remote_array + 2 * elements_per_page; // +2 pages

        uint64_t start, end;
        int dummy = 0;

        // 1. Warm up: 4096 accesses
        for (int i = 0; i < 4000; ++i) {
            dummy += remote_page[i];
        }
        __syncthreads();

        // 2. 100 accesses to remote page, record latency
        for (int i = 0; i < 100; ++i) {
            __syncthreads();
            start = clock64();
            dummy += remote_page[4090];
            end = clock64();
            latencies[i] = end - start;
        }
        __syncthreads();

        // 3. 100 accesses to neighbor pages, record latency
        for (int i = 0; i < 100; ++i) {
            __syncthreads();
            start = clock64();
            dummy += left_neighbor_page[4090];
            dummy += right_neighbor_page[4090];
            end = clock64();
            latencies[100 + i] = end - start;
        }
    }
}

void run_remote_latency_test(int accessing_gpu, int owning_gpu) {
    const size_t page_size = 4096;
    const int num_pages = 3;
    const int total_elements = (page_size * num_pages) / sizeof(int);

    // Set owner device
    CHECK_CUDA(cudaSetDevice(owning_gpu));

    int* d_remote_array;
    CHECK_CUDA(cudaMalloc(&d_remote_array, page_size * num_pages));
    CHECK_CUDA(cudaMemset(d_remote_array, 0, page_size * num_pages));

    // Enable peer access if needed
    CHECK_CUDA(cudaSetDevice(accessing_gpu));
    int can_access_peer = 0;
    CHECK_CUDA(cudaDeviceCanAccessPeer(&can_access_peer, accessing_gpu, owning_gpu));
    if (can_access_peer) {
        CHECK_CUDA(cudaDeviceEnablePeerAccess(owning_gpu, 0));
    } else {
        std::cerr << "Peer access not possible between GPU" << accessing_gpu << " and GPU" << owning_gpu << std::endl;
        exit(EXIT_FAILURE);
    }

    uint64_t* d_latencies;
    CHECK_CUDA(cudaMalloc(&d_latencies, sizeof(uint64_t) * 200));

    // Launch on accessing_gpu
    CHECK_CUDA(cudaSetDevice(accessing_gpu));
    remote_access_latency<<<1, 32>>>(d_latencies, d_remote_array, page_size);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy back results
    uint64_t latencies[200];
    CHECK_CUDA(cudaMemcpy(latencies, d_latencies, sizeof(latencies), cudaMemcpyDeviceToHost));

    uint64_t sum_remote = 0, sum_neighbors = 0;
    for (int i = 0; i < 100; ++i) {
        sum_remote += latencies[i];
    }
    for (int i = 100; i < 200; ++i) {
        sum_neighbors += latencies[i];
    }

    std::cout << "GPU" << accessing_gpu << " accessing GPU" << owning_gpu << "'s memory:\n";
    std::cout << "Average remote page latency: " << (double)sum_remote / 100 << " cycles\n";
    std::cout << "Average neighbor pages latency: " << (double)sum_neighbors / 100 << " cycles\n";

    // Cleanup
    CHECK_CUDA(cudaFree(d_latencies));
    CHECK_CUDA(cudaSetDevice(owning_gpu));
    CHECK_CUDA(cudaFree(d_remote_array));
}

int main() {
    int num_devices = 0;
    CHECK_CUDA(cudaGetDeviceCount(&num_devices));
    if (num_devices < 2) {
        std::cerr << "Need at least two GPUs for remote memory access.\n";
        return -1;
    }

    // GPU0 accessing GPU1's memory
    run_remote_latency_test(0, 1);

    // GPU1 accessing GPU0's memory
    run_remote_latency_test(1, 0);

    return 0;
}
