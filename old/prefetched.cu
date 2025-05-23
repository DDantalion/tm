#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA(call) \
    if ((call) != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(call) << std::endl; \
        exit(EXIT_FAILURE); \
    }

__global__ void remote_access_latency(uint64_t* latencies, int* remote_array, int page_size) {

        int cacheline_size = 128 / sizeof(int);
        int elements_per_page = page_size / sizeof(int);

        int* remote_page = remote_array + elements_per_page;         // +1 page
        int* left_neighbor_page = remote_array;                      // page 0
        int* right_neighbor_page = remote_array + 2 * elements_per_page; // +2 pages

        uint64_t start, end;
        int dummy = 0;

        // 1. 访问同一个page600次，NVIDIA使用的是access-counterbased migration method，threshold=256
        for (int i = 0; i < 600; ++i) {
            dummy += remote_page[i];
            if(dummy>100000){
                latencies[i] -=5167;
            }
        }
        // 2. 一次access未被访问过的cacheline
        for (int i = 0; i < 1; ++i) {
            start = clock64();
            dummy += remote_page[800];
            end = clock64();
            latencies[i] = end - start;
            if(dummy>10000000){
                latencies[i] -=568487;
            }
        }
        for (int i = 0; i < 1; ++i) {
            start = clock64();
            dummy += remote_page[800];
            end = clock64();
            latencies[i+3] = end - start;
            if(dummy>10000000){
                latencies[i+3] -=568487;
            }
        }
        // 3. 一次access访问相邻的page，查看是否在DRAM或CACHE
        for (int i = 0; i < 1; ++i) {
            start = clock64();
            dummy += right_neighbor_page[1];
            end = clock64();
            latencies[1 + i] = end - start;
            if(dummy>1000000000){
                latencies[i] -=56898465;
            }
        }
        for (int i = 0; i < 1; ++i) {
            start = clock64();
            dummy += right_neighbor_page[1];
            end = clock64();
            latencies[2 + i] = end - start;
            if(dummy>1000000000){
                latencies[i] -=56898465;
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
    remote_access_latency<<<1, 1>>>(d_latencies, d_remote_array, page_size);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy back results
    uint64_t latencies[3];
    CHECK_CUDA(cudaMemcpy(latencies, d_latencies, sizeof(latencies), cudaMemcpyDeviceToHost));

    uint64_t sum_remote = 0, sum_neighbors = 0;
    for (int i = 0; i < 1; ++i) {
        sum_remote += latencies[i];
    }
    for (int i = 1; i < 2; ++i) {
        sum_neighbors += latencies[i];
    }

    std::cout << "GPU" << accessing_gpu << " accessing GPU" << owning_gpu << "'s memory:\n";
    std::cout << "Average remote page latency: " << (double)sum_remote / 1 << " cycles\n";
    std::cout << "Average cached remote page latency: " << (double)latencies[3]<< " cycles\n";
    std::cout << "Average neighbor pages latency: " << (double)sum_neighbors / 1 << " cycles\n";
    std::cout << "Average cached neighbor pages latency: " << (double)latencies[2]<< " cycles\n";
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
