// File: migrate_clock64.cu
#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>
#include <cstring>

#define CHECK(call)                                                         \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(_e) << '\n';  \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

//--------------------------------------------------------------------------
//  Kernel: touch the buffer 'count' times and measure the elapsed cycles
//--------------------------------------------------------------------------
__global__ void migrate_kernel(char *buf, size_t size,
                               uint64_t *elapsed_cycles, size_t ini)
{
    uint64_t start = clock64();

    for (size_t j = 0; j < 1; ++j) {
        for (size_t i = ini; i < (ini+size*2/sizeof(char)); i += 131072/sizeof(char)) {
            buf[i] += 1;
            if (buf[i] > 100) buf[i] -= 5;
        }
    }

    uint64_t end = clock64();
    *elapsed_cycles = end - start;          // write result for host
}
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

//--------------------------------------------------------------------------
//  Launch helper – returns the kernel-measured cycles
//--------------------------------------------------------------------------
static void migrate(char* buf, uint64_t* cycles, size_t size, size_t number, size_t order)
{
    // Switch to the GPU that will execute the kernel (remote)
    for(int i =1; i< number; i++){
    //     int canAccess = 0;
    // cudaSetDevice(i);                        // switch to GPU 1
    // cudaDeviceCanAccessPeer(&canAccess, i, 0);
    // if (canAccess) {
    //   cudaDeviceEnablePeerAccess(0, 0);      // allow GPU 1 → GPU 0 access
    // } else {
    //   // fall-back: will route through host (slower)
    // }
    CHECK(cudaSetDevice(i));
    migrate_kernel<<<1, 1>>>(buf, size, &cycles[i-1], (2*size*(order*(number-1) - 1 + i))/sizeof(char));
    }
    CHECK(cudaDeviceSynchronize());
    for(int i =0; i< number-1; i++){
    std::cout << "GPU" << i+1 << "Cycle: " << cycles[i] << '\t';
    float gbps = compute_bandwidth(size, cycles[i]);
    std::cout << "Estimated bandwidth: " << gbps << " GB/s" << std::endl;
    }

}


//--------------------------------------------------------------------------
//  CLI & driver loop
//--------------------------------------------------------------------------
int main(int argc, char **argv)
{
    size_t size      = 64ULL * 1024 * 1024;  // 64 MiB
    int    freq      = 100;
    size_t number = 2;
    int local_gpu = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--freq"  ) && i + 1 < argc) freq      = atoi(argv[++i]);
        if (!strcmp(argv[i], "--size"  ) && i + 1 < argc) size      = atol(argv[++i]);
        if (!strcmp(argv[i], "--local" ) && i + 1 < argc) local_gpu = atoi(argv[++i]);
        // if (!strcmp(argv[i], "--remote") && i + 1 < argc) remote_gpu= atoi(argv[++i]);
        //if (!strcmp(argv[i], "--count" ) && i + 1 < argc) count     = atol(argv[++i]);
        if (!strcmp(argv[i], "--number" ) && i + 1 < argc) number     = atol(argv[++i]);
    }
    size_t SIZE     = 4 * (number) * size * freq;  //  400 - 3200 MiB
    // Select the GPU that owns the allocation (local)
    CHECK(cudaSetDevice(local_gpu));
    char     *buf;
    uint64_t *cycles;      // unified memory so both host and either GPU can read

    CHECK(cudaMallocManaged(&buf,   SIZE));
    CHECK(cudaMallocManaged(&cycles, number * sizeof(uint64_t)));
    for (size_t i = 0; i < freq; ++i) {
        migrate(buf, cycles, size, number, i);
    }
    CHECK(cudaFree(cycles));
    CHECK(cudaFree(buf));
    return 0;
}
