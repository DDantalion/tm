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
__global__ void kernel0(char *buf, size_t size,
                               size_t count, uint64_t *elapsed_cycles)
{
    uint64_t start = clock64();

    for (size_t j = 0; j < 1; ++j) {
        for (size_t i = 0; i < size; i += 4096/sizeof(char)) {
            buf[i] += 1;
            if (buf[i] > 100) buf[i] -= 5;
        }
    }

    uint64_t end = clock64();
    *elapsed_cycles = end - start;          // write result for host
}

__global__ void kernel1(char *buf, size_t size,
                               size_t count, uint64_t *elapsed_cycles)
{
    for (size_t j = 0; j < 1; ++j) {
        for (size_t i = 0; i < size; i += 4096/sizeof(char)) {
            buf[i] += 1;
            if (buf[i] > 100) buf[i] -= 5;
        }
    }
    uint64_t start = clock64();
    for (size_t j = 0; j < 1; ++j) {
        for (size_t i = 0; i < size; i += 4096/sizeof(char)) {
            buf[i] += 1;
            if (buf[i] > 100) buf[i] -= 5;
        }
    }
    uint64_t end = clock64();
    *elapsed_cycles = end - start;          // write result for host
}

__global__ void kernel2(char *buf, size_t size,
                               size_t count, uint64_t *elapsed_cycles)
{
    for (size_t j = 0; j < 300; ++j) {
        for (size_t i = 0; i < size; i += 4096/sizeof(char)) {
            buf[i] += 1;
            if (buf[i] > 100) buf[i] -= 5;
        }
    }
    uint64_t start = clock64();
    for (size_t j = 0; j < 1; ++j) {
        for (size_t i = 0; i < size; i += 4096/sizeof(char)) {
            buf[i] += 1;
            if (buf[i] > 100) buf[i] -= 5;
        }
    }
    uint64_t end = clock64();
    *elapsed_cycles = end - start;          // write result for host
}
//--------------------------------------------------------------------------
//  Launch helper – returns the kernel-measured cycles
//--------------------------------------------------------------------------
static uint64_t migrate(size_t size,
                        int     local_gpu,
                        int     remote_gpu,
                        size_t  count,
                        size_t mode)
{
    // Select the GPU that owns the allocation (local)
    CHECK(cudaSetDevice(local_gpu));

    char     *buf;
    uint64_t *cycles;      // unified memory so both host and either GPU can read

    CHECK(cudaMallocManaged(&buf,    size));
    CHECK(cudaMalloc(&cycles, sizeof(uint64_t)));

    // Switch to the GPU that will execute the kernel (remote)
    //CHECK(cudaSetDevice(remote_gpu));
    switch (mode){
        case 0: 
        kernel0<<<1, 1>>>(buf, size, count, cycles);
        break;
        case 1:
        kernel1<<<1, 1>>>(buf, size, count, cycles);
        break;
        case 2:
        kernel2<<<1, 1>>>(buf, size, count, cycles);
        break;
        default:
        std::cout<<"Illegal Mode!"<<'\n';
        break;
    }
    CHECK(cudaDeviceSynchronize());

    //uint64_t result = *cycles;
    uint64_t result;
    // copy from device to host
    CHECK(cudaMemcpy(&result,
                 cycles,
                 sizeof(uint64_t),
                 cudaMemcpyDeviceToHost));
// now `result` holds the kernel’s elapsed_cycles

    CHECK(cudaFree(cycles));
    CHECK(cudaFree(buf));
    return result;
}

//--------------------------------------------------------------------------
//  CLI & driver loop
//--------------------------------------------------------------------------
int main(int argc, char **argv)
{
    size_t size      = 64ULL * 1024 * 1024;  // 64 MiB
    size_t count     = 300;
    int    freq      = 100;
    int    local_gpu = 0;
    int    remote_gpu= 1;
    int mode = 0;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--freq"  ) && i + 1 < argc) freq      = atoi(argv[++i]);
        if (!strcmp(argv[i], "--size"  ) && i + 1 < argc) size      = atol(argv[++i]);
        if (!strcmp(argv[i], "--local" ) && i + 1 < argc) local_gpu = atoi(argv[++i]);
        if (!strcmp(argv[i], "--remote") && i + 1 < argc) remote_gpu= atoi(argv[++i]);
        if (!strcmp(argv[i], "--count" ) && i + 1 < argc) count     = atol(argv[++i]);
        if (!strcmp(argv[i], "--mode" ) && i + 1 < argc) mode     = atol(argv[++i]);
    }

    for (int i = 0; i < freq; ++i) {
        uint64_t cycles = migrate(size, local_gpu, remote_gpu, count, mode);
        std::cout << "Cycle: " << cycles << '\n';
    }
    return 0;
}
