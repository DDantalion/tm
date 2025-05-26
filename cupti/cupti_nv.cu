#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>
#include <cstring>
#include <iostream>
#include <vector>
#include <chrono>

#define CHECK_CUDA_DRV(call)                                                   \
    do {                                                                       \
        CUresult _status = call;                                               \
        if (_status != CUDA_SUCCESS) {                                         \
            const char *errstr;                                                \
            cuGetErrorString(_status, &errstr);                                \
            std::cerr << "CUDA Driver API error: " << errstr << " at line " << __LINE__ << "\n"; \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CHECK_CUDA_RT(call)                                                    \
    do {                                                                       \
        cudaError_t _status = call;                                            \
        if (_status != cudaSuccess) {                                          \
            std::cerr << "CUDA Runtime API error: " << cudaGetErrorString(_status) << " at line " << __LINE__ << "\n"; \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CHECK_CUPTI(call)                                                      \
    do {                                                                       \
        CUptiResult _status = call;                                            \
        if (_status != CUPTI_SUCCESS) {                                        \
            const char *errstr;                                                \
            cuptiGetResultString(_status, &errstr);                            \
            std::cerr << "CUPTI error: " << errstr << " at line " << __LINE__ << "\n"; \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

__global__ void dummy_kernel(char *buf, size_t size) {
    for (size_t i = 0; i < size; i += 256)
        buf[i]++;
}

int main() {
    CUdevice device;
    CUcontext context;
    int deviceNum = 0;

    CHECK_CUDA_DRV(cuInit(0));
    CHECK_CUDA_DRV(cuDeviceGet(&device, deviceNum));
    CHECK_CUDA_DRV(cuCtxCreate(&context, 0, device));
    CHECK_CUDA_DRV(cuCtxSetCurrent(context)); 

    CUpti_EventGroup eventGroup;
    CUpti_EventID eventId;
    uint64_t eventVal = 0;

    uint32_t numDomains32;
    CHECK_CUPTI(cuptiDeviceGetNumEventDomains(device, &numDomains32));

    size_t domainBufferSize = numDomains32 * sizeof(CUpti_EventDomainID);
    std::vector<CUpti_EventDomainID> domains(numDomains32);
    CHECK_CUPTI(cuptiDeviceEnumEventDomains(device, &domainBufferSize, domains.data()));

        CUpti_EventID ev;
        char name[128];
        strcpy(name, "nvlink_total_data_received");
    CHECK_CUPTI(cuptiEventGetIdFromName(device, name, &ev));
    eventId = ev;
    CHECK_CUPTI(cuptiEventGroupCreate(context, &eventGroup, 0));
    CHECK_CUPTI(cuptiEventGroupAddEvent(eventGroup, eventId));
    CHECK_CUPTI(cuptiEventGroupEnable(eventGroup));

    size_t valueSize = sizeof(uint64_t);
    CHECK_CUPTI(cuptiEventGroupReadEvent(eventGroup, CUPTI_EVENT_READ_FLAG_NONE,
                                         eventId, &valueSize, &eventVal));
    uint64_t startVal = eventVal;

    auto startTime = std::chrono::high_resolution_clock::now();

    char *buf;
    size_t size = 64 * 1024 * 1024; // 64 MB
    CHECK_CUDA_RT(cudaMalloc(&buf, size));
    dummy_kernel<<<128, 256>>>(buf, size);
    CHECK_CUDA_RT(cudaDeviceSynchronize());
    CHECK_CUDA_RT(cudaFree(buf));

    auto endTime = std::chrono::high_resolution_clock::now();

    valueSize = sizeof(uint64_t);
    CHECK_CUPTI(cuptiEventGroupReadEvent(eventGroup, CUPTI_EVENT_READ_FLAG_NONE,
                                         eventId, &valueSize, &eventVal));
    uint64_t endVal = eventVal;

    double elapsedSec = std::chrono::duration<double>(endTime - startTime).count();
    double bytes = static_cast<double>(endVal - startVal) * 32.0; // 32B per flit
    double bandwidthGBs = bytes / elapsedSec / 1e9;

    std::cout << "NVLink Received Bandwidth: " << bandwidthGBs << " GB/s\n";

    CHECK_CUPTI(cuptiEventGroupDisable(eventGroup));
    CHECK_CUPTI(cuptiEventGroupDestroy(eventGroup));
    CHECK_CUDA_DRV(cuCtxDestroy(context));

    return 0;
}
