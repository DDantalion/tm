#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>

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

    // CUDA driver initialization and context setup
    CHECK_CUDA_DRV(cuInit(0));
    CHECK_CUDA_DRV(cuDeviceGet(&device, deviceNum));
    CHECK_CUDA_DRV(cuCtxCreate(&context, 0, device));
    CHECK_CUDA_DRV(cuCtxSetCurrent(context)); // REQUIRED for CUPTI to access event domains

    // CUPTI variables
    CUpti_EventGroup eventGroup;
    CUpti_EventID eventId;
    uint64_t eventVal = 0;

    // Get domain IDs
    uint32_t numDomains = 0;
    CHECK_CUPTI(cuptiDeviceGetNumEventDomains(device, &numDomains));

    std::vector<CUpti_EventDomainID> domains(numDomains);
    size_t sizeDomains = numDomains;
    CHECK_CUPTI(cuptiDeviceEnumEventDomains(device, &sizeDomains, domains.data()));

    // Search for "nvlink_total_data_received"
bool found = false;
for (auto domain : domains) {
    uint32_t numEvents = 0;
    size_t attrSize = sizeof(numEvents);
    CUptiResult attrStatus = cuptiEventDomainGetAttribute(
        domain,
        CUPTI_EVENT_DOMAIN_ATTR_TOTAL_EVENTS,
        &attrSize,
        &numEvents);

    if (attrStatus != CUPTI_SUCCESS || numEvents == 0) continue;

    std::vector<CUpti_EventID> events(numEvents);
    size_t sizeEvents = numEvents;
    CHECK_CUPTI(cuptiEventDomainEnumEvents(domain, &sizeEvents, events.data()));

    for (auto ev : events) {
        char name[128];
        size_t len = sizeof(name);
        CHECK_CUPTI(cuptiEventGetAttribute(ev, CUPTI_EVENT_ATTR_NAME, &len, name));

        if (std::string(name) == "nvlink_total_data_received") {
            eventId = ev;
            found = true;
            break;
        }
    }

    if (found) break;
}
    if (!found) {
        std::cerr << "CUPTI event 'nvlink_total_data_received' not found.\n";
        return -1;
    }

    // Create and enable event group
    CHECK_CUPTI(cuptiEventGroupCreate(context, &eventGroup, 0));
    CHECK_CUPTI(cuptiEventGroupAddEvent(eventGroup, eventId));
    CHECK_CUPTI(cuptiEventGroupEnable(eventGroup));

    // Read event counter before workload
    size_t valueSize = sizeof(uint64_t);
    CHECK_CUPTI(cuptiEventGroupReadEvent(eventGroup, CUPTI_EVENT_READ_FLAG_NONE,
                                         eventId, &valueSize, &eventVal));
    uint64_t startVal = eventVal;

    auto startTime = std::chrono::high_resolution_clock::now();

    // Launch kernel to generate traffic
    char *buf;
    size_t dataSize = 64 * 1024 * 1024; // 64 MB
    CHECK_CUDA_RT(cudaMalloc(&buf, dataSize));
    dummy_kernel<<<128, 256>>>(buf, dataSize);
    CHECK_CUDA_RT(cudaDeviceSynchronize());
    CHECK_CUDA_RT(cudaFree(buf));

    auto endTime = std::chrono::high_resolution_clock::now();

    // Read event counter after workload
    valueSize = sizeof(uint64_t);
    CHECK_CUPTI(cuptiEventGroupReadEvent(eventGroup, CUPTI_EVENT_READ_FLAG_NONE,
                                         eventId, &valueSize, &eventVal));
    uint64_t endVal = eventVal;

    // Compute bandwidth
    double elapsedSec = std::chrono::duration<double>(endTime - startTime).count();
    double bytes = static_cast<double>(endVal - startVal) * 32.0; // 32B per flit
    double bandwidthGBs = bytes / elapsedSec / 1e9;

    std::cout << "NVLink Received Bandwidth: " << bandwidthGBs << " GB/s\n";

    // Cleanup
    CHECK_CUPTI(cuptiEventGroupDisable(eventGroup));
    CHECK_CUPTI(cuptiEventGroupDestroy(eventGroup));
    CHECK_CUDA_DRV(cuCtxDestroy(context));

    return 0;
}
