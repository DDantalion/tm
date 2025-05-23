#include <cupti.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

#define CHECK_CUPTI(call)                                                        \
    do {                                                                         \
        CUptiResult _status = call;                                              \
        if (_status != CUPTI_SUCCESS) {                                          \
            const char *errstr;                                                  \
            cuptiGetResultString(_status, &errstr);                              \
            std::cerr << "CUPTI error: " << errstr << " at " << __LINE__ << "\n";\
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

#define CHECK_CUDA(call)                                                  \
    do {                                                                  \
        cudaError_t _status = call;                                       \
        if (_status != cudaSuccess) {                                     \
            std::cerr << "CUDA error: " << cudaGetErrorString(_status);  \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

// Dummy kernel to create traffic
__global__ void dummy_kernel(char *buf, size_t size) {
    for (int i = 0; i < size; i += 256)
        buf[i]++;
}

int main() {
    CUdevice device;
    CUcontext context;
    int deviceNum = 0;

    // Initialize CUDA & CUPTI
    CHECK_CUDA(cudaSetDevice(deviceNum));
    CHECK_CUDA(cudaFree(0)); // Initializes context
    CHECK_CUPTI(cuInit(0));
    CHECK_CUPTI(cuDeviceGet(&device, deviceNum));
    CHECK_CUPTI(cuCtxCreate(&context, 0, device));

    CUpti_EventGroup eventGroup;
    CUpti_EventID eventId;
    uint64_t eventVal = 0;

    // Find NVLINK counter event
    CUpti_EventDomainID domainId;
    uint32_t numDomains;
    CHECK_CUPTI(cuptiDeviceGetNumEventDomains(device, &numDomains));
    std::vector<CUpti_EventDomainID> domains(numDomains);
    CHECK_CUPTI(cuptiDeviceEnumEventDomains(device, &numDomains, domains.data()));

    // Search for NVLink domain and event
    bool found = false;
    for (auto domain : domains) {
        uint32_t numEvents;
        CHECK_CUPTI(cuptiEventDomainGetNumEvents(domain, &numEvents));
        std::vector<CUpti_EventID> events(numEvents);
        CHECK_CUPTI(cuptiEventDomainEnumEvents(domain, &numEvents, events.data()));

        for (auto ev : events) {
            char name[128];
            size_t len = sizeof(name);
            CHECK_CUPTI(cuptiEventGetAttribute(ev, CUPTI_EVENT_ATTR_NAME, &len, name));
            if (std::string(name) == "nvlink_total_data_received") {
                eventId = ev;
                domainId = domain;
                found = true;
                break;
            }
        }
        if (found) break;
    }

    if (!found) {
        std::cerr << "nvlink_total_data_received not found.\n";
        return -1;
    }

    // Create event group
    CHECK_CUPTI(cuptiEventGroupCreate(context, &eventGroup, 0));
    CHECK_CUPTI(cuptiEventGroupAddEvent(eventGroup, eventId));

    // Enable group and read before
    CHECK_CUPTI(cuptiEventGroupEnable(eventGroup));
    CHECK_CUPTI(cuptiEventGroupReadEvent(eventGroup, CUPTI_EVENT_READ_FLAG_NONE,
                                         eventId, &eventVal));
    uint64_t startVal = eventVal;

    auto start = std::chrono::high_resolution_clock::now();

    // Launch dummy workload to create NVLink traffic
    size_t size = 1 << 26; // 64 MB
    char *devBuf;
    CHECK_CUDA(cudaMalloc(&devBuf, size));
    dummy_kernel<<<128, 256>>>(devBuf, size);
    CHECK_CUDA(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    CHECK_CUDA(cudaFree(devBuf));

    // Read event after
    CHECK_CUPTI(cuptiEventGroupReadEvent(eventGroup, CUPTI_EVENT_READ_FLAG_NONE,
                                         eventId, &eventVal));
    uint64_t endVal = eventVal;

    double durationSec = std::chrono::duration<double>(end - start).count();
    double bytesTransferred = static_cast<double>(endVal - startVal) * 32.0; // 32B per count

    double bandwidthGBs = bytesTransferred / durationSec / 1e9;
    std::cout << "NVLink Received Bandwidth: " << bandwidthGBs << " GB/s\n";

    // Cleanup
    CHECK_CUPTI(cuptiEventGroupDisable(eventGroup));
    CHECK_CUPTI(cuptiEventGroupDestroy(eventGroup));
    CHECK_CUPTI(cuCtxDestroy(context));

    return 0;
}
