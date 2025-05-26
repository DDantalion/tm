#include <cupti.h>
#include <cuda.h>
#include <iostream>
#include <vector>
#include <cstring>

// Error‐checking macros
template <typename T>
void checkCudaDrv(T result, const char* func, int line) {
    if (result != CUDA_SUCCESS) {
        const char *errstr;
        cuGetErrorString(result, &errstr);
        std::cerr << "CUDA Driver error: " << errstr
                  << " at " << func << ":" << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCupti(CUptiResult result, const char* func, int line) {
    if (result != CUPTI_SUCCESS) {
        const char *errstr;
        cuptiGetResultString(result, &errstr);
        std::cerr << "CUPTI error: " << errstr
                  << " at " << func << ":" << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA_DRV(call) checkCudaDrv(call, #call, __LINE__)
#define CHECK_CUPTI(call)    checkCupti(call, #call, __LINE__)

int main() {
    // 1. Initialize CUDA Driver and create a context
    CHECK_CUDA_DRV(cuInit(0));
    CUdevice device;
    CHECK_CUDA_DRV(cuDeviceGet(&device, 0));
    CUcontext context;
    CHECK_CUDA_DRV(cuCtxCreate(&context, 0, device));

    // 2. Enumerate all available metrics on the device
    uint32_t metricCount = 0;
    CHECK_CUPTI(cuptiDeviceGetNumMetrics(device, &metricCount));
    std::vector<CUpti_MetricID> metricIds(metricCount);
    size_t metricsSizeBytes = metricCount * sizeof(CUpti_MetricID);
    CHECK_CUPTI(cuptiDeviceEnumMetrics(device, &metricsSizeBytes, metricIds.data()));

    // 3. For each metric, collect its value
    for (CUpti_MetricID metricId : metricIds) {
        // 3a. Get the metric's name
        char metricName[128] = {0};
        size_t nameLen = sizeof(metricName);
        CHECK_CUPTI(cuptiMetricGetAttribute(
            metricId,
            CUPTI_METRIC_ATTR_NAME,
            &nameLen,
            metricName));
        if(std::strcmp(metricName, "nvlink_total_data_received") != 0){
            continue;
        }
        // 3b. Get IDs of events required by this metric
        uint32_t numEvents = 0;
        CHECK_CUPTI(cuptiMetricGetNumEvents(metricId, &numEvents));
        std::vector<CUpti_EventID> eventIds(numEvents);
        size_t eventsSizeBytes = numEvents * sizeof(CUpti_EventID);
        CHECK_CUPTI(cuptiMetricEnumEvents(
            metricId,
            &eventsSizeBytes,
            eventIds.data()));

        // 3c. Create event group sets (pass grouping)
        CUpti_EventGroupSets *eventGroupSets = nullptr;
        size_t metricIdArraySize = sizeof(metricId);
        CHECK_CUPTI(cuptiMetricCreateEventGroupSets(
            context,
            metricIdArraySize,
            &metricId,
            &eventGroupSets));

        // 3d. Iterate each pass and measure
        for (uint32_t pass = 0; pass < eventGroupSets->numSets; ++pass) {
            auto &groupSet = eventGroupSets->sets[pass];

            // Enable all event groups in this pass
            for (uint32_t i = 0; i < groupSet.numEventGroups; ++i) {
                CHECK_CUPTI(cuptiEventGroupEnable(
                    groupSet.eventGroups[i]));
            }

            // ==== Insert your GPU workload here ==== 
            //    e.g., myKernel<<<blocks, threads>>>(...);
            CHECK_CUDA_DRV(cuCtxSynchronize());

            // Disable event groups
            for (uint32_t i = 0; i < groupSet.numEventGroups; ++i) {
                CHECK_CUPTI(cuptiEventGroupDisable(
                    groupSet.eventGroups[i]));
            }

            // 3e. Compute the metric value
            std::vector<uint64_t> eventValues(numEvents);
            uint64_t durationNs = 1000;
            CUpti_MetricValue metricValue = {0};
            size_t valuesSizeBytes = eventValues.size() * sizeof(uint64_t);

            CHECK_CUPTI(cuptiMetricGetValue(
                device,
                metricId,
                eventsSizeBytes,
                eventIds.data(),
                valuesSizeBytes,
                eventValues.data(),
                durationNs,
                &metricValue));

            std::cout << "Metric " << metricName
                      << " (pass " << pass << ") = "
                      << metricValue.metricValueUint64
                      << " over " << durationNs
                      << " ns" << std::endl;
        }

        // 3f. Clean up the event group sets
        CHECK_CUPTI(cuptiEventGroupSetsDestroy(eventGroupSets));
    }

    // 4. Destroy CUDA context
    CHECK_CUDA_DRV(cuCtxDestroy(context));
    return 0;
}
//nvcc list_metrics.cpp -o list_metrics -lcupti -lcuda
