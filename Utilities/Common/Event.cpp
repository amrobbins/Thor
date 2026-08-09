#include "Event.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Stream.h"
#include "Utilities/Common/SharedOwnership.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <atomic>
#include <memory>
#include <utility>

namespace {

std::atomic<uint64_t> nextEventId{1};

void reportCudaCleanupFailure(const char *operation, cudaError_t status) noexcept {
    if (status == cudaSuccess)
        return;

    ThorImplementation::SharedOwnership::reportCleanupFailure("Event", operation, cudaGetErrorString(status));
}

}  // namespace

struct Event::State {
    State(int32_t gpuNum, bool blockingSync, uint64_t id) : gpuNum(gpuNum), blockingSync(blockingSync), id(id) {}

    ~State() noexcept {
        ThorImplementation::SharedOwnership::cleanupNoThrow("Event", "release CUDA event", [&]() {
            if (cudaEvent == nullptr)
                return;

            int previousGpuNum = -1;
            const cudaError_t getDeviceStatus = cudaGetDevice(&previousGpuNum);
            if (getDeviceStatus != cudaSuccess) {
                reportCudaCleanupFailure("cudaGetDevice before cudaEventDestroy", getDeviceStatus);
                return;
            }

            const bool switchGpu = previousGpuNum != gpuNum;
            if (switchGpu) {
                const cudaError_t setDeviceStatus = cudaSetDevice(gpuNum);
                if (setDeviceStatus != cudaSuccess) {
                    reportCudaCleanupFailure("cudaSetDevice before cudaEventDestroy", setDeviceStatus);
                    return;
                }
            }

            const cudaError_t destroyStatus = cudaEventDestroy(cudaEvent);

            // Restore the caller's active device even when cudaEventDestroy fails.
            cudaError_t restoreStatus = cudaSuccess;
            if (switchGpu)
                restoreStatus = cudaSetDevice(previousGpuNum);

            reportCudaCleanupFailure("cudaEventDestroy", destroyStatus);
            reportCudaCleanupFailure("cudaSetDevice after cudaEventDestroy", restoreStatus);
        });
    }

    int32_t gpuNum;
    cudaEvent_t cudaEvent = nullptr;
    bool blockingSync;
    uint64_t id;
};

Event::Event(int32_t gpuNum, bool enableTiming, bool expectingHostToWaitOnThisOne) {
    construct(gpuNum, enableTiming, expectingHostToWaitOnThisOne);
}

Event::~Event() = default;

void Event::record(Stream stream) { CUDA_CHECK(cudaEventRecord(getEvent(), stream)); }

Event::operator cudaEvent_t() {
    THOR_THROW_IF_FALSE(isInitialized());
    return state->cudaEvent;
}

cudaEvent_t Event::getEvent() {
    THOR_THROW_IF_FALSE(isInitialized());
    return state->cudaEvent;
}

int32_t Event::getGpuNum() const {
    THOR_THROW_IF_FALSE(isInitialized());
    return state->gpuNum;
}

bool Event::isInitialized() const { return state != nullptr; }

bool Event::usesBlockingSync() const {
    THOR_THROW_IF_FALSE(isInitialized());
    return state->blockingSync;
}

void Event::synchronize() {
    THOR_THROW_IF_FALSE(isInitialized());

    ScopedGpu scopedGpu(state->gpuNum);
    CUDA_CHECK(cudaEventSynchronize(state->cudaEvent));
}

float Event::synchronizeAndReportElapsedTimeInMilliseconds(Event startEvent) {
    THOR_THROW_IF_FALSE(isInitialized());

    float milliseconds;

    synchronize();

    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, startEvent, *this));
    return milliseconds;
}

uint64_t Event::getId() const { return state != nullptr ? state->id : 0; }

void Event::construct(int32_t gpuNum, bool enableTiming, bool expectingHostToWaitOnThisOne) {
    ScopedGpu scopedGpu(gpuNum);

    uint32_t flags = 0;
    if (!enableTiming)
        flags |= cudaEventDisableTiming;
    if (expectingHostToWaitOnThisOne)
        flags |= cudaEventBlockingSync;

    auto newState = std::make_shared<State>(gpuNum,
                                            expectingHostToWaitOnThisOne,
                                            nextEventId.fetch_add(1, std::memory_order_relaxed));
    CUDA_CHECK(cudaEventCreateWithFlags(&newState->cudaEvent, flags));
    state = std::move(newState);
}
