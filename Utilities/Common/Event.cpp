#include "Event.h"
#include "Stream.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "DeepLearning/Implementation/ThorError.h"

Event::Event() : ReferenceCounted() {}

Event::Event(int32_t gpuNum, bool enableTiming, bool expectingHostToWaitOnThisOne) {
    construct(gpuNum, enableTiming, expectingHostToWaitOnThisOne);
}

Event::Event(const Event &event) {
    // implemented using operator=
    *this = event;
}

Event &Event::operator=(const Event &other) {
    copyFrom(other);
    return *this;
}

Event::~Event() {
    bool shouldDestroy = ReferenceCounted::removeReference();
    if (shouldDestroy)
        destroy();
}

void Event::record(Stream stream) { CUDA_CHECK(cudaEventRecord(getEvent(), stream)); }

Event::operator cudaEvent_t() {
    THOR_THROW_IF_FALSE(!uninitialized());
    return cudaEvent;
}

cudaEvent_t Event::getEvent() {
    THOR_THROW_IF_FALSE(!uninitialized());
    return cudaEvent;
}

int32_t Event::getGpuNum() const {
    THOR_THROW_IF_FALSE(!uninitialized());
    return gpuNum;
}

bool Event::isInitialized() const { return initialized(); }

void Event::synchronize() {
    THOR_THROW_IF_FALSE(!uninitialized());

    CUDA_CHECK(cudaEventSynchronize(*this));
}

float Event::synchronizeAndReportElapsedTimeInMilliseconds(Event startEvent) {
    THOR_THROW_IF_FALSE(!uninitialized());

    float milliseconds;

    synchronize();

    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, startEvent, *this));
    return milliseconds;
}

uint64_t Event::getId() const { return ReferenceCounted::getReferenceCountedId(); }

void Event::construct(int32_t gpuNum, bool enableTiming, bool expectingHostToWaitOnThisOne) {
    ReferenceCounted::initialize();

    ScopedGpu scopedGpu(gpuNum);

    this->gpuNum = gpuNum;

    uint32_t flags = 0;
    if (!enableTiming)
        flags |= cudaEventDisableTiming;
    if (expectingHostToWaitOnThisOne)
        flags |= cudaEventBlockingSync;

    CUDA_CHECK(cudaEventCreateWithFlags(&cudaEvent, flags));
}

void Event::copyFrom(const Event &other) {
    *((ReferenceCounted *)this) = *((ReferenceCounted *)&other);

    gpuNum = other.gpuNum;
    cudaEvent = other.cudaEvent;
}

void Event::destroy() {
    ScopedGpu scopedGpu(gpuNum);
    CUDA_CHECK(cudaEventDestroy(cudaEvent));
}
