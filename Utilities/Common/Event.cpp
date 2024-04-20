#include "Event.h"
#include "Stream.h"

Event::Event() : ReferenceCounted() {}

Event::Event(int gpuNum, bool enableTiming, bool expectingHostToWaitOnThisOne) {
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

void Event::record(Stream stream) {
    cudaError_t cudaStatus = cudaEventRecord(getEvent(), stream);
    assert(cudaStatus == cudaSuccess);
}

Event::operator cudaEvent_t() {
    assert(!uninitialized());
    return cudaEvent;
}

cudaEvent_t Event::getEvent() {
    assert(!uninitialized());
    return cudaEvent;
}

int Event::getGpuNum() const {
    assert(!uninitialized());
    return gpuNum;
}

void Event::synchronize() {
    assert(!uninitialized());

    cudaError_t cudaStatus;
    cudaStatus = cudaEventSynchronize(*this);
    if (cudaStatus != cudaSuccess) {
        printf("cudaStatus %d\n", cudaStatus);
        fflush(stdout);
    }
    assert(cudaStatus == cudaSuccess);
}

float Event::synchronizeAndReportElapsedTimeInMilliseconds(Event startEvent) {
    assert(!uninitialized());

    float milliseconds;

    synchronize();
    cudaError_t cudaStatus;
    cudaStatus = cudaEventElapsedTime(&milliseconds, startEvent, *this);
    assert(cudaStatus == cudaSuccess);
    return milliseconds;
}

long Event::getReferenceCountedId() { return ReferenceCounted::getReferenceCountedId(); }

void Event::construct(int gpuNum, bool enableTiming, bool expectingHostToWaitOnThisOne) {
    ReferenceCounted::initialize();

    ScopedGpu scopedGpu(gpuNum);

    cudaError_t cudaStatus;
    this->gpuNum = gpuNum;

    unsigned int flags = 0;
    if (!enableTiming)
        flags |= cudaEventDisableTiming;
    if (expectingHostToWaitOnThisOne)
        flags |= cudaEventBlockingSync;

    cudaStatus = cudaEventCreateWithFlags(&cudaEvent, flags);
    assert(cudaStatus == cudaSuccess);
}

void Event::copyFrom(const Event &other) {
    *((ReferenceCounted *)this) = *((ReferenceCounted *)&other);

    gpuNum = other.gpuNum;
    cudaEvent = other.cudaEvent;
}

void Event::destroy() {
    ScopedGpu scopedGpu(gpuNum);
    cudaError_t cudaStatus;
    cudaStatus = cudaEventDestroy(cudaEvent);
    assert(cudaStatus == cudaSuccess);
}
