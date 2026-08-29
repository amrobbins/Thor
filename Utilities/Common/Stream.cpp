#include "Utilities/Common/Stream.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/HostFunctionCleanupQueue.h"
#include "Utilities/Common/SharedOwnership.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <atomic>
#include <cstdio>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

using namespace std;

namespace {

atomic<uint64_t> nextStreamId{1};
atomic<int> numCudnnHandles{0};
atomic<int> numCublasHandles{0};
atomic<int> numCublasLtHandles{0};

void reportCudaCleanupFailure(const char *operation, cudaError_t status) noexcept {
    if (status == cudaSuccess)
        return;
    ThorImplementation::SharedOwnership::reportCleanupFailure("Stream", operation, cudaGetErrorString(status));
}

void reportCudnnCleanupFailure(const char *operation, cudnnStatus_t status) noexcept {
    if (status == CUDNN_STATUS_SUCCESS)
        return;
    ThorImplementation::SharedOwnership::reportCleanupFailure("Stream", operation, cudnnGetErrorString(status));
}

void reportCublasCleanupFailure(const char *operation, cublasStatus_t status) noexcept {
    if (status == CUBLAS_STATUS_SUCCESS)
        return;

    char detail[64];
    std::snprintf(detail, sizeof(detail), "cuBLAS status %d", static_cast<int>(status));
    ThorImplementation::SharedOwnership::reportCleanupFailure("Stream", operation, detail);
}

}  // namespace

struct Stream::HostFunctionFailureState {
    mutex mtx;
    exception_ptr firstFailure;

    void captureCurrentException() noexcept {
        try {
            lock_guard<mutex> lock(mtx);
            if (firstFailure == nullptr)
                firstFailure = current_exception();
        } catch (...) {
            // Nothing may escape a CUDA host callback. Failure to record the
            // original exception leaves no safe recovery path.
            terminate();
        }
    }

    exception_ptr takeFailure() {
        lock_guard<mutex> lock(mtx);
        exception_ptr failure = firstFailure;
        firstFailure = nullptr;
        return failure;
    }
};

struct Stream::State {
    struct PersistingL2AccessPolicyState {
        const void *base = nullptr;
        uint64_t bytes = 0;
        float hitRatio = 0.0f;
        uint64_t sourceIdentity = 0;
        uint64_t policyGeneration = 0;

        bool operator==(const PersistingL2AccessPolicyState &rhs) const = default;
    };

    State(int gpuNum, uint64_t id) : gpuNum(gpuNum), id(id) {}

    ~State() noexcept {
        if (processLifetime ||
            (cudaStream == nullptr && !cudnnHandle.has_value() && !cublasHandle.has_value() && !cublasLtHandle.has_value()))
            return;

        ThorImplementation::SharedOwnership::cleanupNoThrow("Stream", "release CUDA stream state", [&]() {
            int previousGpuNum = -1;
            const cudaError_t getDeviceStatus = cudaGetDevice(&previousGpuNum);
            if (getDeviceStatus != cudaSuccess) {
                reportCudaCleanupFailure("cudaGetDevice before stream cleanup", getDeviceStatus);
                return;
            }

            const bool switchGpu = previousGpuNum != gpuNum;
            if (switchGpu) {
                const cudaError_t setDeviceStatus = cudaSetDevice(gpuNum);
                if (setDeviceStatus != cudaSuccess) {
                    reportCudaCleanupFailure("cudaSetDevice before stream cleanup", setDeviceStatus);
                    return;
                }
            }

            if (cudnnHandle.has_value()) {
                numCudnnHandles.fetch_sub(1, memory_order_relaxed);
                reportCudnnCleanupFailure("cudnnDestroy", cudnnDestroy(cudnnHandle.value()));
                cudnnHandle.reset();
            }

            if (cublasHandle.has_value()) {
                numCublasHandles.fetch_sub(1, memory_order_relaxed);
                reportCublasCleanupFailure("cublasDestroy", cublasDestroy(cublasHandle.value()));
                cublasHandle.reset();
            }

            if (cublasLtHandle.has_value()) {
                numCublasLtHandles.fetch_sub(1, memory_order_relaxed);
                reportCublasCleanupFailure("cublasLtDestroy", cublasLtDestroy(cublasLtHandle.value()));
                cublasLtHandle.reset();
            }

            if (cudaStream != nullptr) {
                reportCudaCleanupFailure("cudaStreamDestroy", cudaStreamDestroy(cudaStream));
                cudaStream = nullptr;
            }

            if (switchGpu)
                reportCudaCleanupFailure("cudaSetDevice after stream cleanup", cudaSetDevice(previousGpuNum));
        });
    }

    int gpuNum;
    cudaStream_t cudaStream = nullptr;
    optional<cudnnHandle_t> cudnnHandle;
    optional<cublasHandle_t> cublasHandle;
    optional<cublasLtHandle_t> cublasLtHandle;

    mutex libraryHandleMutex;
    HostFunctionFailureState hostFunctionFailureState;

    // Access-policy-window configuration is host-side stream state. Stream
    // handles are copyable and share State, so cache it here rather than in a
    // particular caller such as NetworkInput. The mutex keeps concurrent
    // handles from racing the CUDA attribute update and the cached key.
    mutex persistingL2AccessPolicyMutex;
    optional<PersistingL2AccessPolicyState> persistingL2AccessPolicy;

    uint64_t id;
    bool processLifetime = false;
};

Stream::Stream(int gpuNum, Priority priority) { construct(gpuNum, priority); }

Stream::Stream(ThorImplementation::TensorPlacement placement, Priority priority) {
    const int gpuNum = placement.getMemDevice() == ThorImplementation::TensorPlacement::MemDevices::GPU ? placement.getDeviceNum() : 0;
    construct(gpuNum, priority);
}

Stream::operator cudaStream_t() const {
    THOR_THROW_IF_FALSE(!uninitialized());
    return state->cudaStream;
}

Event Stream::putEvent(bool enableTiming, bool expectingHostToWaitOnThisOne) const {
    THOR_THROW_IF_FALSE(!uninitialized());

    ScopedGpu scopedGpu(state->gpuNum);

    Event event(state->gpuNum, enableTiming, expectingHostToWaitOnThisOne);
    event.record(*this);

    return event;
}

void Stream::putEvent(Event &event, bool enableTiming, bool expectingHostToWaitOnThisOne) const {
    THOR_THROW_IF_FALSE(!uninitialized());

    ScopedGpu scopedGpu(state->gpuNum);

    if (!event.isInitialized()) {
        event = Event(state->gpuNum, enableTiming, expectingHostToWaitOnThisOne);
    } else {
        THOR_THROW_IF_FALSE(event.getGpuNum() == state->gpuNum);
        THOR_THROW_IF_FALSE(event.usesTiming() == enableTiming);
        THOR_THROW_IF_FALSE(event.usesBlockingSync() == expectingHostToWaitOnThisOne);
    }
    event.record(*this);
}

Stream GradientUpdateStreamPool::getNext() {
    unique_lock<mutex> lck(mtx);

    uint32_t numGpus = MachineEvaluator::instance().getNumGpus();
    if (deviceNum >= numGpus) {
        printf("Error: trying to get a gradient-update stream for gpu %d but there are only %d gpus\n", deviceNum, numGpus);
        fflush(stdout);
        THOR_THROW_IF_FALSE(deviceNum < numGpus);
    }

    if (streams.size() < MAX_STREAMS) {
        streams.emplace_back(deviceNum);
        return streams.back();
    }

    Stream stream = streams[nextStreamIndex];
    nextStreamIndex = (nextStreamIndex + 1) % MAX_STREAMS;
    return stream;
}

uint32_t GradientUpdateStreamPool::getNumAllocatedStreams() const {
    unique_lock<mutex> lck(mtx);
    return static_cast<uint32_t>(streams.size());
}

// Note: These are global because destroying a stream when static members are destroyed seems to be a problem.
// Also Note: I would rather be able to use unlimited streams to avoid potential false dependencies in very large very branched networks
//            But I need to support whatever hardware limitation that may exist, so I have the ability to set lower limits on the number
//            of streams that are in place. I don't do this for forward/backward (i.e. data) streams because false dependencies along
//            the execution graph could result in deadlock.
vector<deque<Stream>> uploadStreams;
mutex uploadStreamMutex;
uint32_t maxNumUploadStreams = 4;

vector<deque<Stream>> downloadStreams;
mutex downloadStreamMutex;
uint32_t maxNumDownloadStreams = 4;

Stream Stream::getNextUploadStream(uint32_t deviceNum) {
    unique_lock<mutex> lck(uploadStreamMutex);

    uint32_t numGpus = MachineEvaluator::instance().getNumGpus();
    if (deviceNum >= numGpus) {
        printf("Error: trying to get a stream for gpu %d but there are only %d gpus\n", deviceNum, numGpus);
        fflush(stdout);
        THOR_THROW_IF_FALSE(deviceNum < numGpus);
    }

    THOR_THROW_IF_FALSE(maxNumUploadStreams > 0);

    while (uploadStreams.size() < numGpus)
        uploadStreams.emplace_back();

    // I never delete streams since they may be in use. Only ever add new ones.
    if (uploadStreams[deviceNum].size() < maxNumUploadStreams) {
        uploadStreams[deviceNum].emplace_front(deviceNum);
        uploadStreams[deviceNum].front().informIsStatic();
    }

    Stream stream = uploadStreams[deviceNum].front();
    uploadStreams[deviceNum].pop_front();
    uploadStreams[deviceNum].push_back(stream);
    return stream;
}

void Stream::setMaxNumUploadStreams(uint32_t numUploadStreams) {
    unique_lock<mutex> lck(uploadStreamMutex);

    maxNumUploadStreams = numUploadStreams;
}

Stream Stream::getNextDownloadStream(uint32_t deviceNum) {
    unique_lock<mutex> lck(downloadStreamMutex);

    uint32_t numGpus = MachineEvaluator::instance().getNumGpus();
    if (deviceNum >= numGpus) {
        printf("Error: trying to get a stream for gpu %d but there are only %d gpus\n", deviceNum, numGpus);
        fflush(stdout);
        THOR_THROW_IF_FALSE(deviceNum < numGpus);
    }

    THOR_THROW_IF_FALSE(maxNumDownloadStreams > 0);

    while (downloadStreams.size() < numGpus)
        downloadStreams.emplace_back();

    // I never delete streams since they may be in use. Only ever add new ones.
    if (downloadStreams[deviceNum].size() < maxNumDownloadStreams) {
        downloadStreams[deviceNum].emplace_front(deviceNum);
        downloadStreams[deviceNum].front().informIsStatic();
    }

    Stream stream = downloadStreams[deviceNum].front();
    downloadStreams[deviceNum].pop_front();
    downloadStreams[deviceNum].push_back(stream);
    return stream;
}

void Stream::setMaxNumDownloadStreams(uint32_t numDownloadStreams) {
    unique_lock<mutex> lck(downloadStreamMutex);

    maxNumDownloadStreams = numDownloadStreams;
}

void Stream::waitEvent(Event event) const {
    THOR_THROW_IF_FALSE(!uninitialized());

    ScopedGpu scopedGpu(state->gpuNum);

    CUDA_CHECK(cudaStreamWaitEvent(state->cudaStream, event.getEvent(), 0));
}

void Stream::waitFor(const Stream &producer, Event &reusableEvent) const {
    THOR_THROW_IF_FALSE(!uninitialized());
    THOR_THROW_IF_FALSE(producer.isInitialized());

    producer.putEvent(reusableEvent,
                      /*enableTiming=*/false,
                      /*expectingHostToWaitOnThisOne=*/false);
    waitEvent(reusableEvent);
}

void Stream::synchronize() const {
    THOR_THROW_IF_FALSE(!uninitialized());

    // cudaStreamSynchronize follows the process-wide CUDA scheduling policy
    // and may busy-spin. A blocking event preserves stream completion
    // semantics while putting this genuine host wait to sleep.
    Event completionEvent = putEvent(
        /*enableTiming=*/false,
        /*expectingHostToWaitOnThisOne=*/true);
    completionEvent.synchronize();
    rethrowHostFunctionFailure();
}

void Stream::deviceSynchronize(int gpuNum) {
    ScopedGpu scopedGpu(gpuNum);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void CUDART_CB Stream::hostFunctionTrampoline(void *rawArgs) noexcept {
    auto *args = static_cast<HostFunctionArgsBase *>(rawArgs);
    if (args == nullptr || args->function == nullptr || args->failureState == nullptr)
        terminate();

    auto *failureState = static_cast<HostFunctionFailureState *>(args->failureState);
    try {
        args->function(args);
    } catch (...) {
        failureState->captureCurrentException();
    }
}

void Stream::rethrowHostFunctionFailure() const {
    THOR_THROW_IF_FALSE(!uninitialized());
    exception_ptr failure = state->hostFunctionFailureState.takeFailure();
    if (failure != nullptr)
        rethrow_exception(failure);
}

void Stream::enqueueHostFunction(cudaHostFn_t function, std::unique_ptr<HostFunctionArgsBase> &&args) {
    THOR_THROW_IF_FALSE(function != nullptr);
    THOR_THROW_IF_FALSE(args != nullptr);
    THOR_THROW_IF_FALSE(!uninitialized());

    args->function = function;
    args->failureState = &state->hostFunctionFailureState;
    CUDA_CHECK(cudaLaunchHostFunc(*this, &Stream::hostFunctionTrampoline, args.get()));
    HostFunctionCleanupQueue::instance().push(*this, std::move(args));
    THOR_THROW_IF_FALSE(args == nullptr);
}

cudnnHandle_t Stream::getCudnnHandle() const {
    THOR_THROW_IF_FALSE(!uninitialized());
    lock_guard<mutex> lock(state->libraryHandleMutex);

    if (!state->cudnnHandle.has_value()) {
        ScopedGpu scopedGpu(state->gpuNum);
        cudnnHandle_t handle = nullptr;
        const cudnnStatus_t createStatus = cudnnCreate(&handle);
        if (createStatus != CUDNN_STATUS_SUCCESS) {
            printf("cudnnStatus %d : %s   gpu:%d   numCudnnHandles %d\n",
                   createStatus,
                   cudnnGetErrorString(createStatus),
                   state->gpuNum,
                   numCudnnHandles.load(memory_order_relaxed));
            fflush(stdout);
        }
        THOR_THROW_IF_FALSE(createStatus == CUDNN_STATUS_SUCCESS);

        const cudnnStatus_t setStreamStatus = cudnnSetStream(handle, state->cudaStream);
        if (setStreamStatus != CUDNN_STATUS_SUCCESS) {
            reportCudnnCleanupFailure("cudnnDestroy after cudnnSetStream failure", cudnnDestroy(handle));
            THOR_THROW_IF_FALSE(setStreamStatus == CUDNN_STATUS_SUCCESS);
        }

        state->cudnnHandle = handle;
        numCudnnHandles.fetch_add(1, memory_order_relaxed);
    }

    return state->cudnnHandle.value();
}

cudaStream_t Stream::getStream() const {
    THOR_THROW_IF_FALSE(!uninitialized());
    return state->cudaStream;
}

ThorImplementation::PersistingL2OperationResult Stream::trySetPersistingL2AccessPolicyWindow(
    const void *base,
    uint64_t bytes,
    float hitRatio,
    uint64_t sourceIdentity,
    uint64_t policyGeneration) const {
    if (uninitialized()) {
        return ThorImplementation::PersistingL2OperationResult{
            .status = ThorImplementation::PersistingL2OperationStatus::INVALID_ARGUMENT,
            .cuda_status = cudaErrorInvalidResourceHandle,
            .detail = "Stream is uninitialized",
        };
    }

    const State::PersistingL2AccessPolicyState requested{
        .base = base,
        .bytes = bytes,
        .hitRatio = hitRatio,
        .sourceIdentity = sourceIdentity,
        .policyGeneration = policyGeneration,
    };

    lock_guard<mutex> lock(state->persistingL2AccessPolicyMutex);
    if (state->persistingL2AccessPolicy.has_value() &&
        state->persistingL2AccessPolicy.value() == requested) {
        return ThorImplementation::PersistingL2OperationResult{};
    }

    ThorImplementation::PersistingL2OperationResult result =
        ThorImplementation::trySetPersistingL2AccessPolicyWindow(
            state->gpuNum, state->cudaStream, base, bytes, hitRatio);
    if (result.succeeded())
        state->persistingL2AccessPolicy = requested;
    return result;
}

ThorImplementation::PersistingL2OperationResult Stream::tryClearPersistingL2AccessPolicyWindow() const {
    if (uninitialized()) {
        return ThorImplementation::PersistingL2OperationResult{
            .status = ThorImplementation::PersistingL2OperationStatus::INVALID_ARGUMENT,
            .cuda_status = cudaErrorInvalidResourceHandle,
            .detail = "Stream is uninitialized",
        };
    }

    lock_guard<mutex> lock(state->persistingL2AccessPolicyMutex);
    if (!state->persistingL2AccessPolicy.has_value())
        return ThorImplementation::PersistingL2OperationResult{};

    ThorImplementation::PersistingL2OperationResult result =
        ThorImplementation::tryClearPersistingL2AccessPolicyWindow(
            state->gpuNum, state->cudaStream);
    if (result.succeeded())
        state->persistingL2AccessPolicy.reset();
    return result;
}

cublasHandle_t Stream::getCublasHandle() const {
    THOR_THROW_IF_FALSE(!uninitialized());
    lock_guard<mutex> lock(state->libraryHandleMutex);

    if (!state->cublasHandle.has_value()) {
        ScopedGpu scopedGpu(state->gpuNum);
        cublasHandle_t handle = nullptr;
        const cublasStatus_t createStatus = cublasCreate(&handle);
        if (createStatus != CUBLAS_STATUS_SUCCESS) {
            printf("cublasStatus %d    gpu:%d   numcublasHandles %d\n",
                   createStatus,
                   state->gpuNum,
                   numCublasHandles.load(memory_order_relaxed));
            fflush(stdout);
        }
        THOR_THROW_IF_FALSE(createStatus == CUBLAS_STATUS_SUCCESS);

        const cublasStatus_t setStreamStatus = cublasSetStream(handle, state->cudaStream);
        if (setStreamStatus != CUBLAS_STATUS_SUCCESS) {
            reportCublasCleanupFailure("cublasDestroy after cublasSetStream failure", cublasDestroy(handle));
            THOR_THROW_IF_FALSE(setStreamStatus == CUBLAS_STATUS_SUCCESS);
        }

        state->cublasHandle = handle;
        numCublasHandles.fetch_add(1, memory_order_relaxed);
    }

    return state->cublasHandle.value();
}

cublasLtHandle_t Stream::getCublasLtHandle() const { return getCublasLtHandleUnchecked(); }

cublasLtHandle_t Stream::getCublasLtHandleUnchecked() const {
    THOR_THROW_IF_FALSE(!uninitialized());
    THOR_THROW_IF_FALSE(state->cublasLtHandle.has_value());
    return state->cublasLtHandle.value();
}

int Stream::getGpuNum() const {
    THOR_THROW_IF_FALSE(!uninitialized());
    return state->gpuNum;
}

void Stream::informIsStatic() {
    THOR_THROW_IF_FALSE(!uninitialized());
    state->processLifetime = true;
}

uint64_t Stream::getId() const { return state != nullptr ? state->id : 0; }

void Stream::construct(int gpuNum, Priority priority) {
    auto newState = make_shared<State>(gpuNum, nextStreamId.fetch_add(1, memory_order_relaxed));

    ScopedGpu scopedGpu(gpuNum);

    // greatestPriority is given the highest priority in terms of execution, and its numerical value is the minimum of the allowed
    // range.
    int leastPriority, greatestPriority;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    int priorityValue;
    if (priority == Priority::HIGH)
        priorityValue = greatestPriority;
    else if (priority == Priority::REGULAR)
        priorityValue = greatestPriority + 1;
    else
        priorityValue = greatestPriority + 2;

    CUDA_CHECK(cudaStreamCreateWithPriority(&newState->cudaStream, cudaStreamNonBlocking, priorityValue));

    cublasLtHandle_t ltHandle = nullptr;
    const cublasStatus_t cublasStatus = cublasLtCreate(&ltHandle);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        printf("cublasLtStatus %d    gpu:%d   numCublasLtHandles %d\n",
               cublasStatus,
               gpuNum,
               numCublasLtHandles.load(memory_order_relaxed));
        fflush(stdout);
    }
    THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);
    newState->cublasLtHandle = ltHandle;
    numCublasLtHandles.fetch_add(1, memory_order_relaxed);

    state = std::move(newState);
}
