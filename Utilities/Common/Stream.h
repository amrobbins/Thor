#pragma once

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"
#include "Event.h"
#include "ScopedGpu.h"
#include "Utilities/Common/HostFunctionArgs.h"
#include "Utilities/Common/PersistingL2Cache.h"
#include "Utilities/ComputeTopology/MachineEvaluator.h"

#include <cudnn.h>
#include <cublasLt.h>
#include "cublas_v2.h"
#include "cuda.h"
#include "cuda_runtime.h"

#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

/**
 * A shared-ownership container for cudaStream_t and the library handles bound
 * to that stream.
 *
 * Distinct Stream handle objects may be copied, assigned, reset, and destroyed
 * concurrently while referring to the same CUDA stream. Concurrent mutation
 * of the same Stream handle object requires external synchronization, matching
 * the std::shared_ptr ownership contract documented in SharedOwnership.h.
 *
 * Also carries the gpuNum that the stream exists on.
 */
class Stream {
   public:
    Stream() = default;

    enum class Priority { HIGH = 3, REGULAR = 4, LOW = 5 };

    explicit Stream(int gpuNum, Priority priority = Priority::REGULAR);

    Stream(const Stream &other) = default;
    Stream(Stream &&other) noexcept = default;

    explicit Stream(ThorImplementation::TensorPlacement placement, Priority priority = Priority::REGULAR);

    Stream &operator=(const Stream &other) = default;
    Stream &operator=(Stream &&other) noexcept = default;

    operator cudaStream_t() const;

    virtual ~Stream() = default;

    // Value-returning putEvent() creates a distinct CUDA event and transfers
    // ownership of that completion token to the caller. Recurring dependency
    // edges should instead own an Event and use the in-place overload below.
    Event putEvent(bool enableTiming = false, bool expectingHostToWaitOnThisOne = false) const;

    // Lazily creates event once, then re-records the same cudaEvent_t. Creation
    // flags are immutable: GPU, timing, and blocking-sync intent must match on
    // every reuse.
    void putEvent(Event &event, bool enableTiming = false, bool expectingHostToWaitOnThisOne = false) const;

    void waitEvent(Event event) const;

    // Record a reusable synchronization-only event on producer and enqueue this
    // stream's wait for that recorded generation. The caller owns the Event and
    // therefore owns the logical dependency edge; Thor intentionally does not
    // cache events implicitly by stream pair.
    void waitFor(const Stream &producer, Event &reusableEvent) const;

    void synchronize() const;

    static void deviceSynchronize(int gpuNum);

    // Enqueue host work without allowing exceptions to escape through CUDA's C callback boundary.
    // Any exception thrown by the callback is captured and rethrown by synchronize().
    void enqueueHostFunction(cudaHostFn_t function, std::unique_ptr<HostFunctionArgsBase> &&args);

    // cuDNN handles are scoped by requesting host thread within the shared
    // stream state. This keeps each handle permanently bound to this CUDA
    // stream without allowing simultaneous host-thread use of one handle.
    cudnnHandle_t getCudnnHandle() const;

    cudaStream_t getStream() const;

    // Non-throwing CUDA persisting-L2 access-policy helpers. These only
    // configure this stream; they do not decide which allocations should be
    // cached or reserve device-wide L2 capacity.
    // sourceIdentity and policyGeneration are opaque cache-key components for
    // callers that own a higher-level policy. Repeating an identical request
    // on the same shared stream state is a no-op after the first successful
    // CUDA update. Changing the source, geometry, hit ratio, identity, or
    // generation forces the stream attribute to be refreshed.
    [[nodiscard]] ThorImplementation::PersistingL2OperationResult trySetPersistingL2AccessPolicyWindow(
        const void* base,
        uint64_t bytes,
        float hitRatio,
        uint64_t sourceIdentity = 0,
        uint64_t policyGeneration = 0) const;

    // Repeated clears are likewise cached: once this Stream state knows the
    // window is disabled, no CUDA call is issued until a later successful set.
    [[nodiscard]] ThorImplementation::PersistingL2OperationResult tryClearPersistingL2AccessPolicyWindow() const;

    cublasHandle_t getCublasHandle() const;

    cublasLtHandle_t getCublasLtHandle() const;

    cublasLtHandle_t getCublasLtHandleUnchecked() const;

    bool operator==(const Stream &other) const { return state != nullptr && state == other.state; }

    int getGpuNum() const;

    bool isInitialized() const { return !uninitialized(); }

    virtual std::string getObjectName() const { return "Stream"; }

    // Process-lifetime streams cannot safely interact with CUDA from static
    // destruction. Mark the shared resource state so every handle agrees that
    // CUDA/library resources intentionally survive until process exit.
    void informIsStatic();

    uint64_t getId() const;

    static Stream getNextUploadStream(uint32_t deviceNum);
    static void setMaxNumUploadStreams(uint32_t numGradientUpdateStreams);

    static Stream getNextDownloadStream(uint32_t deviceNum);
    static void setMaxNumDownloadStreams(uint32_t numGradientUpdateStreams);

   private:
    struct HostFunctionFailureState;
    struct State;

    static void CUDART_CB hostFunctionTrampoline(void *rawArgs) noexcept;
    void rethrowHostFunctionFailure() const;

    void construct(int gpuNum, Priority priority);
    bool uninitialized() const { return state == nullptr; }

   private:
    std::shared_ptr<State> state;

    static std::unordered_map<uint32_t, Stream> staticDeviceStreams;
};

/**
 * A lazily allocated, owner-scoped pool for gradient-update streams.
 *
 * Each placed model owns one pool per GPU. All physical stamps for that model
 * on the GPU share the pool. The first three requests allocate distinct
 * streams; subsequent requests reuse those streams
 * round-robin. Keeping the pool owner-scoped prevents optimizer work from one
 * model from adding ordering dependencies to another concurrently executing
 * model.
 */
class GradientUpdateStreamPool {
   public:
    static constexpr uint32_t MAX_STREAMS = 3;

    explicit GradientUpdateStreamPool(uint32_t deviceNum) : deviceNum(deviceNum) {}

    Stream getNext();

    uint32_t getDeviceNum() const { return deviceNum; }
    uint32_t getNumAllocatedStreams() const;

   private:
    const uint32_t deviceNum;
    mutable std::mutex mtx;
    std::deque<Stream> streams;
    uint32_t nextStreamIndex = 0;
};
