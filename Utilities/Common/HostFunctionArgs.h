#pragma once

#include "cuda_runtime.h"

#include <atomic>
#include <mutex>

class Stream;

// Base class for state passed to cudaLaunchHostFunc callbacks. CUDA host
// callbacks must not call CUDA APIs, directly or through destructors. Thor
// therefore owns callback state until the callback has completed and destroys
// it on a normal host worker thread. The callback must not delete its args.
struct HostFunctionArgsBase {
    virtual ~HostFunctionArgsBase() = default;

   private:
    void markCallbackCompleted() noexcept {
        callbackCompleted.store(true, std::memory_order_release);
        callbackCompleted.notify_one();
    }

    void waitForCallbackCompletion() noexcept {
        while (!callbackCompleted.load(std::memory_order_acquire))
            callbackCompleted.wait(false, std::memory_order_acquire);

        // callbackCompleted is published while the trampoline owns this mutex.
        // Acquiring it here is the lifetime rendezvous: once this lock succeeds,
        // the trampoline has released its final callback-side access to this
        // argument object, so the cleanup worker may destroy it after returning.
        std::lock_guard<std::mutex> lock(callbackCompletionMutex);
    }

    // Populated by Stream::enqueueHostFunction(). Keeping the dispatch metadata
    // and completion state in the already-allocated callback state lets every
    // host function pass through one noexcept trampoline without adding another
    // allocation or a CUDA completion event.
    cudaHostFn_t function = nullptr;
    void *failureState = nullptr;
    std::atomic<bool> callbackCompleted{false};
    std::mutex callbackCompletionMutex;

    friend class HostFunctionCleanupQueue;
    friend class Stream;
};
