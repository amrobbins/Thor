#pragma once

#include "Utilities/Common/HostFunctionArgs.h"

#include <cstddef>
#include <memory>

class Stream;

/**
 * Bounded owner queue for cudaLaunchHostFunc argument cleanup.
 *
 * A fixed number of workers wait for callbacks by recording reusable blocking
 * CUDA events on the callback streams. This prevents one synchronizing host
 * thread from being created for every callback while still ensuring callback
 * argument destructors run outside CUDA's host-callback context.
 */
class HostFunctionCleanupQueue {
   public:
    static constexpr std::size_t WORKER_COUNT = 4;
    static constexpr std::size_t QUEUE_CAPACITY = 4096;

    static HostFunctionCleanupQueue &instance();

    HostFunctionCleanupQueue(const HostFunctionCleanupQueue &) = delete;
    HostFunctionCleanupQueue &operator=(const HostFunctionCleanupQueue &) = delete;

    ~HostFunctionCleanupQueue();

    // Blocking only when the bounded queue is full. Ownership transfers on
    // return, while the fixed worker set performs the eventual destruction.
    void push(Stream stream, std::unique_ptr<HostFunctionArgsBase> &&args);

    // Wait until all queued and currently processed cleanup work is complete.
    // Intended for explicit lifecycle boundaries and tests, not batch hot paths.
    void waitForEmpty();

    [[nodiscard]] std::size_t getWorkerCount() const;
    [[nodiscard]] std::size_t getQueueCapacity() const;
    [[nodiscard]] std::size_t getPendingCount() const;
    [[nodiscard]] std::size_t getActiveCount() const;

   private:
    struct State;

    HostFunctionCleanupQueue();

    std::unique_ptr<State> state;
};
