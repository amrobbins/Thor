#include "Utilities/Common/HostFunctionCleanupQueue.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/Stream.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

using namespace std;

namespace {

struct PendingHostFunctionCleanup {
    PendingHostFunctionCleanup(Stream stream, unique_ptr<HostFunctionArgsBase> &&args)
        : stream(std::move(stream)), args(std::move(args)) {}

    PendingHostFunctionCleanup(PendingHostFunctionCleanup &&other) noexcept
        : stream(other.stream), args(std::move(other.args)) {}

    PendingHostFunctionCleanup &operator=(PendingHostFunctionCleanup &&) = delete;
    PendingHostFunctionCleanup(const PendingHostFunctionCleanup &) = delete;
    PendingHostFunctionCleanup &operator=(const PendingHostFunctionCleanup &) = delete;

    Stream stream;
    unique_ptr<HostFunctionArgsBase> args;
};

}  // namespace

struct HostFunctionCleanupQueue::State {
    mutable mutex mtx;
    condition_variable notEmpty;
    condition_variable notFull;
    condition_variable becameEmpty;
    deque<PendingHostFunctionCleanup> pending;
    vector<thread> workers;
    size_t active = 0;
    bool stopping = false;

    void workerLoop() {
        static constexpr size_t MAX_BATCH_SIZE = 64;

        while (true) {
            vector<PendingHostFunctionCleanup> cleanupBatch;
            cleanupBatch.reserve(MAX_BATCH_SIZE);
            {
                unique_lock<mutex> lock(mtx);
                notEmpty.wait(lock, [&] { return stopping || !pending.empty(); });

                if (pending.empty()) {
                    THOR_THROW_IF_FALSE(stopping);
                    return;
                }

                // Retain the existing same-stream batching only to amortize
                // queue-lock traffic; callback completion is still checked
                // independently for every argument below.
                const uint64_t streamId = pending.front().stream.getId();
                do {
                    cleanupBatch.emplace_back(std::move(pending.front()));
                    pending.pop_front();
                } while (cleanupBatch.size() < MAX_BATCH_SIZE && !pending.empty() && pending.front().stream.getId() == streamId);

                active += cleanupBatch.size();
                notFull.notify_all();
            }

            // Callback completion is a CPU lifetime notification, not a CUDA
            // dependency. Each argument owns its completion state so cleanup
            // remains correct even when callbacks finish before queue insertion
            // or before this worker begins waiting.
            for (PendingHostFunctionCleanup &cleanup : cleanupBatch)
                HostFunctionCleanupQueue::waitForCallbackCompletion(*cleanup.args);

            // Destruction is deliberately outside the CUDA host callbacks. The
            // retained Stream copy keeps the stream-owned failure state alive
            // through callback completion and argument destruction.
            const size_t completedCount = cleanupBatch.size();
            cleanupBatch.clear();

            {
                unique_lock<mutex> lock(mtx);
                THOR_THROW_IF_FALSE(active >= completedCount);
                active -= completedCount;
                if (pending.empty() && active == 0)
                    becameEmpty.notify_all();
            }
        }
    }
};

HostFunctionCleanupQueue &HostFunctionCleanupQueue::instance() {
    static HostFunctionCleanupQueue singleton;
    return singleton;
}

void HostFunctionCleanupQueue::waitForCallbackCompletion(HostFunctionArgsBase &args) noexcept {
    args.waitForCallbackCompletion();
}

HostFunctionCleanupQueue::HostFunctionCleanupQueue() : state(make_unique<State>()) {
    state->workers.reserve(WORKER_COUNT);
    for (size_t i = 0; i < WORKER_COUNT; ++i)
        state->workers.emplace_back([this] { state->workerLoop(); });
}

HostFunctionCleanupQueue::~HostFunctionCleanupQueue() {
    waitForEmpty();

    {
        unique_lock<mutex> lock(state->mtx);
        state->stopping = true;
        state->notEmpty.notify_all();
        state->notFull.notify_all();
    }

    for (thread &worker : state->workers) {
        if (worker.joinable())
            worker.join();
    }
}

void HostFunctionCleanupQueue::push(Stream stream, unique_ptr<HostFunctionArgsBase> &&args) {
    THOR_THROW_IF_FALSE(args != nullptr);

    unique_lock<mutex> lock(state->mtx);
    state->notFull.wait(lock, [&] { return state->stopping || state->pending.size() < QUEUE_CAPACITY; });
    THOR_THROW_IF_FALSE(!state->stopping);

    state->pending.emplace_back(std::move(stream), std::move(args));
    THOR_THROW_IF_FALSE(args == nullptr);
    state->notEmpty.notify_one();
}

void HostFunctionCleanupQueue::waitForEmpty() {
    unique_lock<mutex> lock(state->mtx);
    state->becameEmpty.wait(lock, [&] { return state->pending.empty() && state->active == 0; });
}

size_t HostFunctionCleanupQueue::getWorkerCount() const { return WORKER_COUNT; }

size_t HostFunctionCleanupQueue::getQueueCapacity() const { return QUEUE_CAPACITY; }

size_t HostFunctionCleanupQueue::getPendingCount() const {
    unique_lock<mutex> lock(state->mtx);
    return state->pending.size();
}

size_t HostFunctionCleanupQueue::getActiveCount() const {
    unique_lock<mutex> lock(state->mtx);
    return state->active;
}
