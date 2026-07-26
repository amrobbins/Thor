#include "Utilities/Common/HostFunctionCleanupQueue.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/Event.h"
#include "Utilities/Common/Stream.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
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

        // Reuse one blocking event per GPU on each worker. A worker never
        // re-records one of its events until the prior wait has completed.
        unordered_map<int32_t, Event> completionEvents;

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

                const uint64_t streamId = pending.front().stream.getId();
                do {
                    cleanupBatch.emplace_back(std::move(pending.front()));
                    pending.pop_front();
                } while (cleanupBatch.size() < MAX_BATCH_SIZE && !pending.empty() && pending.front().stream.getId() == streamId);

                active += cleanupBatch.size();
                notFull.notify_all();
            }

            Stream &stream = cleanupBatch.front().stream;
            Event &completionEvent = completionEvents[stream.getGpuNum()];

            // This event is inserted after every cudaLaunchHostFunc represented
            // by this batch on the same stream. Waiting for it therefore proves
            // all callbacks in the batch returned. The event uses
            // cudaEventBlockingSync, so the cleanup worker sleeps rather than
            // spinning a CPU core.
            stream.putEvent(completionEvent,
                            /*enableTiming=*/false,
                            /*expectingHostToWaitOnThisOne=*/true);
            completionEvent.synchronize();

            // Destruction is deliberately outside the CUDA host callbacks.
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
