#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include "Utilities/ThreadPool/ThreadPool.h"

using namespace std::chrono_literals;

// -------------------------
// 1) Processes N tasks then returns false (normal completion)
// -------------------------
struct CountingProcessor {
    struct State {
        std::atomic<int> remaining{0};
        std::atomic<int> processed{0};
        std::atomic<int> process_calls{0};
    };

    explicit CountingProcessor(std::shared_ptr<State> s) : st(std::move(s)) {}

    bool process() {
        st->process_calls.fetch_add(1, std::memory_order_relaxed);

        int prev = st->remaining.fetch_sub(1, std::memory_order_relaxed);
        if (prev <= 0) {
            // We overshot; restore (optional).
            st->remaining.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        st->processed.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    std::shared_ptr<State> st;
};

TEST(ThreadPool, ProcessesAllWorkAndStopsNaturally) {
    auto state = std::make_shared<CountingProcessor::State>();
    state->remaining.store(10'000, std::memory_order_relaxed);

    {
        ThreadPool<CountingProcessor> pool(CountingProcessor(state), /*numThreads=*/4);
        // Destructor joins. We just let it run to completion.
    }

    EXPECT_EQ(state->processed.load(), 10'000);
    EXPECT_GE(state->process_calls.load(), 10'000);  // may be a bit more due to overshoot attempts
}

// -------------------------
// 2) numThreads==0 coerces to 1 (sanity)
// -------------------------
TEST(ThreadPool, ZeroThreadsBecomesOneThread) {
    auto state = std::make_shared<CountingProcessor::State>();
    state->remaining.store(1'000, std::memory_order_relaxed);

    {
        ThreadPool<CountingProcessor> pool(CountingProcessor(state), /*numThreads=*/0);
    }

    EXPECT_EQ(state->processed.load(), 1'000);
}

// -------------------------
// 3) Verify per-thread copy behavior
//    Each worker thread does: Processor threadsProcessor = processor;
//    So copy-ctor should run once per worker.
// -------------------------
struct CopyTrackingProcessor {
    struct State {
        std::atomic<int> remaining{0};

        std::atomic<int> copy_ctor_calls{0};

        std::mutex m;
        std::set<int> seen_instance_ids;  // unique per threadsProcessor copy
    };

    explicit CopyTrackingProcessor(std::shared_ptr<State> s) : st(std::move(s)), instance_id(0) {}

    // Copy ctor: assign a unique id to each copy
    CopyTrackingProcessor(const CopyTrackingProcessor& other) : st(other.st) {
        instance_id = st->copy_ctor_calls.fetch_add(1, std::memory_order_relaxed) + 1;
    }

    CopyTrackingProcessor(CopyTrackingProcessor&& other) noexcept : st(std::move(other.st)), instance_id(other.instance_id) {}

    CopyTrackingProcessor& operator=(const CopyTrackingProcessor&) = delete;

    bool process() {
        // record this instance id at least once
        {
            std::lock_guard<std::mutex> g(st->m);
            st->seen_instance_ids.insert(instance_id);
        }

        int prev = st->remaining.fetch_sub(1, std::memory_order_relaxed);
        if (prev <= 0) {
            st->remaining.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        return true;
    }

    std::shared_ptr<State> st;
    int instance_id;
};

TEST(ThreadPool, CreatesOneProcessorCopyPerWorkerThread) {
    auto state = std::make_shared<CopyTrackingProcessor::State>();

    const size_t numThreads = 6;
    // ensure at least one task per thread
    state->remaining.store(static_cast<int>(numThreads * 10), std::memory_order_relaxed);

    {
        ThreadPool<CopyTrackingProcessor> pool(CopyTrackingProcessor(state), numThreads);
    }

    // copy ctor should have been called once per worker
    EXPECT_EQ(state->copy_ctor_calls.load(), static_cast<int>(numThreads));

    // and each worker copy should have recorded its unique id
    {
        std::lock_guard<std::mutex> g(state->m);
        EXPECT_EQ(state->seen_instance_ids.size(), numThreads);
    }
}

// -------------------------
// 4) Stop() requests stop and joins promptly IF process() is non-blocking.
//    This test ensures stop can cut work short.
// -------------------------
struct SlowButNonBlockingProcessor {
    struct State {
        std::atomic<int> processed{0};
    };

    explicit SlowButNonBlockingProcessor(std::shared_ptr<State> s) : st(std::move(s)) {}

    bool process() {
        st->processed.fetch_add(1, std::memory_order_relaxed);
        std::this_thread::sleep_for(2ms);  // simulate work
        return true;                       // never naturally finishes
    }

    std::shared_ptr<State> st;
};

TEST(ThreadPool, StopRequestsStopAndJoins) {
    auto state = std::make_shared<SlowButNonBlockingProcessor::State>();

    ThreadPool<SlowButNonBlockingProcessor> pool(SlowButNonBlockingProcessor(state), /*numThreads=*/4);

    std::this_thread::sleep_for(20ms);

    // stop should return (joins), and processed should be > 0 but not enormous.
    pool.stop();

    int processed = state->processed.load(std::memory_order_relaxed);
    EXPECT_GT(processed, 0);
    EXPECT_LT(processed, 10'000);  // very loose bound, just to show it didn't run forever
}
