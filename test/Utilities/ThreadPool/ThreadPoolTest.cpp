#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "Utilities/ThreadPool/ThreadPool.h"

// ---------------------------
// Test scaffolding
// ---------------------------

struct TestContext {
    // number of Processor(context) constructions (should equal numThreads actually used)
    std::atomic<int> processor_ctor_calls{0};

    // how many items were processed total
    std::atomic<int> processed_count{0};

    // used to detect duplicates (same item processed twice)
    std::atomic<int> duplicate_count{0};

    // gate to optionally delay processing (for stop() tests)
    std::atomic<bool> allow_processing{true};

    // Record items seen; protected by mutex (vector<bool> isn't atomic-friendly)
    std::mutex m;
    std::vector<uint8_t> seen;  // 0/1 flags

    explicit TestContext(size_t n = 0) : seen(n, 0) {}

    void reset(size_t n) {
        std::lock_guard<std::mutex> g(m);
        seen.assign(n, 0);
        processor_ctor_calls.store(0);
        processed_count.store(0);
        duplicate_count.store(0);
        allow_processing.store(true);
    }
};

struct TestProcessor {
    TestContext& ctx;

    explicit TestProcessor(TestContext& c) : ctx(c) { ctx.processor_ctor_calls.fetch_add(1, std::memory_order_relaxed); }

    void process(int item) {
        // Optional gate to make stop() behavior more observable
        while (!ctx.allow_processing.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }

        // Mark as seen, detect duplicates
        {
            std::lock_guard<std::mutex> g(ctx.m);
            if (static_cast<size_t>(item) < ctx.seen.size()) {
                if (ctx.seen[item] != 0) {
                    ctx.duplicate_count.fetch_add(1, std::memory_order_relaxed);
                } else {
                    ctx.seen[item] = 1;
                }
            } else {
                // Out-of-range work item would be a test bug; treat as duplicate-like failure.
                ctx.duplicate_count.fetch_add(1, std::memory_order_relaxed);
            }
        }

        ctx.processed_count.fetch_add(1, std::memory_order_relaxed);

        // Add a tiny delay to increase concurrency interleavings (helps catch races)
        // Keep it small to avoid flaky slow tests.
        std::this_thread::yield();
    }
};

// ---------------------------
// Tests
// ---------------------------

TEST(ThreadPool, IsNotCopyable) {
    using Pool = ThreadPool<TestProcessor, int, TestContext>;
    static_assert(!std::is_copy_constructible_v<Pool>, "ThreadPool should not be copy-constructible");
    static_assert(!std::is_copy_assignable_v<Pool>, "ThreadPool should not be copy-assignable");
}

TEST(ThreadPool, ProcessesAllItemsExactlyOnce) {
    constexpr int N = 10'000;
    std::vector<int> items;
    items.reserve(N);
    for (int i = 0; i < N; ++i)
        items.push_back(i);

    TestContext ctx(static_cast<size_t>(N));

    {
        ThreadPool<TestProcessor, int, TestContext> pool(std::move(items), ctx, /*numThreads=*/8);
        pool.wait();
    }

    // No duplicates
    EXPECT_EQ(ctx.duplicate_count.load(), 0);

    // All items processed
    EXPECT_EQ(ctx.processed_count.load(), N);

    // All seen flags set
    {
        std::lock_guard<std::mutex> g(ctx.m);
        for (int i = 0; i < N; ++i) {
            ASSERT_EQ(ctx.seen[i], 1) << "missing item " << i;
        }
    }
}

TEST(ThreadPool, NumThreadsZeroMeansOneThread) {
    constexpr int N = 1000;
    std::vector<int> items;
    items.reserve(N);
    for (int i = 0; i < N; ++i)
        items.push_back(i);

    TestContext ctx(static_cast<size_t>(N));

    {
        ThreadPool<TestProcessor, int, TestContext> pool(std::move(items), ctx, /*numThreads=*/0);
        pool.wait();
    }

    EXPECT_EQ(ctx.processor_ctor_calls.load(), 1) << "Expected exactly one worker when numThreads=0";
    EXPECT_EQ(ctx.duplicate_count.load(), 0);
    EXPECT_EQ(ctx.processed_count.load(), N);
}

TEST(ThreadPool, ConstructsOneProcessorPerWorker) {
    constexpr int N = 5000;
    constexpr uint32_t kThreads = 6;

    std::vector<int> items;
    items.reserve(N);
    for (int i = 0; i < N; ++i)
        items.push_back(i);

    TestContext ctx(static_cast<size_t>(N));

    {
        ThreadPool<TestProcessor, int, TestContext> pool(std::move(items), ctx, kThreads);
        pool.wait();
    }

    EXPECT_EQ(ctx.processor_ctor_calls.load(), static_cast<int>(kThreads));
    EXPECT_EQ(ctx.duplicate_count.load(), 0);
    EXPECT_EQ(ctx.processed_count.load(), N);
}

TEST(ThreadPool, StopRequestsEarlyExit) {
    // Large enough so "stop" has something to stop.
    constexpr int N = 200'000;
    constexpr uint32_t kThreads = 8;

    std::vector<int> items;
    items.reserve(N);
    for (int i = 0; i < N; ++i)
        items.push_back(i);

    TestContext ctx(static_cast<size_t>(N));
    ctx.allow_processing.store(true);

    ThreadPool<TestProcessor, int, TestContext> pool(std::move(items), ctx, kThreads);

    // Wait until some progress is made (avoid stopping before any thread runs)
    while (ctx.processed_count.load(std::memory_order_relaxed) < 2000) {
        std::this_thread::yield();
    }

    pool.stop();

    const int processed = ctx.processed_count.load();
    EXPECT_GE(processed, 2000);
    EXPECT_LT(processed, N) << "stop() should usually prevent processing all items (best-effort)";

    EXPECT_EQ(ctx.duplicate_count.load(), 0);

    // Verify that every processed item is marked seen, and no unprocessed item is incorrectly marked.
    // (Since we set seen[item]=1 only when processing, this mostly validates internal consistency.)
    {
        std::lock_guard<std::mutex> g(ctx.m);
        int seen_count = 0;
        for (uint8_t v : ctx.seen)
            seen_count += (v != 0);
        EXPECT_EQ(seen_count, processed);
    }
}
