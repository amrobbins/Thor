#include "Utilities/Common/Stream.h"
#include "Utilities/ComputeTopology/MachineEvaluator.h"

#include "gtest/gtest.h"

#include <atomic>
#include <cstdint>
#include <thread>
#include <utility>
#include <vector>

namespace {

bool gpuAvailable() { return MachineEvaluator::instance().getNumGpus() > 0; }

}  // namespace

TEST(StreamSharedOwnership, DefaultStreamIsUninitializedWithZeroIdentity) {
    Stream stream;

    EXPECT_FALSE(stream.isInitialized());
    EXPECT_EQ(stream.getId(), 0u);
}

TEST(StreamSharedOwnership, CopiesAndMovesShareOneStreamState) {
    if (!gpuAvailable())
        GTEST_SKIP() << "Stream shared-ownership test requires a GPU";

    Stream original(0);
    const uint64_t originalId = original.getId();
    const cudaStream_t originalCudaStream = original.getStream();
    const cublasLtHandle_t originalCublasLtHandle = original.getCublasLtHandle();

    Stream copied = original;
    EXPECT_EQ(copied.getId(), originalId);
    EXPECT_EQ(copied.getStream(), originalCudaStream);
    EXPECT_EQ(copied.getCublasLtHandle(), originalCublasLtHandle);
    EXPECT_EQ(copied, original);

    Stream moved = std::move(copied);
    EXPECT_FALSE(copied.isInitialized());
    EXPECT_TRUE(moved.isInitialized());
    EXPECT_EQ(moved.getId(), originalId);
    EXPECT_EQ(moved.getStream(), originalCudaStream);

    Stream assigned;
    assigned = moved;
    EXPECT_EQ(assigned.getId(), originalId);
    EXPECT_EQ(assigned.getStream(), originalCudaStream);

    Stream moveAssigned;
    moveAssigned = std::move(assigned);
    EXPECT_FALSE(assigned.isInitialized());
    EXPECT_EQ(moveAssigned.getId(), originalId);
    EXPECT_EQ(moveAssigned.getStream(), originalCudaStream);

    original = Stream();
    moved = Stream();

    // The remaining independently-owned handle must keep the CUDA stream and
    // its eagerly-created cuBLASLt handle alive.
    EXPECT_EQ(moveAssigned.getCublasLtHandle(), originalCublasLtHandle);
    moveAssigned.synchronize();
}

TEST(StreamSharedOwnership, IndependentlyCreatedStreamsHaveDistinctIdentities) {
    if (!gpuAvailable())
        GTEST_SKIP() << "Stream shared-ownership test requires a GPU";

    Stream first(0);
    Stream second(0);

    EXPECT_NE(first.getId(), second.getId());
    EXPECT_NE(first.getStream(), second.getStream());
    EXPECT_FALSE(first == second);

    first.synchronize();
    second.synchronize();
}

TEST(StreamSharedOwnership, CudnnHandlesArePerHostThreadWhileOtherLazyHandlesRemainShared) {
    if (!gpuAvailable())
        GTEST_SKIP() << "Stream shared-ownership test requires a GPU";

    constexpr uint32_t NUM_THREADS = 8;

    Stream root(0);
    std::vector<Stream> stableSources(NUM_THREADS, root);
    std::vector<cudnnHandle_t> cudnnHandles(NUM_THREADS, nullptr);
    std::vector<cudnnHandle_t> repeatedCudnnHandles(NUM_THREADS, nullptr);
    std::vector<cublasHandle_t> cublasHandles(NUM_THREADS, nullptr);
    std::vector<cublasLtHandle_t> cublasLtHandles(NUM_THREADS, nullptr);
    std::vector<std::thread> workers;
    workers.reserve(NUM_THREADS);

    std::atomic<uint32_t> readyThreads = 0;
    std::atomic<uint32_t> acquiredHandles = 0;
    std::atomic<bool> start = false;
    std::atomic<bool> release = false;

    for (uint32_t threadIndex = 0; threadIndex < NUM_THREADS; ++threadIndex) {
        workers.emplace_back([&, threadIndex]() {
            Stream source = stableSources[threadIndex];
            readyThreads.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire))
                std::this_thread::yield();

            cudnnHandles[threadIndex] = source.getCudnnHandle();
            repeatedCudnnHandles[threadIndex] = source.getCudnnHandle();
            cublasHandles[threadIndex] = source.getCublasHandle();
            cublasLtHandles[threadIndex] = source.getCublasLtHandle();
            acquiredHandles.fetch_add(1, std::memory_order_release);

            // Keep every requesting thread alive until all handles have been
            // acquired so a finished thread id cannot be reused by another
            // worker during this test.
            while (!release.load(std::memory_order_acquire))
                std::this_thread::yield();
        });
    }

    while (readyThreads.load(std::memory_order_acquire) != NUM_THREADS)
        std::this_thread::yield();
    start.store(true, std::memory_order_release);
    while (acquiredHandles.load(std::memory_order_acquire) != NUM_THREADS)
        std::this_thread::yield();
    release.store(true, std::memory_order_release);

    for (std::thread &worker : workers)
        worker.join();

    for (uint32_t threadIndex = 0; threadIndex < NUM_THREADS; ++threadIndex) {
        EXPECT_EQ(repeatedCudnnHandles[threadIndex], cudnnHandles[threadIndex]);
        for (uint32_t otherThreadIndex = threadIndex + 1; otherThreadIndex < NUM_THREADS; ++otherThreadIndex)
            EXPECT_NE(cudnnHandles[threadIndex], cudnnHandles[otherThreadIndex]);
    }

    for (uint32_t threadIndex = 1; threadIndex < NUM_THREADS; ++threadIndex) {
        EXPECT_EQ(cublasHandles[threadIndex], cublasHandles[0]);
        EXPECT_EQ(cublasLtHandles[threadIndex], cublasLtHandles[0]);
    }

    root.synchronize();
}

TEST(StreamSharedOwnership, DistinctHandlesMayBeCopiedMovedAndDestroyedConcurrently) {
    if (!gpuAvailable())
        GTEST_SKIP() << "Stream shared-ownership test requires a GPU";

    constexpr uint32_t NUM_THREADS = 8;
    constexpr uint32_t COPIES_PER_THREAD = 10000;

    Stream root(0);
    const uint64_t rootId = root.getId();
    const cudaStream_t rootCudaStream = root.getStream();

    std::vector<Stream> stableSources(NUM_THREADS, root);
    std::vector<std::thread> workers;
    workers.reserve(NUM_THREADS);

    std::atomic<bool> workerFailed = false;
    for (uint32_t threadIndex = 0; threadIndex < NUM_THREADS; ++threadIndex) {
        workers.emplace_back([source = stableSources[threadIndex], rootId, rootCudaStream, &workerFailed]() mutable {
            for (uint32_t copyIndex = 0; copyIndex < COPIES_PER_THREAD; ++copyIndex) {
                Stream copy = source;
                if (copy.getId() != rootId || copy.getStream() != rootCudaStream)
                    workerFailed.store(true, std::memory_order_relaxed);

                Stream moved = std::move(copy);
                if (copy.isInitialized() || moved.getId() != rootId || moved.getStream() != rootCudaStream)
                    workerFailed.store(true, std::memory_order_relaxed);

                copy = source;
                moved = Stream();
            }
        });
    }

    // Independently-owned source handles keep the state alive while the root
    // handle is reset and worker-local handles churn concurrently.
    root = Stream();

    for (std::thread &worker : workers)
        worker.join();

    EXPECT_FALSE(workerFailed.load(std::memory_order_relaxed));
    ASSERT_FALSE(stableSources.empty());
    EXPECT_EQ(stableSources.front().getId(), rootId);
    EXPECT_EQ(stableSources.front().getStream(), rootCudaStream);
    stableSources.front().synchronize();
}
