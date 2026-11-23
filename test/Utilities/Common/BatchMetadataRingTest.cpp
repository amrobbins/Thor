#include "Utilities/Common/BatchMetadataRing.h"

#include "Utilities/Common/Event.h"
#include "Utilities/Common/Stream.h"
#include "gtest/gtest.h"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                \
    do {                                                                                      \
        int cudaDeviceCountForTest = 0;                                                       \
        if (cudaGetDeviceCount(&cudaDeviceCountForTest) != cudaSuccess || cudaDeviceCountForTest <= 0) { \
            GTEST_SKIP() << "CUDA device is required for BatchMetadataRing tests.";          \
        }                                                                                     \
    } while (false)

// The bytes of a table for a batch.
std::vector<uint8_t> tableBytes(uint32_t batch, uint32_t table, uint64_t bytes) {
    std::vector<uint8_t> values(bytes);
    for (uint64_t i = 0; i < bytes; ++i) values[i] = static_cast<uint8_t>(31 * batch + 7 * table + 3 * i + 1);
    return values;
}

void fillTable(BatchMetadataRing &ring, BatchMetadataRing::TableId table, const std::vector<uint8_t> &values) {
    uint8_t *host = ring.openTable(table);
    if (!values.empty()) std::memcpy(host, values.data(), values.size());
    ring.closeTable(table, values.size());
}

// The device bytes at `device`, read on a stream of their own that waits on nothing.
std::vector<uint8_t> readDevice(const void *device, uint64_t bytes) {
    cudaStream_t stream = nullptr;
    EXPECT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    std::vector<uint8_t> values(bytes);
    if (bytes != 0) EXPECT_EQ(cudaMemcpyAsync(values.data(), device, bytes, cudaMemcpyDeviceToHost, stream), cudaSuccess);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    return values;
}

// A host function that holds its stream until the gate opens. The destructor opens it, so a failing test cannot leave
// the stream held.
struct Gate {
    std::atomic<bool> open{false};
    ~Gate() { open = true; }
    static void CUDART_CB hold(void *gate) {
        while (!static_cast<Gate *>(gate)->open.load()) std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
};

}  // namespace

TEST(BatchMetadataRing, TablesPackIntoTheSlotOneAfterAnotherOnSixteenByteBoundaries) {
    REQUIRE_CUDA_DEVICE();
    BatchMetadataRing ring;
    const BatchMetadataRing::TableId first = ring.declareTable(100);
    const BatchMetadataRing::TableId second = ring.declareTable(33);
    const BatchMetadataRing::TableId third = ring.declareTable(16);
    EXPECT_EQ(ring.getNumTables(), 3U);
    EXPECT_EQ(ring.getSlotBytes(), 112U + 48U + 16U);
    ring.reserveSlots(0, 2);
    EXPECT_EQ(ring.getDepth(), 2U);
    EXPECT_THROW(ring.declareTable(8), std::logic_error);

    // Filled in another order than declared, the second table holding nothing.
    ring.beginBatch(1);
    fillTable(ring, third, tableBytes(0, 2, 16));
    fillTable(ring, second, {});
    fillTable(ring, first, tableBytes(0, 0, 37));
    const uint8_t *thirdDevice = static_cast<const uint8_t *>(ring.deviceTable(third));
    EXPECT_EQ(static_cast<const uint8_t *>(ring.deviceTable(second)), thirdDevice + 16);
    EXPECT_EQ(static_cast<const uint8_t *>(ring.deviceTable(first)), thirdDevice + 16);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(thirdDevice) % BatchMetadataRing::kTableAlignmentBytes, 0U);
    ring.commitBatch();
    EXPECT_EQ(ring.getLastCommittedBytes(), 16U + 37U);
    Event uploaded = ring.uploadedEvent();
    uploaded.synchronize();
    EXPECT_EQ(readDevice(ring.deviceTable(third), 16), tableBytes(0, 2, 16));
    EXPECT_EQ(readDevice(ring.deviceTable(first), 37), tableBytes(0, 0, 37));
    Stream stream(0);
    ring.releaseBatch(stream);
    stream.synchronize();
}

TEST(BatchMetadataRing, RejectsMisuse) {
    REQUIRE_CUDA_DEVICE();
    BatchMetadataRing ring;
    EXPECT_THROW(ring.declareTable(0), std::invalid_argument);
    const BatchMetadataRing::TableId table = ring.declareTable(32);
    const BatchMetadataRing::TableId other = ring.declareTable(32);
    EXPECT_THROW(ring.beginBatch(0), std::logic_error);  // no slots yet
    EXPECT_THROW(ring.reserveSlots(0, 0), std::invalid_argument);
    ring.reserveSlots(0, 2);
    EXPECT_THROW(ring.beginBatch(2), std::out_of_range);
    EXPECT_THROW((void)ring.openTable(table), std::logic_error);  // outside a batch
    ring.beginBatch(0);
    EXPECT_THROW((void)ring.openTable(7), std::out_of_range);
    EXPECT_THROW((void)ring.deviceTable(table), std::logic_error);  // not written
    (void)ring.openTable(table);
    EXPECT_THROW((void)ring.openTable(other), std::logic_error);  // one at a time
    EXPECT_THROW(ring.closeTable(other, 8), std::logic_error);    // not the open one
    EXPECT_THROW(ring.commitBatch(), std::logic_error);           // a table still open
    EXPECT_THROW(ring.closeTable(table, 33), std::length_error);  // past its capacity
    ring.closeTable(table, 32);
    EXPECT_THROW((void)ring.openTable(table), std::logic_error);  // twice in one batch
    ring.commitBatch();
    EXPECT_THROW(ring.commitBatch(), std::logic_error);
    ring.uploadedEvent().synchronize();
    Stream stream(0);
    ring.releaseBatch(stream);
    ring.releaseBatch(stream);  // a second release of the same batch does nothing
    stream.synchronize();
    ring.clear();
    EXPECT_EQ(ring.getDepth(), 0U);
    EXPECT_EQ(ring.getNumTables(), 0U);
}

TEST(BatchMetadataRing, ARingWithoutTablesDoesNothing) {
    REQUIRE_CUDA_DEVICE();
    BatchMetadataRing ring;
    ring.reserveSlots(0, 1);
    ring.beginBatch(5);  // any queue slot: there is nothing to keep apart
    ring.commitBatch();
    EXPECT_THROW((void)ring.uploadedEvent(), std::logic_error);
    Stream stream(0);
    ring.releaseBatch(stream);
    EXPECT_FALSE(ring.synchronizeEvent().has_value());
}

TEST(BatchMetadataRing, EveryBatchReadsItsOwnTablesAcrossTheQueue) {
    REQUIRE_CUDA_DEVICE();
    constexpr uint32_t kDepth = 3;
    constexpr uint32_t kBatches = 11;
    BatchMetadataRing ring;
    const BatchMetadataRing::TableId plan = ring.declareTable(4000);
    const BatchMetadataRing::TableId small = ring.declareTable(24);
    ring.reserveSlots(0, kDepth);

    // Every batch is enqueued before any is read back: the reads, on a processing stream, wait only on their batch's
    // upload, and release the slot after reading it.
    Stream processing(0);
    std::vector<std::vector<uint8_t>> planRead(kBatches), smallRead(kBatches);
    std::vector<uint64_t> planBytes(kBatches);
    for (uint32_t batch = 0; batch < kBatches; ++batch) {
        planBytes[batch] = 16 * (1 + (batch * 37) % 250) + batch % 3;
        planRead[batch].resize(planBytes[batch]);
        smallRead[batch].resize(24);
        ring.beginBatch(batch % kDepth);
        fillTable(ring, plan, tableBytes(batch, 0, planBytes[batch]));
        fillTable(ring, small, tableBytes(batch, 1, 24));
        ring.commitBatch();
        processing.waitEvent(ring.uploadedEvent());
        ASSERT_EQ(cudaMemcpyAsync(planRead[batch].data(), ring.deviceTable(plan), planBytes[batch], cudaMemcpyDeviceToHost,
                                  processing.getStream()),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(smallRead[batch].data(), ring.deviceTable(small), 24, cudaMemcpyDeviceToHost, processing.getStream()),
                  cudaSuccess);
        ring.releaseBatch(processing);
    }
    processing.synchronize();
    for (uint32_t batch = 0; batch < kBatches; ++batch) {
        EXPECT_EQ(planRead[batch], tableBytes(batch, 0, planBytes[batch])) << "batch " << batch;
        EXPECT_EQ(smallRead[batch], tableBytes(batch, 1, 24)) << "batch " << batch;
    }
}

TEST(BatchMetadataRing, ASlotIsRewrittenOnlyOnceItsBatchIsReleased) {
    REQUIRE_CUDA_DEVICE();
    BatchMetadataRing ring;
    const BatchMetadataRing::TableId table = ring.declareTable(64);
    ring.reserveSlots(0, 2);
    Stream processing(0);
    Stream other(0);
    Gate gate;

    // Batch 0 on slot 0: its processing is held at the gate before it releases the slot.
    ring.beginBatch(0);
    fillTable(ring, table, tableBytes(0, 0, 64));
    ring.commitBatch();
    const void *slot0 = ring.deviceTable(table);
    processing.waitEvent(ring.uploadedEvent());
    ASSERT_EQ(cudaLaunchHostFunc(processing.getStream(), &Gate::hold, &gate), cudaSuccess);
    ring.releaseBatch(processing);

    // Batch 1 on slot 1, processed and released on another stream.
    ring.beginBatch(1);
    fillTable(ring, table, tableBytes(1, 0, 64));
    ring.commitBatch();
    other.waitEvent(ring.uploadedEvent());
    ring.releaseBatch(other);
    other.synchronize();

    // Batch 2 takes slot 0 again. The host may fill the slot's pinned arena (its previous copy is done), but the copy
    // waits on batch 0's release, behind the gate: slot 0 still holds batch 0's bytes.
    ring.beginBatch(0);
    fillTable(ring, table, tableBytes(2, 0, 64));
    ring.commitBatch();
    EXPECT_EQ(ring.deviceTable(table), slot0);
    Event uploaded = ring.uploadedEvent();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_EQ(cudaEventQuery(uploaded.getEvent()), cudaErrorNotReady);
    EXPECT_EQ(readDevice(slot0, 64), tableBytes(0, 0, 64));

    gate.open = true;
    uploaded.synchronize();
    EXPECT_EQ(readDevice(slot0, 64), tableBytes(2, 0, 64));
    processing.waitEvent(uploaded);
    ring.releaseBatch(processing);
    processing.synchronize();
}
