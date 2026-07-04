#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Loaders/IndexedLocalNamedExampleReader.h"
#include "Utilities/Loaders/LocalNamedExampleLayout.h"
#include "Utilities/Random/FullPeriodRandom.h"
#include "Utilities/WorkQueue/AsyncQueue.h"
#include "Utilities/WorkQueue/AsyncTensorQueue.h"

#include <atomic>
#include <cstdint>
#include <condition_variable>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>


struct IndexedLocalNamedBatchAssemblerStats {
    std::string splitName;
    uint64_t recordsRequested = 0;
    uint64_t logicalRecordBytesRequested = 0;
    uint64_t readCallsSubmitted = 0;
    uint64_t readBytesSubmitted = 0;
    uint64_t readCallsCompleted = 0;
    uint64_t readBytesCompleted = 0;
    uint64_t recordsCopied = 0;
    uint64_t recordCopyBytes = 0;
    uint64_t recordCopyMemcpyCalls = 0;
    uint64_t recordCopyActiveNanoseconds = 0;
    uint64_t recordCopyPopWaitNanoseconds = 0;
    uint64_t completedRecordQueuePushWaitNanoseconds = 0;
    uint64_t copiedRecordQueuePushWaitNanoseconds = 0;
    uint64_t recordBufferPoolCapacity = 0;
    uint64_t currentRecordBufferPoolDepth = 0;
    uint64_t batchesAssembled = 0;
    uint64_t batchesDelivered = 0;
    uint64_t batchBuffersReturned = 0;
    uint64_t currentReadyBatches = 0;
    uint64_t currentPendingBatches = 0;
    uint64_t currentCompletedRecordQueueDepth = 0;
    uint64_t currentCopiedRecordQueueDepth = 0;
    uint64_t targetBatchQueueDepth = 0;
    uint64_t shardReadQueueDepth = 0;
    uint64_t shardRequestQueueDepth = 0;
    uint64_t completedRecordQueueDepth = 0;
    uint64_t recordCopyThreadCount = 0;
    uint64_t recordSizeBytes = 0;
    std::string resolvedIoBackend;

    double readAmplification() const {
        // read_amplification is a physical-read/logical-read ratio.  With the
        // asynchronous indexed loader, records can be planned before their
        // reader-session stats have been flushed back into this snapshot, so
        // logicalRecordBytesRequested can legitimately lead readBytesSubmitted.
        // Use the logical byte count represented by submitted read calls instead
        // of the broader planning counter, otherwise a normal prefetch lead makes
        // a direct 1:1 read path appear to have amplification below 1.0.
        if (readCallsSubmitted == 0 || recordSizeBytes == 0) {
            return 0.0;
        }
        const double logicalBytesSubmitted =
            static_cast<double>(readCallsSubmitted) * static_cast<double>(recordSizeBytes);
        if (logicalBytesSubmitted == 0.0) {
            return 0.0;
        }
        return static_cast<double>(readBytesSubmitted) / logicalBytesSubmitted;
    }

    double planningLeadRecords() const {
        if (recordsRequested < readCallsSubmitted) {
            return 0.0;
        }
        return static_cast<double>(recordsRequested - readCallsSubmitted);
    }

    double averageCopyNanosecondsPerRecord() const {
        if (recordsCopied == 0) {
            return 0.0;
        }
        return static_cast<double>(recordCopyActiveNanoseconds) / static_cast<double>(recordsCopied);
    }

    double averageCopyMemcpyCallsPerRecord() const {
        if (recordsCopied == 0) {
            return 0.0;
        }
        return static_cast<double>(recordCopyMemcpyCalls) / static_cast<double>(recordsCopied);
    }

    double averageCopyBytesPerRecord() const {
        if (recordsCopied == 0) {
            return 0.0;
        }
        return static_cast<double>(recordCopyBytes) / static_cast<double>(recordsCopied);
    }
};

struct IndexedLocalNamedBatchState;

struct IndexedLocalNamedCompletedBatch {
    uint64_t batchOrdinal = 0;
};

struct IndexedLocalNamedBatchLoadWork {
    IndexedLocalNamedBatchState *batchState = nullptr;
    uint64_t batchOrdinal = 0;
    uint64_t slotBegin = 0;
    uint64_t slotEnd = 0;
};

struct IndexedLocalNamedBatchState {
    uint64_t batchOrdinal = 0;
    uint64_t batchNum = 0;
    uint64_t expectedRecords = 0;
    uint64_t expectedLoadChunks = 0;
    std::atomic<uint64_t> completedLoadChunks{0};
    bool loadComplete = false;
    std::map<std::string, ThorImplementation::Tensor> tensors;
    std::vector<uint8_t *> tensorBasePointers;
    std::vector<uint64_t> globalExampleIndices;
};

/**
 * Indexed split assembler for IndexedLocalNamedBatchLoader.
 *
 * Owns the per-split async read/prefetch pipeline:
 *   logical split positions -> global row ids -> reader sessions direct-load into
 *   dense named CPU batch tensors -> ready batch queues consumed by getBatch().
 */
class IndexedLocalNamedBatchAssembler {
   public:
    IndexedLocalNamedBatchAssembler(std::shared_ptr<IndexedLocalNamedExampleReader> reader,
                                    LocalNamedExampleLayout layout,
                                    std::vector<uint64_t> indices,
                                    std::string splitName,
                                    uint64_t batchSize,
                                    uint64_t batchQueueDepth,
                                    bool randomized = false,
                                    std::optional<uint64_t> seed = std::nullopt);
    ~IndexedLocalNamedBatchAssembler();

    IndexedLocalNamedBatchAssembler(const IndexedLocalNamedBatchAssembler &) = delete;
    IndexedLocalNamedBatchAssembler &operator=(const IndexedLocalNamedBatchAssembler &) = delete;
    IndexedLocalNamedBatchAssembler(IndexedLocalNamedBatchAssembler &&) = delete;
    IndexedLocalNamedBatchAssembler &operator=(IndexedLocalNamedBatchAssembler &&) = delete;

    uint64_t getNumBatchesPerEpoch() const;
    uint64_t getNumExamples() const;
    uint64_t getNextBatchNum();
    IndexedLocalNamedBatchAssemblerStats getStatsSnapshot();

#ifdef THOR_GTEST
    uint64_t getReadyBatchCountForTesting() {
        throwIfWorkerFailed();
        const int occupancy = batchNumQueue.occupancy();
        THOR_THROW_IF_FALSE(occupancy >= 0);
        return static_cast<uint64_t>(occupancy);
    }
#endif

    void getBatch(std::map<std::string, ThorImplementation::Tensor> &tensors, uint64_t &batchNum);
    void returnBuffers(const std::map<std::string, ThorImplementation::Tensor> &tensors);

   private:
    std::shared_ptr<IndexedLocalNamedExampleReader> reader;
    LocalNamedExampleLayout layout;
    std::vector<uint64_t> indices;
    std::string splitName;
    std::map<std::string, ThorImplementation::TensorDescriptor> batchTensorDescriptors;
    std::map<std::string, std::unique_ptr<AsyncTensorQueue>> batchTensorQueues;
    std::vector<uint64_t> layoutTensorOrdinals;

    uint64_t batchSize;
    uint64_t batchQueueDepth;
    uint64_t shardReadQueueDepth;
    uint64_t shardRequestQueueDepth;
    uint64_t completedBatchQueueDepth;
    uint64_t recordSizeBytes;
    uint64_t batchesPerEpoch;
    uint64_t numDatasetExamples;
    uint64_t nextBatchNum;
    uint64_t nextLogicalPosition;
    uint64_t nextBatchOrdinal;
    uint64_t nextPublishOrdinal;
    bool randomized;
    std::unique_ptr<FullPeriodRandom> randomizer;

    AsyncQueue<IndexedLocalNamedBatchLoadWork> loadWorkQueue;
    AsyncQueue<IndexedLocalNamedCompletedBatch> completedBatchQueue;
    AsyncQueue<uint64_t> batchNumQueue;
    std::vector<std::thread> loadWorkerThreads;
    std::thread assemblerThread;

    std::map<uint64_t, std::shared_ptr<IndexedLocalNamedBatchState>> pendingBatches;
    mutable std::mutex pendingBatchesMutex;

    std::atomic<uint64_t> statsRecordsRequested{0};
    std::atomic<uint64_t> statsLogicalRecordBytesRequested{0};
    std::atomic<uint64_t> statsReadCallsSubmitted{0};
    std::atomic<uint64_t> statsReadBytesSubmitted{0};
    std::atomic<uint64_t> statsReadCallsCompleted{0};
    std::atomic<uint64_t> statsReadBytesCompleted{0};
    std::atomic<uint64_t> statsRecordsCopied{0};
    std::atomic<uint64_t> statsRecordCopyBytes{0};
    std::atomic<uint64_t> statsRecordCopyMemcpyCalls{0};
    std::atomic<uint64_t> statsRecordCopyActiveNanoseconds{0};
    std::atomic<uint64_t> statsRecordCopyPopWaitNanoseconds{0};
    std::atomic<uint64_t> statsCompletedRecordQueuePushWaitNanoseconds{0};
    std::atomic<uint64_t> statsCopiedRecordQueuePushWaitNanoseconds{0};
    std::atomic<uint64_t> statsBatchesAssembled{0};
    std::atomic<uint64_t> statsBatchesDelivered{0};
    std::atomic<uint64_t> statsBatchBuffersReturned{0};
    mutable std::mutex statsMutex;
    std::string resolvedIoBackend;

    mutable std::mutex workerExceptionMutex;
    std::exception_ptr workerException;
    std::mutex batchDeliveryMutex;
    std::mutex returnBuffersMutex;
    uint64_t loadWorkerThreadCount;
    uint64_t loadWorkQueueDepth;
    uint64_t recordCopyThreadCount;
    uint64_t recordBufferPoolDepth;

    void open();
    void close();

    void loadWorkerThread(uint64_t workerIndex);
    void loadWorkerThreadMain(uint64_t workerIndex);
    void batchAssemblerThread();
    void batchAssemblerThreadMain();

    bool startNextBatch();
    bool canStartNextBatchWithoutBlocking();
    bool pushLoadWorkWithDrain(const IndexedLocalNamedBatchLoadWork &work);
    bool waitForCompletedBatch();
    void markAvailableCompletedBatches();
    bool markBatchLoaded(uint64_t batchOrdinal);
    bool publishCompletedBatches();
    uint64_t pendingBatchCount() const;
    uint64_t nextLogicalSplitPosition();
    void validateGlobalIndex(uint64_t index, const char *context) const;
    void validateReturnedTensorMapExact(const std::map<std::string, ThorImplementation::Tensor> &tensors) const;
    void recordWorkerException(std::exception_ptr exception);
    void throwIfWorkerFailed() const;
    void setResolvedIoBackend(const std::string &backendName);
    void emitStatsIfEnabled(const char *event, uint64_t batchNum = 0);
};
