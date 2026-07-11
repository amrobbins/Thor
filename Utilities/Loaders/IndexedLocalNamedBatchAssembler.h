#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Loaders/IndexedLocalNamedExampleReader.h"
#include "DeepLearning/Api/Data/DatasetLayout.h"
#include "Utilities/Random/FullPeriodRandom.h"
#include "Utilities/WorkQueue/AsyncQueue.h"
#include "Utilities/WorkQueue/AsyncTensorQueue.h"

#include <atomic>
#include <chrono>
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
    uint64_t windowedSourceReadCalls = 0;
    uint64_t windowedSourceReadBytes = 0;
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

    uint64_t loadWorkPopWaitNanoseconds = 0;
    uint64_t loadWorkPopCalls = 0;
    uint64_t loadWorkerBatches = 0;
    uint64_t loadWorkerActiveNanoseconds = 0;
    uint64_t loadWorkerReadSubmitNanoseconds = 0;
    uint64_t loadWorkerReadDrainNanoseconds = 0;
    uint64_t loadWorkerCompletedBatchPushWaitNanoseconds = 0;
    uint64_t readvSubmitNanoseconds = 0;
    uint64_t readvSubmitBackpressureCount = 0;
    uint64_t readvSubmitBackpressureNanoseconds = 0;
    uint64_t readvCompletionWaitCalls = 0;
    uint64_t readvCompletionWaitNanoseconds = 0;
    uint64_t readerDrainCalls = 0;
    uint64_t readerDrainNanoseconds = 0;
    uint64_t readerDrainContextVisits = 0;
    uint64_t readerDrainSubmitCalls = 0;
    uint64_t readerDrainSubmitNanoseconds = 0;
    uint64_t readerDrainWaitLoopNanoseconds = 0;
    uint64_t readerDrainCompletionProcessNanoseconds = 0;
    uint64_t readerDrainCompletions = 0;
    uint64_t readerDrainMaxInflightReads = 0;
    uint64_t readerShardContextOpenCount = 0;
    uint64_t readerMaxOpenShardContexts = 0;
    uint64_t readerLoadExampleCalls = 0;
    uint64_t readerLoadExampleNanoseconds = 0;
    uint64_t readerResolveShardNanoseconds = 0;
    uint64_t readerShardContextLookupCalls = 0;
    uint64_t readerShardContextCacheHits = 0;
    uint64_t readerShardContextCacheMisses = 0;
    uint64_t readerShardContextLookupNanoseconds = 0;
    uint64_t readerShardReadRequestNanoseconds = 0;
    uint64_t readerIovecSlotAcquireNanoseconds = 0;
    uint64_t readerIovecFillNanoseconds = 0;
    uint64_t readerReadvSubmitCallNanoseconds = 0;
    uint64_t getBatchCalls = 0;
    uint64_t getBatchReadyQueueEmptyCount = 0;
    uint64_t getBatchImmediateCount = 0;
    uint64_t getBatchWaitNanoseconds = 0;
    uint64_t getBatchTensorUnloadWaitNanoseconds = 0;
    uint64_t returnBufferCalls = 0;
    uint64_t returnBufferWaitNanoseconds = 0;
    uint64_t startBatchCalls = 0;
    uint64_t startBatchTensorAcquireNanoseconds = 0;
    uint64_t startBatchPlanningNanoseconds = 0;
    uint64_t pushLoadWorkWaitNanoseconds = 0;
    uint64_t waitForCompletedBatchCalls = 0;
    uint64_t waitForCompletedBatchNanoseconds = 0;
    uint64_t publishCompletedBatchCalls = 0;
    uint64_t publishCompletedBatchNanoseconds = 0;
    uint64_t currentPendingLoadedBatches = 0;
    uint64_t currentPendingUnloadedBatches = 0;
    uint64_t oldestPendingBatchAgeNanoseconds = 0;
    uint64_t averagePendingBatchAgeNanoseconds = 0;

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
    std::vector<uint8_t *> windowedTensorBasePointers;
    std::vector<uint8_t *> windowedMaskBasePointers;
    std::vector<uint64_t> globalExampleIndices;
    std::chrono::steady_clock::time_point pendingSince;
    std::chrono::steady_clock::time_point loadedAt;
};

/**
 * Indexed split assembler for IndexedNamedBatchSession.
 *
 * Owns the per-split async read/prefetch pipeline:
 *   logical split positions -> global row ids -> reader sessions direct-load into
 *   dense named CPU batch tensors -> ready batch queues consumed by getBatch().
 */
class IndexedLocalNamedBatchAssembler {
   public:
    IndexedLocalNamedBatchAssembler(std::shared_ptr<IndexedLocalNamedExampleReader> reader,
                                    DatasetLayout layout,
                                    std::shared_ptr<const std::vector<uint64_t>> indices,
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
    [[nodiscard]] const std::vector<uint64_t> &getIndices() const { return *indices; }
    [[nodiscard]] bool isRandomized() const { return randomized; }
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
    DatasetLayout layout;
    std::shared_ptr<const std::vector<uint64_t>> indices;
    std::string splitName;
    std::map<std::string, ThorImplementation::TensorDescriptor> batchTensorDescriptors;
    std::map<std::string, std::unique_ptr<AsyncTensorQueue>> batchTensorQueues;
    std::vector<uint64_t> layoutTensorOrdinals;
    std::vector<uint64_t> layoutWindowedTensorOrdinals;

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
    std::atomic<uint64_t> statsWindowedSourceReadCalls{0};
    std::atomic<uint64_t> statsWindowedSourceReadBytes{0};
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
    std::atomic<uint64_t> statsLoadWorkPopWaitNanoseconds{0};
    std::atomic<uint64_t> statsLoadWorkPopCalls{0};
    std::atomic<uint64_t> statsLoadWorkerBatches{0};
    std::atomic<uint64_t> statsLoadWorkerActiveNanoseconds{0};
    std::atomic<uint64_t> statsLoadWorkerReadSubmitNanoseconds{0};
    std::atomic<uint64_t> statsLoadWorkerReadDrainNanoseconds{0};
    std::atomic<uint64_t> statsLoadWorkerCompletedBatchPushWaitNanoseconds{0};
    std::atomic<uint64_t> statsReadvSubmitNanoseconds{0};
    std::atomic<uint64_t> statsReadvSubmitBackpressureCount{0};
    std::atomic<uint64_t> statsReadvSubmitBackpressureNanoseconds{0};
    std::atomic<uint64_t> statsReadvCompletionWaitCalls{0};
    std::atomic<uint64_t> statsReadvCompletionWaitNanoseconds{0};
    std::atomic<uint64_t> statsReaderDrainCalls{0};
    std::atomic<uint64_t> statsReaderDrainNanoseconds{0};
    std::atomic<uint64_t> statsReaderDrainContextVisits{0};
    std::atomic<uint64_t> statsReaderDrainSubmitCalls{0};
    std::atomic<uint64_t> statsReaderDrainSubmitNanoseconds{0};
    std::atomic<uint64_t> statsReaderDrainWaitLoopNanoseconds{0};
    std::atomic<uint64_t> statsReaderDrainCompletionProcessNanoseconds{0};
    std::atomic<uint64_t> statsReaderDrainCompletions{0};
    std::atomic<uint64_t> statsReaderDrainMaxInflightReads{0};
    std::atomic<uint64_t> statsReaderShardContextOpenCount{0};
    std::atomic<uint64_t> statsReaderMaxOpenShardContexts{0};
    std::atomic<uint64_t> statsReaderLoadExampleCalls{0};
    std::atomic<uint64_t> statsReaderLoadExampleNanoseconds{0};
    std::atomic<uint64_t> statsReaderResolveShardNanoseconds{0};
    std::atomic<uint64_t> statsReaderShardContextLookupCalls{0};
    std::atomic<uint64_t> statsReaderShardContextCacheHits{0};
    std::atomic<uint64_t> statsReaderShardContextCacheMisses{0};
    std::atomic<uint64_t> statsReaderShardContextLookupNanoseconds{0};
    std::atomic<uint64_t> statsReaderShardReadRequestNanoseconds{0};
    std::atomic<uint64_t> statsReaderIovecSlotAcquireNanoseconds{0};
    std::atomic<uint64_t> statsReaderIovecFillNanoseconds{0};
    std::atomic<uint64_t> statsReaderReadvSubmitCallNanoseconds{0};
    std::atomic<uint64_t> statsGetBatchCalls{0};
    std::atomic<uint64_t> statsGetBatchReadyQueueEmptyCount{0};
    std::atomic<uint64_t> statsGetBatchImmediateCount{0};
    std::atomic<uint64_t> statsGetBatchWaitNanoseconds{0};
    std::atomic<uint64_t> statsGetBatchTensorUnloadWaitNanoseconds{0};
    std::atomic<uint64_t> statsReturnBufferCalls{0};
    std::atomic<uint64_t> statsReturnBufferWaitNanoseconds{0};
    std::atomic<uint64_t> statsStartBatchCalls{0};
    std::atomic<uint64_t> statsStartBatchTensorAcquireNanoseconds{0};
    std::atomic<uint64_t> statsStartBatchPlanningNanoseconds{0};
    std::atomic<uint64_t> statsPushLoadWorkWaitNanoseconds{0};
    std::atomic<uint64_t> statsWaitForCompletedBatchCalls{0};
    std::atomic<uint64_t> statsWaitForCompletedBatchNanoseconds{0};
    std::atomic<uint64_t> statsPublishCompletedBatchCalls{0};
    std::atomic<uint64_t> statsPublishCompletedBatchNanoseconds{0};
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
    void fillPendingBatchAgeStats(IndexedLocalNamedBatchAssemblerStats &stats) const;
    uint64_t nextLogicalSplitPosition();
    void validateGlobalIndex(uint64_t index, const char *context) const;
    void validateReturnedTensorMapExact(const std::map<std::string, ThorImplementation::Tensor> &tensors) const;
    void recordWorkerException(std::exception_ptr exception);
    void throwIfWorkerFailed() const;
    void setResolvedIoBackend(const std::string &backendName);
    void emitStatsIfEnabled(const char *event, uint64_t batchNum = 0);
};
