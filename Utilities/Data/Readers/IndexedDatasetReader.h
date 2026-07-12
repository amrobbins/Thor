#pragma once

#include "DeepLearning/Api/Data/DatasetLayout.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

struct IndexedDatasetReaderSessionStats {
    uint64_t readCallsSubmitted = 0;
    uint64_t readBytesSubmitted = 0;
    uint64_t readCallsCompleted = 0;
    uint64_t readBytesCompleted = 0;
    uint64_t windowedSourceReadCalls = 0;
    uint64_t windowedSourceReadBytes = 0;
    uint64_t readvSubmitNanoseconds = 0;
    uint64_t readvSubmitBackpressureCount = 0;
    uint64_t readvSubmitBackpressureNanoseconds = 0;
    uint64_t readvCompletionWaitCalls = 0;
    uint64_t readvCompletionWaitNanoseconds = 0;
    uint64_t drainCalls = 0;
    uint64_t drainNanoseconds = 0;
    uint64_t drainContextVisits = 0;
    uint64_t drainSubmitCalls = 0;
    uint64_t drainSubmitNanoseconds = 0;
    uint64_t drainWaitLoopNanoseconds = 0;
    uint64_t drainCompletionProcessNanoseconds = 0;
    uint64_t drainCompletions = 0;
    uint64_t drainMaxInflightReads = 0;
    uint64_t shardContextOpenCount = 0;
    uint64_t maxOpenShardContexts = 0;
    uint64_t loadExampleCalls = 0;
    uint64_t loadExampleNanoseconds = 0;
    uint64_t resolveShardNanoseconds = 0;
    uint64_t shardContextLookupCalls = 0;
    uint64_t shardContextCacheHits = 0;
    uint64_t shardContextCacheMisses = 0;
    uint64_t shardContextLookupNanoseconds = 0;
    uint64_t shardReadRequestNanoseconds = 0;
    uint64_t iovecSlotAcquireNanoseconds = 0;
    uint64_t iovecFillNanoseconds = 0;
    uint64_t readvSubmitCallNanoseconds = 0;
    std::vector<std::string> resolvedIoBackends;
};

/**
 * Passive storage reader for an indexed file dataset.
 *
 * The batch session/assembler owns batching and worker scheduling.  The reader owns
 * the dataset storage layout, shard resolution, file offsets, and per-worker
 * I/O sessions.  Callers pass final tensor batch destinations; sessions read
 * directly into those destinations without exposing shards to the batch session.
 */
class IndexedDatasetReader : public std::enable_shared_from_this<IndexedDatasetReader> {
   public:
    class Session {
       public:
        ~Session();

        Session(const Session &) = delete;
        Session &operator=(const Session &) = delete;
        Session(Session &&) = delete;
        Session &operator=(Session &&) = delete;

        void loadExampleInto(uint64_t globalExampleIndex, uint64_t batchSlot, const std::vector<uint8_t *> &tensorBasePointers);
        void loadDirectExampleInto(uint64_t globalExampleIndex, uint64_t batchSlot, const std::vector<uint8_t *> &tensorBasePointers);
        void loadExampleInto(uint64_t globalExampleIndex,
                             uint64_t batchSlot,
                             const std::vector<uint8_t *> &tensorBasePointers,
                             const std::vector<uint8_t *> &windowedTensorBasePointers,
                             const std::vector<uint8_t *> &windowedMaskBasePointers);
        void drain();
        IndexedDatasetReaderSessionStats takeStats();

       private:
        friend class IndexedDatasetReader;

        class Impl;
        std::unique_ptr<Impl> impl;

        Session(std::shared_ptr<IndexedDatasetReader> reader, uint64_t queueDepth);
    };

    static std::shared_ptr<IndexedDatasetReader> openDataset(const std::filesystem::path &datasetPath);
    static std::shared_ptr<IndexedDatasetReader> openDataset(const std::filesystem::path &datasetPath,
                                                                       const DatasetLayout &requestedLayout);
    ~IndexedDatasetReader();

    IndexedDatasetReader(const IndexedDatasetReader &) = delete;
    IndexedDatasetReader &operator=(const IndexedDatasetReader &) = delete;
    IndexedDatasetReader(IndexedDatasetReader &&) = delete;
    IndexedDatasetReader &operator=(IndexedDatasetReader &&) = delete;

    std::unique_ptr<Session> createSession(uint64_t queueDepth);

    [[nodiscard]] const DatasetLayout &getLayout() const;
    [[nodiscard]] uint64_t getNumExamples() const;
    [[nodiscard]] uint64_t getRecordSizeBytes() const;
    [[nodiscard]] uint64_t getTensorCount() const;
    [[nodiscard]] uint64_t getWindowedTensorCount() const;
    [[nodiscard]] uint64_t getLayoutTensorOrdinal(std::string_view tensorName) const;
    [[nodiscard]] uint64_t getLayoutWindowedTensorOrdinal(std::string_view tensorName) const;
    void validateGlobalIndex(uint64_t index, const char *context) const;

   private:
    class Impl;
    std::unique_ptr<Impl> impl;

    explicit IndexedDatasetReader(std::unique_ptr<Impl> impl);
};
