#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Loaders/LocalNamedExampleLayout.h"
#include "Utilities/Loaders/Shard.h"
#include "Utilities/Random/FullPeriodRandom.h"
#include "Utilities/WorkQueue/AsyncQueue.h"
#include "Utilities/WorkQueue/AsyncTensorQueue.h"

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

struct LocalNamedExampleRecord {
    std::vector<uint8_t> data;
};

/**
 * Constructs batches of named tensors from local named-example shard records.
 *
 * Each shard record is one full, fixed-size example laid out according to a
 * LocalNamedExampleLayout.  The assembler reads full contiguous records through
 * the same Shard/UringDirect path as LocalBatchLoader, then memcpy's each named
 * slice into its own dense [batch, *shape] CPU tensor buffer.
 */
class LocalNamedBatchAssembler {
   public:
    LocalNamedBatchAssembler(std::vector<std::shared_ptr<Shard>> shards,
                             ExampleType exampleType,
                             LocalNamedExampleLayout layout,
                             uint64_t batchSize,
                             uint64_t batchQueueDepth = 32,
                             bool randomizeExamples = true,
                             std::optional<uint64_t> seed = std::nullopt);
    ~LocalNamedBatchAssembler();

    LocalNamedBatchAssembler(const LocalNamedBatchAssembler &) = delete;
    LocalNamedBatchAssembler &operator=(const LocalNamedBatchAssembler &) = delete;
    LocalNamedBatchAssembler(LocalNamedBatchAssembler &&) = delete;
    LocalNamedBatchAssembler &operator=(LocalNamedBatchAssembler &&) = delete;

    uint64_t getNumBatchesPerEpoch() const;
    uint64_t getNumExamples() const;
    uint64_t getNextBatchNum();

    void getBatch(std::map<std::string, ThorImplementation::Tensor> &tensors, uint64_t &batchNum);
    void returnBuffers(const std::map<std::string, ThorImplementation::Tensor> &tensors);

   private:
    std::vector<std::shared_ptr<Shard>> shards;
    std::vector<uint64_t> numExamplesPerShard;
    std::vector<std::unique_ptr<FullPeriodRandom>> randomizers;
    ExampleType exampleType;
    LocalNamedExampleLayout layout;
    std::map<std::string, ThorImplementation::TensorDescriptor> batchTensorDescriptors;

    uint64_t batchSize;
    uint64_t batchQueueDepth;
    uint64_t shardReadQueueDepth;
    uint64_t shardExampleQueueDepth;
    uint64_t recordSizeBytes;
    uint64_t numExamples;
    uint64_t batchesPerEpoch;
    bool randomizeExamples;
    std::optional<uint64_t> seed;
    std::unique_ptr<FullPeriodRandom> shardSelectionRandomizer;
    uint64_t sequentialGlobalExampleIndex = 0;

    std::vector<std::unique_ptr<AsyncQueue<LocalNamedExampleRecord>>> shardQueues;
    std::map<std::string, std::unique_ptr<AsyncTensorQueue>> batchTensorQueues;
    AsyncQueue<uint64_t> batchNumQueue;

    std::vector<std::thread> shardThreads;
    std::thread assemblerThread;

    void open();
    void close();

    void shardReaderThread(uint64_t shardIndex);
    void batchAssemblerThread();
    uint64_t nextShardSelection();
    void validateReturnedTensorMapExact(const std::map<std::string, ThorImplementation::Tensor> &tensors) const;
};
