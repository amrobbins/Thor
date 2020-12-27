#pragma once

#include "Utilities/Loaders/MemMappedFileTypes.h"
#include "Utilities/Loaders/ShardedRawDatasetCreator.h"
#include "Utilities/Random/FullPeriodRandom.h"
#include "Utilities/WorkQueue/AsyncQueue.h"
#include "Utilities/WorkQueue/AsyncTensorQueue.h"

#include <memory>
#include <thread>
#include <vector>

struct LabeledExample {
    std::string label;
    std::string filename;
    std::vector<uint8_t> data;
};

/**
 * Constructs batches of random examples from the data in all shards.
 * All examples from all shards are consumed exactly once per epoch,
 * except for the examples in the last batch of the epoch which may have some
 * examples that have been seen already this epoch.
 *
 * Each epoch is randomized differently from the previous one.
 *
 * The batches of examples are written into pinned memory buffers,
 * the buffers need to be returned to the assembler once they have been read out
 * so that they can be reused.
 */
class BatchAssembler {
   public:
    BatchAssembler(std::vector<std::shared_ptr<Shard>> shards,
                   ExampleType exampleType,
                   ThorImplementation::TensorDescriptor exampleDescriptor,
                   uint64_t batchSize);
    virtual ~BatchAssembler();

    uint64_t getNumBatchesPerEpoch();

    void getBatch(ThorImplementation::Tensor &batchTensor,
                  ThorImplementation::Tensor &labelTensor,
                  uint64_t &batchNum,
                  uint64_t &numBatchesInEpoch);
    void returnBuffer(ThorImplementation::Tensor &batchTensor, ThorImplementation::Tensor &labelTensor);

   private:
    std::vector<std::shared_ptr<Shard>> shards;
    std::vector<std::unique_ptr<FullPeriodRandom>> randomizers;
    ExampleType exampleType;
    std::vector<uint64_t> numExamplesPerShard;
    ThorImplementation::TensorDescriptor exampleDescriptor;
    ThorImplementation::TensorDescriptor batchDataTensorDescriptor;
    ThorImplementation::TensorDescriptor batchLabelTensorDescriptor;
    uint64_t batchSize;
    uint64_t numExamples;
    uint64_t batchesPerEpoch;
    uint64_t currentBatchNum;

    std::unordered_map<std::string, uint64_t> classIndexes;

    std::vector<std::unique_ptr<AsyncQueue<LabeledExample>>> shardQueues;
    AsyncTensorQueue batchDataQueue;
    AsyncTensorQueue batchLabelQueue;
    deque<uint64_t> batchNumQueue;

    std::vector<std::thread> shardThreads;
    std::thread assemblerThread;

    void open();
    void close();

    void shardReaderThread(uint64_t shard);
    void batchAssemblerThread();
};
