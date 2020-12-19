/*

#pragma once

#include "Utilities/Loaders/ShardedRawDatasetCreator.h"
#include "Utilities/Random/FullPeriodRandom.h"
#include "Utilities/WorkQueue/AsyncQueue.h"
#include "Utilities/WorkQueue/AsyncTensorQueue.h"

#include <thread>

struct LabeledExample {
    string label;
    string filename;
    vector<uint8_t> data;
};
*/
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
/*
class BatchAssembler {
  public:
   BatchAssembler(vector<Shard *> shards,
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
   vector<Shard *> shards;
   vector<FullPeriodRandom> randomizers;
   ExampleType exampleType;
   vector<uint64_t> numExamplesPerShard;
   ThorImplementation::TensorDescriptor exampleDescriptor;
   ThorImplementation::TensorDescriptor batchDataTensorDescriptor;
   ThorImplementation::TensorDescriptor batchLabelTensorDescriptor;
   uint64_t batchSize;
   uint64_t numExamples;
   uint64_t batchesPerEpoch;
   uint64_t currentBatchNum;

   unordered_map<string, uint64_t> classIndexes;

   vector<AsyncQueue<LabeledExample>> shardQueues;
   AsyncTensorQueue batchDataQueue;
   AsyncTensorQueue batchLabelQueue;
   deque<uint64_t> batchNumQueue;

   vector<std::thread> shardThreads;
   std::thread assemblerThread;

   void open();
   void close();

   void shardReaderThread(uint64_t shard);
   void batchAssemblerThread();
};
*/
