#pragma once

#include "DeepLearning/Api/Loaders/Loader.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Loaders/BatchAssembler.h"
#include "Utilities/Loaders/Shard.h"

#include <memory>

class LocalBatchLoader : public Loader {
   public:
    LocalBatchLoader(set<std::string> shardPaths, ThorImplementation::TensorDescriptor exampleDescriptor, uint64_t batchSize);

    virtual std::map<std::string, ThorImplementation::Tensor> getBatch(ExampleType exampleType, uint64_t &batchNum);
    virtual void returnBatchBuffers(ExampleType exampleType, std::map<std::string, ThorImplementation::Tensor> tensorMap);

    virtual uint64_t getNumBatchesPerEpoch();

   private:
    vector<shared_ptr<Shard>> shards;

    shared_ptr<BatchAssembler> batchAssemblerTrain;
    shared_ptr<BatchAssembler> batchAssemblerValidate;
    shared_ptr<BatchAssembler> batchAssemblerTest;
};
