#pragma once

#include "DeepLearning/Api/Loaders/Loader.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Loaders/BatchAssembler.h"
#include "Utilities/Loaders/Shard.h"

#include <memory>

class LocalBatchLoader : public Loader {
   public:
    LocalBatchLoader(std::set<std::string> shardPaths,
                     ThorImplementation::TensorDescriptor exampleDescriptor,
                     ThorImplementation::TensorDescriptor labelDescriptor,
                     uint64_t batchSize);

    virtual std::map<std::string, ThorImplementation::Tensor> getBatch(ExampleType exampleType, uint64_t &batchNum);
    virtual void returnBatchBuffers(ExampleType exampleType, std::map<std::string, ThorImplementation::Tensor> tensorMap);

    virtual uint64_t getNumBatchesPerEpoch(ExampleType exampleType);
    virtual uint64_t getNumExamples(ExampleType exampleType);
    virtual uint64_t getNextBatchNum(ExampleType exampleType);

   private:
    std::vector<std::shared_ptr<Shard>> shards;

    std::shared_ptr<BatchAssembler> batchAssemblerTrain;
    std::shared_ptr<BatchAssembler> batchAssemblerValidate;
    std::shared_ptr<BatchAssembler> batchAssemblerTest;
};
