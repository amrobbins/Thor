#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Loaders/Shard.h"

#include <assert.h>
#include <map>
#include <memory>
#include <string>

using std::shared_ptr;

class Loader {
   public:
    virtual ~Loader() {}

    virtual std::map<std::string, ThorImplementation::Tensor> getBatch(ExampleType exampleType, uint64_t &batchNum) = 0;
    virtual void returnBatchBuffers(ExampleType exampleType, std::map<std::string, ThorImplementation::Tensor> tensorMap) = 0;

    virtual uint64_t getBatchSize() { return batchSize; }
    virtual uint64_t getNumBatchesPerEpoch() = 0;

   protected:
    uint64_t batchSize;
};
