#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Loaders/Shard.h"

#include <assert.h>
#include <map>
#include <memory>
#include <string>

class Loader {
   public:
    virtual ~Loader() {}

    virtual std::map<std::string, ThorImplementation::Tensor> getBatch(ExampleType exampleType, uint64_t &batchNum) = 0;
    virtual void returnBatchBuffers(ExampleType exampleType, std::map<std::string, ThorImplementation::Tensor> tensorMap) = 0;

    virtual uint64_t getBatchSize() { return batchSize; }
    virtual uint64_t getNumBatchesPerEpoch(ExampleType exampleType) = 0;
    virtual uint64_t getNumExamples(ExampleType exampleType) = 0;
    virtual uint64_t getNextBatchNum(ExampleType exampleType) = 0;

    virtual void setDatasetName(std::string datasetName) { this->datasetName = datasetName; }
    virtual std::string getDatasetName() { return datasetName; }

   protected:
    uint64_t batchSize;
    std::string datasetName;
};
