#pragma once

#include "DeepLearning/Api/Loaders/Batch.h"
#include "Utilities/Loaders/Shard.h"

#include <memory>
#include <string>

class Loader {
   public:
    virtual ~Loader() {}

    virtual Batch getBatch(ExampleType exampleType, uint64_t &batchNum) = 0;
    virtual void returnBatchBuffers(ExampleType exampleType, Batch&& batch) = 0;

    virtual uint64_t getBatchSize() const { return batchSize; }
    virtual uint64_t getNumBatchesPerEpoch(ExampleType exampleType) = 0;
    virtual uint64_t getNumExamples(ExampleType exampleType) = 0;
    virtual uint64_t getNextBatchNum(ExampleType exampleType) = 0;

    virtual void setDatasetName(std::string datasetName) { this->datasetName = datasetName; }
    virtual std::string getDatasetName() const { return datasetName; }

   protected:
    uint64_t batchSize;
    std::string datasetName;
};
