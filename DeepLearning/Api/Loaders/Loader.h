#pragma once

#include "DeepLearning/Api/Loaders/Batch.h"
#include "DeepLearning/Api/Loaders/DeviceDatasetMaterialization.h"
#include "Utilities/Loaders/Shard.h"

#include <memory>
#include <stdexcept>
#include <string>

class Loader {
   public:
    virtual ~Loader() {}

    virtual Batch getBatch(ExampleType exampleType, uint64_t &batchNum) = 0;
    virtual void returnBatchBuffers(ExampleType exampleType, Batch&& batch) = 0;

    virtual uint64_t getBatchSize() { return batchSize; }
    virtual uint64_t getNumBatchesPerEpoch(ExampleType exampleType) = 0;
    virtual uint64_t getNumExamples(ExampleType exampleType) = 0;
    virtual uint64_t getNextBatchNum(ExampleType exampleType) = 0;

    [[nodiscard]] virtual bool supportsDeviceDatasetMaterialization() const { return false; }
    [[nodiscard]] virtual std::string getDeviceDatasetMaterializationUnsupportedReason() const {
        return "loader_not_materializable";
    }
    [[nodiscard]] virtual DeviceDatasetMaterializationView describeDeviceDatasetMaterialization() const {
        throw std::runtime_error("Loader does not support device dataset materialization.");
    }

    virtual void setDatasetName(std::string datasetName) { this->datasetName = datasetName; }
    virtual std::string getDatasetName() { return datasetName; }

   protected:
    uint64_t batchSize;
    std::string datasetName;
};
