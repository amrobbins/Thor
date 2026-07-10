#pragma once

#include "DeepLearning/Api/Loaders/Batch.h"
#include "DeepLearning/Api/Loaders/DeviceDatasetMaterialization.h"
#include "Utilities/Loaders/Shard.h"

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace Thor {
class NamedDataset;
}

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
    [[nodiscard]] virtual Thor::DatasetMaterializationDescription describeDeviceDatasetMaterialization() const {
        throw std::runtime_error("Loader does not support device dataset materialization.");
    }
    [[nodiscard]] virtual Thor::DeviceDatasetSessionDescription describeDeviceDatasetSession() const {
        throw std::runtime_error("Loader does not expose a device dataset session description.");
    }
    [[nodiscard]] virtual std::shared_ptr<const Thor::NamedDataset> getNamedDataset() const {
        return nullptr;
    }

    /**
     * Returns the placement of a named dense tensor produced by getBatch(), when
     * the loader can state that contract before a batch is requested.  The
     * queued trainer uses this to configure same-GPU NetworkInputs for direct
     * device-to-featureOutput copies without allocating input staging rings.
     */
    [[nodiscard]] virtual std::optional<ThorImplementation::TensorPlacement> getBatchTensorPlacement(
        const std::string &tensorName) const {
        (void)tensorName;
        return std::nullopt;
    }

    virtual void setDatasetName(std::string datasetName) { this->datasetName = datasetName; }
    virtual std::string getDatasetName() { return datasetName; }

   protected:
    uint64_t batchSize;
    std::string datasetName;
};
