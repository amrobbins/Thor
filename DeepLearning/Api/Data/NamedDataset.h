#pragma once

#include "DeepLearning/Api/Data/DatasetId.h"
#include "DeepLearning/Api/Data/DatasetSchema.h"

#include <cstdint>
#include <memory>
#include <set>
#include <string_view>

namespace Thor {

class BatchSession;
class BatchPolicy;
class DatasetAccessPolicy;
class DatasetSplitManifest;
class DeviceDatasetResidencyCache;
class TrainingData;

class NamedDataset : public std::enable_shared_from_this<NamedDataset> {
   public:
    NamedDataset();
    virtual ~NamedDataset();
    NamedDataset(const NamedDataset &) = delete;
    NamedDataset &operator=(const NamedDataset &) = delete;
    NamedDataset(NamedDataset &&) = delete;
    NamedDataset &operator=(NamedDataset &&) = delete;

    [[nodiscard]] virtual const DatasetId &getId() const = 0;
    [[nodiscard]] virtual uint64_t getNumExamples() const = 0;
    [[nodiscard]] virtual const DatasetSchema &getSchema() const = 0;
    [[nodiscard]] virtual const DatasetField &getField(std::string_view name) const = 0;

    /** Internal shared runtime for per-device immutable replicas. */
    [[nodiscard]] DeviceDatasetResidencyCache &getDeviceDatasetResidencyCache() const;

   protected:
    /** Backend hook used only by TrainingData to create fresh mutable iteration state. */
    [[nodiscard]] virtual std::shared_ptr<BatchSession> openBatchSession(
        const DatasetSplitManifest &splits,
        const BatchPolicy &batching,
        const DatasetAccessPolicy &accessPolicy,
        uint64_t maxInFlightBatches,
        const std::set<DatasetFieldId> &requiredFieldIds) const = 0;

   private:
    friend class TrainingData;
    std::shared_ptr<DeviceDatasetResidencyCache> deviceResidencyCache;
};

}  // namespace Thor
