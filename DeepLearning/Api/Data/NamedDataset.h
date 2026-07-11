#pragma once

#include "DeepLearning/Api/Data/DatasetId.h"
#include "DeepLearning/Api/Data/DatasetSchema.h"

#include <cstdint>
#include <memory>
#include <string_view>

namespace Thor {

class DeviceDatasetResidencyCache;

class NamedDataset {
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

   private:
    std::shared_ptr<DeviceDatasetResidencyCache> deviceResidencyCache;
};

}  // namespace Thor
