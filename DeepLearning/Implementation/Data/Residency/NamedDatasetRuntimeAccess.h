#pragma once

#include "DeepLearning/Implementation/Data/Residency/DeviceDatasetResidency.h"

#include <cstdint>
#include <memory>

struct MaterializedNamedDatasetSnapshot;

namespace Thor {
class NamedDataset;
struct DatasetMaterializationDescription;
}

namespace ThorImplementation {

/** Internal bridge to NamedDataset-owned mutable runtime and materialization state. */
class NamedDatasetRuntimeAccess {
   public:
    [[nodiscard]] static Thor::DeviceDatasetResidencyCache &residencyCache(
        const Thor::NamedDataset &dataset);

    [[nodiscard]] static std::unique_ptr<Thor::DatasetMaterializationDescription>
    describeMaterialization(const Thor::NamedDataset &dataset);

    [[nodiscard]] static MaterializedNamedDatasetSnapshot materializeSnapshot(
        const Thor::NamedDataset &dataset,
        uint64_t readerQueueDepth = 32);
};

}  // namespace ThorImplementation
