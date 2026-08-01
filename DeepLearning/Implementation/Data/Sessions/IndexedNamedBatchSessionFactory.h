#pragma once

#include "DeepLearning/Api/Data/DatasetSchema.h"
#include "DeepLearning/Api/Data/DatasetFieldMaterializationRequirement.h"

#include <cstdint>
#include <memory>
#include <set>

namespace Thor {
class BatchPolicy;
class BatchSession;
class DatasetSplitManifest;
class FileDataset;
}

namespace ThorImplementation {

[[nodiscard]] std::shared_ptr<Thor::BatchSession> openIndexedNamedBatchSession(
    std::shared_ptr<const Thor::FileDataset> dataset,
    const Thor::DatasetSplitManifest &splits,
    const Thor::BatchPolicy &batching,
    uint64_t maxInFlightBatches,
    const Thor::DatasetFieldMaterializationRequirements &fieldRequirements);

}  // namespace ThorImplementation
