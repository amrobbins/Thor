#include "DeepLearning/Api/Data/NamedDataset.h"

#include "DeepLearning/Implementation/Data/Materialization/DeviceDatasetMaterialization.h"
#include "DeepLearning/Implementation/Data/Materialization/MaterializedNamedDatasetSnapshot.h"
#include "DeepLearning/Implementation/Data/Residency/DeviceDatasetResidency.h"
#include "DeepLearning/Implementation/Data/Residency/NamedDatasetRuntimeAccess.h"

#include <memory>
#include <stdexcept>

namespace Thor {

class NamedDataset::Runtime {
   public:
    DeviceDatasetResidencyCache deviceResidencyCache;
};

NamedDataset::NamedDataset() : runtime(std::make_unique<Runtime>()) {}

NamedDataset::~NamedDataset() = default;

std::unique_ptr<DatasetMaterializationDescription>
NamedDataset::describeMaterializationForRuntime() const {
    return nullptr;
}

MaterializedNamedDatasetSnapshot NamedDataset::materializeSnapshotForRuntime(
    uint64_t) const {
    throw std::runtime_error(
        "NamedDataset backend does not support device materialization.");
}

}  // namespace Thor

namespace ThorImplementation {

Thor::DeviceDatasetResidencyCache &NamedDatasetRuntimeAccess::residencyCache(
    const Thor::NamedDataset &dataset) {
    return dataset.runtime->deviceResidencyCache;
}

std::unique_ptr<Thor::DatasetMaterializationDescription>
NamedDatasetRuntimeAccess::describeMaterialization(
    const Thor::NamedDataset &dataset) {
    return dataset.describeMaterializationForRuntime();
}

MaterializedNamedDatasetSnapshot NamedDatasetRuntimeAccess::materializeSnapshot(
    const Thor::NamedDataset &dataset,
    uint64_t readerQueueDepth) {
    return dataset.materializeSnapshotForRuntime(readerQueueDepth);
}

}  // namespace ThorImplementation
