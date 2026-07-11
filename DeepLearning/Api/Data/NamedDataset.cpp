#include "DeepLearning/Api/Data/NamedDataset.h"

#include "DeepLearning/Api/Training/DeviceDatasetResidency.h"

#include <memory>

namespace Thor {

NamedDataset::NamedDataset()
    : deviceResidencyCache(std::make_shared<DeviceDatasetResidencyCache>()) {}

NamedDataset::~NamedDataset() = default;

DeviceDatasetResidencyCache &NamedDataset::getDeviceDatasetResidencyCache() const {
    return *deviceResidencyCache;
}

}  // namespace Thor
