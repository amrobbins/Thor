#pragma once

#include "DeepLearning/Api/Loaders/Loader.h"
#include "DeepLearning/Api/Training/DeviceDatasetStorage.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"

#include <cstdint>
#include <memory>
#include <optional>

namespace Thor {

struct DeviceDatasetStorageSelection {
    std::shared_ptr<Loader> loader;
    DeviceDatasetStorageReport report{};
};

/**
 * Select the effective loader for a training request under the device dataset
 * storage policy. Persistent materialization is canonical and split-independent;
 * the returned BatchSession applies the source session's manifest and batching
 * policy over that shared row-ordered storage. The Loader-typed return remains
 * temporarily for compatibility with legacy training entry points.
 */
[[nodiscard]] DeviceDatasetStorageSelection selectDeviceDatasetStorageLoader(
    const std::shared_ptr<Loader> &sourceLoader,
    DeviceDatasetStorage requested,
    ThorImplementation::TensorPlacement devicePlacement,
    uint64_t batchQueueDepth,
    std::optional<uint64_t> availableBytesOverride = std::nullopt);

[[nodiscard]] uint64_t estimateDeviceResidentNamedDatasetRequiredBytes(
    const DatasetMaterializationDescription &dataset,
    const DeviceDatasetSessionDescription &session,
    uint64_t batchQueueDepth);

[[nodiscard]] uint64_t estimateDeviceResidentNamedDatasetStorageBytes(
    const DatasetMaterializationDescription &dataset);

}  // namespace Thor
