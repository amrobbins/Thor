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
 * storage policy.  OFF always returns the source loader. BEST_EFFORT falls back
 * to the source loader with telemetry on unsupported/OOM/materialization errors.
 * STRICT throws on any failure to create the device-resident loader.
 *
 * availableBytesOverride exists only for deterministic unit tests of the memory
 * decision path; normal callers should leave it unset so CUDA memory is queried
 * after model placement/workspace reservation.
 */
[[nodiscard]] DeviceDatasetStorageSelection selectDeviceDatasetStorageLoader(
    const std::shared_ptr<Loader> &sourceLoader,
    DeviceDatasetStorage requested,
    ThorImplementation::TensorPlacement devicePlacement,
    uint64_t batchQueueDepth,
    std::optional<uint64_t> availableBytesOverride = std::nullopt);

[[nodiscard]] uint64_t estimateDeviceResidentNamedDatasetRequiredBytes(const DeviceDatasetMaterializationView &view,
                                                                       uint64_t batchQueueDepth);

}  // namespace Thor
