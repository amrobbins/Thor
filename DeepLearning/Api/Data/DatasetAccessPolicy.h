#pragma once

#include "DeepLearning/Api/Training/DeviceDatasetStorage.h"
#include "DeepLearning/Api/Training/WindowedDeviceCache.h"

namespace Thor {

/**
 * Immutable policy controlling how a TrainingData recipe accesses its dataset.
 *
 * This belongs to the data recipe rather than TrainerFitOptions because these
 * settings control the physical access path for the immutable dataset. They are
 * execution/data-access policy, not model-training hyperparameters.
 */
struct DatasetAccessPolicy {
    DeviceDatasetStorage deviceStorage = DeviceDatasetStorage::BEST_EFFORT;
    // AUTO is intentionally narrow: only compact device-resident window source
    // payloads are candidates for CUDA persisting L2.
    WindowedDeviceCache windowedDeviceCache = WindowedDeviceCache::AUTO;

    friend bool operator==(const DatasetAccessPolicy&, const DatasetAccessPolicy&) = default;
};

}  // namespace Thor
