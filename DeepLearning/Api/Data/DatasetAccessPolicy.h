#pragma once

#include "DeepLearning/Api/Training/DeviceDatasetStorage.h"

namespace Thor {

/**
 * Immutable policy controlling how a TrainingData recipe accesses its dataset.
 *
 * This belongs to the data recipe rather than TrainerFitOptions because device
 * residency changes the physical access path for the immutable dataset; it is
 * not an optimization or model-training hyperparameter.
 */
struct DatasetAccessPolicy {
    DeviceDatasetStorage deviceStorage = DeviceDatasetStorage::BEST_EFFORT;

    friend bool operator==(const DatasetAccessPolicy&, const DatasetAccessPolicy&) = default;
};

}  // namespace Thor
