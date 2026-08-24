#pragma once

#include "DeepLearning/Api/Data/BatchSession.h"
#include "DeepLearning/Api/Data/TrainingData.h"
#include "DeepLearning/Implementation/Data/Materialization/DeviceDatasetMaterialization.h"
#include "DeepLearning/Api/Training/DeviceDatasetStorage.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"

#include <cstdint>
#include <memory>
#include <optional>

namespace Thor {

class FileDataset;

struct DeviceDatasetStorageSelection {
    std::shared_ptr<BatchSession> session;
    DeviceDatasetStorageReport report{};
};

/**
 * Select the effective per-run session under the immutable TrainingData access
 * policy. Dataset identity/schema/backend come from TrainingData::dataset;
 * split membership and batching come from TrainingData itself; only live queue
 * state and required-field reporting come from sourceSession.
 */
[[nodiscard]] DeviceDatasetStorageSelection selectDeviceDatasetStorageSession(
    const std::shared_ptr<BatchSession>& sourceSession,
    const TrainingData& trainingData,
    ThorImplementation::TensorPlacement devicePlacement,
    uint64_t batchQueueDepth,
    std::optional<uint64_t> availableBytesOverride = std::nullopt);

/**
 * Select device storage for a session whose split view differs from the
 * TrainingData default (for example, a named validation population). Dataset,
 * batching, access policy, and storage ownership still come from TrainingData.
 */
[[nodiscard]] DeviceDatasetStorageSelection selectDeviceDatasetStorageSession(
    const std::shared_ptr<BatchSession>& sourceSession,
    const TrainingData& trainingData,
    const DatasetSplitManifest& sessionSplits,
    ThorImplementation::TensorPlacement devicePlacement,
    uint64_t batchQueueDepth,
    std::optional<uint64_t> availableBytesOverride = std::nullopt);

[[nodiscard]] DatasetMaterializationDescription describeDatasetMaterialization(
    const FileDataset& dataset);

[[nodiscard]] DatasetMaterializationDescription describeDatasetMaterialization(
    const TrainingData& trainingData);

[[nodiscard]] DeviceDatasetSessionDescription describeDeviceDatasetSession(
    const DatasetSplitManifest& splits,
    const BatchPolicy& batching,
    const DatasetFieldMaterializationRequirements& fieldRequirements = {},
    WindowedDeviceCache windowedDeviceCache = WindowedDeviceCache::AUTO);

[[nodiscard]] DeviceDatasetSessionDescription describeDeviceDatasetSession(
    const TrainingData& trainingData,
    const DatasetFieldMaterializationRequirements& fieldRequirements = {});

[[nodiscard]] uint64_t estimateDeviceResidentNamedDatasetRequiredBytes(
    const DatasetMaterializationDescription& dataset,
    const DeviceDatasetSessionDescription& session,
    uint64_t batchQueueDepth);

[[nodiscard]] uint64_t estimateDeviceResidentNamedDatasetStorageBytes(
    const DatasetMaterializationDescription& dataset);

}  // namespace Thor
