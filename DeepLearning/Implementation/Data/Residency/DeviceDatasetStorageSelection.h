#pragma once

#include "DeepLearning/Api/Data/BatchSession.h"
#include "DeepLearning/Api/Data/TrainingData.h"
#include "DeepLearning/Implementation/Data/Materialization/DeviceDatasetMaterialization.h"
#include "DeepLearning/Api/Training/DeviceDatasetStorage.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <set>

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

[[nodiscard]] DatasetMaterializationDescription describeDatasetMaterialization(
    const FileDataset& dataset);

[[nodiscard]] DatasetMaterializationDescription describeDatasetMaterialization(
    const TrainingData& trainingData);

[[nodiscard]] DeviceDatasetSessionDescription describeDeviceDatasetSession(
    const DatasetSplitManifest& splits,
    const BatchPolicy& batching,
    const std::set<DatasetFieldId>& requiredFieldIds = {});

[[nodiscard]] DeviceDatasetSessionDescription describeDeviceDatasetSession(
    const TrainingData& trainingData,
    const std::set<DatasetFieldId>& requiredFieldIds = {});

[[nodiscard]] uint64_t estimateDeviceResidentNamedDatasetRequiredBytes(
    const DatasetMaterializationDescription& dataset,
    const DeviceDatasetSessionDescription& session,
    uint64_t batchQueueDepth);

[[nodiscard]] uint64_t estimateDeviceResidentNamedDatasetStorageBytes(
    const DatasetMaterializationDescription& dataset);

}  // namespace Thor
