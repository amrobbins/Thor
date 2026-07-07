#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Loaders/LocalNamedExampleLayout.h"
#include "Utilities/Loaders/MaterializedNamedDatasetSnapshot.h"
#include "Utilities/Loaders/Shard.h"

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <set>
#include <string>
#include <vector>

struct DeviceResidentNamedSplit {
    ExampleType exampleType = ExampleType::TRAIN;
    std::string splitName;
    std::vector<uint64_t> sourceIndices;
    std::map<std::string, ThorImplementation::Tensor> tensors;
    bool randomized = false;
    std::optional<uint64_t> seed{};
    uint64_t batchesPerEpoch = 0;

    [[nodiscard]] uint64_t numExamples() const { return static_cast<uint64_t>(sourceIndices.size()); }
    [[nodiscard]] uint64_t totalBytes() const;
    [[nodiscard]] const ThorImplementation::Tensor &tensor(const std::string &name) const;
};

/**
 * Device-resident copy of a dense materialized named dataset.
 *
 * This owns persistent split tensors on one GPU. NetworkInput tensors remain
 * current-batch tensors; DeviceResidentNamedBatchLoader gathers rows from this
 * dataset into reusable device batch tensors.
 */
class DeviceResidentNamedDataset {
   public:
    DeviceResidentNamedDataset() = default;

    [[nodiscard]] static std::shared_ptr<DeviceResidentNamedDataset> fromSnapshot(
        const MaterializedNamedDatasetSnapshot &snapshot,
        ThorImplementation::TensorPlacement devicePlacement);
    [[nodiscard]] static std::shared_ptr<DeviceResidentNamedDataset> fromSnapshot(
        const MaterializedNamedDatasetSnapshot &snapshot,
        ThorImplementation::TensorPlacement devicePlacement,
        const std::set<std::string> &tensorNamesToUpload);

    [[nodiscard]] const LocalNamedExampleLayout &getLayout() const { return layout; }
    [[nodiscard]] uint64_t getNumDatasetExamples() const { return numDatasetExamples; }
    [[nodiscard]] uint64_t getBatchSize() const { return batchSize; }
    [[nodiscard]] ThorImplementation::TensorPlacement getPlacement() const { return placement; }
    [[nodiscard]] double getUploadSeconds() const { return uploadSeconds; }
    [[nodiscard]] uint64_t totalExamples() const;
    [[nodiscard]] uint64_t totalBytes() const;
    [[nodiscard]] const DeviceResidentNamedSplit *findSplit(ExampleType exampleType) const;
    [[nodiscard]] const DeviceResidentNamedSplit &split(ExampleType exampleType) const;

   private:
    LocalNamedExampleLayout layout;
    uint64_t numDatasetExamples = 0;
    uint64_t batchSize = 0;
    ThorImplementation::TensorPlacement placement;
    std::vector<DeviceResidentNamedSplit> splits;
    double uploadSeconds = 0.0;
};
