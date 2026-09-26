#pragma once

#include "DeepLearning/Api/Data/BatchPolicy.h"
#include "DeepLearning/Api/Data/BatchSession.h"
#include "DeepLearning/Api/Data/DatasetSplitManifest.h"
#include "DeepLearning/Api/Data/DatasetAccessPolicy.h"
#include "DeepLearning/Api/Data/NamedDataset.h"
#include "DeepLearning/Api/Data/DatasetFieldMaterializationRequirement.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace Thor {

/**
 * Immutable recipe connecting one dataset, an optional split manifest and a
 * batching policy. Opening the recipe creates a fresh mutable BatchSession.
 *
 * The split may be absent only while constructing a repository-backed resume.
 * TrainingRuns installs the persisted original split before any training
 * session is opened. A new run and direct Trainer::fit both require a split.
 */
class TrainingData {
   public:
    TrainingData(std::shared_ptr<const NamedDataset> dataset,
                 DatasetSplitManifest splits,
                 BatchPolicy batching,
                 DatasetAccessPolicy accessPolicy = {},
                 std::string datasetName = "dataset");
    TrainingData(std::shared_ptr<const NamedDataset> dataset,
                 std::optional<DatasetSplitManifest> splits,
                 BatchPolicy batching,
                 DatasetAccessPolicy accessPolicy = {},
                 std::string datasetName = "dataset");

    [[nodiscard]] std::shared_ptr<BatchSession> openSession(uint64_t maxInFlightBatches = 32) const;
    [[nodiscard]] std::shared_ptr<BatchSession> openSession(
        uint64_t maxInFlightBatches,
        const DatasetFieldMaterializationRequirements& fieldRequirements) const;
    [[nodiscard]] std::shared_ptr<BatchSession> openValidationSession(
        const std::string& validationPopulation,
        uint64_t maxInFlightBatches = 32) const;
    [[nodiscard]] std::shared_ptr<BatchSession> openValidationSession(
        const std::string& validationPopulation,
        uint64_t maxInFlightBatches,
        const DatasetFieldMaterializationRequirements& fieldRequirements) const;

    [[nodiscard]] const std::shared_ptr<const NamedDataset> &getDataset() const { return dataset; }
    [[nodiscard]] bool hasSplits() const { return splits.has_value(); }
    [[nodiscard]] const DatasetSplitManifest &getSplits() const;
    [[nodiscard]] const BatchPolicy &getBatching() const { return batching; }
    [[nodiscard]] const DatasetAccessPolicy &getAccessPolicy() const { return accessPolicy; }
    [[nodiscard]] const std::string &getDatasetName() const { return datasetName; }

    void requireNonEmptyPartition(ExampleType exampleType, const std::string& context) const;

   private:
    std::shared_ptr<const NamedDataset> dataset;
    std::optional<DatasetSplitManifest> splits;
    BatchPolicy batching;
    DatasetAccessPolicy accessPolicy;
    std::string datasetName;
};

}  // namespace Thor
