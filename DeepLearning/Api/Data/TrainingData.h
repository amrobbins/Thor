#pragma once

#include "DeepLearning/Api/Data/BatchPolicy.h"
#include "DeepLearning/Api/Data/BatchSession.h"
#include "DeepLearning/Api/Data/DatasetSplitManifest.h"
#include "DeepLearning/Api/Data/DatasetAccessPolicy.h"
#include "DeepLearning/Api/Data/NamedDataset.h"

#include <cstdint>
#include <memory>
#include <set>
#include <string>

namespace Thor {

/**
 * Immutable recipe connecting one dataset, one split manifest and batching
 * policy.  Opening the recipe creates a fresh mutable BatchSession.
 */
class TrainingData {
   public:
    TrainingData(std::shared_ptr<const NamedDataset> dataset,
                 DatasetSplitManifest splits,
                 BatchPolicy batching,
                 DatasetAccessPolicy accessPolicy = {},
                 std::string datasetName = "indexed_named_examples");

    [[nodiscard]] std::shared_ptr<BatchSession> openSession(uint64_t maxInFlightBatches = 32) const;
    [[nodiscard]] std::shared_ptr<BatchSession> openSession(
        uint64_t maxInFlightBatches,
        const std::set<DatasetFieldId>& requiredFieldIds) const;

    [[nodiscard]] const std::shared_ptr<const NamedDataset> &getDataset() const { return dataset; }
    [[nodiscard]] const DatasetSplitManifest &getSplits() const { return splits; }
    [[nodiscard]] const BatchPolicy &getBatching() const { return batching; }
    [[nodiscard]] const DatasetAccessPolicy &getAccessPolicy() const { return accessPolicy; }
    [[nodiscard]] const std::string &getDatasetName() const { return datasetName; }

   private:
    std::shared_ptr<const NamedDataset> dataset;
    DatasetSplitManifest splits;
    BatchPolicy batching;
    DatasetAccessPolicy accessPolicy;
    std::string datasetName;
};

}  // namespace Thor
