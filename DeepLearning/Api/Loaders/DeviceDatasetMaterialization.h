#pragma once

#include "DeepLearning/Api/Data/BatchPolicy.h"
#include "DeepLearning/Api/Data/DatasetId.h"
#include "DeepLearning/Api/Data/DatasetSplitManifest.h"
#include "DeepLearning/Api/Data/DatasetSchema.h"
#include "DeepLearning/Api/Data/DatasetLayout.h"

#include <cstdint>
#include <filesystem>
#include <set>
#include <utility>

namespace Thor {

/**
 * Canonical, split-independent description of one immutable named dataset.
 *
 * This is the only input to CPU/GPU persistent dataset materialization. It
 * deliberately contains no split membership, batch size, randomization,
 * queue depth, or live loader state.
 */
struct DatasetMaterializationDescription {
    DatasetMaterializationDescription(std::filesystem::path datasetPath,
                                      DatasetId datasetId,
                                      DatasetSchema schema,
                                      DatasetLayout layout,
                                      uint64_t numExamples)
        : datasetPath(std::move(datasetPath)),
          datasetId(std::move(datasetId)),
          schema(std::move(schema)),
          layout(std::move(layout)),
          numExamples(numExamples) {}

    std::filesystem::path datasetPath;
    DatasetId datasetId;
    DatasetSchema schema;
    DatasetLayout layout;
    uint64_t numExamples = 0;
};

/**
 * Immutable per-session iteration recipe used after a canonical resident
 * dataset has been acquired. This state must never be copied into a persistent
 * materialized dataset.
 */
class DeviceDatasetSessionDescription {
   public:
    DeviceDatasetSessionDescription(DatasetSplitManifest splits,
                                    BatchPolicy batching,
                                    std::set<DatasetFieldId> requiredFieldIds = {})
        : splits(std::move(splits)),
          batching(std::move(batching)),
          requiredFieldIds(std::move(requiredFieldIds)) {}

    [[nodiscard]] const DatasetSplitManifest &getSplits() const { return splits; }
    [[nodiscard]] const BatchPolicy &getBatching() const { return batching; }
    [[nodiscard]] const std::set<DatasetFieldId>& getRequiredFieldIds() const { return requiredFieldIds; }

   private:
    DatasetSplitManifest splits;
    BatchPolicy batching;
    std::set<DatasetFieldId> requiredFieldIds;
};

}  // namespace Thor
