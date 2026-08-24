#pragma once

#include "DeepLearning/Api/Data/BatchPolicy.h"
#include "DeepLearning/Api/Data/DatasetId.h"
#include "DeepLearning/Api/Data/DatasetSplitManifest.h"
#include "DeepLearning/Api/Data/DatasetSchema.h"
#include "DeepLearning/Api/Data/DatasetLayout.h"
#include "DeepLearning/Api/Training/WindowedDeviceCache.h"

#include <cstdint>
#include <filesystem>
#include <utility>

namespace Thor {

enum class DatasetMaterializationSource { FILE_DATASET, MEMORY };

/**
 * Canonical, split-independent description of one immutable named dataset.
 *
 * This is the only input to CPU/GPU persistent dataset materialization. It
 * deliberately contains no split membership, batch size, randomization,
 * queue depth, or live session state.
 */
struct DatasetMaterializationDescription {
    DatasetMaterializationDescription(std::filesystem::path datasetPath,
                                      DatasetId datasetId,
                                      DatasetSchema schema,
                                      DatasetLayout layout,
                                      uint64_t numExamples,
                                      DatasetMaterializationSource source =
                                          DatasetMaterializationSource::FILE_DATASET)
        : datasetPath(std::move(datasetPath)),
          datasetId(std::move(datasetId)),
          schema(std::move(schema)),
          layout(std::move(layout)),
          numExamples(numExamples),
          source(source) {}

    std::filesystem::path datasetPath;
    DatasetId datasetId;
    DatasetSchema schema;
    DatasetLayout layout;
    uint64_t numExamples = 0;
    DatasetMaterializationSource source = DatasetMaterializationSource::FILE_DATASET;
};

/**
 * Immutable per-session iteration recipe used after a canonical resident
 * dataset has been acquired. This state must never be copied into a persistent
 * materialized dataset.
 */
class DeviceDatasetSessionDescription {
   public:
    DeviceDatasetSessionDescription(
        DatasetSplitManifest splits,
        BatchPolicy batching,
        DatasetFieldMaterializationRequirements fieldRequirements = {},
        WindowedDeviceCache windowedDeviceCache = WindowedDeviceCache::AUTO)
        : splits(std::move(splits)),
          batching(std::move(batching)),
          fieldRequirements(std::move(fieldRequirements)),
          windowedDeviceCache(windowedDeviceCache) {}

    [[nodiscard]] const DatasetSplitManifest &getSplits() const { return splits; }
    [[nodiscard]] const BatchPolicy &getBatching() const { return batching; }
    [[nodiscard]] const DatasetFieldMaterializationRequirements& getFieldRequirements() const { return fieldRequirements; }
    [[nodiscard]] WindowedDeviceCache getWindowedDeviceCache() const { return windowedDeviceCache; }

   private:
    DatasetSplitManifest splits;
    BatchPolicy batching;
    DatasetFieldMaterializationRequirements fieldRequirements;
    WindowedDeviceCache windowedDeviceCache = WindowedDeviceCache::AUTO;
};

}  // namespace Thor
