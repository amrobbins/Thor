#pragma once

#include "DeepLearning/Api/Data/DatasetId.h"
#include "DeepLearning/Api/Data/DatasetSchema.h"
#include "DeepLearning/Api/Data/DatasetLayout.h"
#include "DeepLearning/Implementation/Data/Materialization/DeviceDatasetMaterialization.h"
#include "DeepLearning/Implementation/Data/Materialization/MaterializedNamedDatasetSnapshot.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>

class Stream;

/**
 * Canonical device-resident storage for one immutable named dataset.
 *
 * Dense-only datasets may store canonical tensors in dataset row order. For a
 * file dataset with windowed fields, the compact representation instead stores
 * the physical indexed records, source sequences, and affine-reference
 * metadata. Direct and window outputs are materialized only for the selected
 * batch rows.
 */
class DeviceResidentNamedDataset {
   public:
    [[nodiscard]] static std::shared_ptr<DeviceResidentNamedDataset> fromSnapshot(
        const MaterializedNamedDatasetSnapshot &snapshot,
        ThorImplementation::TensorPlacement devicePlacement);

    [[nodiscard]] static std::shared_ptr<DeviceResidentNamedDataset>
    fromCompactFileDataset(
        const Thor::DatasetMaterializationDescription &description,
        ThorImplementation::TensorPlacement devicePlacement,
        const std::set<std::string> &fieldNamesToExpose);

    [[nodiscard]] static uint64_t estimateCompactFileDatasetBytes(
        const Thor::DatasetMaterializationDescription &description,
        const std::set<std::string> &fieldNamesToExpose);

    [[nodiscard]] const Thor::DatasetId &getDatasetId() const { return datasetId; }
    [[nodiscard]] const Thor::DatasetSchema &getSchema() const { return schema; }
    [[nodiscard]] const DatasetLayout &getLayout() const { return layout; }
    [[nodiscard]] uint64_t getNumExamples() const { return numExamples; }
    [[nodiscard]] uint64_t totalExamples() const { return numExamples; }
    [[nodiscard]] ThorImplementation::TensorPlacement getPlacement() const { return placement; }
    [[nodiscard]] double getUploadSeconds() const { return uploadSeconds; }
    [[nodiscard]] uint64_t totalBytes() const;
    [[nodiscard]] uint64_t compactRecordBytes() const;
    [[nodiscard]] uint64_t compactSourceBytes() const;
    [[nodiscard]] uint64_t compactMetadataBytes() const;
    [[nodiscard]] bool usesCompactFileStorage() const { return compactFileStorage; }
    [[nodiscard]] bool hasField(Thor::DatasetFieldId id) const;
    [[nodiscard]] bool hasTensor(const std::string &name) const;
    [[nodiscard]] bool hasCompactField(const std::string &name) const;
    [[nodiscard]] bool hasCompactDirectField(const std::string &name) const;
    [[nodiscard]] bool hasCompactWindowField(const std::string &name) const;
    [[nodiscard]] const ThorImplementation::Tensor &field(Thor::DatasetFieldId id) const;
    [[nodiscard]] const ThorImplementation::Tensor &tensor(const std::string &name) const;

    void enqueueCompactFieldMaterialization(
        const std::string &fieldName,
        const ThorImplementation::Tensor &rowIndicesDevice,
        ThorImplementation::Tensor &destination,
        Stream &stream) const;

   private:
    struct CompactDirectFieldStorage {
        DatasetLayout::TensorSpec spec;
    };

    struct CompactWindowSourceStorage {
        DatasetLayout::WindowedTensorSourceSpec spec;
        ThorImplementation::Tensor bytes;
        ThorImplementation::Tensor sequences;
        uint64_t sequenceCount = 0;
    };

    struct CompactWindowFieldStorage {
        DatasetLayout::WindowedTensorSpec spec;
        bool materializeMask = false;
    };

    struct CompactAffineFieldStorage {
        ThorImplementation::Tensor segments;
        uint64_t segmentCount = 0;
    };

    DeviceResidentNamedDataset(Thor::DatasetId datasetId,
                               Thor::DatasetSchema schema,
                               DatasetLayout layout,
                               uint64_t numExamples,
                               ThorImplementation::TensorPlacement placement)
        : datasetId(std::move(datasetId)),
          schema(std::move(schema)),
          layout(std::move(layout)),
          numExamples(numExamples),
          placement(placement) {}

    Thor::DatasetId datasetId;
    Thor::DatasetSchema schema;
    DatasetLayout layout;
    uint64_t numExamples = 0;
    ThorImplementation::TensorPlacement placement;
    std::map<Thor::DatasetFieldId, ThorImplementation::Tensor> fields;

    bool compactFileStorage = false;
    ThorImplementation::Tensor compactRecords;
    std::map<std::string, CompactDirectFieldStorage> compactDirectFields;
    std::map<std::string, CompactWindowSourceStorage> compactSources;
    std::map<std::string, CompactWindowFieldStorage> compactWindowFields;
    std::map<std::string, CompactAffineFieldStorage> compactAffineFields;
    std::set<Thor::DatasetFieldId> compactFieldIds;
    double uploadSeconds = 0.0;
};
