#pragma once

#include "DeepLearning/Api/Data/DatasetId.h"
#include "DeepLearning/Api/Data/DatasetSchema.h"
#include "DeepLearning/Api/Data/DatasetLayout.h"
#include "DeepLearning/Implementation/Data/Materialization/DeviceDatasetMaterialization.h"
#include "DeepLearning/Implementation/Data/Materialization/MaterializedNamedDatasetSnapshot.h"
#include "DeepLearning/Implementation/Data/Residency/DeviceResidentWindowMaterializationKernel.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensor.h"

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>
#include <utility>

class Stream;

/**
 * Canonical device-resident storage for one immutable named dataset.
 *
 * Canonical snapshots may store dense tensors and packed ragged fields in
 * dataset row order. For a file dataset with windowed or ragged fields, the
 * compact representation keeps only the device records/source sidecars that
 * value materialization still consumes. Dense window placement metadata is
 * resolved once on the host and selected row plans are staged per batch.
 * Direct/window outputs are deferred references; ragged outputs are
 * gathered into reusable batch-capacity buffers for the selected rows.
 */
struct RaggedBatchExtent {
    std::vector<uint64_t> hostOffsets;

    bool operator==(const RaggedBatchExtent&) const = default;
};

class DeviceResidentNamedDataset {
   public:
    // Stable for the lifetime of this immutable resident dataset. The access
    // describes only the compact immutable source payload read by dense or
    // source-backed ragged window materialization; reference/sequence metadata
    // and generated masks are intentionally not L2 cache candidates here.
    struct CompactWindowSourceAccess {
        std::string sourceName;
        uint64_t tensorId = 0;
        const void *base = nullptr;
        uint64_t bytes = 0;

        bool operator==(const CompactWindowSourceAccess &rhs) const = default;
    };

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
    [[nodiscard]] bool hasCompactRaggedField(const std::string &name) const;
    [[nodiscard]] bool hasSnapshotRaggedField(const std::string &name) const;
    // Dense window payloads and source-backed ragged window fields resolve to
    // their shared compact source allocation. A dense mask-only field returns
    // nullopt because its materialization kernel does not dereference the source
    // payload. Ordinary direct/ragged fields likewise have no compact window
    // source access.
    [[nodiscard]] std::optional<CompactWindowSourceAccess>
    compactWindowSourceAccessForField(const std::string &fieldName) const;
    // Returns one entry per unique source allocation referenced by at least one
    // exposed dense window payload or source-backed ragged window field.
    [[nodiscard]] std::vector<CompactWindowSourceAccess> compactWindowSourceAccesses() const;
    [[nodiscard]] const ThorImplementation::Tensor &field(Thor::DatasetFieldId id) const;
    [[nodiscard]] const ThorImplementation::Tensor &tensor(const std::string &name) const;

    [[nodiscard]] RaggedBatchExtent validateCompactRaggedBatchCapacity(
        const std::string &fieldName,
        const ThorImplementation::Tensor &rowIndicesHost,
        uint64_t logicalRows,
        uint64_t maxTotalValues,
        uint64_t maxValuesPerRow = 0) const;

    // destination offsets must already contain the device representation of the
    // authoritative host partition for this selected batch.
    void enqueueCompactRaggedFieldMaterialization(
        const std::string &fieldName,
        const ThorImplementation::Tensor &rowIndicesDevice,
        uint64_t logicalRows,
        ThorImplementation::RaggedTensor &destination,
        Stream &stream) const;

    [[nodiscard]] RaggedBatchExtent validateSnapshotRaggedBatchCapacity(
        const std::string &fieldName,
        const ThorImplementation::Tensor &rowIndicesHost,
        uint64_t logicalRows,
        uint64_t maxTotalValues,
        uint64_t maxValuesPerRow = 0) const;

    // destination offsets must already contain the device representation of the
    // authoritative host partition for this selected batch.
    void enqueueSnapshotRaggedFieldMaterialization(
        const std::string &fieldName,
        const ThorImplementation::Tensor &rowIndicesDevice,
        uint64_t logicalRows,
        ThorImplementation::RaggedTensor &destination,
        Stream &stream) const;

    [[nodiscard]] std::vector<std::string> compactWindowPlanNames() const;
    [[nodiscard]] std::optional<std::string> compactWindowPlanNameForField(
        const std::string &fieldName) const;
    [[nodiscard]] uint64_t compactWindowPlanBytesPerRow(const std::string &planName) const;
    // Stage exactly logicalRows descriptors.  The destination may be a byte
    // tensor or an aligned UINT64 alias into a larger selection-metadata block.
    void stageCompactWindowPlan(
        const std::string &planName,
        const ThorImplementation::Tensor &rowIndicesHost,
        uint64_t logicalRows,
        ThorImplementation::Tensor &planHostStaging) const;

    void enqueueCompactFieldMaterialization(
        const std::string &fieldName,
        const ThorImplementation::Tensor &rowIndicesDevice,
        const ThorImplementation::Tensor &windowPlanDevice,
        uint64_t logicalRows,
        ThorImplementation::Tensor &destination,
        Stream &stream) const;

   private:
    struct CompactDirectFieldStorage {
        DatasetLayout::TensorSpec spec;
    };

    struct CompactRaggedFieldStorage {
        DatasetLayout::RaggedTensorSpec spec;
        // Owned only by ordinary ragged fields. Source-backed ragged windows
        // read the canonical CompactWindowSourceStorage::bytes allocation.
        ThorImplementation::Tensor values;
        std::vector<uint64_t> valueCounts;
    };

    struct SnapshotRaggedFieldStorage {
        DatasetLayout::RaggedTensorSpec spec;
        ThorImplementation::Tensor references;
        ThorImplementation::Tensor values;
        std::vector<uint64_t> valueCounts;
        uint64_t storedValueCount = 0;
        uint64_t valueBytes = 0;
    };

    struct CompactWindowSourceStorage {
        DatasetLayout::WindowedTensorSourceSpec spec;
        ThorImplementation::Tensor bytes;
    };

    struct CompactWindowFieldStorage {
        DatasetLayout::WindowedTensorSpec spec;
        bool materializeMask = false;
    };

    struct CompactWindowPlanStorage {
        bool use32BitSteps = true;
        std::vector<DeviceResidentWindowRowPlan32> plans32;
        std::vector<DeviceResidentWindowRowPlan64> plans64;
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
    std::map<std::string, CompactRaggedFieldStorage> compactRaggedFields;
    std::map<std::string, SnapshotRaggedFieldStorage> snapshotRaggedFields;
    std::map<std::string, CompactWindowSourceStorage> compactSources;
    std::map<std::string, CompactWindowFieldStorage> compactWindowFields;
    std::map<std::string, CompactWindowPlanStorage> compactWindowPlans;
    std::set<Thor::DatasetFieldId> compactFieldIds;
    double uploadSeconds = 0.0;
};
