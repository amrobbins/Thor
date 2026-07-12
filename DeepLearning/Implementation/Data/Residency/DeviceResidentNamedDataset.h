#pragma once

#include "DeepLearning/Api/Data/DatasetId.h"
#include "DeepLearning/Api/Data/DatasetSchema.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Api/Data/DatasetLayout.h"
#include "DeepLearning/Implementation/Data/Materialization/MaterializedNamedDatasetSnapshot.h"

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>

/**
 * Canonical device-resident storage for one immutable named dataset.
 *
 * Fields are keyed by immutable DatasetFieldId and stored once in dataset row
 * order. This object contains no split membership, batch size, randomization,
 * cursor, or queue state.
 */
class DeviceResidentNamedDataset {
   public:
    [[nodiscard]] static std::shared_ptr<DeviceResidentNamedDataset> fromSnapshot(
        const MaterializedNamedDatasetSnapshot &snapshot,
        ThorImplementation::TensorPlacement devicePlacement);
    [[nodiscard]] static std::shared_ptr<DeviceResidentNamedDataset> fromSnapshot(
        const MaterializedNamedDatasetSnapshot &snapshot,
        ThorImplementation::TensorPlacement devicePlacement,
        const std::set<std::string> &tensorNamesToUpload);

    [[nodiscard]] const Thor::DatasetId &getDatasetId() const { return datasetId; }
    [[nodiscard]] const Thor::DatasetSchema &getSchema() const { return schema; }
    [[nodiscard]] const DatasetLayout &getLayout() const { return layout; }
    [[nodiscard]] uint64_t getNumExamples() const { return numExamples; }
    [[nodiscard]] uint64_t totalExamples() const { return numExamples; }
    [[nodiscard]] ThorImplementation::TensorPlacement getPlacement() const { return placement; }
    [[nodiscard]] double getUploadSeconds() const { return uploadSeconds; }
    [[nodiscard]] uint64_t totalBytes() const;
    [[nodiscard]] bool hasField(Thor::DatasetFieldId id) const;
    [[nodiscard]] bool hasTensor(const std::string &name) const;
    [[nodiscard]] const ThorImplementation::Tensor &field(Thor::DatasetFieldId id) const;
    [[nodiscard]] const ThorImplementation::Tensor &tensor(const std::string &name) const;

   private:
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
    double uploadSeconds = 0.0;
};
