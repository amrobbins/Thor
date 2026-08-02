#pragma once

#include "DeepLearning/Api/Data/DatasetId.h"
#include "DeepLearning/Api/Data/DatasetSchema.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/DataType.h"
#include "DeepLearning/Api/Data/DatasetLayout.h"

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct MaterializedRaggedFieldSnapshot {
    ThorImplementation::DataType valuesDataType = ThorImplementation::DataType::FP32;
    std::vector<uint64_t> trailingDimensions;
    uint64_t storedValueCount = 0;
    uint64_t valueBytes = 0;
    // All-empty ragged datasets have no values allocation. Offsets are always
    // present and retain the source canonical UINT32/UINT64 dtype.
    ThorImplementation::Tensor values;
    ThorImplementation::Tensor offsets;

    [[nodiscard]] uint64_t totalBytes() const {
        return (values.isInitialized() ? values.getArraySizeInBytes() : 0) +
               (offsets.isInitialized() ? offsets.getArraySizeInBytes() : 0);
    }
};

/**
 * Canonical CPU snapshot of one immutable named dataset.
 *
 * Dense fields are stored exactly once in dataset row order with shape
 * [num_examples, *example_shape]. Ragged fields are stored as one packed
 * values array plus canonical row-partition offsets [num_examples + 1].
 * Fields are keyed by their immutable DatasetFieldId. No split, batching,
 * randomization, or queue metadata is permitted in this representation.
 */
struct MaterializedNamedDatasetSnapshot {
    MaterializedNamedDatasetSnapshot(Thor::DatasetId datasetId,
                                     Thor::DatasetSchema schema,
                                     DatasetLayout layout,
                                     uint64_t numExamples)
        : datasetId(std::move(datasetId)),
          schema(std::move(schema)),
          layout(std::move(layout)),
          numExamples(numExamples) {}

    Thor::DatasetId datasetId;
    Thor::DatasetSchema schema;
    DatasetLayout layout;
    uint64_t numExamples = 0;
    std::map<Thor::DatasetFieldId, ThorImplementation::Tensor> fields;
    std::map<Thor::DatasetFieldId, MaterializedRaggedFieldSnapshot> raggedFields;
    double materializationSeconds = 0.0;

    [[nodiscard]] uint64_t totalExamples() const { return numExamples; }
    [[nodiscard]] uint64_t totalBytes() const;
    [[nodiscard]] bool hasField(Thor::DatasetFieldId id) const;
    [[nodiscard]] bool hasField(const std::string &name) const;
    [[nodiscard]] bool hasRaggedField(Thor::DatasetFieldId id) const;
    [[nodiscard]] bool hasRaggedField(const std::string &name) const;
    [[nodiscard]] const ThorImplementation::Tensor &field(Thor::DatasetFieldId id) const;
    [[nodiscard]] const ThorImplementation::Tensor &tensor(const std::string &name) const;
    [[nodiscard]] const MaterializedRaggedFieldSnapshot &raggedField(Thor::DatasetFieldId id) const;
    [[nodiscard]] const MaterializedRaggedFieldSnapshot &raggedTensor(const std::string &name) const;
};
