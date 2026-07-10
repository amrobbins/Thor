#pragma once

#include "DeepLearning/Api/Data/DatasetId.h"
#include "DeepLearning/Api/Data/DatasetSchema.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Loaders/LocalNamedExampleLayout.h"

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

/**
 * Canonical CPU snapshot of one immutable named dataset.
 *
 * Every field is stored exactly once in dataset row order with shape
 * [num_examples, *example_shape]. Fields are keyed by their immutable
 * DatasetFieldId. No split, batching, randomization, or queue metadata is
 * permitted in this representation.
 */
struct MaterializedNamedDatasetSnapshot {
    MaterializedNamedDatasetSnapshot(Thor::DatasetId datasetId,
                                     Thor::DatasetSchema schema,
                                     LocalNamedExampleLayout layout,
                                     uint64_t numExamples)
        : datasetId(std::move(datasetId)),
          schema(std::move(schema)),
          layout(std::move(layout)),
          numExamples(numExamples) {}

    Thor::DatasetId datasetId;
    Thor::DatasetSchema schema;
    LocalNamedExampleLayout layout;
    uint64_t numExamples = 0;
    std::map<Thor::DatasetFieldId, ThorImplementation::Tensor> fields;
    double materializationSeconds = 0.0;

    [[nodiscard]] uint64_t totalExamples() const { return numExamples; }
    [[nodiscard]] uint64_t totalBytes() const;
    [[nodiscard]] bool hasField(Thor::DatasetFieldId id) const;
    [[nodiscard]] bool hasField(const std::string &name) const;
    [[nodiscard]] const ThorImplementation::Tensor &field(Thor::DatasetFieldId id) const;
    [[nodiscard]] const ThorImplementation::Tensor &tensor(const std::string &name) const;
};
