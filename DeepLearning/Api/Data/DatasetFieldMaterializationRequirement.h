#pragma once

#include "DeepLearning/Api/Data/DatasetSchema.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensorDescriptor.h"

#include <map>
#include <optional>
#include <set>
#include <utility>

namespace Thor {

/**
 * Per-field runtime materialization contract selected by the consuming graph.
 *
 * Dense/windowed fields need only their immutable dataset field identity.
 * Ragged fields additionally require the exact fixed-capacity runtime descriptor
 * expected by the logical RaggedNetworkInput. The dataset stores variable-length
 * examples; this descriptor tells a BatchSession how those examples must be
 * packed for this particular network/batch size.
 */
struct DatasetFieldMaterializationRequirement {
    DatasetFieldId fieldId = 0;
    std::optional<ThorImplementation::RaggedTensorDescriptor> raggedTensorDescriptor;

    [[nodiscard]] static DatasetFieldMaterializationRequirement dense(DatasetFieldId fieldId) {
        return DatasetFieldMaterializationRequirement{.fieldId = fieldId, .raggedTensorDescriptor = std::nullopt};
    }

    [[nodiscard]] static DatasetFieldMaterializationRequirement ragged(
        DatasetFieldId fieldId,
        ThorImplementation::RaggedTensorDescriptor descriptor) {
        return DatasetFieldMaterializationRequirement{
            .fieldId = fieldId,
            .raggedTensorDescriptor = std::move(descriptor)};
    }

    [[nodiscard]] bool isRagged() const { return raggedTensorDescriptor.has_value(); }

    bool operator==(const DatasetFieldMaterializationRequirement&) const = default;
};

using DatasetFieldMaterializationRequirements =
    std::map<DatasetFieldId, DatasetFieldMaterializationRequirement>;

[[nodiscard]] inline std::set<DatasetFieldId> datasetFieldIds(
    const DatasetFieldMaterializationRequirements& requirements) {
    std::set<DatasetFieldId> ids;
    for (const auto& [fieldId, requirement] : requirements) {
        (void)requirement;
        ids.insert(fieldId);
    }
    return ids;
}

}  // namespace Thor
