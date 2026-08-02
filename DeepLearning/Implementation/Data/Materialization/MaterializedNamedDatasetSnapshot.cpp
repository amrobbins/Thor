#include "DeepLearning/Implementation/Data/Materialization/MaterializedNamedDatasetSnapshot.h"

uint64_t MaterializedNamedDatasetSnapshot::totalBytes() const {
    uint64_t total = 0;
    for (const auto &entry : fields) {
        total += entry.second.getDescriptor().getArraySizeInBytes();
    }
    for (const auto &entry : raggedFields) {
        total += entry.second.totalBytes();
    }
    return total;
}

bool MaterializedNamedDatasetSnapshot::hasField(Thor::DatasetFieldId id) const {
    return fields.find(id) != fields.end() || raggedFields.find(id) != raggedFields.end();
}

bool MaterializedNamedDatasetSnapshot::hasRaggedField(Thor::DatasetFieldId id) const {
    return raggedFields.find(id) != raggedFields.end();
}

bool MaterializedNamedDatasetSnapshot::hasField(const std::string &name) const {
    return schema.contains(name) && hasField(schema.getField(name).id);
}

bool MaterializedNamedDatasetSnapshot::hasRaggedField(const std::string &name) const {
    return schema.contains(name) && hasRaggedField(schema.getField(name).id);
}

const ThorImplementation::Tensor &MaterializedNamedDatasetSnapshot::field(
    Thor::DatasetFieldId id) const {
    const auto it = fields.find(id);
    if (it == fields.end()) {
        throw std::runtime_error("Materialized named dataset snapshot is missing field id.");
    }
    return it->second;
}

const ThorImplementation::Tensor &MaterializedNamedDatasetSnapshot::tensor(
    const std::string &name) const {
    return field(schema.getField(name).id);
}

const MaterializedRaggedFieldSnapshot &MaterializedNamedDatasetSnapshot::raggedField(
    Thor::DatasetFieldId id) const {
    const auto it = raggedFields.find(id);
    if (it == raggedFields.end()) {
        throw std::runtime_error("Materialized named dataset snapshot is missing ragged field id.");
    }
    return it->second;
}

const MaterializedRaggedFieldSnapshot &MaterializedNamedDatasetSnapshot::raggedTensor(
    const std::string &name) const {
    return raggedField(schema.getField(name).id);
}
