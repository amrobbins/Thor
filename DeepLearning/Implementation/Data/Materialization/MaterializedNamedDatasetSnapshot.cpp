#include "DeepLearning/Implementation/Data/Materialization/MaterializedNamedDatasetSnapshot.h"

uint64_t MaterializedNamedDatasetSnapshot::totalBytes() const {
    uint64_t total = 0;
    for (const auto &entry : fields) {
        total += entry.second.getDescriptor().getArraySizeInBytes();
    }
    return total;
}

bool MaterializedNamedDatasetSnapshot::hasField(Thor::DatasetFieldId id) const {
    return fields.find(id) != fields.end();
}

bool MaterializedNamedDatasetSnapshot::hasField(const std::string &name) const {
    return schema.contains(name) && hasField(schema.getField(name).id);
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
