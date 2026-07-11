#include "DeepLearning/Api/Data/DatasetSchema.h"

#include <stdexcept>
#include <utility>

namespace Thor {

DatasetSchema::DatasetSchema(std::vector<DatasetField> fields) : fields(std::move(fields)) {
    if (this->fields.empty()) {
        throw std::runtime_error("DatasetSchema must contain at least one field.");
    }
    for (uint64_t ordinal = 0; ordinal < this->fields.size(); ++ordinal) {
        const DatasetField &field = this->fields.at(ordinal);
        if (field.name.empty()) {
            throw std::runtime_error("DatasetSchema field names must not be empty.");
        }
        if (field.dimensions.empty()) {
            throw std::runtime_error("DatasetSchema field dimensions must not be empty: " + field.name);
        }
        for (uint64_t dimension : field.dimensions) {
            if (dimension == 0) {
                throw std::runtime_error("DatasetSchema field dimensions must be positive: " + field.name);
            }
        }
        if (!ordinalByName.emplace(field.name, ordinal).second) {
            throw std::runtime_error("DatasetSchema contains duplicate field name: " + field.name);
        }
        if (!ordinalById.emplace(field.id, ordinal).second) {
            throw std::runtime_error("DatasetSchema contains duplicate field id.");
        }
    }
}

const DatasetField &DatasetSchema::getField(std::string_view name) const {
    const auto it = ordinalByName.find(name);
    if (it == ordinalByName.end()) {
        throw std::runtime_error("DatasetSchema has no field named: " + std::string(name));
    }
    return fields.at(it->second);
}

const DatasetField &DatasetSchema::getField(DatasetFieldId id) const {
    const auto it = ordinalById.find(id);
    if (it == ordinalById.end()) {
        throw std::runtime_error("DatasetSchema has no field with the requested id.");
    }
    return fields.at(it->second);
}

bool DatasetSchema::contains(std::string_view name) const { return ordinalByName.find(name) != ordinalByName.end(); }

}  // namespace Thor
