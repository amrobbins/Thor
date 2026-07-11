#pragma once

#include "DeepLearning/Implementation/Tensor/DataType.h"

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace Thor {

using DatasetFieldId = uint64_t;

enum class DatasetFieldKind { DENSE, WINDOWED, WINDOW_MASK };

struct DatasetField {
    DatasetFieldId id = 0;
    std::string name;
    ThorImplementation::DataType dataType = ThorImplementation::DataType::FP32;
    std::vector<uint64_t> dimensions;
    DatasetFieldKind kind = DatasetFieldKind::DENSE;

    bool operator==(const DatasetField &rhs) const = default;
    bool operator!=(const DatasetField &rhs) const = default;
};

class DatasetSchema {
   public:
    explicit DatasetSchema(std::vector<DatasetField> fields);

    [[nodiscard]] const std::vector<DatasetField> &getFields() const { return fields; }
    [[nodiscard]] const DatasetField &getField(std::string_view name) const;
    [[nodiscard]] const DatasetField &getField(DatasetFieldId id) const;
    [[nodiscard]] bool contains(std::string_view name) const;
    [[nodiscard]] uint64_t size() const { return static_cast<uint64_t>(fields.size()); }

    bool operator==(const DatasetSchema &rhs) const { return fields == rhs.fields; }
    bool operator!=(const DatasetSchema &rhs) const { return !(*this == rhs); }

   private:
    std::vector<DatasetField> fields;
    std::map<std::string, uint64_t, std::less<>> ordinalByName;
    std::map<DatasetFieldId, uint64_t> ordinalById;
};

}  // namespace Thor
