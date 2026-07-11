#pragma once

#include <string>
#include <string_view>

namespace Thor {

class DatasetId {
   public:
    explicit DatasetId(std::string value);

    [[nodiscard]] static DatasetId generate();
    [[nodiscard]] static DatasetId fromStableMaterial(std::string_view material);

    [[nodiscard]] const std::string &str() const { return value; }

    bool operator==(const DatasetId &rhs) const = default;
    bool operator!=(const DatasetId &rhs) const = default;
    bool operator<(const DatasetId &rhs) const { return value < rhs.value; }

   private:
    std::string value;
};

}  // namespace Thor
