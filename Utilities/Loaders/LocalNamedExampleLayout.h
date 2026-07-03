#pragma once

#include "DeepLearning/Implementation/Tensor/DataType.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

class LocalNamedExampleLayout {
   public:
    static constexpr const char *FORMAT = "thor.local_named_example_dataset.v1";

    struct TensorSpec {
        std::string name;
        ThorImplementation::DataType dataType;
        std::vector<uint64_t> dimensions;
        uint64_t offsetBytes;
        uint64_t numBytes;

        bool operator==(const TensorSpec &rhs) const;
        bool operator!=(const TensorSpec &rhs) const { return !(*this == rhs); }
    };

    LocalNamedExampleLayout();
    LocalNamedExampleLayout(ThorImplementation::DataType dataType, uint64_t recordSizeBytes, std::vector<TensorSpec> tensors);

    [[nodiscard]] ThorImplementation::DataType dataType() const;
    [[nodiscard]] uint64_t recordSizeBytes() const;
    [[nodiscard]] const TensorSpec &tensor(std::string_view name) const;
    [[nodiscard]] const std::vector<TensorSpec> &tensors() const;

    void validate() const;
    void validateRequestedLayoutExact(const LocalNamedExampleLayout &requested) const;

    [[nodiscard]] nlohmann::json toJson() const;
    static LocalNamedExampleLayout fromJson(const nlohmann::json &j);

    void writeManifest(const std::filesystem::path &path) const;
    static LocalNamedExampleLayout readManifest(const std::filesystem::path &path);

    static LocalNamedExampleLayout fromTensorShapes(const std::map<std::string, std::vector<uint64_t>> &tensors,
                                                    ThorImplementation::DataType dataType);
    static LocalNamedExampleLayout fromTensorShapes(const std::vector<std::pair<std::string, std::vector<uint64_t>>> &tensors,
                                                    ThorImplementation::DataType dataType);

   private:
    ThorImplementation::DataType layoutDataType;
    uint64_t layoutRecordSizeBytes;
    std::vector<TensorSpec> layoutTensors;
};
