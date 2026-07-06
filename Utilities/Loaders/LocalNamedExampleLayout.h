#pragma once

#include "DeepLearning/Implementation/Tensor/DataType.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
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

    struct WindowedTensorSourceSequence {
        std::string keyHex;
        int64_t startIndex = 0;
        int64_t endIndexExclusive = 0;
        uint64_t offsetBytes = 0;
        uint64_t numSteps = 0;
        uint64_t numBytes = 0;

        bool operator==(const WindowedTensorSourceSequence &rhs) const;
        bool operator!=(const WindowedTensorSourceSequence &rhs) const { return !(*this == rhs); }
    };

    struct WindowedTensorSpec {
        std::string name;
        ThorImplementation::DataType dataType;
        std::vector<uint64_t> dimensions;
        ThorImplementation::DataType keyDataType;
        ThorImplementation::DataType indexDataType;
        double padValue = 0.0;
        std::optional<std::string> maskName;
        uint64_t referenceOffsetBytes = 0;
        uint64_t referenceNumBytes = 0;
        std::optional<std::string> sourceFilename;
        uint64_t sourceNumBytes = 0;
        std::vector<WindowedTensorSourceSequence> sourceSequences;

        [[nodiscard]] uint64_t windowLength() const;
        [[nodiscard]] std::vector<uint64_t> sourceStepDimensions() const;
        [[nodiscard]] uint64_t sourceStepNumBytes() const;
        [[nodiscard]] uint64_t outputNumBytes() const;
        [[nodiscard]] uint64_t keyNumBytes() const;
        [[nodiscard]] uint64_t indexNumBytes() const;

        bool operator==(const WindowedTensorSpec &rhs) const;
        bool operator!=(const WindowedTensorSpec &rhs) const { return !(*this == rhs); }
        [[nodiscard]] bool contractEquals(const WindowedTensorSpec &rhs) const;
    };

    struct WindowedTensorShape {
        std::string name;
        std::vector<uint64_t> dimensions;
        ThorImplementation::DataType keyDataType;
        ThorImplementation::DataType indexDataType;
        double padValue = 0.0;
        std::optional<std::string> maskName;
        ThorImplementation::DataType dataType;

        WindowedTensorShape(std::string name,
                            std::vector<uint64_t> dimensions,
                            ThorImplementation::DataType keyDataType,
                            ThorImplementation::DataType indexDataType,
                            double padValue = 0.0,
                            std::optional<std::string> maskName = std::nullopt,
                            ThorImplementation::DataType dataType = ThorImplementation::DataType::FP32);
    };

    LocalNamedExampleLayout();
    LocalNamedExampleLayout(ThorImplementation::DataType dataType, uint64_t recordSizeBytes, std::vector<TensorSpec> tensors);
    LocalNamedExampleLayout(ThorImplementation::DataType dataType,
                            uint64_t recordSizeBytes,
                            std::vector<TensorSpec> tensors,
                            std::vector<WindowedTensorSpec> windowedTensors);

    [[nodiscard]] ThorImplementation::DataType dataType() const;
    [[nodiscard]] uint64_t recordSizeBytes() const;
    [[nodiscard]] const TensorSpec &tensor(std::string_view name) const;
    [[nodiscard]] const std::vector<TensorSpec> &tensors() const;
    [[nodiscard]] bool hasWindowedTensors() const;
    [[nodiscard]] const WindowedTensorSpec &windowedTensor(std::string_view name) const;
    [[nodiscard]] const std::vector<WindowedTensorSpec> &windowedTensors() const;

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
    static LocalNamedExampleLayout fromTensorShapes(
        const std::vector<std::pair<std::string, std::vector<uint64_t>>> &tensors,
        const std::vector<WindowedTensorShape> &windowedTensors,
        ThorImplementation::DataType dataType);

   private:
    ThorImplementation::DataType layoutDataType;
    uint64_t layoutRecordSizeBytes;
    std::vector<TensorSpec> layoutTensors;
    std::vector<WindowedTensorSpec> layoutWindowedTensors;
};
