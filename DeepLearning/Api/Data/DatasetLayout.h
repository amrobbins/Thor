#pragma once

#include "DeepLearning/Implementation/Tensor/DataType.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

class DatasetLayout {
   public:
    enum class WindowedTensorReferenceMode { INDEXED, AFFINE };

    static constexpr const char *FORMAT = "thor.dataset.v1";

    struct TensorSpec {
        std::string name;
        ThorImplementation::DataType dataType;
        std::vector<uint64_t> dimensions;
        uint64_t offsetBytes;
        uint64_t numBytes;

        bool operator==(const TensorSpec &rhs) const;
        bool operator!=(const TensorSpec &rhs) const { return !(*this == rhs); }
    };

    struct TensorShape {
        std::string name;
        std::vector<uint64_t> dimensions;
        ThorImplementation::DataType dataType;

        TensorShape(std::string name,
                    std::vector<uint64_t> dimensions,
                    ThorImplementation::DataType dataType);
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

    /** One immutable named source that may feed any number of windowed output fields. */
    struct WindowedTensorSourceSpec {
        std::string name;
        ThorImplementation::DataType dataType;
        std::vector<uint64_t> stepDimensions;
        ThorImplementation::DataType keyDataType;
        std::optional<std::string> sourceFilename;
        uint64_t sourceNumBytes = 0;
        std::vector<WindowedTensorSourceSequence> sourceSequences;

        [[nodiscard]] uint64_t stepNumBytes() const;
        [[nodiscard]] uint64_t keyNumBytes() const;

        bool operator==(const WindowedTensorSourceSpec &rhs) const;
        bool operator!=(const WindowedTensorSourceSpec &rhs) const { return !(*this == rhs); }
        [[nodiscard]] bool contractEquals(const WindowedTensorSourceSpec &rhs) const;
    };

    struct WindowedTensorSpec {
        std::string name;
        std::string sourceName;
        ThorImplementation::DataType dataType;
        std::vector<uint64_t> dimensions;
        ThorImplementation::DataType keyDataType;
        ThorImplementation::DataType indexDataType;
        double padValue = 0.0;
        std::optional<std::string> maskName;
        WindowedTensorReferenceMode referenceMode = WindowedTensorReferenceMode::INDEXED;
        uint64_t referenceOffsetBytes = 0;
        uint64_t referenceNumBytes = 0;

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

    struct WindowedTensorSourceShape {
        std::string name;
        std::vector<uint64_t> stepDimensions;
        ThorImplementation::DataType dataType;
        ThorImplementation::DataType keyDataType;

        WindowedTensorSourceShape(std::string name,
                                  std::vector<uint64_t> stepDimensions,
                                  ThorImplementation::DataType dataType,
                                  ThorImplementation::DataType keyDataType);
    };

    struct WindowedTensorShape {
        std::string name;
        std::vector<uint64_t> dimensions;
        std::string sourceName;
        ThorImplementation::DataType indexDataType;
        double padValue = 0.0;
        std::optional<std::string> maskName;
        WindowedTensorReferenceMode referenceMode = WindowedTensorReferenceMode::INDEXED;

        WindowedTensorShape(std::string name,
                            std::vector<uint64_t> dimensions,
                            std::string sourceName,
                            ThorImplementation::DataType indexDataType,
                            double padValue = 0.0,
                            std::optional<std::string> maskName = std::nullopt,
                            WindowedTensorReferenceMode referenceMode = WindowedTensorReferenceMode::INDEXED);
    };

    DatasetLayout();
    DatasetLayout(uint64_t recordSizeBytes, std::vector<TensorSpec> tensors);
    DatasetLayout(uint64_t recordSizeBytes,
                  std::vector<TensorSpec> tensors,
                  std::vector<WindowedTensorSourceSpec> windowedTensorSources,
                  std::vector<WindowedTensorSpec> windowedTensors);

    [[nodiscard]] uint64_t recordSizeBytes() const;
    [[nodiscard]] const TensorSpec &tensor(std::string_view name) const;
    [[nodiscard]] const std::vector<TensorSpec> &tensors() const;
    [[nodiscard]] bool hasWindowedTensors() const;
    [[nodiscard]] bool hasAffineWindowedTensors() const;
    [[nodiscard]] bool hasIndexedWindowedTensors() const;
    [[nodiscard]] const WindowedTensorSourceSpec &windowedTensorSource(std::string_view name) const;
    [[nodiscard]] const std::vector<WindowedTensorSourceSpec> &windowedTensorSources() const;
    [[nodiscard]] const WindowedTensorSpec &windowedTensor(std::string_view name) const;
    [[nodiscard]] const std::vector<WindowedTensorSpec> &windowedTensors() const;

    void validate() const;
    void validateRequestedLayoutExact(const DatasetLayout &requested) const;

    [[nodiscard]] nlohmann::json toJson() const;
    static DatasetLayout fromJson(const nlohmann::json &j);

    void writeManifest(const std::filesystem::path &path) const;
    static DatasetLayout readManifest(const std::filesystem::path &path);

    static DatasetLayout fromTensorShapes(const std::vector<TensorShape> &tensors);
    static DatasetLayout fromTensorShapes(const std::vector<TensorShape> &tensors,
                                          const std::vector<WindowedTensorSourceShape> &windowedTensorSources,
                                          const std::vector<WindowedTensorShape> &windowedTensors);

   private:
    uint64_t layoutRecordSizeBytes;
    std::vector<TensorSpec> layoutTensors;
    std::vector<WindowedTensorSourceSpec> layoutWindowedTensorSources;
    std::vector<WindowedTensorSpec> layoutWindowedTensors;
};
