#pragma once

#include "DeepLearning/Api/DataType.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Api/Network/Network.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace Thor::SegmentedPrimitiveDetail {

inline bool isSupportedValueDataType(DataType dataType) {
    return dataType == DataType::FP16 || dataType == DataType::BF16 || dataType == DataType::FP32;
}

inline void requireSupportedValueDataType(DataType dataType, const char* layerName) {
    if (!isSupportedValueDataType(dataType)) {
        throw std::invalid_argument(std::string(layerName) +
                                    " supports only FP16, BF16, and FP32 values; FP64 and non-floating dtypes are intentionally unsupported.");
    }
}

inline uint64_t elementsPerValue(const std::vector<uint64_t>& trailingDimensions, const char* layerName) {
    uint64_t elements = 1;
    for (uint64_t dim : trailingDimensions) {
        if (dim == 0 || elements > std::numeric_limits<uint64_t>::max() / dim) {
            throw std::overflow_error(std::string(layerName) + " trailing value size overflows uint64_t.");
        }
        elements *= dim;
    }
    return elements;
}

inline constexpr const char* kNeutralLayerRaggedReferenceVersion = "1.0.0";

enum class SerializedRaggedGeometry { LEGACY_FIXED, PLACEMENT_NEUTRAL };

inline SerializedRaggedGeometry serializedRaggedGeometry(const nlohmann::json& raggedJson, const char* layerName) {
    const bool hasReferenceVersion = raggedJson.contains("layer_ragged_reference_version");
    const bool hasGeometryMode = raggedJson.contains("geometry_mode");
    if (!hasReferenceVersion && !hasGeometryMode) return SerializedRaggedGeometry::LEGACY_FIXED;
    if (!hasReferenceVersion || !hasGeometryMode) {
        throw std::runtime_error(std::string(layerName) + " serialized ragged reference has an incomplete geometry schema.");
    }
    if (raggedJson.at("layer_ragged_reference_version").get<std::string>() != kNeutralLayerRaggedReferenceVersion) {
        throw std::runtime_error(std::string(layerName) + " serialized ragged reference has an unsupported helper version.");
    }
    if (raggedJson.at("geometry_mode").get<std::string>() != "placement_neutral") {
        throw std::runtime_error(std::string(layerName) + " serialized ragged reference has an unsupported geometry mode.");
    }
    return SerializedRaggedGeometry::PLACEMENT_NEUTRAL;
}

// Common layer-local ragged reference writer. Existing fixed-geometry callers
// retain the historical RaggedTensor JSON byte-for-byte. Placement-neutral
// callers use an explicitly versioned semantic reference whose nested logical
// tensors carry only [M, ...trailing] and canonical [1] graph-token shapes.
inline nlohmann::json architectureJsonForLayerRaggedReference(const RaggedTensor& input) {
    if (!input.isPlacementNeutral()) return input.architectureJson();

    THOR_THROW_IF_FALSE(input.hasMaxValuesPerRow());
    return nlohmann::json{{"layer_ragged_reference_version", kNeutralLayerRaggedReferenceVersion},
                          {"geometry_mode", "placement_neutral"},
                          {"ragged_rank", input.getRaggedRank()},
                          {"values", input.getValues().architectureJson()},
                          {"offsets", input.getRowPartitionToken().architectureJson()},
                          {"max_values_per_row", input.getMaxValuesPerRow()}};
}

inline RaggedTensor reconstructInput(const nlohmann::json& inputJson, Network* network, const char* layerName) {
    Tensor values = network->getApiTensorByOriginalId(inputJson.at("values").at("id").get<uint64_t>());
    Tensor offsets = network->getApiTensorByOriginalId(inputJson.at("offsets").at("id").get<uint64_t>());

    if (serializedRaggedGeometry(inputJson, layerName) == SerializedRaggedGeometry::PLACEMENT_NEUTRAL) {
        if (inputJson.at("ragged_rank").get<uint32_t>() != 1 || !inputJson.contains("max_values_per_row") ||
            inputJson.contains("batch_size") || inputJson.contains("max_total_values")) {
            throw std::runtime_error(std::string(layerName) + " serialized placement-neutral ragged reference is malformed.");
        }
        return RaggedTensor::placementNeutral(values, offsets, inputJson.at("max_values_per_row").get<uint64_t>());
    }

    RaggedTensor input = inputJson.contains("max_values_per_row")
        ? RaggedTensor(values, offsets, inputJson.at("max_values_per_row").get<uint64_t>())
        : RaggedTensor(values, offsets);
    if (input.getBatchSize() != inputJson.at("batch_size").get<uint64_t>() ||
        input.getMaxTotalValues() != inputJson.at("max_total_values").get<uint64_t>()) {
        throw std::runtime_error(std::string(layerName) + " serialized ragged metadata does not match reconstructed tensors.");
    }
    return input;
}

inline void validateSerializedPreservedPartition(const nlohmann::json& outputJson,
                                                 const nlohmann::json& inputJson,
                                                 const RaggedTensor& input,
                                                 const char* layerName) {
    const SerializedRaggedGeometry inputGeometry = serializedRaggedGeometry(inputJson, layerName);
    const SerializedRaggedGeometry outputGeometry = serializedRaggedGeometry(outputJson, layerName);
    if (inputGeometry != outputGeometry ||
        outputJson.at("offsets").at("id").get<uint64_t>() != inputJson.at("offsets").at("id").get<uint64_t>()) {
        throw std::runtime_error(std::string(layerName) +
                                 " serialized output must preserve the exact input row partition metadata.");
    }

    if (inputGeometry == SerializedRaggedGeometry::PLACEMENT_NEUTRAL) {
        if (!input.isPlacementNeutral() || !input.hasMaxValuesPerRow() || outputJson.contains("batch_size") ||
            outputJson.contains("max_total_values") || !outputJson.contains("max_values_per_row") ||
            outputJson.at("max_values_per_row").get<uint64_t>() != input.getMaxValuesPerRow()) {
            throw std::runtime_error(std::string(layerName) +
                                     " serialized output must preserve the exact input row partition metadata.");
        }
        return;
    }

    if (outputJson.at("batch_size").get<uint64_t>() != input.getBatchSize() ||
        outputJson.at("max_total_values").get<uint64_t>() != input.getMaxTotalValues() ||
        outputJson.contains("max_values_per_row") != input.hasMaxValuesPerRow() ||
        (input.hasMaxValuesPerRow() &&
         outputJson.at("max_values_per_row").get<uint64_t>() != input.getMaxValuesPerRow())) {
        throw std::runtime_error(std::string(layerName) +
                                 " serialized output must preserve the exact input row partition metadata.");
    }
}

}  // namespace Thor::SegmentedPrimitiveDetail
