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

inline RaggedTensor reconstructInput(const nlohmann::json& inputJson, Network* network, const char* layerName) {
    Tensor values = network->getApiTensorByOriginalId(inputJson.at("values").at("id").get<uint64_t>());
    Tensor offsets = network->getApiTensorByOriginalId(inputJson.at("offsets").at("id").get<uint64_t>());
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
    if (outputJson.at("offsets").at("id").get<uint64_t>() != inputJson.at("offsets").at("id").get<uint64_t>() ||
        outputJson.at("batch_size").get<uint64_t>() != input.getBatchSize() ||
        outputJson.at("max_total_values").get<uint64_t>() != input.getMaxTotalValues() ||
        outputJson.contains("max_values_per_row") != input.hasMaxValuesPerRow() ||
        (input.hasMaxValuesPerRow() &&
         outputJson.at("max_values_per_row").get<uint64_t>() != input.getMaxValuesPerRow())) {
        throw std::runtime_error(std::string(layerName) +
                                 " serialized output must preserve the exact input row partition metadata.");
    }
}

}  // namespace Thor::SegmentedPrimitiveDetail
