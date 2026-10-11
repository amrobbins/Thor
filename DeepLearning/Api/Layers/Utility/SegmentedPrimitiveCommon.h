#pragma once

#include "DeepLearning/Api/DataType.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Api/Network/Network.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
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

inline constexpr const char* kCanonicalLayerRaggedReferenceVersion = "1.0.0";

// These values identify archived reference *grammars*, not model execution modes.
// Both forms reconstruct exactly one logical ragged tensor representation.
enum class LayerRaggedReferenceGrammar { HISTORICAL_PACKED, CANONICAL_LOGICAL };

// A canonical logical reference has five required fields. Previously emitted
// references may carry the now-obsolete geometry_mode marker as a sixth field.
// Recognize that field only as legacy JSON syntax; it never selects a model
// geometry or changes placement. The grammar reader separately validates its
// archived value, while this exact key set rejects unexpected properties.
inline bool hasCanonicalLayerRaggedReferenceFields(const nlohmann::json& reference) {
    if (!reference.is_object()) return false;
    const bool hasObsoleteMarker = reference.contains("geometry_mode");
    return reference.size() == (hasObsoleteMarker ? 6U : 5U) &&
           reference.contains("layer_ragged_reference_version") &&
           reference.contains("ragged_rank") &&
           reference.contains("values") &&
           reference.contains("offsets") &&
           reference.contains("max_values_per_row");
}

inline LayerRaggedReferenceGrammar layerRaggedReferenceGrammar(const nlohmann::json& raggedJson, const char* layerName) {
    const bool hasReferenceVersion = raggedJson.contains("layer_ragged_reference_version");
    const bool hasHistoricalModeMarker = raggedJson.contains("geometry_mode");
    if (!hasReferenceVersion) {
        if (hasHistoricalModeMarker) {
            throw std::runtime_error(std::string(layerName) +
                                     " serialized ragged reference has an obsolete mode marker without its reference version.");
        }
        return LayerRaggedReferenceGrammar::HISTORICAL_PACKED;
    }
    if (raggedJson.at("layer_ragged_reference_version").get<std::string>() != kCanonicalLayerRaggedReferenceVersion) {
        throw std::runtime_error(std::string(layerName) + " serialized ragged reference has an unsupported helper version.");
    }
    // Historical canonical writers included a redundant marker describing the
    // old fixed/neutral model split. Accept that old *grammar* only; new
    // references do not carry any model geometry mode.
    if (hasHistoricalModeMarker && raggedJson.at("geometry_mode").get<std::string>() != "placement_neutral") {
        throw std::runtime_error(std::string(layerName) + " serialized ragged reference has an invalid archived mode marker.");
    }
    // Enforce the canonical reference grammar at the shared entry point, including
    // callers whose owning layer has no additional schema-specific validator.
    // This is strictly a field-set check: the six-field archived canonical form
    // remains readable, and the unversioned historical packed grammar is not
    // subjected to new canonical requirements.
    if (!hasCanonicalLayerRaggedReferenceFields(raggedJson)) {
        throw std::runtime_error(std::string(layerName) + " serialized canonical ragged reference has unexpected fields.");
    }
    return LayerRaggedReferenceGrammar::CANONICAL_LOGICAL;
}

// Durable layer references always carry logical geometry. Historical formats
// are accepted by the reader, but must never reintroduce fixed-batch models.
inline nlohmann::json architectureJsonForLayerRaggedReference(const RaggedTensor& input) {
    THOR_THROW_IF_FALSE(input.hasMaxValuesPerRow() &&
                        input.getOffsetsDimensions() == std::vector<uint64_t>{1} &&
                        input.getValuesDimensions().front() == input.getMaxValuesPerRow());
    return nlohmann::json{{"layer_ragged_reference_version", kCanonicalLayerRaggedReferenceVersion},
                          {"ragged_rank", input.getRaggedRank()},
                          {"values", input.getValues().architectureJson()},
                          {"offsets", input.getRowPartitionToken().architectureJson()},
                          {"max_values_per_row", input.getMaxValuesPerRow()}};
}

inline RaggedTensor reconstructInput(const nlohmann::json& inputJson, Network* network, const char* layerName) {
    const LayerRaggedReferenceGrammar grammar = layerRaggedReferenceGrammar(inputJson, layerName);
    if (inputJson.value("ragged_rank", 1U) != 1U) {
        throw std::runtime_error(std::string(layerName) + " serialized ragged rank must be one.");
    }
    const auto& savedValues = inputJson.at("values");
    const auto& savedOffsets = inputJson.at("offsets");
    Tensor values = network->getApiTensorByOriginalId(savedValues.at("id").get<uint64_t>());
    Tensor offsets = network->getApiTensorByOriginalId(savedOffsets.at("id").get<uint64_t>());
    // Look up semantic geometry in the owning live graph, which U1 reconstructs
    // before loading dependent layers. Archived physical dimensions are not M.
    const auto dimensions = values.getDimensions();
    if (dimensions.empty() || dimensions.front() == 0 || offsets.getDimensions() != std::vector<uint64_t>{1}) {
        throw std::runtime_error(std::string(layerName) +
                                 " requires logical source M and a canonical [1] partition token.");
    }
    const uint64_t m = dimensions.front();
    if (inputJson.contains("max_values_per_row") &&
        inputJson.at("max_values_per_row").get<uint64_t>() != m) {
        throw std::runtime_error(std::string(layerName) + " serialized semantic M does not match the source graph.");
    }
    if (grammar == LayerRaggedReferenceGrammar::CANONICAL_LOGICAL && !inputJson.contains("max_values_per_row")) {
        throw std::runtime_error(std::string(layerName) + " canonical ragged reference is missing semantic M.");
    }
    if (values.getOriginalId() != savedValues.at("id").get<uint64_t>() ||
        offsets.getOriginalId() != savedOffsets.at("id").get<uint64_t>() ||
        savedValues.at("data_type").get<DataType>() != values.getDataType() ||
        savedOffsets.at("data_type").get<DataType>() != offsets.getDataType()) {
        throw std::runtime_error(std::string(layerName) + " ragged tensor identity or dtype mismatch.");
    }
    const auto archivedDimensions = savedValues.at("dimensions").get<std::vector<uint64_t>>();
    const auto archivedOffsetsDimensions = savedOffsets.at("dimensions").get<std::vector<uint64_t>>();
    if (archivedDimensions.size() != dimensions.size() || archivedDimensions.empty() ||
        archivedOffsetsDimensions.size() != 1 || archivedOffsetsDimensions.front() == 0 ||
        !std::equal(archivedDimensions.begin() + 1, archivedDimensions.end(), dimensions.begin() + 1)) {
        throw std::runtime_error(std::string(layerName) + " ragged trailing geometry mismatch.");
    }
    // New references encode logical [M,features] and [1]. Older references
    // may record [B*M,features] and [B+1]; those leading extents are discarded.
    if (grammar == LayerRaggedReferenceGrammar::CANONICAL_LOGICAL &&
        (archivedDimensions.front() != m || archivedOffsetsDimensions.front() != 1)) {
        throw std::runtime_error(std::string(layerName) + " canonical ragged logical dimensions mismatch.");
    }
    return RaggedTensor::logical(std::move(values), std::move(offsets), m);
}

// Output values are owned by the producing layer, not by the archived physical
// packing capacity. Keep their original ID, dtype and trailing shape while
// reconstructing the leading dimension from the input's authoritative M.
// This is deliberately owner-local: ordinary dense Tensor JSON is untouched.
inline Tensor reconstructPreservedOutputValues(const nlohmann::json& featureOutputJson,
                                              const nlohmann::json& raggedOutputJson,
                                              const RaggedTensor& logicalInput,
                                              const char* layerName) {
    const LayerRaggedReferenceGrammar grammar = layerRaggedReferenceGrammar(raggedOutputJson, layerName);
    const auto& raggedValuesJson = raggedOutputJson.at("values");
    if (featureOutputJson.at("id").get<uint64_t>() != raggedValuesJson.at("id").get<uint64_t>() ||
        featureOutputJson.at("data_type").get<DataType>() != raggedValuesJson.at("data_type").get<DataType>()) {
        throw std::runtime_error(std::string(layerName) + " serialized output values identity or dtype mismatch.");
    }
    auto dimensions = featureOutputJson.at("dimensions").get<std::vector<uint64_t>>();
    const auto referenceDimensions = raggedValuesJson.at("dimensions").get<std::vector<uint64_t>>();
    if (dimensions.empty() || referenceDimensions.size() != dimensions.size() ||
        dimensions.front() == 0 || referenceDimensions.front() == 0 ||
        !std::equal(dimensions.begin() + 1, dimensions.end(), referenceDimensions.begin() + 1)) {
        throw std::runtime_error(std::string(layerName) + " serialized output values trailing geometry mismatch.");
    }
    const uint64_t m = logicalInput.getMaxValuesPerRow();
    if (grammar == LayerRaggedReferenceGrammar::CANONICAL_LOGICAL &&
        (dimensions.front() != m || referenceDimensions.front() != m)) {
        throw std::runtime_error(std::string(layerName) + " serialized logical output M mismatch.");
    }
    // Historical packed leading extents are physical hints; neither is a
    // trustworthy source from which to infer or validate the logical M.
    dimensions.front() = m;
    return Tensor::deserializeWithDimensions(featureOutputJson, dimensions);
}

// Single-owner ragged output (e.g. broadcast): the archived leading packed
// extent is never a semantic source of M. Validate trailing shape at the owner.
inline Tensor reconstructRaggedOutputValues(const nlohmann::json& outputJson,
                                            const RaggedTensor& logicalPartition,
                                            const char* layerName) {
    const LayerRaggedReferenceGrammar grammar = layerRaggedReferenceGrammar(outputJson, layerName);
    const nlohmann::json& valuesJson = outputJson.at("values");
    auto dimensions = valuesJson.at("dimensions").get<std::vector<uint64_t>>();
    if (dimensions.empty() || dimensions.front() == 0) {
        throw std::runtime_error(std::string(layerName) + " serialized output values geometry is invalid.");
    }
    const uint64_t m = logicalPartition.getMaxValuesPerRow();
    if (grammar == LayerRaggedReferenceGrammar::CANONICAL_LOGICAL && dimensions.front() != m) {
        throw std::runtime_error(std::string(layerName) + " serialized logical output M mismatch.");
    }
    dimensions.front() = m;
    return Tensor::deserializeWithDimensions(valuesJson, dimensions);
}

// Layers such as sequence slice/concatenate create a new logical partition.
// Their semantic output M comes from the operation, never from historical
// [B*M] or [B+1] output JSON. Reconstruct only the layer-owned tensors here;
// makeLayer subsequently validates new partition identity and semantics.
inline RaggedTensor reconstructNewPartitionOutput(const nlohmann::json& outputJson,
                                                  uint64_t semanticM,
                                                  DataType valuesDType,
                                                  DataType offsetsDType,
                                                  const std::vector<uint64_t>& trailing,
                                                  const char* layerName) {
    const LayerRaggedReferenceGrammar grammar = layerRaggedReferenceGrammar(outputJson, layerName);
    if (semanticM == 0 || outputJson.value("ragged_rank", 1U) != 1U ||
        (outputJson.contains("max_values_per_row") &&
         outputJson.at("max_values_per_row").get<uint64_t>() != semanticM)) {
        throw std::runtime_error(std::string(layerName) + " output semantic M/rank mismatch.");
    }
    const auto& valuesJson = outputJson.at("values");
    const auto& offsetsJson = outputJson.at("offsets");
    auto archivedValuesDimensions = valuesJson.at("dimensions").get<std::vector<uint64_t>>();
    const auto archivedOffsetsDimensions = offsetsJson.at("dimensions").get<std::vector<uint64_t>>();
    if (archivedValuesDimensions.size() != trailing.size() + 1 ||
        archivedValuesDimensions.front() == 0 ||
        !std::equal(archivedValuesDimensions.begin() + 1, archivedValuesDimensions.end(), trailing.begin()) ||
        archivedOffsetsDimensions.size() != 1 || archivedOffsetsDimensions.front() == 0 ||
        valuesJson.at("data_type").get<DataType>() != valuesDType ||
        offsetsJson.at("data_type").get<DataType>() != offsetsDType ||
        valuesJson.at("id").get<uint64_t>() == offsetsJson.at("id").get<uint64_t>() ||
        (grammar == LayerRaggedReferenceGrammar::CANONICAL_LOGICAL &&
         (archivedValuesDimensions.front() != semanticM || archivedOffsetsDimensions.front() != 1))) {
        throw std::runtime_error(std::string(layerName) + " output tensor identity, dtype, or trailing geometry mismatch.");
    }
    archivedValuesDimensions.front() = semanticM;
    Tensor values = Tensor::deserializeWithDimensions(valuesJson, archivedValuesDimensions);
    Tensor partition = Tensor::deserializeWithDimensions(offsetsJson, {1});
    return RaggedTensor::logical(std::move(values), std::move(partition), semanticM);
}

inline void validateSerializedPreservedPartition(const nlohmann::json& outputJson,
                                                 const nlohmann::json& inputJson,
                                                 const RaggedTensor& input,
                                                 const char* layerName) {
    const LayerRaggedReferenceGrammar grammar = layerRaggedReferenceGrammar(outputJson, layerName);
    (void)layerRaggedReferenceGrammar(inputJson, layerName); // Validate the input grammar too.
    if (!input.hasMaxValuesPerRow() || input.getOffsetsDimensions() != std::vector<uint64_t>{1} ||
        outputJson.at("offsets").at("id").get<uint64_t>() != input.getRowPartitionToken().getOriginalId() ||
        outputJson.at("offsets").at("id").get<uint64_t>() !=
            inputJson.at("offsets").at("id").get<uint64_t>() ||
        outputJson.at("offsets").at("data_type").get<DataType>() != input.getOffsetsDataType() ||
        (outputJson.contains("max_values_per_row") &&
         outputJson.at("max_values_per_row").get<uint64_t>() != input.getMaxValuesPerRow()) ||
        outputJson.value("ragged_rank", 1U) != 1U) {
        throw std::runtime_error(std::string(layerName) +
                                 " serialized output must preserve the canonical logical row partition.");
    }
    const auto savedOffsetsDims = outputJson.at("offsets").at("dimensions").get<std::vector<uint64_t>>();
    if (savedOffsetsDims.size() != 1 || savedOffsetsDims.front() == 0 ||
        (grammar == LayerRaggedReferenceGrammar::CANONICAL_LOGICAL &&
         (savedOffsetsDims.front() != 1 || !outputJson.contains("max_values_per_row")))) {
        throw std::runtime_error(std::string(layerName) + " malformed serialized output partition token.");
    }
}

}  // namespace Thor::SegmentedPrimitiveDetail
