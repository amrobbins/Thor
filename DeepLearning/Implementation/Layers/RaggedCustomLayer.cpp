#include "DeepLearning/Implementation/Layers/RaggedCustomLayer.h"

#include <set>
#include <stdexcept>
#include <utility>

namespace ThorImplementation {

namespace {

std::vector<bool> raggedInputDimensionsIncludeBatch(size_t inputPortCount, const std::vector<uint32_t>& valuesInputPorts) {
    // Packed-values tensors use their leading full-capacity row dimension as
    // CustomLayer's physical batch-like dimension. Structural inputs such as
    // offsets already include their complete physical shape and must not
    // participate in batch-capacity agreement/inference.
    std::vector<bool> dimensionsIncludeBatch(inputPortCount, true);
    for (uint32_t valuesInputPort : valuesInputPorts) {
        if (valuesInputPort < inputPortCount) {
            dimensionsIncludeBatch[valuesInputPort] = false;
        }
    }
    return dimensionsIncludeBatch;
}

std::set<std::string> trustedReservedRaggedInputNames(const std::vector<std::string>& inputNames) {
    // Legacy ragged Expression-backed API layers use ordinary structural-port
    // names such as "feature_offsets" or "offsets".  The generic API
    // CustomLayer uses Thor's reserved hidden port instead.  Only authorize
    // that reserved name when the caller actually declared it; otherwise the
    // base CustomLayer validation would incorrectly require every
    // RaggedCustomLayer caller to expose the generic CustomLayer port name.
    for (const std::string& inputName : inputNames) {
        if (inputName == RaggedCustomLayer::RAGGED_OFFSETS_INPUT_NAME) {
            return {RaggedCustomLayer::RAGGED_OFFSETS_INPUT_NAME};
        }
    }
    return {};
}

}  // namespace

RaggedCustomLayer::RaggedCustomLayer(DynamicExpression expression,
                                     std::vector<std::string> inputNames,
                                     std::vector<std::string> outputNames,
                                     TensorPlacement placement,
                                     bool inferenceOnly,
                                     uint64_t fullCapacityRows,
                                     uint64_t inputElementsPerValue,
                                     uint64_t outputElementsPerValue,
                                     uint32_t valuesInputPort,
                                     uint32_t offsetsInputPort,
                                     int64_t stampedId)
    : RaggedCustomLayer(std::move(expression),
                        std::move(inputNames),
                        std::move(outputNames),
                        placement,
                        std::vector<std::shared_ptr<PhysicalParameter>>{},
                        inferenceOnly,
                        fullCapacityRows,
                        std::vector<uint64_t>{inputElementsPerValue},
                        std::vector<uint64_t>{outputElementsPerValue},
                        std::vector<uint32_t>{valuesInputPort},
                        offsetsInputPort,
                        stampedId) {}

RaggedCustomLayer::RaggedCustomLayer(DynamicExpression expression,
                                     std::vector<std::string> inputNames,
                                     std::vector<std::string> outputNames,
                                     TensorPlacement placement,
                                     bool inferenceOnly,
                                     uint64_t fullCapacityRows,
                                     std::vector<uint64_t> inputElementsPerValue,
                                     uint64_t outputElementsPerValue,
                                     std::vector<uint32_t> valuesInputPorts,
                                     uint32_t offsetsInputPort,
                                     int64_t stampedId)
    : RaggedCustomLayer(std::move(expression),
                        std::move(inputNames),
                        std::move(outputNames),
                        placement,
                        std::vector<std::shared_ptr<PhysicalParameter>>{},
                        inferenceOnly,
                        fullCapacityRows,
                        std::move(inputElementsPerValue),
                        std::vector<uint64_t>{outputElementsPerValue},
                        std::move(valuesInputPorts),
                        offsetsInputPort,
                        stampedId) {}

RaggedCustomLayer::RaggedCustomLayer(DynamicExpression expression,
                                     std::vector<std::string> inputNames,
                                     std::vector<std::string> outputNames,
                                     TensorPlacement placement,
                                     std::vector<std::shared_ptr<PhysicalParameter>> parameters,
                                     bool inferenceOnly,
                                     uint64_t fullCapacityRows,
                                     std::vector<uint64_t> inputElementsPerValue,
                                     std::vector<uint64_t> outputElementsPerValue,
                                     std::vector<uint32_t> valuesInputPorts,
                                     uint32_t offsetsInputPort,
                                     int64_t stampedId,
                                     std::vector<DeclaredOutputDescriptor> declaredOutputDescriptors)
    : CustomLayer(std::move(expression),
                  inputNames,
                  outputNames,
                  placement,
                  parameters,
                  inferenceOnly,
                  stampedId,
                  std::move(declaredOutputDescriptors),
                  false,
                  false,
                  raggedInputDimensionsIncludeBatch(inputNames.size(), valuesInputPorts),
                  std::nullopt,
                  trustedReservedRaggedInputNames(inputNames)) {
    const uint32_t inputPortCount = static_cast<uint32_t>(inputNames.size());
    const uint32_t outputPortCount = static_cast<uint32_t>(outputNames.size());
    if (fullCapacityRows == 0) {
        throw std::invalid_argument("RaggedCustomLayer packed capacity must be non-zero.");
    }
    if (inputPortCount == 0 || outputPortCount == 0 || valuesInputPorts.empty() ||
        inputElementsPerValue.size() != valuesInputPorts.size() ||
        outputElementsPerValue.size() != outputPortCount || offsetsInputPort >= inputPortCount) {
        throw std::invalid_argument(
            "RaggedCustomLayer requires packed-values metadata for at least one input and every output.");
    }
    for (uint64_t elementsPerValue : outputElementsPerValue) {
        if (elementsPerValue == 0) {
            throw std::invalid_argument("RaggedCustomLayer output row widths must be non-zero.");
        }
    }
    for (size_t i = 0; i < valuesInputPorts.size(); ++i) {
        if (valuesInputPorts[i] >= inputPortCount || valuesInputPorts[i] == offsetsInputPort ||
            inputElementsPerValue[i] == 0) {
            throw std::invalid_argument("RaggedCustomLayer packed-values input metadata is invalid.");
        }
        for (size_t j = 0; j < i; ++j) {
            if (valuesInputPorts[j] == valuesInputPorts[i]) {
                throw std::invalid_argument("RaggedCustomLayer packed-values input ports must be unique.");
            }
        }
    }
}

}  // namespace ThorImplementation
