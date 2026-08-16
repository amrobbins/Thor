#include "DeepLearning/Implementation/Layers/RaggedCustomLayer.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"

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
                  trustedReservedRaggedInputNames(inputNames)),
      fullCapacityRows(fullCapacityRows),
      inputElementsPerValue(std::move(inputElementsPerValue)),
      outputElementsPerValue(std::move(outputElementsPerValue)),
      valuesInputPorts(std::move(valuesInputPorts)),
      offsetsInputPort(offsetsInputPort),
      inputPortCount(static_cast<uint32_t>(inputNames.size())),
      outputPortCount(static_cast<uint32_t>(outputNames.size())) {
    if (fullCapacityRows == 0) {
        throw std::invalid_argument("RaggedCustomLayer packed capacity must be non-zero.");
    }
    if (inputPortCount == 0 || outputPortCount == 0 || this->valuesInputPorts.empty() ||
        this->inputElementsPerValue.size() != this->valuesInputPorts.size() ||
        this->outputElementsPerValue.size() != outputPortCount || this->offsetsInputPort >= inputPortCount) {
        throw std::invalid_argument(
            "RaggedCustomLayer requires packed-values metadata for at least one input and every output.");
    }
    for (uint64_t elementsPerValue : this->outputElementsPerValue) {
        if (elementsPerValue == 0) {
            throw std::invalid_argument("RaggedCustomLayer output row widths must be non-zero.");
        }
    }
    for (size_t i = 0; i < this->valuesInputPorts.size(); ++i) {
        if (this->valuesInputPorts[i] >= inputPortCount || this->valuesInputPorts[i] == this->offsetsInputPort ||
            this->inputElementsPerValue[i] == 0) {
            throw std::invalid_argument("RaggedCustomLayer packed-values input metadata is invalid.");
        }
        for (size_t j = 0; j < i; ++j) {
            if (this->valuesInputPorts[j] == this->valuesInputPorts[i]) {
                throw std::invalid_argument("RaggedCustomLayer packed-values input ports must be unique.");
            }
        }
    }
}

uint32_t RaggedCustomLayer::applicationIndexForConnection(uint32_t connectionNumber) const {
    return connectionNumber / inputPortCount;
}

uint64_t RaggedCustomLayer::requireActiveRows(uint32_t applicationIndex) const {
    const uint32_t offsetsFlatIndex = applicationIndex * inputPortCount + offsetsInputPort;
    if (offsetsFlatIndex >= featureInputs.size() || !featureInputs[offsetsFlatIndex].has_value()) {
        throw std::runtime_error("RaggedCustomLayer row-partition offsets input is not connected for this application.");
    }

    const Tensor offsets = featureInputs[offsetsFlatIndex].value();
    const TensorDescriptor offsetsDescriptor = offsets.getDescriptor();
    if (offsetsDescriptor.getNumDimensions() != 1 || offsetsDescriptor.getDimensions()[0] == 0 ||
        !RowPartitionDescriptor::isValidOffsetsDataType(offsetsDescriptor.getDataType())) {
        throw std::runtime_error("RaggedCustomLayer row-partition offsets input is not canonical.");
    }

    const uint64_t batchSize = offsetsDescriptor.getDimensions()[0] - 1;
    RowPartitionRuntime rowPartition(
        offsets, RowPartitionDescriptor(batchSize, fullCapacityRows, offsetsDescriptor.getDataType()));
    const std::optional<uint64_t> activeRows = rowPartition.getHostActiveValueCountIfAvailable();
    if (!activeRows.has_value()) {
        throw std::runtime_error(
            "RaggedCustomLayer requires a host-known active-value count on its row partition for tail canonicalization.");
    }
    if (activeRows.value() > fullCapacityRows) {
        throw std::runtime_error("RaggedCustomLayer active row count exceeds its packed capacity.");
    }
    return activeRows.value();
}

void RaggedCustomLayer::zeroInactiveTail(Tensor tensor, uint64_t activeRows, uint64_t rowWidth, Stream stream) const {
    THOR_THROW_IF_FALSE(activeRows <= fullCapacityRows);
    THOR_THROW_IF_FALSE(rowWidth > 0);
    THOR_THROW_IF_FALSE(tensor.getTotalNumElements() == fullCapacityRows * rowWidth);
    if (activeRows == fullCapacityRows) {
        return;
    }
    Tensor tail =
        tensor.aliasView({fullCapacityRows - activeRows, rowWidth}, {rowWidth, 1}, activeRows * rowWidth);
    tail.memsetAsync(stream, 0);
}

void RaggedCustomLayer::beforeForwardExpressionRun(uint32_t connectionNumber, Stream& stream) {
    (void)stream;
    const uint32_t applicationIndex = applicationIndexForConnection(connectionNumber);
    const uint64_t activeRows = requireActiveRows(applicationIndex);
    if (activeRowsByApplication.size() <= applicationIndex) {
        activeRowsByApplication.resize(applicationIndex + 1, 0);
    }
    activeRowsByApplication[applicationIndex] = activeRows;
}

void RaggedCustomLayer::afterForwardExpressionRun(uint32_t connectionNumber, Stream& stream) {
    const uint32_t applicationIndex = applicationIndexForConnection(connectionNumber);
    if (applicationIndex >= activeRowsByApplication.size()) {
        throw std::runtime_error("RaggedCustomLayer has no active-row state for this application.");
    }
    const uint64_t activeRows = activeRowsByApplication[applicationIndex];
    for (uint32_t outputPort = 0; outputPort < outputPortCount; ++outputPort) {
        const uint32_t outputFlatIndex = applicationIndex * outputPortCount + outputPort;
        if (outputFlatIndex >= featureOutputs.size() || !featureOutputs[outputFlatIndex].has_value()) {
            throw std::runtime_error("RaggedCustomLayer output is not connected for this application.");
        }
        Tensor output = featureOutputs[outputFlatIndex].value();
        zeroInactiveTail(output, activeRows, outputElementsPerValue[outputPort], stream);
    }
}

void RaggedCustomLayer::afterBackwardErrorExpressionRun(uint32_t connectionNumber, Stream& stream) {
    const uint32_t applicationIndex = applicationIndexForConnection(connectionNumber);
    if (applicationIndex >= activeRowsByApplication.size()) {
        return;
    }
    const uint64_t activeRows = activeRowsByApplication[applicationIndex];
    for (size_t i = 0; i < valuesInputPorts.size(); ++i) {
        const uint32_t valuesFlatIndex = applicationIndex * inputPortCount + valuesInputPorts[i];
        if (valuesFlatIndex >= errorOutputs.size() || !errorOutputs[valuesFlatIndex].has_value()) {
            continue;
        }
        Tensor dValues = errorOutputs[valuesFlatIndex].value();
        zeroInactiveTail(dValues, activeRows, inputElementsPerValue[i], stream);
    }
}

}  // namespace ThorImplementation
