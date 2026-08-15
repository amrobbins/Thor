#include "DeepLearning/Implementation/Layers/RaggedExpressionLayer.h"

#include "DeepLearning/Implementation/ThorError.h"

#include <stdexcept>
#include <utility>

namespace ThorImplementation {

RaggedExpressionLayer::RaggedExpressionLayer(DynamicExpression expression,
                                             std::vector<std::string> inputNames,
                                             std::vector<std::string> outputNames,
                                             TensorPlacement placement,
                                             bool inferenceOnly,
                                             uint64_t fullCapacityRows,
                                             uint64_t inputElementsPerValue,
                                             uint64_t outputElementsPerValue,
                                             uint32_t valuesInputPort,
                                             int64_t stampedId)
    : CustomLayer(std::move(expression),
                  inputNames,
                  outputNames,
                  placement,
                  std::vector<std::shared_ptr<PhysicalParameter>>{},
                  inferenceOnly,
                  stampedId),
      fullCapacityRows(fullCapacityRows),
      inputElementsPerValue(inputElementsPerValue),
      outputElementsPerValue(outputElementsPerValue),
      valuesInputPort(valuesInputPort),
      inputPortCount(static_cast<uint32_t>(inputNames.size())),
      outputPortCount(static_cast<uint32_t>(outputNames.size())) {
    if (fullCapacityRows == 0 || inputElementsPerValue == 0 || outputElementsPerValue == 0) {
        throw std::invalid_argument("RaggedExpressionLayer dimensions must be non-zero.");
    }
    if (inputPortCount == 0 || outputPortCount != 1 || valuesInputPort >= inputPortCount) {
        throw std::invalid_argument("RaggedExpressionLayer requires at least one input and exactly one output.");
    }
}

uint32_t RaggedExpressionLayer::applicationIndexForConnection(uint32_t connectionNumber) const {
    return connectionNumber / inputPortCount;
}

uint64_t RaggedExpressionLayer::requireActiveRows(uint32_t applicationIndex) const {
    const uint32_t flatIndex = applicationIndex * inputPortCount + valuesInputPort;
    if (flatIndex >= featureInputs.size() || !featureInputs[flatIndex].has_value()) {
        throw std::runtime_error("RaggedExpressionLayer values input is not connected for this application.");
    }
    const Tensor values = featureInputs[flatIndex].value();
    const std::optional<uint64_t> activeRows = values.getRaggedActiveRows();
    if (!activeRows.has_value()) {
        throw std::runtime_error("RaggedExpressionLayer requires host-known active-row metadata on packed values.");
    }
    if (activeRows.value() > fullCapacityRows) {
        throw std::runtime_error("RaggedExpressionLayer active row count exceeds its packed capacity.");
    }
    return activeRows.value();
}

void RaggedExpressionLayer::zeroInactiveTail(Tensor tensor, uint64_t activeRows, uint64_t rowWidth, Stream stream) const {
    THOR_THROW_IF_FALSE(activeRows <= fullCapacityRows);
    THOR_THROW_IF_FALSE(rowWidth > 0);
    if (activeRows == fullCapacityRows) {
        return;
    }
    Tensor tail =
        tensor.aliasView({fullCapacityRows - activeRows, rowWidth}, {rowWidth, 1}, activeRows * rowWidth);
    tail.memsetAsync(stream, 0);
}

void RaggedExpressionLayer::beforeForwardExpressionRun(uint32_t connectionNumber, Stream& stream) {
    (void)stream;
    const uint32_t applicationIndex = applicationIndexForConnection(connectionNumber);
    const uint64_t activeRows = requireActiveRows(applicationIndex);
    if (activeRowsByApplication.size() <= applicationIndex) {
        activeRowsByApplication.resize(applicationIndex + 1, 0);
    }
    activeRowsByApplication[applicationIndex] = activeRows;
}

void RaggedExpressionLayer::afterForwardExpressionRun(uint32_t connectionNumber, Stream& stream) {
    const uint32_t applicationIndex = applicationIndexForConnection(connectionNumber);
    if (applicationIndex >= activeRowsByApplication.size()) {
        throw std::runtime_error("RaggedExpressionLayer has no active-row state for this application.");
    }
    const uint32_t outputFlatIndex = applicationIndex * outputPortCount;
    if (outputFlatIndex >= featureOutputs.size() || !featureOutputs[outputFlatIndex].has_value()) {
        throw std::runtime_error("RaggedExpressionLayer output is not connected for this application.");
    }
    Tensor output = featureOutputs[outputFlatIndex].value();
    const uint64_t activeRows = activeRowsByApplication[applicationIndex];
    zeroInactiveTail(output, activeRows, outputElementsPerValue, stream);
    output.setRaggedActiveRows(activeRows);
}

void RaggedExpressionLayer::afterBackwardErrorExpressionRun(uint32_t connectionNumber, Stream& stream) {
    const uint32_t applicationIndex = applicationIndexForConnection(connectionNumber);
    if (applicationIndex >= activeRowsByApplication.size()) {
        return;
    }
    const uint32_t valuesFlatIndex = applicationIndex * inputPortCount + valuesInputPort;
    if (valuesFlatIndex >= errorOutputs.size() || !errorOutputs[valuesFlatIndex].has_value()) {
        return;
    }
    Tensor dValues = errorOutputs[valuesFlatIndex].value();
    const uint64_t activeRows = activeRowsByApplication[applicationIndex];
    zeroInactiveTail(dValues, activeRows, inputElementsPerValue, stream);
    dValues.setRaggedActiveRows(activeRows);
}

}  // namespace ThorImplementation
