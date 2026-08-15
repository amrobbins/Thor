#include "DeepLearning/Implementation/Layers/NeuralNetwork/RaggedFullyConnected.h"

#include "DeepLearning/Implementation/ThorError.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace ThorImplementation {

RaggedFullyConnected::RaggedFullyConnected(DynamicExpression expression,
                                           TensorPlacement placement,
                                           std::vector<std::shared_ptr<PhysicalParameter>> physicalParameters,
                                           uint64_t inputFeatures,
                                           uint64_t outputFeatures,
                                           uint64_t fullCapacityRows,
                                           DataType inputDataType,
                                           DataType outputDataType,
                                           bool inferenceOnly,
                                           int64_t stampedId)
    : CustomLayer(std::move(expression), placement, physicalParameters, inferenceOnly, stampedId),
      inputFeatures(inputFeatures),
      outputFeatures(outputFeatures),
      fullCapacityRows(fullCapacityRows),
      inputDataType(inputDataType),
      outputDataType(outputDataType) {
    if (inputFeatures == 0 || outputFeatures == 0 || fullCapacityRows == 0) {
        throw std::invalid_argument("RaggedFullyConnected dimensions must be non-zero.");
    }
}

uint64_t RaggedFullyConnected::requireActiveRows(const Tensor& packedValues) const {
    const std::optional<uint64_t> activeRows = packedValues.getRaggedActiveRows();
    if (!activeRows.has_value()) {
        throw std::runtime_error("RaggedFullyConnected requires the packed values tensor to carry host-known ragged active-row metadata.");
    }
    if (activeRows.value() > fullCapacityRows) {
        throw std::runtime_error("RaggedFullyConnected active row count exceeds fullCapacityRows.");
    }
    return activeRows.value();
}

void RaggedFullyConnected::zeroRows(Tensor tensor, uint64_t firstRow, uint64_t endRow, uint64_t rowWidth, Stream stream) const {
    THOR_THROW_IF_FALSE(firstRow <= endRow);
    THOR_THROW_IF_FALSE(endRow <= fullCapacityRows);
    if (firstRow == endRow)
        return;
    Tensor tail = tensor.aliasView({endRow - firstRow, rowWidth}, {rowWidth, 1}, firstRow * rowWidth);
    tail.memsetAsync(stream, 0);
}

uint64_t RaggedFullyConnected::activeRowsForConnection(uint32_t connectionNumber) const {
    if (connectionNumber >= activeRowsByConnection.size()) {
        throw std::runtime_error("RaggedFullyConnected has no active-row state for this application.");
    }
    return activeRowsByConnection[connectionNumber];
}

void RaggedFullyConnected::beforeForwardExpressionRun(uint32_t connectionNumber, Stream& stream) {
    THOR_THROW_IF_FALSE(connectionNumber < featureInputs.size());
    THOR_THROW_IF_FALSE(featureInputs[connectionNumber].has_value());
    Tensor input = featureInputs[connectionNumber].value();
    if (input.getDimensions() != std::vector<uint64_t>{fullCapacityRows, inputFeatures} || input.getDataType() != inputDataType) {
        throw std::runtime_error("RaggedFullyConnected packed input does not match the configured [capacity, features] tensor.");
    }

    const uint64_t activeRows = requireActiveRows(input);
    if (activeRowsByConnection.size() <= connectionNumber) {
        activeRowsByConnection.resize(connectionNumber + 1, 0);
    }
    activeRowsByConnection[connectionNumber] = activeRows;

    // Canonical zero padding is part of the packed-ragged contract. It makes the
    // capacity bucket safe for dW and for full-capacity fused valuewise tails.
    zeroRows(input, activeRows, fullCapacityRows, inputFeatures, stream);
}

void RaggedFullyConnected::afterForwardExpressionRun(uint32_t connectionNumber, Stream& stream) {
    THOR_THROW_IF_FALSE(connectionNumber < featureOutputs.size());
    THOR_THROW_IF_FALSE(featureOutputs[connectionNumber].has_value());
    Tensor output = featureOutputs[connectionNumber].value();
    if (output.getDimensions() != std::vector<uint64_t>{fullCapacityRows, outputFeatures} || output.getDataType() != outputDataType) {
        throw std::runtime_error("RaggedFullyConnected expression output does not match [capacity, outputFeatures].");
    }
    const uint64_t activeRows = activeRowsForConnection(connectionNumber);
    zeroRows(output, activeRows, fullCapacityRows, outputFeatures, stream);
    output.setRaggedActiveRows(activeRows);
}

void RaggedFullyConnected::backward(std::optional<Tensor> errorInput, uint32_t batchSize) {
    if (errorInput.has_value()) {
        uint32_t connectionNumber = 0;
        for (; connectionNumber < errorInputs.size(); ++connectionNumber) {
            if (errorInputs[connectionNumber].has_value() && errorInputs[connectionNumber].value() == errorInput.value())
                break;
        }
        if (connectionNumber == errorInputs.size()) {
            throw std::runtime_error("RaggedFullyConnected backward received an unknown error tensor.");
        }
        const uint64_t activeRows = activeRowsForConnection(connectionNumber);
        zeroRows(errorInput.value(), activeRows, fullCapacityRows, outputFeatures, streams[connectionNumber]);
        Tensor annotatedError = errorInput.value();
        annotatedError.setRaggedActiveRows(activeRows);
    }
    CustomLayer::backward(errorInput, batchSize);
}

void RaggedFullyConnected::afterBackwardErrorExpressionRun(uint32_t connectionNumber, Stream& stream) {
    if (connectionNumber >= errorOutputs.size() || !errorOutputs[connectionNumber].has_value())
        return;
    Tensor dX = errorOutputs[connectionNumber].value();
    const uint64_t activeRows = activeRowsForConnection(connectionNumber);
    zeroRows(dX, activeRows, fullCapacityRows, inputFeatures, stream);
    dX.setRaggedActiveRows(activeRows);
}

uint64_t RaggedFullyConnected::selectedCapacityRows(uint64_t activeRows) const {
    return chooseRaggedMatmulCapacityBucket(activeRows, makeRaggedMatmulCapacityBuckets(fullCapacityRows));
}

}  // namespace ThorImplementation
