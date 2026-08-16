#include "DeepLearning/Implementation/Layers/NeuralNetwork/RaggedFullyConnected.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"

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
    : CustomLayer(std::move(expression),
                  std::vector<std::string>{"feature_input", ROW_PARTITION_INPUT_NAME},
                  std::vector<std::string>{"feature_output"},
                  placement,
                  physicalParameters,
                  inferenceOnly,
                  stampedId,
                  {},
                  false,
                  false,
                  std::vector<bool>{false, true},
                  std::nullopt),
      inputFeatures(inputFeatures),
      outputFeatures(outputFeatures),
      fullCapacityRows(fullCapacityRows),
      inputDataType(inputDataType),
      outputDataType(outputDataType) {
    if (inputFeatures == 0 || outputFeatures == 0 || fullCapacityRows == 0) {
        throw std::invalid_argument("RaggedFullyConnected dimensions must be non-zero.");
    }
}

uint32_t RaggedFullyConnected::applicationIndexForConnection(uint32_t connectionNumber) const {
    return connectionNumber / INPUT_PORT_COUNT;
}

Tensor RaggedFullyConnected::packedValuesForApplication(uint32_t applicationIndex) const {
    const uint32_t valuesFlatIndex = applicationIndex * INPUT_PORT_COUNT + VALUES_INPUT_PORT;
    if (valuesFlatIndex >= featureInputs.size() || !featureInputs[valuesFlatIndex].has_value()) {
        throw std::runtime_error("RaggedFullyConnected packed values input is not connected for this application.");
    }
    return featureInputs[valuesFlatIndex].value();
}

uint64_t RaggedFullyConnected::requireActiveRows(uint32_t applicationIndex) const {
    const uint32_t offsetsFlatIndex = applicationIndex * INPUT_PORT_COUNT + ROW_PARTITION_INPUT_PORT;
    if (offsetsFlatIndex >= featureInputs.size() || !featureInputs[offsetsFlatIndex].has_value()) {
        throw std::runtime_error("RaggedFullyConnected row-partition offsets input is not connected for this application.");
    }

    const Tensor offsets = featureInputs[offsetsFlatIndex].value();
    const TensorDescriptor offsetsDescriptor = offsets.getDescriptor();
    if (offsetsDescriptor.getNumDimensions() != 1 || offsetsDescriptor.getDimensions()[0] == 0 ||
        !RowPartitionDescriptor::isValidOffsetsDataType(offsetsDescriptor.getDataType())) {
        throw std::runtime_error("RaggedFullyConnected row-partition offsets input is not canonical.");
    }

    const uint64_t batchSize = offsetsDescriptor.getDimensions()[0] - 1;
    RowPartitionRuntime rowPartition(
        offsets, RowPartitionDescriptor(batchSize, fullCapacityRows, offsetsDescriptor.getDataType()));
    const uint64_t activeRows = rowPartition.requireHostActiveValueCount();
    if (activeRows > fullCapacityRows) {
        throw std::runtime_error("RaggedFullyConnected active row count exceeds fullCapacityRows.");
    }
    return activeRows;
}

void RaggedFullyConnected::zeroRows(Tensor tensor, uint64_t firstRow, uint64_t endRow, uint64_t rowWidth, Stream stream) const {
    THOR_THROW_IF_FALSE(firstRow <= endRow);
    THOR_THROW_IF_FALSE(endRow <= fullCapacityRows);
    if (firstRow == endRow)
        return;
    Tensor tail = tensor.aliasView({endRow - firstRow, rowWidth}, {rowWidth, 1}, firstRow * rowWidth);
    tail.memsetAsync(stream, 0);
}

uint64_t RaggedFullyConnected::activeRowsForApplication(uint32_t applicationIndex) const {
    if (applicationIndex >= activeRowsByApplication.size()) {
        throw std::runtime_error("RaggedFullyConnected has no active-row state for this application.");
    }
    return activeRowsByApplication[applicationIndex];
}

void RaggedFullyConnected::beforeForwardExpressionRun(uint32_t connectionNumber, Stream& stream) {
    const uint32_t applicationIndex = applicationIndexForConnection(connectionNumber);
    Tensor input = packedValuesForApplication(applicationIndex);
    if (input.getDimensions() != std::vector<uint64_t>{fullCapacityRows, inputFeatures} || input.getDataType() != inputDataType) {
        throw std::runtime_error("RaggedFullyConnected packed input does not match the configured [capacity, features] tensor.");
    }

    const uint64_t activeRows = requireActiveRows(applicationIndex);
    if (activeRowsByApplication.size() <= applicationIndex) {
        activeRowsByApplication.resize(applicationIndex + 1, 0);
    }
    activeRowsByApplication[applicationIndex] = activeRows;

    // Canonical zero padding is part of the packed-ragged contract. It makes the
    // capacity bucket safe for dW and for full-capacity fused valuewise tails.
    zeroRows(input, activeRows, fullCapacityRows, inputFeatures, stream);

}

void RaggedFullyConnected::afterForwardExpressionRun(uint32_t connectionNumber, Stream& stream) {
    const uint32_t applicationIndex = applicationIndexForConnection(connectionNumber);
    THOR_THROW_IF_FALSE(applicationIndex < featureOutputs.size());
    THOR_THROW_IF_FALSE(featureOutputs[applicationIndex].has_value());
    Tensor output = featureOutputs[applicationIndex].value();
    if (output.getDimensions() != std::vector<uint64_t>{fullCapacityRows, outputFeatures} || output.getDataType() != outputDataType) {
        throw std::runtime_error("RaggedFullyConnected expression output does not match [capacity, outputFeatures].");
    }
    const uint64_t activeRows = activeRowsForApplication(applicationIndex);
    zeroRows(output, activeRows, fullCapacityRows, outputFeatures, stream);
}

void RaggedFullyConnected::backward(std::optional<Tensor> errorInput, uint32_t batchSize) {
    if (errorInput.has_value()) {
        uint32_t applicationIndex = 0;
        for (; applicationIndex < errorInputs.size(); ++applicationIndex) {
            if (errorInputs[applicationIndex].has_value() && errorInputs[applicationIndex].value() == errorInput.value())
                break;
        }
        if (applicationIndex == errorInputs.size()) {
            throw std::runtime_error("RaggedFullyConnected backward received an unknown error tensor.");
        }
        const uint64_t activeRows = activeRowsForApplication(applicationIndex);
        const uint32_t streamFlatIndex = applicationIndex * INPUT_PORT_COUNT + VALUES_INPUT_PORT;
        THOR_THROW_IF_FALSE(streamFlatIndex < streams.size());
        zeroRows(errorInput.value(), activeRows, fullCapacityRows, outputFeatures, streams[streamFlatIndex]);
    }
    CustomLayer::backward(errorInput, batchSize);
}

void RaggedFullyConnected::afterBackwardErrorExpressionRun(uint32_t connectionNumber, Stream& stream) {
    const uint32_t applicationIndex = applicationIndexForConnection(connectionNumber);
    const uint32_t valuesFlatIndex = applicationIndex * INPUT_PORT_COUNT + VALUES_INPUT_PORT;
    if (valuesFlatIndex >= errorOutputs.size() || !errorOutputs[valuesFlatIndex].has_value())
        return;
    Tensor dX = errorOutputs[valuesFlatIndex].value();
    const uint64_t activeRows = activeRowsForApplication(applicationIndex);
    zeroRows(dX, activeRows, fullCapacityRows, inputFeatures, stream);
}

uint64_t RaggedFullyConnected::selectedCapacityRows(uint64_t activeRows) const {
    return chooseRaggedMatmulCapacityBucket(activeRows, makeRaggedMatmulCapacityBuckets(fullCapacityRows));
}

}  // namespace ThorImplementation
