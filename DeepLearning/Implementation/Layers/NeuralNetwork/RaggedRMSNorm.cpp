#include "DeepLearning/Implementation/Layers/NeuralNetwork/RaggedRMSNorm.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/RaggedMatmulCapacityBuckets.h"

#include <stdexcept>
#include <utility>

namespace ThorImplementation {

RaggedRMSNorm::RaggedRMSNorm(DynamicExpression expression,
                             std::vector<std::string> inputNames,
                             std::vector<std::string> outputNames,
                             TensorPlacement placement,
                             std::vector<std::shared_ptr<PhysicalParameter>> physicalParameters,
                             uint64_t fullCapacityRows,
                             uint64_t elementsPerValue,
                             bool inferenceOnly,
                             int64_t stampedId)
    : CustomLayer(std::move(expression),
                  std::move(inputNames),
                  std::move(outputNames),
                  placement,
                  std::move(physicalParameters),
                  inferenceOnly,
                  stampedId),
      fullCapacityRows(fullCapacityRows),
      elementsPerValue(elementsPerValue) {
    if (fullCapacityRows == 0 || elementsPerValue == 0) {
        throw std::invalid_argument("RaggedRMSNorm dimensions must be non-zero.");
    }
}

uint64_t RaggedRMSNorm::requireActiveRows(uint32_t connectionNumber) const {
    if (connectionNumber >= featureInputs.size() || !featureInputs[connectionNumber].has_value()) {
        throw std::runtime_error("RaggedRMSNorm packed values input is not connected for this application.");
    }
    const Tensor values = featureInputs[connectionNumber].value();
    validatePackedTensor(values, "feature input");
    const std::optional<uint64_t> activeRows = values.getRaggedActiveRows();
    if (!activeRows.has_value()) {
        throw std::runtime_error("RaggedRMSNorm requires host-known ragged active-row metadata on packed values.");
    }
    if (activeRows.value() > fullCapacityRows) {
        throw std::runtime_error("RaggedRMSNorm active row count exceeds its packed capacity.");
    }
    return activeRows.value();
}

void RaggedRMSNorm::validatePackedTensor(const Tensor& tensor, const char* what) const {
    const std::vector<uint64_t> dims = tensor.getDimensions();
    const uint64_t totalElements = tensor.getTotalNumElements();
    if (dims.empty() || dims[0] != fullCapacityRows || totalElements % fullCapacityRows != 0 ||
        totalElements / fullCapacityRows != elementsPerValue) {
        throw std::runtime_error(std::string("RaggedRMSNorm ") + what +
                                 " does not match its configured packed [capacity, trailing-value-domain] geometry.");
    }
}

uint64_t RaggedRMSNorm::activeRowsForConnection(uint32_t connectionNumber) const {
    if (connectionNumber >= activeRowsByConnection.size()) {
        throw std::runtime_error("RaggedRMSNorm has no active-row state for this application.");
    }
    return activeRowsByConnection[connectionNumber];
}

void RaggedRMSNorm::zeroInactiveTail(Tensor tensor, uint64_t activeRows, Stream stream) const {
    THOR_THROW_IF_FALSE(activeRows <= fullCapacityRows);
    if (activeRows == fullCapacityRows)
        return;
    Tensor tail = tensor.aliasView({fullCapacityRows - activeRows, elementsPerValue},
                                   {elementsPerValue, 1},
                                   activeRows * elementsPerValue);
    tail.memsetAsync(stream, 0);
}

void RaggedRMSNorm::beforeForwardExpressionRun(uint32_t connectionNumber, Stream& stream) {
    const uint64_t activeRows = requireActiveRows(connectionNumber);
    if (activeRowsByConnection.size() <= connectionNumber) {
        activeRowsByConnection.resize(connectionNumber + 1, 0);
    }
    activeRowsByConnection[connectionNumber] = activeRows;

    // The selected cuDNN bucket may extend beyond the logical active prefix, and
    // RMSNorm autodiff later performs full-capacity reductions for dscale. Keep
    // every invalid packed row canonical so neither path can observe padding.
    zeroInactiveTail(featureInputs[connectionNumber].value(), activeRows, stream);
}

void RaggedRMSNorm::afterForwardExpressionRun(uint32_t connectionNumber, Stream& stream) {
    THOR_THROW_IF_FALSE(connectionNumber < featureOutputs.size());
    THOR_THROW_IF_FALSE(featureOutputs[connectionNumber].has_value());
    Tensor output = featureOutputs[connectionNumber].value();
    validatePackedTensor(output, "feature output");
    const uint64_t activeRows = activeRowsForConnection(connectionNumber);
    zeroInactiveTail(output, activeRows, stream);
    output.setRaggedActiveRows(activeRows);
}

void RaggedRMSNorm::backward(std::optional<Tensor> errorInput, uint32_t batchSize) {
    if (errorInput.has_value()) {
        uint32_t connectionNumber = 0;
        for (; connectionNumber < errorInputs.size(); ++connectionNumber) {
            if (errorInputs[connectionNumber].has_value() && errorInputs[connectionNumber].value() == errorInput.value())
                break;
        }
        if (connectionNumber == errorInputs.size()) {
            throw std::runtime_error("RaggedRMSNorm backward received an unknown error tensor.");
        }
        const uint64_t activeRows = activeRowsForConnection(connectionNumber);
        validatePackedTensor(errorInput.value(), "incoming gradient");
        zeroInactiveTail(errorInput.value(), activeRows, streams[connectionNumber]);
        Tensor annotatedError = errorInput.value();
        annotatedError.setRaggedActiveRows(activeRows);
    }
    CustomLayer::backward(errorInput, batchSize);
}

void RaggedRMSNorm::afterBackwardErrorExpressionRun(uint32_t connectionNumber, Stream& stream) {
    if (connectionNumber >= errorOutputs.size() || !errorOutputs[connectionNumber].has_value())
        return;
    Tensor dValues = errorOutputs[connectionNumber].value();
    validatePackedTensor(dValues, "input gradient");
    const uint64_t activeRows = activeRowsForConnection(connectionNumber);
    zeroInactiveTail(dValues, activeRows, stream);
    dValues.setRaggedActiveRows(activeRows);
}

uint64_t RaggedRMSNorm::selectedCapacityRows(uint64_t activeRows) const {
    const std::vector<uint64_t> buckets = makeRaggedMatmulCapacityBuckets(fullCapacityRows);
    if (activeRows == 0)
        return buckets.front();
    return chooseRaggedMatmulCapacityBucket(activeRows, buckets);
}

}  // namespace ThorImplementation
