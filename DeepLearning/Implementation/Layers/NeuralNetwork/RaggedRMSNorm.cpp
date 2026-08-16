#include "DeepLearning/Implementation/Layers/NeuralNetwork/RaggedRMSNorm.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
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
                  stampedId,
                  {},
                  false,
                  false,
                  std::vector<bool>{false, true},
                  std::nullopt),
      fullCapacityRows(fullCapacityRows),
      elementsPerValue(elementsPerValue) {
    if (fullCapacityRows == 0 || elementsPerValue == 0) {
        throw std::invalid_argument("RaggedRMSNorm dimensions must be non-zero.");
    }
}

uint32_t RaggedRMSNorm::applicationIndexForConnection(uint32_t connectionNumber) const {
    return connectionNumber / INPUT_PORT_COUNT;
}

Tensor RaggedRMSNorm::packedValuesForApplication(uint32_t applicationIndex) const {
    const uint32_t valuesFlatIndex = applicationIndex * INPUT_PORT_COUNT + VALUES_INPUT_PORT;
    if (valuesFlatIndex >= featureInputs.size() || !featureInputs[valuesFlatIndex].has_value()) {
        throw std::runtime_error("RaggedRMSNorm packed values input is not connected for this application.");
    }
    return featureInputs[valuesFlatIndex].value();
}

uint64_t RaggedRMSNorm::requireActiveRows(uint32_t applicationIndex) const {
    const uint32_t offsetsFlatIndex = applicationIndex * INPUT_PORT_COUNT + ROW_PARTITION_INPUT_PORT;
    if (offsetsFlatIndex >= featureInputs.size() || !featureInputs[offsetsFlatIndex].has_value()) {
        throw std::runtime_error("RaggedRMSNorm row-partition offsets input is not connected for this application.");
    }

    const Tensor offsets = featureInputs[offsetsFlatIndex].value();
    const TensorDescriptor offsetsDescriptor = offsets.getDescriptor();
    if (offsetsDescriptor.getNumDimensions() != 1 || offsetsDescriptor.getDimensions()[0] == 0 ||
        !RowPartitionDescriptor::isValidOffsetsDataType(offsetsDescriptor.getDataType())) {
        throw std::runtime_error("RaggedRMSNorm row-partition offsets input is not canonical.");
    }

    const uint64_t batchSize = offsetsDescriptor.getDimensions()[0] - 1;
    RowPartitionRuntime rowPartition(
        offsets, RowPartitionDescriptor(batchSize, fullCapacityRows, offsetsDescriptor.getDataType()));
    const uint64_t activeRows = rowPartition.requireHostActiveValueCount();
    if (activeRows > fullCapacityRows) {
        throw std::runtime_error("RaggedRMSNorm active row count exceeds its packed capacity.");
    }
    return activeRows;
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

uint64_t RaggedRMSNorm::activeRowsForApplication(uint32_t applicationIndex) const {
    if (applicationIndex >= activeRowsByApplication.size()) {
        throw std::runtime_error("RaggedRMSNorm has no active-row state for this application.");
    }
    return activeRowsByApplication[applicationIndex];
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
    const uint32_t applicationIndex = applicationIndexForConnection(connectionNumber);
    Tensor input = packedValuesForApplication(applicationIndex);
    validatePackedTensor(input, "feature input");
    const uint64_t activeRows = requireActiveRows(applicationIndex);

    if (activeRowsByApplication.size() <= applicationIndex) {
        activeRowsByApplication.resize(applicationIndex + 1, 0);
    }
    activeRowsByApplication[applicationIndex] = activeRows;

    // The selected cuDNN bucket may extend beyond the logical active prefix, and
    // RMSNorm autodiff later performs full-capacity reductions for dscale. Keep
    // every invalid packed row canonical so neither path can observe padding.
    zeroInactiveTail(input, activeRows, stream);

}

void RaggedRMSNorm::afterForwardExpressionRun(uint32_t connectionNumber, Stream& stream) {
    const uint32_t applicationIndex = applicationIndexForConnection(connectionNumber);
    THOR_THROW_IF_FALSE(applicationIndex < featureOutputs.size());
    THOR_THROW_IF_FALSE(featureOutputs[applicationIndex].has_value());
    Tensor output = featureOutputs[applicationIndex].value();
    validatePackedTensor(output, "feature output");
    const uint64_t activeRows = activeRowsForApplication(applicationIndex);
    zeroInactiveTail(output, activeRows, stream);
}

void RaggedRMSNorm::backward(std::optional<Tensor> errorInput, uint32_t batchSize) {
    if (!errorInput.has_value()) {
        CustomLayer::backward(errorInput, batchSize);
        return;
    }

    uint32_t applicationIndex = 0;
    for (; applicationIndex < errorInputs.size(); ++applicationIndex) {
        if (errorInputs[applicationIndex].has_value() && errorInputs[applicationIndex].value() == errorInput.value())
            break;
    }
    if (applicationIndex == errorInputs.size()) {
        throw std::runtime_error("RaggedRMSNorm backward received an unknown error tensor.");
    }

    const uint64_t activeRows = activeRowsForApplication(applicationIndex);
    validatePackedTensor(errorInput.value(), "incoming gradient");
    const uint32_t streamFlatIndex = applicationIndex * INPUT_PORT_COUNT + VALUES_INPUT_PORT;
    THOR_THROW_IF_FALSE(streamFlatIndex < streams.size());
    zeroInactiveTail(errorInput.value(), activeRows, streams[streamFlatIndex]);

    CustomLayer::backward(errorInput, batchSize);
}

void RaggedRMSNorm::afterBackwardErrorExpressionRun(uint32_t connectionNumber, Stream& stream) {
    const uint32_t applicationIndex = applicationIndexForConnection(connectionNumber);
    const uint32_t valuesFlatIndex = applicationIndex * INPUT_PORT_COUNT + VALUES_INPUT_PORT;
    if (valuesFlatIndex >= errorOutputs.size() || !errorOutputs[valuesFlatIndex].has_value())
        return;
    Tensor dValues = errorOutputs[valuesFlatIndex].value();
    validatePackedTensor(dValues, "input gradient");
    const uint64_t activeRows = activeRowsForApplication(applicationIndex);
    zeroInactiveTail(dValues, activeRows, stream);
}

uint64_t RaggedRMSNorm::selectedCapacityRows(uint64_t activeRows) const {
    const std::vector<uint64_t> buckets = makeRaggedMatmulCapacityBuckets(fullCapacityRows);
    if (activeRows == 0)
        return buckets.front();
    return chooseRaggedMatmulCapacityBucket(activeRows, buckets);
}

}  // namespace ThorImplementation
