#include "Utilities/TensorOperations/Ragged/PaddedRaggedSequence.h"

#include "Utilities/TensorOperations/Ragged/PaddedRaggedSequenceKernel.h"
#include "Utilities/TensorOperations/Ragged/RowPartitionDTypePolicy.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ThorImplementation {
namespace {

uint64_t checkedMul(uint64_t a, uint64_t b, const char* label) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        throw std::invalid_argument(std::string(label) + " overflows uint64_t.");
    }
    return a * b;
}

uint64_t elementSizeBytes(DataType dtype) {
    const float bytes = TensorDescriptor::getElementSizeInBytes(dtype);
    const uint64_t rounded = static_cast<uint64_t>(bytes);
    if (rounded == 0 || static_cast<float>(rounded) != bytes) {
        throw std::invalid_argument("Padded ragged sequence requires a whole-byte value dtype.");
    }
    return rounded;
}

std::vector<uint64_t> denseStrides(const std::vector<uint64_t>& dims) {
    std::vector<uint64_t> strides(dims.size(), 1);
    for (size_t i = dims.size(); i-- > 1;) {
        strides[i - 1] = checkedMul(strides[i], dims[i], "padded ragged dense stride");
    }
    return strides;
}

}  // namespace

uint64_t PaddedRaggedSequencePlan::logicalValueElements() const {
    return checkedMul(activeValues, channels, "padded ragged logical value elements");
}

uint64_t PaddedRaggedSequencePlan::denseCapacityValues() const {
    return checkedMul(batchSize, widthCapacity, "padded ragged dense capacity values");
}

uint64_t PaddedRaggedSequencePlan::paddingValueCapacity() const {
    const uint64_t capacity = denseCapacityValues();
    if (capacity < activeValues) {
        throw std::logic_error("PaddedRaggedSequencePlan capacity is smaller than its active population.");
    }
    return capacity - activeValues;
}

PaddedRaggedSequencePlan preparePaddedRaggedSequencePlan(const RowPartitionRuntime& rowPartition,
                                                         uint64_t channels,
                                                         DataType valuesDataType,
                                                         uint64_t widthCapacity) {
    if (!rowPartition.isInitialized()) {
        throw std::invalid_argument("preparePaddedRaggedSequencePlan requires an initialized row partition.");
    }
    if (!rowPartition.hasMaxValuesPerRow()) {
        throw std::invalid_argument("Padded ragged sequence requires max_values_per_row structural metadata.");
    }
    if (channels == 0) {
        throw std::invalid_argument("Padded ragged sequence requires channels > 0.");
    }
    if (!isCanonicalRowPartitionOffsetDataType(rowPartition.getOffsetsDataType())) {
        throw std::invalid_argument("Padded ragged sequence offsets must use UINT32 or UINT64.");
    }

    const uint64_t activeValues = rowPartition.requireHostActiveValueCount();
    const uint64_t maxActiveRowLength = rowPartition.requireHostMaxActiveRowLength();
    const uint64_t maxValuesPerRow = rowPartition.getMaxValuesPerRow();
    if (widthCapacity < maxActiveRowLength) {
        throw std::invalid_argument("Padded ragged width capacity is smaller than max_active_row_length.");
    }
    const uint64_t denseCapacityValues = checkedMul(rowPartition.getBatchSize(), widthCapacity,
                                                    "padded ragged B*W capacity");
    if (activeValues > denseCapacityValues) {
        throw std::invalid_argument(
            "Padded ragged host extent metadata is inconsistent: active_value_count exceeds B*width_capacity.");
    }
    PaddedRaggedSequencePlan plan;
    plan.valuesDataType = valuesDataType;
    plan.offsetsDataType = rowPartition.getOffsetsDataType();
    plan.batchSize = rowPartition.getBatchSize();
    plan.maxTotalValues = rowPartition.getMaxTotalValues();
    plan.maxValuesPerRow = maxValuesPerRow;
    plan.channels = channels;
    plan.activeValues = activeValues;
    plan.widthCapacity = widthCapacity;
    plan.valueElements = checkedMul(
        checkedMul(plan.batchSize, plan.channels, "padded ragged B*C"), widthCapacity, "padded ragged B*C*W");
    plan.valueBytes = checkedMul(plan.valueElements, elementSizeBytes(valuesDataType), "padded ragged value bytes");
    return plan;
}

PaddedRaggedSequence::PaddedRaggedSequence(PaddedRaggedSequencePlan plan,
                                           Tensor rowOffsets,
                                           TensorPlacement placement,
                                           uint64_t reservedWidthCapacity)
    : plan(std::move(plan)), rowOffsets(std::move(rowOffsets)), reservedWidthCapacity(reservedWidthCapacity) {
    if (!this->rowOffsets.isInitialized()) {
        throw std::invalid_argument("PaddedRaggedSequence requires initialized row offsets.");
    }
    if (placement.getMemDevice() != TensorPlacement::MemDevices::GPU || this->rowOffsets.getPlacement() != placement) {
        throw std::invalid_argument("PaddedRaggedSequence requires offsets and storage on one GPU.");
    }
    if (this->rowOffsets.getDataType() != this->plan.offsetsDataType ||
        this->rowOffsets.getDimensions() != std::vector<uint64_t>({this->plan.batchSize + 1})) {
        throw std::invalid_argument("PaddedRaggedSequence row offsets do not match the prepared plan.");
    }
    if (reservedWidthCapacity < this->plan.widthCapacity) {
        throw std::invalid_argument("PaddedRaggedSequence reserved width is smaller than the prepared plan.");
    }

    const uint64_t allocationElements = checkedMul(
        checkedMul(this->plan.batchSize, this->plan.channels, "padded ragged reserved B*C"),
        reservedWidthCapacity,
        "padded ragged reserved B*C*W");
    if (allocationElements > 0) {
        paddedValues = Tensor(placement, TensorDescriptor(this->plan.valuesDataType, {allocationElements}));
    }
}

void PaddedRaggedSequence::reconfigure(PaddedRaggedSequencePlan newPlan) {
    if (newPlan.valuesDataType != plan.valuesDataType || newPlan.offsetsDataType != plan.offsetsDataType ||
        newPlan.batchSize != plan.batchSize || newPlan.maxTotalValues != plan.maxTotalValues ||
        newPlan.maxValuesPerRow != plan.maxValuesPerRow || newPlan.channels != plan.channels) {
        throw std::invalid_argument("PaddedRaggedSequence::reconfigure cannot change the static representation contract.");
    }
    if (newPlan.widthCapacity > reservedWidthCapacity) {
        throw std::invalid_argument("PaddedRaggedSequence::reconfigure width exceeds reserved storage.");
    }
    plan = std::move(newPlan);
}

Tensor PaddedRaggedSequence::paddedTensor() const { return paddedTensorForWidth(plan.widthCapacity); }

Tensor PaddedRaggedSequence::paddedTensorForWidth(uint64_t widthCapacity) const {
    if (widthCapacity == 0) {
        throw std::runtime_error("PaddedRaggedSequence has no dense tensor for an all-empty width-0 batch.");
    }
    if (widthCapacity > reservedWidthCapacity) {
        throw std::invalid_argument("PaddedRaggedSequence requested view width exceeds reserved storage.");
    }
    if (!paddedValues.isInitialized()) {
        throw std::runtime_error("PaddedRaggedSequence value storage is not initialized.");
    }
    const std::vector<uint64_t> dims{plan.batchSize, plan.channels, 1, widthCapacity};
    return paddedValues.aliasView(dims, denseStrides(dims), 0);
}

void PaddedRaggedSequence::packFrom(const Tensor& packedValues, Stream& stream) {
    if (packedValues.getPlacement() != rowOffsets.getPlacement() || packedValues.getDataType() != plan.valuesDataType ||
        packedValues.getDimensions() != std::vector<uint64_t>({plan.maxTotalValues, plan.channels})) {
        throw std::invalid_argument("PaddedRaggedSequence::packFrom packed values do not match the prepared plan.");
    }
    if (stream.getGpuNum() != packedValues.getPlacement().getDeviceNum()) {
        throw std::invalid_argument("PaddedRaggedSequence::packFrom stream GPU does not match tensor placement.");
    }
    if (plan.widthCapacity == 0) {
        return;
    }
    launchPackedToPaddedRaggedSequence(
        packedValues, rowOffsets, paddedValues, plan.batchSize, plan.channels, plan.widthCapacity, stream);
}

void PaddedRaggedSequence::sanitizedCopyFrom(const PaddedRaggedSequence& source, Stream& stream) {
    if (source.getRowOffsets() != rowOffsets) {
        throw std::invalid_argument("PaddedRaggedSequence::sanitizedCopyFrom requires the same canonical offsets tensor.");
    }
    if (source.getPlan() != plan) {
        throw std::invalid_argument("PaddedRaggedSequence::sanitizedCopyFrom requires identical selected representation plans.");
    }
    if (stream.getGpuNum() != rowOffsets.getPlacement().getDeviceNum()) {
        throw std::invalid_argument("PaddedRaggedSequence::sanitizedCopyFrom stream GPU does not match tensor placement.");
    }
    if (plan.widthCapacity == 0) {
        return;
    }
    Tensor sourceStorage = source.getPaddedValuesStorage();
    launchSanitizedPaddedRaggedSequenceCopy(
        sourceStorage, rowOffsets, paddedValues, plan.batchSize, plan.channels, plan.widthCapacity, stream);
}

void PaddedRaggedSequence::unpackTo(Tensor& packedValues, Stream& stream) const {
    if (packedValues.getPlacement() != rowOffsets.getPlacement() || packedValues.getDataType() != plan.valuesDataType ||
        packedValues.getDimensions() != std::vector<uint64_t>({plan.maxTotalValues, plan.channels})) {
        throw std::invalid_argument("PaddedRaggedSequence::unpackTo packed values do not match the prepared plan.");
    }
    if (stream.getGpuNum() != packedValues.getPlacement().getDeviceNum()) {
        throw std::invalid_argument("PaddedRaggedSequence::unpackTo stream GPU does not match tensor placement.");
    }
    if (plan.widthCapacity == 0) {
        return;
    }
    launchPaddedToPackedRaggedSequence(
        paddedValues, rowOffsets, packedValues, plan.batchSize, plan.channels, plan.widthCapacity, stream);
}

}  // namespace ThorImplementation
