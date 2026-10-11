#pragma once

#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/TensorFanout.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace Thor::SegmentedPrimitivePlacementDetail {

inline ThorImplementation::TensorDescriptor physicalDescriptorForStamp(
    const std::shared_ptr<ThorImplementation::Layer>& drivingLayer,
    const char* layerName) {
    THOR_THROW_IF_FALSE(drivingLayer != nullptr);

    if (const auto networkInput = std::dynamic_pointer_cast<ThorImplementation::NetworkInput>(drivingLayer);
        networkInput != nullptr) {
        const std::optional<ThorImplementation::TensorDescriptor> descriptor =
            networkInput->getContentDescriptorForStamp();
        if (descriptor.has_value()) return descriptor.value();
    }

    const std::optional<ThorImplementation::Tensor> output = drivingLayer->getFeatureOutput();
    if (output.has_value()) return output->getDescriptor();

    throw std::runtime_error(std::string(layerName) +
                             " placement-neutral input requires an already-materialized physical descriptor.");
}

inline uint32_t placementBatchSizeForStamp(
    const Tensor& logicalDenseInput,
    const RaggedTensor& logicalPartitionInput,
    const std::shared_ptr<ThorImplementation::Layer>& drivingLayer,
    const Tensor& connectingApiTensor,
    const char* layerName) {
    THOR_THROW_IF_FALSE(logicalPartitionInput.hasMaxValuesPerRow() &&
                        logicalPartitionInput.getOffsetsDimensions() == std::vector<uint64_t>{1});
    const ThorImplementation::TensorDescriptor physicalInput =
        physicalDescriptorForStamp(drivingLayer, layerName);
    const std::vector<uint64_t> physicalDimensions = physicalInput.getDimensions();

    uint64_t placementBatchSize = 0;
    if (connectingApiTensor == logicalDenseInput) {
        const std::vector<uint64_t> logicalDimensions = logicalDenseInput.getDimensions();
        if (physicalInput.getDataType() != logicalDenseInput.getDataType() ||
            physicalDimensions.size() != logicalDimensions.size() + 1 || physicalDimensions.empty() ||
            !std::equal(logicalDimensions.begin(), logicalDimensions.end(), physicalDimensions.begin() + 1)) {
            throw std::runtime_error(std::string(layerName) +
                                     " dense driving descriptor does not match the logical dense input.");
        }
        placementBatchSize = physicalDimensions.front();
    } else if (connectingApiTensor == logicalPartitionInput.getOffsets()) {
        if (physicalInput.getDataType() != logicalPartitionInput.getOffsetsDataType() ||
            physicalDimensions.size() != 1 || physicalDimensions.front() < 2) {
            throw std::runtime_error(std::string(layerName) +
                                     " partition driving descriptor must be a [B+1] offsets vector.");
        }
        placementBatchSize = physicalDimensions.front() - 1;
    } else {
        throw std::runtime_error(std::string(layerName) +
                                 " stamp was driven by an input that is not part of the segmented broadcast contract.");
    }

    if (placementBatchSize == 0 || placementBatchSize > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error(std::string(layerName) +
                                 " placement batch size exceeds Thor's uint32 placement capacity.");
    }
    return static_cast<uint32_t>(placementBatchSize);
}

inline std::optional<ThorImplementation::RowPartitionDescriptor> rowPartitionDescriptorForStamp(
    const std::shared_ptr<ThorImplementation::Layer>& drivingLayer) {
    if (drivingLayer == nullptr) return std::nullopt;
    const auto fanout = std::dynamic_pointer_cast<ThorImplementation::TensorFanout>(drivingLayer);
    if (fanout == nullptr) return std::nullopt;
    return fanout->getRowPartitionDescriptorForStamp();
}

inline ThorImplementation::RaggedTensorDescriptor physicalRaggedDescriptorForStamp(
    const RaggedTensor& logicalInput,
    const std::shared_ptr<ThorImplementation::Layer>& drivingLayer,
    const char* layerName,
    bool drivingStructuralPartitionPort = false) {
    THOR_THROW_IF_FALSE(logicalInput.hasMaxValuesPerRow() &&
                        logicalInput.getOffsetsDimensions() == std::vector<uint64_t>{1});
    if (const std::optional<ThorImplementation::RowPartitionDescriptor> placedPartition =
            rowPartitionDescriptorForStamp(drivingLayer);
        placedPartition.has_value()) {
        const ThorImplementation::RowPartitionDescriptor& runtimePartition = placedPartition.value();
        const uint64_t placementBatchSize = runtimePartition.getBatchSize();
        if (placementBatchSize == 0) {
            throw std::runtime_error(std::string(layerName) +
                                     " managed row-partition metadata has zero placement batch size.");
        }
        if (runtimePartition.hasMaxValuesPerRow() &&
            runtimePartition.getMaxValuesPerRow() != logicalInput.getMaxValuesPerRow()) {
            throw std::runtime_error(std::string(layerName) +
                                     " managed row-partition metadata disagrees with semantic max_values_per_row.");
        }
        if (runtimePartition.getOffsetsDataType() != logicalInput.getOffsetsDataType()) {
            throw std::runtime_error(std::string(layerName) +
                                     " managed row-partition metadata disagrees with the logical offsets dtype.");
        }

        const ThorImplementation::RaggedTensorDescriptor descriptor =
            logicalInput.getDescriptorForPlacement(placementBatchSize);
        if (descriptor.getRowPartition() != runtimePartition) {
            throw std::runtime_error(std::string(layerName) +
                                     " managed row-partition metadata disagrees with the neutral placement descriptor.");
        }

        const ThorImplementation::TensorDescriptor carrier = physicalDescriptorForStamp(drivingLayer, layerName);
        const std::vector<uint64_t> carrierDimensions = carrier.getDimensions();
        const bool offsetsCarrier =
            carrier.getDataType() == runtimePartition.getOffsetsDataType() && carrierDimensions.size() == 1 &&
            (carrierDimensions.front() == 1 ||
             carrierDimensions == runtimePartition.getOffsetsDescriptor().getDimensions());
        // HOST_EXTENT is represented by the *canonical source values* carrier,
        // whose trailing dimensions and dtype may differ from this consumer's
        // values after intervening transforms (Flatten, Reshape, TypeConverter).
        // Only a structural-port stamp may use that carrier as the source of B.
        // An actual values-input stamp still requires its complete descriptor.
        const bool hostExtentCarrier =
            !carrierDimensions.empty() && carrierDimensions.front() == descriptor.getMaxTotalValues() &&
            (drivingStructuralPartitionPort || carrier == descriptor.getValuesDescriptor());
        if (!offsetsCarrier && !hostExtentCarrier) {
            throw std::runtime_error(std::string(layerName) +
                " managed row-partition carrier does not represent the placed partition or its packed values.");
        }
        return descriptor;
    }

    // Structural carriers must supply explicit stamp-time partition metadata.
    // Their leading dimension and dtype are not necessarily the consumer's
    // values geometry; guessing from that shape would couple the model to the
    // wrong physical tensor.
    if (drivingStructuralPartitionPort) {
        throw std::runtime_error(std::string(layerName) +
            " structural partition carrier is missing its runtime row-partition descriptor.");
    }
    const ThorImplementation::TensorDescriptor physicalValues =
        physicalDescriptorForStamp(drivingLayer, layerName);
    const std::vector<uint64_t> physicalDimensions = physicalValues.getDimensions();
    const std::vector<uint64_t> logicalTrailingDimensions = logicalInput.getTrailingDimensions();
    if (physicalValues.getDataType() != logicalInput.getValuesDataType() ||
        physicalDimensions.size() != logicalTrailingDimensions.size() + 1 || physicalDimensions.empty() ||
        !std::equal(logicalTrailingDimensions.begin(),
                    logicalTrailingDimensions.end(),
                    physicalDimensions.begin() + 1)) {
        throw std::runtime_error(std::string(layerName) +
                                 " placement-neutral physical values descriptor does not match the logical input.");
    }

    const uint64_t maxValuesPerRow = logicalInput.getMaxValuesPerRow();
    const uint64_t physicalCapacity = physicalDimensions.front();
    // 1B already chose physicalCapacity = placementBatch * M for this runtime
    // stamp. Recovering B here uses explicit semantic M plus runtime materialized
    // geometry; it does not infer M or persist either value back into the model.
    if (physicalCapacity == 0 || physicalCapacity % maxValuesPerRow != 0) {
        throw std::runtime_error(std::string(layerName) +
                                 " placement-neutral physical capacity is not an integral placement batch times max_values_per_row.");
    }
    const uint64_t placementBatchSize = physicalCapacity / maxValuesPerRow;
    const ThorImplementation::RaggedTensorDescriptor descriptor =
        logicalInput.getDescriptorForPlacement(placementBatchSize);
    if (descriptor.getValuesDescriptor() != physicalValues) {
        throw std::runtime_error(std::string(layerName) +
                                 " placement-neutral descriptor disagrees with the already-materialized physical values input.");
    }
    return descriptor;
}

}  // namespace Thor::SegmentedPrimitivePlacementDetail
