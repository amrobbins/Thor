#pragma once

#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionDescriptor.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/Stream.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace ThorTest {

// A stamp fills its managed row-partition carriers from the batch metadata ring, in one launch, before the carrier
// inputs run. A test that drives a physical carrier input directly writes the carrier's bytes itself (the [B+1]
// offsets, or the [1] active count: the last offset), on the input's stream, then forwards it.
inline void forwardManagedRowPartitionCarrierForTest(
    ThorImplementation::NetworkInput &input,
    bool validationPass,
    const ThorImplementation::RowPartitionDescriptor &descriptor,
    uint32_t batchSize,
    const std::vector<uint64_t> &hostOffsets,
    std::optional<ThorImplementation::RowPartitionId> logicalRowPartitionId = std::nullopt) {
    using ThorImplementation::DataType;
    THOR_THROW_IF_FALSE(input.getFeatureOutput().has_value());
    THOR_THROW_IF_FALSE(!hostOffsets.empty());
    ThorImplementation::Tensor carrier = input.getFeatureOutput().value();
    const uint64_t count = carrier.getTotalNumElements();
    THOR_THROW_IF_FALSE(count == 1 || count == hostOffsets.size());
    const uint64_t first = hostOffsets.size() - count;
    ThorImplementation::Tensor host(ThorImplementation::TensorPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU),
                                    carrier.getDescriptor());
    for (uint64_t i = 0; i < count; ++i) {
        if (carrier.getDataType() == DataType::UINT32)
            host.getMemPtr<uint32_t>()[i] = static_cast<uint32_t>(hostOffsets[first + i]);
        else
            host.getMemPtr<uint64_t>()[i] = hostOffsets[first + i];
    }
    Stream stream = input.getStream();
    carrier.copyFromAsync(host, stream);
    stream.synchronize();
    input.forwardManagedRowPartitionCarrier(validationPass, descriptor, batchSize, hostOffsets, logicalRowPartitionId);
}

}  // namespace ThorTest
