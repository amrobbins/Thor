#include "Utilities/TensorOperations/Ragged/PaddedRaggedSequenceKernel.h"

#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/Ragged/RowPartitionDTypePolicy.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ThorImplementation {
namespace {

template <typename ElementT, typename OffsetT>
__global__ void packedToPaddedRaggedSequenceKernel(const ElementT* packedValues,
                                                    const OffsetT* rowOffsets,
                                                    ElementT* paddedValues,
                                                    uint64_t batchSize,
                                                    uint64_t channels,
                                                    uint64_t widthCapacity) {
    const uint64_t totalElements = batchSize * channels * widthCapacity;
    for (uint64_t flat = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         flat < totalElements;
         flat += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        const uint64_t timestep = flat % widthCapacity;
        const uint64_t rowChannel = flat / widthCapacity;
        const uint64_t channel = rowChannel % channels;
        const uint64_t row = rowChannel / channels;
        const uint64_t begin = static_cast<uint64_t>(rowOffsets[row]);
        const uint64_t end = static_cast<uint64_t>(rowOffsets[row + 1]);
        const uint64_t rowLength = end >= begin ? end - begin : 0;

        if (timestep < rowLength) {
            const uint64_t packedIndex = (begin + timestep) * channels + channel;
            paddedValues[flat] = packedValues[packedIndex];
        } else {
            paddedValues[flat] = ElementT{};
        }
    }
}

template <typename ElementT, typename OffsetT>
__global__ void sanitizedPaddedRaggedSequenceCopyKernel(const ElementT* sourcePaddedValues,
                                                         const OffsetT* rowOffsets,
                                                         ElementT* destinationPaddedValues,
                                                         uint64_t batchSize,
                                                         uint64_t channels,
                                                         uint64_t widthCapacity) {
    const uint64_t totalElements = batchSize * channels * widthCapacity;
    for (uint64_t flat = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         flat < totalElements;
         flat += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        const uint64_t timestep = flat % widthCapacity;
        const uint64_t row = flat / (channels * widthCapacity);
        const uint64_t begin = static_cast<uint64_t>(rowOffsets[row]);
        const uint64_t end = static_cast<uint64_t>(rowOffsets[row + 1]);
        const uint64_t rowLength = end >= begin ? end - begin : 0;
        destinationPaddedValues[flat] = timestep < rowLength ? sourcePaddedValues[flat] : ElementT{};
    }
}

template <typename ElementT, typename OffsetT>
__global__ void paddedToPackedRaggedSequenceKernel(const ElementT* paddedValues,
                                                    const OffsetT* rowOffsets,
                                                    ElementT* packedValues,
                                                    uint64_t channels,
                                                    uint64_t widthCapacity) {
    const uint64_t row = static_cast<uint64_t>(blockIdx.x);
    const uint64_t begin = static_cast<uint64_t>(rowOffsets[row]);
    const uint64_t end = static_cast<uint64_t>(rowOffsets[row + 1]);
    if (end <= begin) {
        return;
    }

    const uint64_t rowLength = end - begin;
    if (rowLength > widthCapacity) {
        return;  // Defensive: host dispatch metadata must agree with canonical offsets.
    }
    const uint64_t paddedRowBase = row * channels * widthCapacity;
    const uint64_t rowElements = rowLength * channels;
    for (uint64_t flat = static_cast<uint64_t>(threadIdx.x); flat < rowElements; flat += blockDim.x) {
        const uint64_t timestep = flat / channels;
        const uint64_t channel = flat - timestep * channels;
        const uint64_t packedIndex = (begin + timestep) * channels + channel;
        const uint64_t paddedIndex = paddedRowBase + channel * widthCapacity + timestep;
        packedValues[packedIndex] = paddedValues[paddedIndex];
    }
}

void validateAdapterTensors(const Tensor& packedValues,
                            const Tensor& rowOffsets,
                            const Tensor& paddedValues,
                            uint64_t batchSize,
                            uint64_t channels,
                            uint64_t widthCapacity,
                            Stream& stream) {
    if (batchSize == 0 || batchSize > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) || channels == 0 ||
        widthCapacity == 0) {
        throw std::invalid_argument("Padded ragged adapter geometry is invalid.");
    }
    if (packedValues.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU ||
        packedValues.getPlacement().getDeviceNum() != stream.getGpuNum() ||
        rowOffsets.getPlacement() != packedValues.getPlacement() || paddedValues.getPlacement() != packedValues.getPlacement()) {
        throw std::invalid_argument("Padded ragged adapter tensors must share the execution GPU.");
    }
    if (!packedValues.isDenseContiguous() || !rowOffsets.isDenseContiguous() || !paddedValues.isDenseContiguous()) {
        throw std::invalid_argument("Padded ragged adapter requires dense contiguous backing tensors.");
    }
    if (packedValues.getDataType() != paddedValues.getDataType()) {
        throw std::invalid_argument("Padded ragged adapter packed and padded values must share one dtype.");
    }
    if (!isCanonicalRowPartitionOffsetDataType(rowOffsets.getDataType())) {
        throw std::invalid_argument("Padded ragged adapter offsets must use UINT32 or UINT64.");
    }
    if (rowOffsets.getDimensions() != std::vector<uint64_t>({batchSize + 1}) || packedValues.getNumDimensions() != 2 ||
        packedValues.getDimensions()[1] != channels || paddedValues.getNumDimensions() != 1 ||
        paddedValues.getTotalNumElements() < batchSize * channels * widthCapacity) {
        throw std::invalid_argument("Padded ragged adapter tensor shapes do not match its contract.");
    }
}

uint32_t directLoaderBlocks(uint64_t totalElements) {
    constexpr uint32_t threads = 256;
    constexpr uint32_t maxBlocks = 65535;
    const uint64_t requested = (totalElements + threads - 1) / threads;
    return static_cast<uint32_t>(std::min<uint64_t>(requested, maxBlocks));
}

template <typename ElementT, typename OffsetT>
void launchTypedPackedToPadded(const Tensor& packedValues,
                               Tensor& paddedValues,
                               const Tensor& rowOffsets,
                               uint64_t batchSize,
                               uint64_t channels,
                               uint64_t widthCapacity,
                               Stream& stream) {
    constexpr uint32_t threads = 256;
    const uint64_t totalElements = batchSize * channels * widthCapacity;
    const uint32_t blocks = directLoaderBlocks(totalElements);
    const ElementT* packed = reinterpret_cast<const ElementT*>(packedValues.getMemPtr<void>());
    ElementT* padded = reinterpret_cast<ElementT*>(paddedValues.getMemPtr<void>());
    packedToPaddedRaggedSequenceKernel<ElementT, OffsetT><<<blocks, threads, 0, stream.getStream()>>>(
        packed, rowOffsets.getMemPtr<OffsetT>(), padded, batchSize, channels, widthCapacity);
}

template <typename ElementT, typename OffsetT>
void launchTypedSanitizedPaddedCopy(const Tensor& sourcePaddedValues,
                                    Tensor& destinationPaddedValues,
                                    const Tensor& rowOffsets,
                                    uint64_t batchSize,
                                    uint64_t channels,
                                    uint64_t widthCapacity,
                                    Stream& stream) {
    constexpr uint32_t threads = 256;
    const uint64_t totalElements = batchSize * channels * widthCapacity;
    const uint32_t blocks = directLoaderBlocks(totalElements);
    // ElementT is selected only from the storage byte width so one bitwise
    // sanitation kernel can cover FP16/BF16/FP32 (and other same-width value
    // types). Do not ask Tensor for a semantically typed ElementT pointer: for
    // example FP32 storage dispatched as a 4-byte uint32_t must remain valid.
    const ElementT* source = reinterpret_cast<const ElementT*>(sourcePaddedValues.getMemPtr<void>());
    ElementT* destination = reinterpret_cast<ElementT*>(destinationPaddedValues.getMemPtr<void>());
    sanitizedPaddedRaggedSequenceCopyKernel<ElementT, OffsetT><<<blocks, threads, 0, stream.getStream()>>>(
        source, rowOffsets.getMemPtr<OffsetT>(), destination, batchSize, channels, widthCapacity);
}

template <typename ElementT, typename OffsetT>
void launchTypedPaddedToPacked(const Tensor& paddedValues,
                               Tensor& packedValues,
                               const Tensor& rowOffsets,
                               uint64_t batchSize,
                               uint64_t channels,
                               uint64_t widthCapacity,
                               Stream& stream) {
    constexpr uint32_t threads = 256;
    const uint32_t blocks = static_cast<uint32_t>(batchSize);
    const ElementT* padded = reinterpret_cast<const ElementT*>(paddedValues.getMemPtr<void>());
    ElementT* packed = reinterpret_cast<ElementT*>(packedValues.getMemPtr<void>());
    paddedToPackedRaggedSequenceKernel<ElementT, OffsetT><<<blocks, threads, 0, stream.getStream()>>>(
        padded, rowOffsets.getMemPtr<OffsetT>(), packed, channels, widthCapacity);
}

template <typename ElementT, typename OffsetT, bool PackedToPadded>
void launchTypedAdapter(const Tensor& sourceValues,
                        Tensor& destinationValues,
                        const Tensor& rowOffsets,
                        uint64_t batchSize,
                        uint64_t channels,
                        uint64_t widthCapacity,
                        Stream& stream) {
    if constexpr (PackedToPadded) {
        launchTypedPackedToPadded<ElementT, OffsetT>(
            sourceValues, destinationValues, rowOffsets, batchSize, channels, widthCapacity, stream);
    } else {
        launchTypedPaddedToPacked<ElementT, OffsetT>(
            sourceValues, destinationValues, rowOffsets, batchSize, channels, widthCapacity, stream);
    }
}

template <typename OffsetT, bool PackedToPadded>
void dispatchElementSize(const Tensor& sourceValues,
                         Tensor& destinationValues,
                         const Tensor& rowOffsets,
                         DataType valuesDataType,
                         uint64_t batchSize,
                         uint64_t channels,
                         uint64_t widthCapacity,
                         Stream& stream) {
    const float elementBytesFloat = TensorDescriptor::getElementSizeInBytes(valuesDataType);
    const uint64_t elementBytes = static_cast<uint64_t>(elementBytesFloat);
    if (static_cast<float>(elementBytes) != elementBytesFloat) {
        throw std::invalid_argument("Padded ragged adapter requires a whole-byte value dtype.");
    }
    switch (elementBytes) {
        case 1:
            launchTypedAdapter<uint8_t, OffsetT, PackedToPadded>(
                sourceValues, destinationValues, rowOffsets, batchSize, channels, widthCapacity, stream);
            break;
        case 2:
            launchTypedAdapter<uint16_t, OffsetT, PackedToPadded>(
                sourceValues, destinationValues, rowOffsets, batchSize, channels, widthCapacity, stream);
            break;
        case 4:
            launchTypedAdapter<uint32_t, OffsetT, PackedToPadded>(
                sourceValues, destinationValues, rowOffsets, batchSize, channels, widthCapacity, stream);
            break;
        case 8:
            launchTypedAdapter<uint64_t, OffsetT, PackedToPadded>(
                sourceValues, destinationValues, rowOffsets, batchSize, channels, widthCapacity, stream);
            break;
        default:
            throw std::invalid_argument("Padded ragged adapter supports value element sizes of 1, 2, 4, or 8 bytes.");
    }
    CUDA_CHECK(cudaGetLastError());
}

template <bool PackedToPadded>
void dispatchOffsets(const Tensor& sourceValues,
                     Tensor& destinationValues,
                     const Tensor& rowOffsets,
                     DataType valuesDataType,
                     uint64_t batchSize,
                     uint64_t channels,
                     uint64_t widthCapacity,
                     Stream& stream) {
    switch (rowOffsets.getDataType()) {
        case DataType::UINT32:
            dispatchElementSize<uint32_t, PackedToPadded>(sourceValues, destinationValues, rowOffsets, valuesDataType, batchSize, channels, widthCapacity, stream);
            break;
        case DataType::UINT64:
            dispatchElementSize<uint64_t, PackedToPadded>(sourceValues, destinationValues, rowOffsets, valuesDataType, batchSize, channels, widthCapacity, stream);
            break;
        default: throw std::invalid_argument("Padded ragged adapter offsets must use UINT32 or UINT64.");
    }
}

void validatePaddedSanitationTensors(const Tensor& sourcePaddedValues,
                                      const Tensor& rowOffsets,
                                      const Tensor& destinationPaddedValues,
                                      uint64_t batchSize,
                                      uint64_t channels,
                                      uint64_t widthCapacity,
                                      Stream& stream) {
    if (batchSize == 0 || channels == 0 || widthCapacity == 0) {
        throw std::invalid_argument("Padded ragged sanitation geometry is invalid.");
    }
    if (sourcePaddedValues.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU ||
        sourcePaddedValues.getPlacement().getDeviceNum() != stream.getGpuNum() ||
        rowOffsets.getPlacement() != sourcePaddedValues.getPlacement() ||
        destinationPaddedValues.getPlacement() != sourcePaddedValues.getPlacement()) {
        throw std::invalid_argument("Padded ragged sanitation tensors must share the execution GPU.");
    }
    if (!sourcePaddedValues.isDenseContiguous() || !destinationPaddedValues.isDenseContiguous() ||
        !rowOffsets.isDenseContiguous() || sourcePaddedValues.getDataType() != destinationPaddedValues.getDataType()) {
        throw std::invalid_argument("Padded ragged sanitation requires dense, same-dtype source and destination storage.");
    }
    if (!isCanonicalRowPartitionOffsetDataType(rowOffsets.getDataType()) ||
        rowOffsets.getDimensions() != std::vector<uint64_t>({batchSize + 1})) {
        throw std::invalid_argument("Padded ragged sanitation offsets do not match its contract.");
    }
    const uint64_t requiredElements = batchSize * channels * widthCapacity;
    if (sourcePaddedValues.getNumDimensions() != 1 || destinationPaddedValues.getNumDimensions() != 1 ||
        sourcePaddedValues.getTotalNumElements() < requiredElements ||
        destinationPaddedValues.getTotalNumElements() < requiredElements) {
        throw std::invalid_argument("Padded ragged sanitation storage is smaller than the selected dense prefix.");
    }
    if (sourcePaddedValues.getMemPtr<void>() == destinationPaddedValues.getMemPtr<void>()) {
        throw std::invalid_argument("Padded ragged sanitation must use consumer-owned destination storage.");
    }
}

template <typename ElementT, typename OffsetT>
void launchSanitizedPaddedCopyTyped(const Tensor& sourcePaddedValues,
                                    const Tensor& rowOffsets,
                                    Tensor& destinationPaddedValues,
                                    uint64_t batchSize,
                                    uint64_t channels,
                                    uint64_t widthCapacity,
                                    Stream& stream) {
    launchTypedSanitizedPaddedCopy<ElementT, OffsetT>(
        sourcePaddedValues, destinationPaddedValues, rowOffsets, batchSize, channels, widthCapacity, stream);
}

template <typename OffsetT>
void dispatchSanitizedPaddedCopyElementSize(const Tensor& sourcePaddedValues,
                                            const Tensor& rowOffsets,
                                            Tensor& destinationPaddedValues,
                                            uint64_t batchSize,
                                            uint64_t channels,
                                            uint64_t widthCapacity,
                                            Stream& stream) {
    const float elementBytesFloat = TensorDescriptor::getElementSizeInBytes(sourcePaddedValues.getDataType());
    const uint64_t elementBytes = static_cast<uint64_t>(elementBytesFloat);
    if (static_cast<float>(elementBytes) != elementBytesFloat) {
        throw std::invalid_argument("Padded ragged sanitation requires a whole-byte value dtype.");
    }
    switch (elementBytes) {
        case 1:
            launchSanitizedPaddedCopyTyped<uint8_t, OffsetT>(sourcePaddedValues, rowOffsets, destinationPaddedValues, batchSize, channels, widthCapacity, stream);
            break;
        case 2:
            launchSanitizedPaddedCopyTyped<uint16_t, OffsetT>(sourcePaddedValues, rowOffsets, destinationPaddedValues, batchSize, channels, widthCapacity, stream);
            break;
        case 4:
            launchSanitizedPaddedCopyTyped<uint32_t, OffsetT>(sourcePaddedValues, rowOffsets, destinationPaddedValues, batchSize, channels, widthCapacity, stream);
            break;
        case 8:
            launchSanitizedPaddedCopyTyped<uint64_t, OffsetT>(sourcePaddedValues, rowOffsets, destinationPaddedValues, batchSize, channels, widthCapacity, stream);
            break;
        default:
            throw std::invalid_argument("Padded ragged sanitation supports value element sizes of 1, 2, 4, or 8 bytes.");
    }
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void launchSanitizedPaddedRaggedSequenceCopy(const Tensor& sourcePaddedValues,
                                             const Tensor& rowOffsets,
                                             Tensor& destinationPaddedValues,
                                             uint64_t batchSize,
                                             uint64_t channels,
                                             uint64_t widthCapacity,
                                             Stream& stream) {
    validatePaddedSanitationTensors(
        sourcePaddedValues, rowOffsets, destinationPaddedValues, batchSize, channels, widthCapacity, stream);
    switch (rowOffsets.getDataType()) {
        case DataType::UINT32:
            dispatchSanitizedPaddedCopyElementSize<uint32_t>(
                sourcePaddedValues, rowOffsets, destinationPaddedValues, batchSize, channels, widthCapacity, stream);
            break;
        case DataType::UINT64:
            dispatchSanitizedPaddedCopyElementSize<uint64_t>(
                sourcePaddedValues, rowOffsets, destinationPaddedValues, batchSize, channels, widthCapacity, stream);
            break;
        default:
            throw std::invalid_argument("Padded ragged sanitation offsets must use UINT32 or UINT64.");
    }
}

void launchPackedToPaddedRaggedSequence(const Tensor& packedValues,
                                        const Tensor& rowOffsets,
                                        Tensor& paddedValues,
                                        uint64_t batchSize,
                                        uint64_t channels,
                                        uint64_t widthCapacity,
                                        Stream& stream) {
    validateAdapterTensors(packedValues, rowOffsets, paddedValues, batchSize, channels, widthCapacity, stream);
    dispatchOffsets<true>(packedValues, paddedValues, rowOffsets, packedValues.getDataType(), batchSize, channels, widthCapacity, stream);
}

void launchPaddedToPackedRaggedSequence(const Tensor& paddedValues,
                                        const Tensor& rowOffsets,
                                        Tensor& packedValues,
                                        uint64_t batchSize,
                                        uint64_t channels,
                                        uint64_t widthCapacity,
                                        Stream& stream) {
    validateAdapterTensors(packedValues, rowOffsets, paddedValues, batchSize, channels, widthCapacity, stream);
    dispatchOffsets<false>(paddedValues, packedValues, rowOffsets, packedValues.getDataType(), batchSize, channels, widthCapacity, stream);
}

}  // namespace ThorImplementation
