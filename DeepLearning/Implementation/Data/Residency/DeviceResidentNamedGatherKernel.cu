#include "DeepLearning/Implementation/Data/Residency/DeviceResidentNamedGatherKernel.h"

#include "DeepLearning/Implementation/Data/Residency/DeviceResidentRowGatherKernel.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <cstdint>
#include <vector>

using ThorImplementation::Tensor;
using ThorImplementation::TensorPlacement;
using ThorImplementation::TensorSpan;

namespace {

void validateGatherTensorShapes(const Tensor &source, const Tensor &destination, const TensorSpan &rowIndicesDevice) {
    THOR_THROW_IF_FALSE(source.isInitialized());
    THOR_THROW_IF_FALSE(destination.isInitialized());
    rowIndicesDevice.validate("Device resident gather row indices");
    THOR_THROW_IF_FALSE(source.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    THOR_THROW_IF_FALSE(destination.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    THOR_THROW_IF_FALSE(rowIndicesDevice.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    THOR_THROW_IF_FALSE(source.getPlacement() == destination.getPlacement());
    THOR_THROW_IF_FALSE(source.getPlacement() == rowIndicesDevice.getPlacement());
    THOR_THROW_IF_FALSE(source.getDataType() == destination.getDataType());
    THOR_THROW_IF_FALSE(rowIndicesDevice.getDataType() == ThorImplementation::DataType::UINT64);

    const std::vector<uint64_t> sourceDims = source.getDimensions();
    const std::vector<uint64_t> destDims = destination.getDimensions();
    THOR_THROW_IF_FALSE(!sourceDims.empty());
    THOR_THROW_IF_FALSE(!destDims.empty());
    THOR_THROW_IF_FALSE(sourceDims.size() == destDims.size());
    THOR_THROW_IF_FALSE(rowIndicesDevice.count == destDims.at(0));
    THOR_THROW_IF_FALSE(sourceDims.at(0) > 0);
    THOR_THROW_IF_FALSE(destDims.at(0) > 0);
    for (uint64_t i = 1; i < sourceDims.size(); ++i) {
        THOR_THROW_IF_FALSE(sourceDims.at(i) == destDims.at(i));
    }
}

}  // namespace

void launchDeviceResidentNamedGatherKernel(
    const Tensor &source, Tensor &destination, const TensorSpan &rowIndicesDevice, uint64_t logicalRows, Stream &stream) {
    validateGatherTensorShapes(source, destination, rowIndicesDevice);

    const uint64_t batchCapacity = destination.getDimensions().at(0);
    THOR_THROW_IF_FALSE(logicalRows >= 1 && logicalRows <= batchCapacity);
    THOR_THROW_IF_FALSE(logicalRows <= rowIndicesDevice.count);
    const uint64_t sourceRows = source.getDimensions().at(0);
    const uint64_t rowBytes = destination.getArraySizeInBytes() / batchCapacity;
    THOR_THROW_IF_FALSE(rowBytes > 0);
    THOR_THROW_IF_FALSE(destination.getArraySizeInBytes() == batchCapacity * rowBytes);
    THOR_THROW_IF_FALSE(source.getArraySizeInBytes() == sourceRows * rowBytes);

    launchDeviceResidentRowGatherKernel(static_cast<const uint8_t *>(source.getMemPtr()),
                                        sourceRows,
                                        rowBytes,
                                        static_cast<uint8_t *>(destination.getMemPtr()),
                                        rowIndicesDevice.data<uint64_t>(),
                                        logicalRows,
                                        rowBytes,
                                        stream.getStream());
}
