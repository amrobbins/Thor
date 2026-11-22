#include "DeepLearning/Implementation/Data/Residency/DeviceResidentDirectMaterializationKernel.h"

#include "DeepLearning/Implementation/Data/Residency/DeviceResidentRowGatherKernel.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <cstdint>

using ThorImplementation::DataType;
using ThorImplementation::Tensor;
using ThorImplementation::TensorPlacement;
using ThorImplementation::TensorSpan;

void launchDeviceResidentDirectMaterializationKernel(
    const Tensor &recordStorage,
    uint64_t numExamples,
    uint64_t recordSizeBytes,
    uint64_t fieldOffsetBytes,
    uint64_t fieldBytes,
    uint64_t logicalRows,
    Tensor &destination,
    const TensorSpan &rowIndicesDevice,
    Stream &stream) {
    THOR_THROW_IF_FALSE(recordStorage.isInitialized());
    THOR_THROW_IF_FALSE(destination.isInitialized());
    rowIndicesDevice.validate("Device resident direct materialization row indices");
    THOR_THROW_IF_FALSE(
        recordStorage.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    THOR_THROW_IF_FALSE(destination.getPlacement() == recordStorage.getPlacement());
    THOR_THROW_IF_FALSE(rowIndicesDevice.getPlacement() == recordStorage.getPlacement());
    THOR_THROW_IF_FALSE(recordStorage.getDataType() == DataType::UINT8);
    THOR_THROW_IF_FALSE(rowIndicesDevice.getDataType() == DataType::UINT64);
    THOR_THROW_IF_FALSE(recordStorage.getDimensions().size() == 1);
    THOR_THROW_IF_FALSE(!destination.getDimensions().empty());
    THOR_THROW_IF_FALSE(numExamples > 0);
    THOR_THROW_IF_FALSE(recordSizeBytes > 0);
    THOR_THROW_IF_FALSE(fieldBytes > 0);
    THOR_THROW_IF_FALSE(fieldOffsetBytes <= recordSizeBytes);
    THOR_THROW_IF_FALSE(fieldBytes <= recordSizeBytes - fieldOffsetBytes);
    THOR_THROW_IF_FALSE(
        recordStorage.getArraySizeInBytes() % recordSizeBytes == 0);
    THOR_THROW_IF_FALSE(
        recordStorage.getArraySizeInBytes() / recordSizeBytes == numExamples);

    const uint64_t batchCapacity = destination.getDimensions().front();
    const uint64_t rowIndexCapacity = rowIndicesDevice.count;
    THOR_THROW_IF_FALSE(batchCapacity > 0);
    THOR_THROW_IF_FALSE(logicalRows >= 1 && logicalRows <= batchCapacity);
    THOR_THROW_IF_FALSE(logicalRows <= rowIndexCapacity);
    THOR_THROW_IF_FALSE(destination.getArraySizeInBytes() % fieldBytes == 0);
    THOR_THROW_IF_FALSE(destination.getArraySizeInBytes() / fieldBytes == batchCapacity);

    // The field of record i is fieldBytes at fieldOffsetBytes of record i: rows of fieldBytes, a record apart.
    launchDeviceResidentRowGatherKernel(recordStorage.getMemPtr<uint8_t>() + fieldOffsetBytes,
                                        numExamples,
                                        recordSizeBytes,
                                        static_cast<uint8_t *>(destination.getMemPtr()),
                                        rowIndicesDevice.data<uint64_t>(),
                                        logicalRows,
                                        fieldBytes,
                                        stream.getStream());
}
