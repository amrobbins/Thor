#pragma once

#include "DeepLearning/Api/Data/DatasetLayout.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include <cstdint>
#include <type_traits>

struct DeviceResidentWindowSourceSequence {
    uint64_t keyBits = 0;
    int64_t startIndex = 0;
    int64_t endIndexExclusive = 0;
    uint64_t offsetBytes = 0;
};

struct DeviceResidentAffineWindowSegment {
    uint64_t rowStart = 0;
    uint64_t count = 0;
    uint64_t keyBits = 0;
    int64_t base = 0;
    int64_t stride = 1;
    int64_t fieldOffset = 0;
};

static_assert(sizeof(DeviceResidentWindowSourceSequence) == 32);
static_assert(std::is_trivially_copyable_v<DeviceResidentWindowSourceSequence>);
static_assert(sizeof(DeviceResidentAffineWindowSegment) == 48);
static_assert(std::is_trivially_copyable_v<DeviceResidentAffineWindowSegment>);

struct DeviceResidentWindowMaterializationSpec {
    DatasetLayout::WindowedTensorReferenceMode referenceMode =
        DatasetLayout::WindowedTensorReferenceMode::INDEXED;
    ThorImplementation::DataType dataType = ThorImplementation::DataType::FP32;
    ThorImplementation::DataType keyDataType = ThorImplementation::DataType::UINT64;
    ThorImplementation::DataType indexDataType = ThorImplementation::DataType::INT64;
    uint64_t numExamples = 0;
    uint64_t recordSizeBytes = 0;
    uint64_t referenceOffsetBytes = 0;
    uint64_t windowLength = 0;
    uint64_t sourceStepBytes = 0;
    double padValue = 0.0;
    bool materializeMask = false;
};

void launchDeviceResidentWindowMaterializationKernel(
    const ThorImplementation::Tensor &recordStorage,
    const ThorImplementation::Tensor &sourceStorage,
    const ThorImplementation::Tensor &sourceSequences,
    uint64_t sourceSequenceCount,
    const ThorImplementation::Tensor &affineSegments,
    uint64_t affineSegmentCount,
    const DeviceResidentWindowMaterializationSpec &spec,
    ThorImplementation::Tensor &destination,
    const ThorImplementation::Tensor &rowIndicesDevice,
    Stream &stream);
