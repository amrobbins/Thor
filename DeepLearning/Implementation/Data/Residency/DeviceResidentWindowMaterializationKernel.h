#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/TensorSpan.h"
#include "Utilities/Common/Stream.h"

#include <cstdint>
#include <type_traits>

struct DeviceResidentWindowRowPlan32 {
    uint64_t sourceOffsetBytes = 0;
    uint32_t validStepBegin = 0;
    uint32_t validStepCount = 0;
};

struct DeviceResidentWindowRowPlan64 {
    uint64_t sourceOffsetBytes = 0;
    uint64_t validStepBegin = 0;
    uint64_t validStepCount = 0;
};

static_assert(sizeof(DeviceResidentWindowRowPlan32) == 16);
static_assert(std::is_trivially_copyable_v<DeviceResidentWindowRowPlan32>);
static_assert(sizeof(DeviceResidentWindowRowPlan64) == 24);
static_assert(std::is_trivially_copyable_v<DeviceResidentWindowRowPlan64>);

struct DeviceResidentWindowMaterializationSpec {
    ThorImplementation::DataType dataType = ThorImplementation::DataType::FP32;
    uint64_t windowLength = 0;
    uint64_t sourceStepBytes = 0;
    double padValue = 0.0;
    bool materializeMask = false;
    // A power of two that divides the sourceOffsetBytes of every row plan of the launch. 1 is always true. The kernel
    // moves the widest packet, up to 16 bytes, that divides this, the step bytes and the tensors' addresses, so a
    // caller that knows its plans start on wider boundaries says so here.
    uint64_t sourceOffsetAlignmentBytes = 1;
};

/**
 * Materialize windows of a resident source into a device batch tensor. Batch row r is spec.windowLength steps of
 * spec.sourceStepBytes: steps [validStepBegin, validStepBegin + validStepCount) of its row plan are the bytes at the
 * plan's sourceOffsetBytes of sourceStorage, and the steps around them are spec.padValue. With spec.materializeMask
 * the destination is one UINT8 per step instead, 1 where the step is valid, and sourceStorage is not read.
 *
 * rowPlans holds logicalRows plans, DeviceResidentWindowRowPlan32 for a window of up to 2^32 - 1 steps and
 * DeviceResidentWindowRowPlan64 above, at an address aligned to 8 bytes. Only the leading logicalRows batch rows are
 * written; logicalRows 0 writes nothing.
 */
void launchDeviceResidentWindowMaterializationKernel(
    const ThorImplementation::Tensor &sourceStorage,
    const ThorImplementation::TensorSpan &rowPlans,
    uint64_t logicalRows,
    const DeviceResidentWindowMaterializationSpec &spec,
    ThorImplementation::Tensor &destination,
    Stream &stream);
