#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
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
};

void launchDeviceResidentWindowMaterializationKernel(
    const ThorImplementation::Tensor &sourceStorage,
    const ThorImplementation::Tensor &rowPlans,
    uint64_t logicalRows,
    const DeviceResidentWindowMaterializationSpec &spec,
    ThorImplementation::Tensor &destination,
    Stream &stream);

/** Benchmark-only launch hook that forces a specific rows-per-CTA specialization. */
void launchDeviceResidentWindowMaterializationKernelWithRowsPerCtaForBenchmark(
    const ThorImplementation::Tensor &sourceStorage,
    const ThorImplementation::Tensor &rowPlans,
    uint64_t logicalRows,
    const DeviceResidentWindowMaterializationSpec &spec,
    ThorImplementation::Tensor &destination,
    uint32_t rowsPerCta,
    Stream &stream);
