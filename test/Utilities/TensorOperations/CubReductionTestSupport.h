#pragma once

#include "Utilities/TensorOperations/Cub/CubDataTypePolicy.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <vector>

#define REQUIRE_CUDA_DEVICE()                                                                                          \
    do {                                                                                                                \
        int cuda_device_count_for_test = 0;                                                                             \
        const cudaError_t cuda_status_for_test = cudaGetDeviceCount(&cuda_device_count_for_test);                       \
        if (cuda_status_for_test != cudaSuccess || cuda_device_count_for_test <= 0) {                                    \
            GTEST_SKIP() << "CUDA device is required for CUB reduction tests.";                                         \
        }                                                                                                               \
    } while (false)

namespace ThorImplementation::CubReductionTestSupport {

extern TensorPlacement cpuPlacement;
extern TensorPlacement gpuPlacement;

Tensor makeGpuTensor(const std::vector<float>& values,
                     const std::vector<uint64_t>& dimensions,
                     Stream& stream,
                     DataType dtype = DataType::FP32);
Tensor makeGpuUnsignedTensor(const std::vector<uint64_t>& values,
                             const std::vector<uint64_t>& dimensions,
                             Stream& stream,
                             DataType dtype = DataType::UINT32);
void overwriteGpuUnsignedTensor(Tensor& gpu, const std::vector<uint64_t>& values, Stream& stream);

std::vector<float> copyGpuTensorAsFloat(const Tensor& gpu, Stream& stream);
std::vector<uint64_t> copyGpuTensorAsUnsigned(const Tensor& gpu, Stream& stream);

void expectFloatVectorNear(const std::vector<float>& actual,
                           const std::vector<float>& expected,
                           float tolerance = 0.0f);

std::vector<float> executeFp32Output(const Tensor& input, CubReductionOp op, uint32_t axis, Stream& stream);
std::vector<float> executeFp32Output(const Tensor& input,
                                     CubReductionOp op,
                                     const std::vector<uint32_t>& axes,
                                     Stream& stream);

struct OperationExpectation {
    CubReductionOp op;
    std::vector<float> expected;
    float tolerance;
};

void expectOperations(const Tensor& input,
                      uint32_t axis,
                      const std::vector<OperationExpectation>& expectations,
                      Stream& stream);

void expectInputStorageAccumulatesInFp32(DataType input_dtype, Stream& stream);

}  // namespace ThorImplementation::CubReductionTestSupport
