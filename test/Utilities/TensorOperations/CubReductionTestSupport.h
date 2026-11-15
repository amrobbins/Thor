#pragma once

#include "Utilities/TensorOperations/Cub/CubDataTypePolicy.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <optional>
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

std::vector<float> executeFp32Output(const StridedSpan& input, CubReductionOp op, uint32_t axis, Stream& stream);
std::vector<float> executeFp32Output(const StridedSpan& input,
                                     CubReductionOp op,
                                     const std::vector<uint32_t>& axes,
                                     Stream& stream);

struct OperationExpectation {
    CubReductionOp op;
    std::vector<float> expected;
    float tolerance;
};

void expectOperations(const StridedSpan& input,
                      uint32_t axis,
                      const std::vector<OperationExpectation>& expectations,
                      Stream& stream);

void expectInputStorageAccumulatesInFp32(DataType input_dtype, Stream& stream);

/**
 * 1390: a view the reducer refuses (CubReduction::analyzeGeometry throws NotImplementedException, which this checks),
 * run in the two parts a caller runs (CubReduction::splitRefusedView of the span, as Expression does): the reduction
 * of its bytes in physical order, then the dense copy of that output read rotated (F / G) or, 1391, its columns sliced
 * (H read at its row pitch); or the dense copy of the view, then the reduction of the copy (H). Returns the output,
 * synchronized, with the dimensions the reducer would have given it, and the split's kind in `*kind` when given.
 * Throws std::logic_error when splitRefusedView has nothing for the view. 1392: a pitched view the reducer would read
 * in place (CubReduction::pitchedCompleteLaunch) runs the split it has otherwise.
 */
Tensor runRefusedViewInTwoParts(const StridedSpan& view,
                                CubReductionOp op,
                                const std::vector<uint32_t>& axes,
                                Stream& stream,
                                std::optional<DataType> output_dtype = std::nullopt,
                                std::optional<CubReductionViewSplit::Kind>* kind = nullptr);
std::vector<float> executeRefusedViewInTwoParts(const StridedSpan& view,
                                                CubReductionOp op,
                                                const std::vector<uint32_t>& axes,
                                                Stream& stream);

}  // namespace ThorImplementation::CubReductionTestSupport
