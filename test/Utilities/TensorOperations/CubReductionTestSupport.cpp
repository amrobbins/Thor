#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>

namespace ThorImplementation::CubReductionTestSupport {
namespace {

void storeCpuValues(Tensor& cpu, DataType dtype, const std::vector<float>& values) {
    void* storage = cpu.getMemPtr<void>();
    switch (dtype) {
        case DataType::FP32: {
            float* typed = static_cast<float*>(storage);
            for (size_t i = 0; i < values.size(); ++i) {
                typed[i] = values[i];
            }
            return;
        }
        case DataType::FP16: {
            __half* typed = static_cast<__half*>(storage);
            for (size_t i = 0; i < values.size(); ++i) {
                typed[i] = __float2half_rn(values[i]);
            }
            return;
        }
        case DataType::BF16: {
            __nv_bfloat16* typed = static_cast<__nv_bfloat16*>(storage);
            for (size_t i = 0; i < values.size(); ++i) {
                typed[i] = __float2bfloat16_rn(values[i]);
            }
            return;
        }
#if THOR_CUB_ENABLE_FP8_TYPES
        case DataType::FP8_E4M3: {
            __nv_fp8_e4m3* typed = static_cast<__nv_fp8_e4m3*>(storage);
            for (size_t i = 0; i < values.size(); ++i) {
                typed[i].__x = __nv_cvt_float_to_fp8(values[i], __NV_SATFINITE, __NV_E4M3);
            }
            return;
        }
        case DataType::FP8_E5M2: {
            __nv_fp8_e5m2* typed = static_cast<__nv_fp8_e5m2*>(storage);
            for (size_t i = 0; i < values.size(); ++i) {
                typed[i].__x = __nv_cvt_float_to_fp8(values[i], __NV_SATFINITE, __NV_E5M2);
            }
            return;
        }
#endif
#if THOR_CUB_ENABLE_64BIT_TYPES
        case DataType::FP64: {
            double* typed = static_cast<double*>(storage);
            for (size_t i = 0; i < values.size(); ++i) {
                typed[i] = static_cast<double>(values[i]);
            }
            return;
        }
#endif
        default:
            throw std::invalid_argument("Unsupported CUB reduction test storage dtype.");
    }
}

std::vector<float> readCpuValues(const Tensor& cpu) {
    const DataType dtype = cpu.getDataType();
    const size_t num_elements = cpu.getTotalNumElements();
    const void* storage = cpu.getMemPtr<void>();
    std::vector<float> values(num_elements);

    switch (dtype) {
        case DataType::FP32: {
            const float* typed = static_cast<const float*>(storage);
            for (size_t i = 0; i < num_elements; ++i) {
                values[i] = typed[i];
            }
            return values;
        }
        case DataType::FP16: {
            const __half* typed = static_cast<const __half*>(storage);
            for (size_t i = 0; i < num_elements; ++i) {
                values[i] = __half2float(typed[i]);
            }
            return values;
        }
        case DataType::BF16: {
            const __nv_bfloat16* typed = static_cast<const __nv_bfloat16*>(storage);
            for (size_t i = 0; i < num_elements; ++i) {
                values[i] = __bfloat162float(typed[i]);
            }
            return values;
        }
#if THOR_CUB_ENABLE_FP8_TYPES
        case DataType::FP8_E4M3: {
            const __nv_fp8_e4m3* typed = static_cast<const __nv_fp8_e4m3*>(storage);
            for (size_t i = 0; i < num_elements; ++i) {
                values[i] = static_cast<float>(typed[i]);
            }
            return values;
        }
        case DataType::FP8_E5M2: {
            const __nv_fp8_e5m2* typed = static_cast<const __nv_fp8_e5m2*>(storage);
            for (size_t i = 0; i < num_elements; ++i) {
                values[i] = static_cast<float>(typed[i]);
            }
            return values;
        }
#endif
#if THOR_CUB_ENABLE_64BIT_TYPES
        case DataType::FP64: {
            const double* typed = static_cast<const double*>(storage);
            for (size_t i = 0; i < num_elements; ++i) {
                values[i] = static_cast<float>(typed[i]);
            }
            return values;
        }
#endif
        default:
            throw std::invalid_argument("Unsupported CUB reduction test output dtype.");
    }
}

}  // namespace

TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

Tensor makeGpuTensor(const std::vector<float>& values,
                     const std::vector<uint64_t>& dimensions,
                     Stream& stream,
                     DataType dtype) {
    TensorDescriptor descriptor(dtype, dimensions);
    if (descriptor.getTotalNumElements() != values.size()) {
        throw std::invalid_argument("Test tensor value count does not match dimensions.");
    }

    Tensor cpu(cpuPlacement, descriptor);
    storeCpuValues(cpu, dtype, values);

    Tensor gpu(gpuPlacement, descriptor);
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();
    return gpu;
}

std::vector<float> copyGpuTensorAsFloat(const Tensor& gpu, Stream& stream) {
    Tensor cpu = gpu.clone(cpuPlacement);
    cpu.copyFromAsync(gpu, stream);
    stream.synchronize();
    return readCpuValues(cpu);
}

void expectFloatVectorNear(const std::vector<float>& actual,
                           const std::vector<float>& expected,
                           float tolerance) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::isnan(expected[i])) {
            EXPECT_TRUE(std::isnan(actual[i])) << "index " << i << " actual=" << actual[i];
        } else if (std::isinf(expected[i])) {
            EXPECT_EQ(actual[i], expected[i]) << "index " << i;
        } else {
            EXPECT_NEAR(actual[i], expected[i], tolerance) << "index " << i;
        }
    }
}

std::vector<float> executeFp32Output(const Tensor& input, CubReductionOp op, uint32_t axis, Stream& stream) {
    std::shared_ptr<StampedCubReduction> stamped = CubReduction(op, axis, DataType::FP32).stamp(input, stream);
    stamped->run();
    stream.synchronize();
    return copyGpuTensorAsFloat(stamped->getOutputTensor(), stream);
}

std::vector<float> executeFp32Output(const Tensor& input,
                                     CubReductionOp op,
                                     const std::vector<uint32_t>& axes,
                                     Stream& stream) {
    std::shared_ptr<StampedCubReduction> stamped = CubReduction(op, axes, DataType::FP32).stamp(input, stream);
    stamped->run();
    stream.synchronize();
    return copyGpuTensorAsFloat(stamped->getOutputTensor(), stream);
}

void expectOperations(const Tensor& input,
                      uint32_t axis,
                      const std::vector<OperationExpectation>& expectations,
                      Stream& stream) {
    for (const OperationExpectation& expectation : expectations) {
        SCOPED_TRACE(static_cast<int>(expectation.op));
        expectFloatVectorNear(executeFp32Output(input, expectation.op, axis, stream),
                              expectation.expected,
                              expectation.tolerance);
    }
}

void expectInputStorageAccumulatesInFp32(DataType input_dtype, Stream& stream) {
    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, stream, input_dtype);
    std::shared_ptr<StampedCubReduction> stamped =
        CubReduction(CubReductionOp::Sum, 1, DataType::FP32).stamp(input, stream);
    EXPECT_EQ(stamped->getInputDataType(), input_dtype);
    EXPECT_EQ(stamped->getAccumulatorDataType(), DataType::FP32);
    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), {3.0f, 7.0f});
}

}  // namespace ThorImplementation::CubReductionTestSupport
