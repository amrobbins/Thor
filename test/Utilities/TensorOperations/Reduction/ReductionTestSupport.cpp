#include "test/Utilities/TensorOperations/Reduction/ReductionTestSupport.h"

#include "Utilities/Common/LowPrecisionFloat.h"
#include "Utilities/Exceptions.h"
#include "Utilities/TensorOperations/Copy/StridedCopy.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace ThorImplementation::ReductionTestSupport {
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
                typed[i] = ThorLowPrecision::toFp8E4M3Satfinite(values[i]);
            }
            return;
        }
        case DataType::FP8_E5M2: {
            __nv_fp8_e5m2* typed = static_cast<__nv_fp8_e5m2*>(storage);
            for (size_t i = 0; i < values.size(); ++i) {
                typed[i] = ThorLowPrecision::toFp8E5M2Satfinite(values[i]);
            }
            return;
        }
#endif
        default:
            throw std::invalid_argument("Unsupported reduction test storage dtype.");
    }
}

template <typename T>
void storeCpuUnsignedValuesAs(Tensor& cpu, const std::vector<uint64_t>& values, const char* dtypeName) {
    T* typed = cpu.getMemPtr<T>();
    for (size_t i = 0; i < values.size(); ++i) {
        if (values[i] > static_cast<uint64_t>(std::numeric_limits<T>::max())) {
            throw std::invalid_argument(std::string("Test ") + dtypeName + " value is out of range.");
        }
        typed[i] = static_cast<T>(values[i]);
    }
}

void storeCpuUnsignedValues(Tensor& cpu, const std::vector<uint64_t>& values) {
    switch (cpu.getDataType()) {
        case DataType::UINT8:
            storeCpuUnsignedValuesAs<uint8_t>(cpu, values, "UINT8");
            return;
        case DataType::UINT16:
            storeCpuUnsignedValuesAs<uint16_t>(cpu, values, "UINT16");
            return;
        case DataType::UINT32:
            storeCpuUnsignedValuesAs<uint32_t>(cpu, values, "UINT32");
            return;
        case DataType::UINT64: {
            uint64_t* typed = cpu.getMemPtr<uint64_t>();
            for (size_t i = 0; i < values.size(); ++i) {
                typed[i] = values[i];
            }
            return;
        }
        default:
            throw std::invalid_argument("Unsigned test tensors must use UINT8, UINT16, UINT32, or UINT64.");
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
        default:
            throw std::invalid_argument("Unsupported reduction test output dtype.");
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

Tensor makeGpuUnsignedTensor(const std::vector<uint64_t>& values,
                             const std::vector<uint64_t>& dimensions,
                             Stream& stream,
                             DataType dtype) {
    TensorDescriptor descriptor(dtype, dimensions);
    if (descriptor.getTotalNumElements() != values.size()) {
        throw std::invalid_argument("Test unsigned tensor value count does not match dimensions.");
    }
    Tensor cpu(cpuPlacement, descriptor);
    storeCpuUnsignedValues(cpu, values);
    Tensor gpu(gpuPlacement, descriptor);
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();
    return gpu;
}

void overwriteGpuUnsignedTensor(Tensor& gpu, const std::vector<uint64_t>& values, Stream& stream) {
    if (gpu.getTotalNumElements() != values.size()) {
        throw std::invalid_argument("Replacement unsigned value count does not match tensor size.");
    }
    Tensor cpu(cpuPlacement, gpu.getDescriptor());
    storeCpuUnsignedValues(cpu, values);
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();
}

std::vector<float> copyGpuTensorAsFloat(const Tensor& gpu, Stream& stream) {
    Tensor cpu = gpu.clone(cpuPlacement);
    cpu.copyFromAsync(gpu, stream);
    stream.synchronize();
    return readCpuValues(cpu);
}

std::vector<uint64_t> copyGpuTensorAsUnsigned(const Tensor& gpu, Stream& stream) {
    Tensor cpu = gpu.clone(cpuPlacement);
    cpu.copyFromAsync(gpu, stream);
    stream.synchronize();

    const size_t num_elements = cpu.getTotalNumElements();
    std::vector<uint64_t> values(num_elements);
    switch (cpu.getDataType()) {
        case DataType::UINT8: {
            const uint8_t* typed = cpu.getMemPtr<uint8_t>();
            for (size_t i = 0; i < num_elements; ++i) {
                values[i] = typed[i];
            }
            return values;
        }
        case DataType::UINT16: {
            const uint16_t* typed = cpu.getMemPtr<uint16_t>();
            for (size_t i = 0; i < num_elements; ++i) {
                values[i] = typed[i];
            }
            return values;
        }
        case DataType::UINT32: {
            const uint32_t* typed = cpu.getMemPtr<uint32_t>();
            for (size_t i = 0; i < num_elements; ++i) {
                values[i] = typed[i];
            }
            return values;
        }
        case DataType::UINT64: {
            const uint64_t* typed = cpu.getMemPtr<uint64_t>();
            for (size_t i = 0; i < num_elements; ++i) {
                values[i] = typed[i];
            }
            return values;
        }
        default:
            throw std::invalid_argument("Unsigned test tensors must use UINT8, UINT16, UINT32, or UINT64.");
    }
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

std::vector<float> executeFp32Output(const StridedSpan& input, ReductionOp op, uint32_t axis, Stream& stream) {
    std::shared_ptr<StampedReducer> stamped = Reducer(op, axis, DataType::FP32).stamp(input, stream);
    stamped->run();
    stream.synchronize();
    return copyGpuTensorAsFloat(stamped->getOutputTensor(), stream);
}

std::vector<float> executeFp32Output(const StridedSpan& input,
                                     ReductionOp op,
                                     const std::vector<uint32_t>& axes,
                                     Stream& stream) {
    std::shared_ptr<StampedReducer> stamped = Reducer(op, axes, DataType::FP32).stamp(input, stream);
    stamped->run();
    stream.synchronize();
    return copyGpuTensorAsFloat(stamped->getOutputTensor(), stream);
}

void expectOperations(const StridedSpan& input,
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
    std::shared_ptr<StampedReducer> stamped =
        Reducer(ReductionOp::Sum, 1, DataType::FP32).stamp(input, stream);
    EXPECT_EQ(stamped->getInputDataType(), input_dtype);
    EXPECT_EQ(stamped->getAccumulatorDataType(), DataType::FP32);
    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), {3.0f, 7.0f});
}

Tensor runRefusedViewInTwoParts(const StridedSpan& view,
                                ReductionOp op,
                                const std::vector<uint32_t>& axes,
                                Stream& stream,
                                std::optional<DataType> output_dtype,
                                std::optional<ViewSplit::Kind>* kind) {
    const std::vector<uint64_t>& dims = view.getDimensions();
    const std::vector<uint64_t>& strides = view.getStridesElements();
    EXPECT_THROW(static_cast<void>(Reducer::analyzeGeometry(dims, strides, axes)), NotImplementedException);
    const DataType resolved_output = Reducer(op, axes, output_dtype).resolveOutputDataType(view.getDataType());
    // The split the view has without the reducer's in-place pitched read (Reducer::pitchedCompleteLaunch).
    const std::optional<ViewSplit> split =
        Reducer::splitRefusedView(view, axes, resolved_output, false);
    if (!split.has_value()) throw std::logic_error("runRefusedViewInTwoParts: the reducer does not refuse the view.");
    if (kind != nullptr) *kind = split->kind;
    const Reducer reduction(op, split->dense_axes, output_dtype);
    if (split->kind == ViewSplit::Kind::PackedInput) {
        Tensor packed(view.getPlacement(), TensorDescriptor(view.getDataType(), split->dense_dimensions));
        copyToDenseAsync(view, packed, stream);
        const std::shared_ptr<StampedReducer> stamped = reduction.stamp(StridedSpan(packed), stream);
        stamped->run();
        stream.synchronize();
        return stamped->getOutputTensor();
    }
    // Rotated (F / G) or sliced (H read at its row pitch): the reduction of the bytes, then the copy of its output.
    const std::shared_ptr<StampedReducer> stamped = reduction.stamp(split->reducedInput(view), stream);
    stamped->run();
    const Tensor& buffer = stamped->getOutputTensor();  // buffer_dimensions, with the reduced axis kept as extent 1
    Tensor output(view.getPlacement(), TensorDescriptor(buffer.getDataType(), split->copy_dimensions));
    copyToDenseAsync(split->copySource(buffer), output, stream);
    stream.synchronize();
    std::vector<uint64_t> output_dims = dims;  // the reducer keeps a reduced axis as extent 1
    for (const uint32_t axis : axes) output_dims[axis] = 1;
    output.reshape(output_dims);
    return output;
}

std::vector<float> executeRefusedViewInTwoParts(const StridedSpan& view,
                                                ReductionOp op,
                                                const std::vector<uint32_t>& axes,
                                                Stream& stream) {
    return copyGpuTensorAsFloat(runRefusedViewInTwoParts(view, op, axes, stream, DataType::FP32), stream);
}

}  // namespace ThorImplementation::ReductionTestSupport
