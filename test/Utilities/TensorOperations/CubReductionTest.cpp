#include "Utilities/TensorOperations/Cub/CubReduction.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                                          \
    do {                                                                                                                \
        int cuda_device_count_for_test = 0;                                                                             \
        const cudaError_t cuda_status_for_test = cudaGetDeviceCount(&cuda_device_count_for_test);                       \
        if (cuda_status_for_test != cudaSuccess || cuda_device_count_for_test <= 0) {                                    \
            GTEST_SKIP() << "CUDA device is required for CUB reduction tests.";                                         \
        }                                                                                                               \
    } while (false)

TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

template <typename T>
DataType dtypeFor();

template <>
DataType dtypeFor<float>() {
    return DataType::FP32;
}

template <>
DataType dtypeFor<__half>() {
    return DataType::FP16;
}

template <>
DataType dtypeFor<__nv_bfloat16>() {
    return DataType::BF16;
}

#if THOR_CUB_ENABLE_FP8_TYPES
template <>
DataType dtypeFor<__nv_fp8_e4m3>() {
    return DataType::FP8_E4M3;
}

template <>
DataType dtypeFor<__nv_fp8_e5m2>() {
    return DataType::FP8_E5M2;
}
#endif

template <typename T>
Tensor makeGpuTensor(const std::vector<T>& values, const std::vector<uint64_t>& dimensions, Stream& stream) {
    TensorDescriptor descriptor(dtypeFor<T>(), dimensions);
    if (descriptor.getTotalNumElements() != values.size()) {
        throw std::invalid_argument("Test tensor value count does not match dimensions.");
    }

    Tensor cpu(cpuPlacement, descriptor);
    T* cpu_ptr = cpu.getMemPtr<T>();
    for (size_t i = 0; i < values.size(); ++i) {
        cpu_ptr[i] = values[i];
    }

    Tensor gpu(gpuPlacement, descriptor);
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();
    return gpu;
}

template <typename T>
std::vector<float> copyGpuTensorAsFloat(const Tensor& gpu, Stream& stream) {
    Tensor cpu = gpu.clone(cpuPlacement);
    cpu.copyFromAsync(gpu, stream);
    stream.synchronize();

    std::vector<float> values(cpu.getTotalNumElements());
    const T* cpu_ptr = cpu.getMemPtr<T>();
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(cpu_ptr[i]);
    }
    return values;
}

std::vector<__nv_bfloat16> toBf16(const std::vector<float>& values) {
    std::vector<__nv_bfloat16> converted;
    converted.reserve(values.size());
    for (float value : values) {
        converted.emplace_back(value);
    }
    return converted;
}

void expectFloatVectorNear(const std::vector<float>& actual, const std::vector<float>& expected, float tolerance = 0.0f) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "index " << i;
    }
}

}  // namespace

TEST(CubReductionGeometry, SelectsBestSingleAxisPath) {
    const CubReductionGeometry scalar = CubReduction::analyzeGeometry({257}, 0);
    EXPECT_EQ(scalar.path, CubReductionPath::DeviceTransformReduce);
    EXPECT_EQ(scalar.outer_size, 1U);
    EXPECT_EQ(scalar.reduction_size, 257U);
    EXPECT_EQ(scalar.inner_size, 1U);
    EXPECT_EQ(scalar.output_elements, 1U);
    EXPECT_EQ(scalar.output_dimensions, (std::vector<uint64_t>{1}));

    const CubReductionGeometry contiguous = CubReduction::analyzeGeometry({2, 3, 4}, 2);
    EXPECT_EQ(contiguous.path, CubReductionPath::ContiguousFixedSegment);
    EXPECT_EQ(contiguous.outer_size, 6U);
    EXPECT_EQ(contiguous.reduction_size, 4U);
    EXPECT_EQ(contiguous.inner_size, 1U);
    EXPECT_EQ(contiguous.output_elements, 6U);
    EXPECT_EQ(contiguous.output_dimensions, (std::vector<uint64_t>{2, 3, 1}));

    const CubReductionGeometry strided = CubReduction::analyzeGeometry({2, 3, 4}, 1);
    EXPECT_EQ(strided.path, CubReductionPath::StridedFixedSegment);
    EXPECT_EQ(strided.outer_size, 2U);
    EXPECT_EQ(strided.reduction_size, 3U);
    EXPECT_EQ(strided.inner_size, 4U);
    EXPECT_EQ(strided.output_elements, 8U);
    EXPECT_EQ(strided.output_dimensions, (std::vector<uint64_t>{2, 1, 4}));
}

TEST(CubReductionGeometry, RejectsInvalidSingleAxisGeometry) {
    EXPECT_THROW(static_cast<void>(CubReduction::analyzeGeometry({}, 0)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(CubReduction::analyzeGeometry({2, 3}, 2)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(CubReduction::analyzeGeometry({2, 0, 3}, 1)), std::invalid_argument);
}

TEST(CubReduction, WholeTensorBf16SumAccumulatesInFp32AndDefaultsOutputToInputDtype) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    std::vector<float> values(257, 1.0f);
    values[0] = 256.0f;
    Tensor input = makeGpuTensor(toBf16(values), {257}, stream);

    CubReduction reduction(CubReductionOp::Sum, 0);
    std::shared_ptr<StampedCubReduction> stamped = reduction.stamp(input, stream);

    EXPECT_EQ(stamped->getPath(), CubReductionPath::DeviceTransformReduce);
    EXPECT_EQ(stamped->getInputDataType(), DataType::BF16);
    EXPECT_EQ(stamped->getOutputDataType(), DataType::BF16);
    EXPECT_EQ(stamped->getAccumulatorDataType(), DataType::FP32);
    EXPECT_EQ(stamped->getOutputTensor().getDimensions(), (std::vector<uint64_t>{1}));
    EXPECT_GT(stamped->getWorkspaceSizeInBytes(), 0U);

    stamped->run();
    stream.synchronize();

    expectFloatVectorNear(copyGpuTensorAsFloat<__nv_bfloat16>(stamped->getOutputTensor(), stream), {512.0f});
}

TEST(CubReduction, ContiguousAxisUsesFixedSegmentReductionAndExplicitFp32Output) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuTensor(toBf16({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}), {2, 3}, stream);
    CubReduction reduction(CubReductionOp::Sum, 1, DataType::FP32);
    std::shared_ptr<StampedCubReduction> stamped = reduction.stamp(input, stream);

    EXPECT_EQ(stamped->getPath(), CubReductionPath::ContiguousFixedSegment);
    EXPECT_EQ(stamped->getOutputDataType(), DataType::FP32);
    EXPECT_EQ(stamped->getGeometry().reduction_size, 3U);
    EXPECT_EQ(stamped->getGeometry().output_elements, 2U);
    EXPECT_EQ(stamped->getOutputTensor().getDimensions(), (std::vector<uint64_t>{2, 1}));

    stamped->runOn(stream);
    stream.synchronize();

    expectFloatVectorNear(copyGpuTensorAsFloat<float>(stamped->getOutputTensor(), stream), {6.0f, 15.0f});
}

TEST(CubReduction, StridedAxisUsesCountingTransformIterator) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuTensor(
        toBf16({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}),
        {2, 3, 2},
        stream);
    CubReduction reduction(CubReductionOp::Sum, 1, DataType::FP32);
    std::shared_ptr<StampedCubReduction> stamped = reduction.stamp(input, stream);

    EXPECT_EQ(stamped->getPath(), CubReductionPath::StridedFixedSegment);
    EXPECT_EQ(stamped->getGeometry().outer_size, 2U);
    EXPECT_EQ(stamped->getGeometry().reduction_size, 3U);
    EXPECT_EQ(stamped->getGeometry().inner_size, 2U);
    EXPECT_EQ(stamped->getOutputTensor().getDimensions(), (std::vector<uint64_t>{2, 1, 2}));

    stamped->run();
    stream.synchronize();

    expectFloatVectorNear(copyGpuTensorAsFloat<float>(stamped->getOutputTensor(), stream), {9.0f, 12.0f, 27.0f, 30.0f});
}

TEST(CubReduction, SupportsMinAndMaxWithFp32Accumulation) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuTensor(toBf16({4.0f, -2.0f, 7.0f, 3.0f, 8.0f, 1.0f}), {2, 3}, stream);

    std::shared_ptr<StampedCubReduction> minimum = CubReduction(CubReductionOp::Min, 1, DataType::FP32).stamp(input, stream);
    std::shared_ptr<StampedCubReduction> maximum = CubReduction(CubReductionOp::Max, 1, DataType::FP32).stamp(input, stream);

    minimum->run();
    maximum->run();
    stream.synchronize();

    expectFloatVectorNear(copyGpuTensorAsFloat<float>(minimum->getOutputTensor(), stream), {-2.0f, 1.0f});
    expectFloatVectorNear(copyGpuTensorAsFloat<float>(maximum->getOutputTensor(), stream), {7.0f, 8.0f});
}

TEST(CubReduction, ReusesStampedOutputAndWorkspaceAcrossExecutions) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuTensor(toBf16({1.0f, 2.0f, 3.0f, 4.0f}), {2, 2}, stream);
    std::shared_ptr<StampedCubReduction> stamped = CubReduction(CubReductionOp::Sum, 1, DataType::FP32).stamp(input, stream);
    const uint64_t output_tensor_id = stamped->getOutputTensor().getTensorId();
    const size_t workspace_bytes = stamped->getWorkspaceSizeInBytes();

    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat<float>(stamped->getOutputTensor(), stream), {3.0f, 7.0f});

    Tensor replacement = makeGpuTensor(toBf16({10.0f, 20.0f, 30.0f, 40.0f}), {2, 2}, stream);
    input.copyFromAsync(replacement, stream);
    stamped->run();
    stream.synchronize();

    EXPECT_EQ(stamped->getOutputTensor().getTensorId(), output_tensor_id);
    EXPECT_EQ(stamped->getWorkspaceSizeInBytes(), workspace_bytes);
    expectFloatVectorNear(copyGpuTensorAsFloat<float>(stamped->getOutputTensor(), stream), {30.0f, 70.0f});
}

TEST(CubReduction, ValidatesPreallocatedOutputContract) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuTensor(toBf16({1.0f, 2.0f, 3.0f, 4.0f}), {2, 2}, stream);
    CubReduction reduction(CubReductionOp::Sum, 1, DataType::FP32);

    Tensor correct_output(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 1}));
    std::shared_ptr<StampedCubReduction> stamped = reduction.stamp(input, correct_output, stream);
    EXPECT_EQ(stamped->getOutputTensor().getTensorId(), correct_output.getTensorId());

    Tensor wrong_dtype(gpuPlacement, TensorDescriptor(DataType::BF16, {2, 1}));
    EXPECT_THROW(static_cast<void>(reduction.stamp(input, wrong_dtype, stream)), std::invalid_argument);

    Tensor wrong_shape(gpuPlacement, TensorDescriptor(DataType::FP32, {1, 2}));
    EXPECT_THROW(static_cast<void>(reduction.stamp(input, wrong_shape, stream)), std::invalid_argument);

    Tensor scalar = makeGpuTensor(toBf16({1.0f}), {1}, stream);
    EXPECT_THROW(static_cast<void>(CubReduction(CubReductionOp::Sum, 0).stamp(scalar, scalar, stream)), std::invalid_argument);
}

#if THOR_CUB_ENABLE_FP8_TYPES
TEST(CubReduction, Fp8OutputsUseExplicitSaturatingFp32Conversion) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuTensor<float>({40000.0f, 40000.0f, -40000.0f, -40000.0f}, {2, 2}, stream);

    std::shared_ptr<StampedCubReduction> e4m3 =
        CubReduction(CubReductionOp::Sum, 1, DataType::FP8_E4M3).stamp(input, stream);
    std::shared_ptr<StampedCubReduction> e5m2 =
        CubReduction(CubReductionOp::Sum, 1, DataType::FP8_E5M2).stamp(input, stream);

    e4m3->run();
    e5m2->run();
    stream.synchronize();

    expectFloatVectorNear(copyGpuTensorAsFloat<__nv_fp8_e4m3>(e4m3->getOutputTensor(), stream), {448.0f, -448.0f});
    expectFloatVectorNear(copyGpuTensorAsFloat<__nv_fp8_e5m2>(e5m2->getOutputTensor(), stream), {57344.0f, -57344.0f});
}
#endif
