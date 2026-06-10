#include "Utilities/Expression/FlatScatterAddKernel.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                                          \
    do {                                                                                                                \
        int cuda_device_count_for_test = 0;                                                                             \
        const cudaError_t cuda_status_for_test = cudaGetDeviceCount(&cuda_device_count_for_test);                       \
        if (cuda_status_for_test != cudaSuccess || cuda_device_count_for_test <= 0) {                                    \
            GTEST_SKIP() << "CUDA device is required for flat scatter-add tests.";                                      \
        }                                                                                                               \
    } while (false)

TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

uint64_t tensorNumel(const Tensor& tensor) {
    uint64_t n = 1;
    for (uint64_t d : tensor.getDimensions()) {
        n *= d;
    }
    return n;
}

Tensor makeGpuFloatVector(const std::vector<float>& values, Stream& stream) {
    Tensor cpu(cpuPlacement, TensorDescriptor(DataType::FP32, {static_cast<uint64_t>(values.size())}));
    auto* cpu_ptr = cpu.getMemPtr<float>();
    for (size_t i = 0; i < values.size(); ++i) {
        cpu_ptr[i] = values[i];
    }

    Tensor gpu(gpuPlacement, TensorDescriptor(DataType::FP32, {static_cast<uint64_t>(values.size())}));
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();
    return gpu;
}

Tensor makeGpuUint32Vector(const std::vector<uint32_t>& values, Stream& stream) {
    Tensor cpu(cpuPlacement, TensorDescriptor(DataType::UINT32, {static_cast<uint64_t>(values.size())}));
    auto* cpu_ptr = cpu.getMemPtr<uint32_t>();
    for (size_t i = 0; i < values.size(); ++i) {
        cpu_ptr[i] = values[i];
    }

    Tensor gpu(gpuPlacement, TensorDescriptor(DataType::UINT32, {static_cast<uint64_t>(values.size())}));
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();
    return gpu;
}

std::vector<float> copyGpuFloatVector(const Tensor& gpu, Stream& stream) {
    Tensor cpu = gpu.clone(cpuPlacement);
    cpu.copyFromAsync(gpu, stream);
    stream.synchronize();

    std::vector<float> values(tensorNumel(cpu));
    const auto* ptr = cpu.getMemPtr<float>();
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = ptr[i];
    }
    return values;
}

void expectFloatVectorEq(const std::vector<float>& actual, const std::vector<float>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "index " << i;
    }
}

}  // namespace

TEST(FlatScatterAddKernel, UniqueValidIndicesUseDirectAxpbyPath) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor updates = makeGpuFloatVector({1.5f, -2.0f, 3.0f}, stream);
    Tensor indices = makeGpuUint32Vector({4U, 0U, 2U}, stream);
    Tensor output = makeGpuFloatVector({9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f}, stream);

    auto plan = prepareFlatScatterAdd(updates, indices, output, FlatScatterAddIndexPolicy::IndicesAreUniqueAndValid);
    EXPECT_EQ(plan->index_policy, FlatScatterAddIndexPolicy::IndicesAreUniqueAndValid);
    EXPECT_EQ(plan->sort_temp_storage_bytes, 0U);
    EXPECT_EQ(plan->reduce_temp_storage_bytes, 0U);

    runFlatScatterAdd(plan, updates, indices, output, stream);

    expectFloatVectorEq(copyGpuFloatVector(output, stream), {-2.0f, 0.0f, 3.0f, 0.0f, 1.5f, 0.0f});
}

TEST(FlatScatterAddKernel, CanonicalizedPathSumsDuplicatesAndSkipsInvalidIndices) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor updates = makeGpuFloatVector({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, stream);
    Tensor indices = makeGpuUint32Vector({2U, std::numeric_limits<uint32_t>::max(), 2U, 6U, 0U}, stream);
    Tensor output = makeGpuFloatVector({9.0f, 9.0f, 9.0f, 9.0f, 9.0f}, stream);

    auto plan = prepareFlatScatterAdd(updates, indices, output);
    EXPECT_EQ(plan->index_policy, FlatScatterAddIndexPolicy::CanonicalizeDuplicatesAndSkipInvalid);
    EXPECT_GT(plan->sort_temp_storage_bytes, 0U);
    EXPECT_GT(plan->reduce_temp_storage_bytes, 0U);

    runFlatScatterAdd(plan, updates, indices, output, stream);

    expectFloatVectorEq(copyGpuFloatVector(output, stream), {5.0f, 0.0f, 4.0f, 0.0f, 0.0f});
}
