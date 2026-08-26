#include "Utilities/Expression/AutoDiff.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/ExpressionDTypeResolution.h"
#include "Utilities/Expression/FusedEquation.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                                          \
    do {                                                                                                               \
        int cuda_device_count_for_test = 0;                                                                            \
        const cudaError_t cuda_status_for_test = cudaGetDeviceCount(&cuda_device_count_for_test);                      \
        if (cuda_status_for_test != cudaSuccess || cuda_device_count_for_test <= 0) {                                  \
            GTEST_SKIP() << "CUDA device is required for output materialization execution tests.";                    \
        }                                                                                                              \
    } while (false)

TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

Tensor makeInput(Stream& stream) {
    Tensor cpu(cpuPlacement, TensorDescriptor(DataType::FP32, {4}));
    auto* ptr = static_cast<float*>(cpu.getMemPtr());
    ptr[0] = 1.0f;
    ptr[1] = -2.0f;
    ptr[2] = 3.5f;
    ptr[3] = 7.0f;

    Tensor gpu(gpuPlacement, TensorDescriptor(DataType::FP32, {4}));
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();
    return gpu;
}

std::vector<float> copyToCpu(const Tensor& gpu, Stream& stream) {
    Tensor cpu(cpuPlacement, gpu.getDescriptor());
    cpu.copyFromAsync(gpu, stream);
    stream.synchronize();
    const auto* ptr = static_cast<const float*>(cpu.getMemPtr());
    return {ptr[0], ptr[1], ptr[2], ptr[3]};
}

void expectInputValues(const Tensor& tensor, Stream& stream) {
    const std::vector<float> actual = copyToCpu(tensor, stream);
    const std::vector<float> expected{1.0f, -2.0f, 3.5f, 7.0f};
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "index " << i;
    }
}

PhysicalOutputs duplicateInputOutputs(bool first_requires_distinct) {
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    PhysicalOutputs outputs = Expression::outputs({{"a", x}, {"b", x}}).physicalOutputs();
    if (first_requires_distinct) {
        outputs.outputs[0].materialization.require_distinct_storage = true;
    }
    return outputs;
}

PhysicalOutputs duplicateComputedOutputs(bool first_requires_distinct) {
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression y = x + Expression::constantScalar(1.0);
    PhysicalOutputs outputs = Expression::outputs({{"a", y}, {"b", y}}).physicalOutputs();
    if (first_requires_distinct) {
        outputs.outputs[0].materialization.require_distinct_storage = true;
    }
    return outputs;
}

}  // namespace

TEST(ExpressionOutputMaterialization, DuplicateLogicalOutputsMayAliasWithoutDistinctStorageContract) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeInput(stream);

    FusedEquation equation = FusedEquation::compile(duplicateInputOutputs(false), 0);
    StampedExecutionPlan plan = equation.stamp({{"x", input}}, stream);

    EXPECT_EQ(plan.output("a").getTensorId(), input.getTensorId());
    EXPECT_EQ(plan.output("b").getTensorId(), input.getTensorId());
    EXPECT_EQ(plan.output("a").getTensorId(), plan.output("b").getTensorId());
}

TEST(ExpressionOutputMaterialization, DistinctStorageContractAllocatesSeparateStampedOutputWithoutChangingValue) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeInput(stream);

    FusedEquation equation = FusedEquation::compile(duplicateInputOutputs(true), 0);
    StampedExecutionPlan plan = equation.stamp({{"x", input}}, stream);

    const uint64_t a_id_before_run = plan.output("a").getTensorId();
    const uint64_t b_id_before_run = plan.output("b").getTensorId();
    EXPECT_NE(a_id_before_run, 0U);
    EXPECT_NE(b_id_before_run, 0U);
    EXPECT_NE(a_id_before_run, b_id_before_run);
    EXPECT_NE(a_id_before_run, input.getTensorId());
    EXPECT_EQ(b_id_before_run, input.getTensorId());

    plan.run();
    stream.synchronize();

    EXPECT_EQ(plan.output("a").getTensorId(), a_id_before_run)
        << "distinct output storage must be allocated while stamping, not during execution";
    EXPECT_EQ(plan.output("b").getTensorId(), b_id_before_run);
    expectInputValues(plan.output("a"), stream);
    expectInputValues(plan.output("b"), stream);
}

TEST(ExpressionOutputMaterialization, DistinctCallerBuffersAreHonored) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeInput(stream);
    Tensor a_destination(gpuPlacement, TensorDescriptor(DataType::FP32, {4}));
    Tensor b_destination(gpuPlacement, TensorDescriptor(DataType::FP32, {4}));

    FusedEquation equation = FusedEquation::compile(duplicateInputOutputs(true), 0);
    StampedExecutionPlan plan = equation.stamp(
        {{"x", input}}, stream, {}, {{"a", a_destination}, {"b", b_destination}});

    EXPECT_EQ(plan.output("a").getTensorId(), a_destination.getTensorId());
    EXPECT_EQ(plan.output("b").getTensorId(), b_destination.getTensorId());
    EXPECT_NE(plan.output("a").getTensorId(), plan.output("b").getTensorId());

    plan.run();
    stream.synchronize();
    expectInputValues(plan.output("a"), stream);
    expectInputValues(plan.output("b"), stream);
}


TEST(ExpressionOutputMaterialization, PreallocatedDistinctOutputDoesNotForceSiblingToAliasDirectDestination) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeInput(stream);
    Tensor a_destination(gpuPlacement, TensorDescriptor(DataType::FP32, {4}));

    FusedEquation equation = FusedEquation::compile(duplicateComputedOutputs(true), 0);
    StampedExecutionPlan plan = equation.stamp({{"x", input}}, stream, {}, {{"a", a_destination}});

    EXPECT_EQ(plan.output("a").getTensorId(), a_destination.getTensorId());
    EXPECT_NE(plan.output("a").getTensorId(), plan.output("b").getTensorId())
        << "a direct producer write into a distinct preallocated output must not make an unpreallocated sibling alias it";

    const uint64_t b_id_before_run = plan.output("b").getTensorId();
    plan.run();
    stream.synchronize();
    EXPECT_EQ(plan.output("b").getTensorId(), b_id_before_run);

    const std::vector<float> a_values = copyToCpu(plan.output("a"), stream);
    const std::vector<float> b_values = copyToCpu(plan.output("b"), stream);
    const std::vector<float> expected{2.0f, -1.0f, 4.5f, 8.0f};
    ASSERT_EQ(a_values.size(), expected.size());
    ASSERT_EQ(b_values.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(a_values[i], expected[i]) << "a index " << i;
        EXPECT_FLOAT_EQ(b_values[i], expected[i]) << "b index " << i;
    }
}

TEST(ExpressionOutputMaterialization, AutoDiffDuplicateGradientsShareValueButReturnDistinctTensorAllocations) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs_value = makeInput(stream);
    Tensor rhs_value = makeInput(stream);
    Tensor dy = makeInput(stream);

    const Expression lhs = Expression::input("lhs", DataType::FP32, DataType::FP32);
    const Expression rhs = Expression::input("rhs", DataType::FP32, DataType::FP32);
    PhysicalOutputs forward = Expression::outputs({{"y", lhs + rhs}}).physicalOutputs();
    resolveOutputsDTypesInPlace(forward, {DataType::FP32, DataType::FP32});
    PhysicalOutputs backward = buildBackwardOutputs(
        forward,
        {"lhs", "rhs"},
        std::optional<std::string>{"dy"},
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"lhs", {4}},
            {"rhs", {4}},
        });

    ASSERT_EQ(backward.outputs.size(), 2u);
    ASSERT_EQ(backward.outputs[0].node_idx, backward.outputs[1].node_idx);
    ASSERT_FALSE(backward.outputs[0].materialization.require_distinct_storage);
    ASSERT_TRUE(backward.outputs[1].materialization.require_distinct_storage);

    FusedEquation equation = FusedEquation::compile(backward, 0);
    StampedExecutionPlan plan = equation.stamp(
        {{"lhs", lhs_value}, {"rhs", rhs_value}, {"dy", dy}}, stream);

    const uint64_t lhs_grad_id = plan.output("lhs_grad").getTensorId();
    const uint64_t rhs_grad_id = plan.output("rhs_grad").getTensorId();
    EXPECT_NE(lhs_grad_id, rhs_grad_id);

    plan.run();
    stream.synchronize();

    EXPECT_EQ(plan.output("lhs_grad").getTensorId(), lhs_grad_id);
    EXPECT_EQ(plan.output("rhs_grad").getTensorId(), rhs_grad_id);
    expectInputValues(plan.output("lhs_grad"), stream);
    expectInputValues(plan.output("rhs_grad"), stream);
}

TEST(ExpressionOutputMaterialization, AliasedCallerBuffersRejectDistinctStorageContract) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeInput(stream);
    Tensor shared_destination(gpuPlacement, TensorDescriptor(DataType::FP32, {4}));

    FusedEquation equation = FusedEquation::compile(duplicateInputOutputs(true), 0);
    EXPECT_THROW(
        (void)equation.stamp(
            {{"x", input}}, stream, {}, {{"a", shared_destination}, {"b", shared_destination}}),
        std::runtime_error);
}
