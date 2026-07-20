#include "Utilities/Expression/CompiledEquation.h"
#include "Utilities/Expression/StampedEquation.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <memory>
#include <utility>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                                          \
    do {                                                                                                                \
        int cuda_device_count_for_test = 0;                                                                             \
        const cudaError_t cuda_status_for_test = cudaGetDeviceCount(&cuda_device_count_for_test);                       \
        if (cuda_status_for_test != cudaSuccess || cuda_device_count_for_test <= 0) {                                    \
            GTEST_SKIP() << "CUDA device is required for reduction plan tests.";                                     \
        }                                                                                                               \
    } while (false)

struct ReductionCase {
    ExprOp expression_op;
    CubReductionOp cub_op;
};

const std::vector<ReductionCase> valueReductionCases = {
    {ExprOp::REDUCE_SUM, CubReductionOp::Sum},       {ExprOp::REDUCE_PROD, CubReductionOp::Product},
    {ExprOp::REDUCE_MIN, CubReductionOp::Min},       {ExprOp::REDUCE_MAX, CubReductionOp::Max},
    {ExprOp::REDUCE_AVG, CubReductionOp::Mean},      {ExprOp::REDUCE_NORM1, CubReductionOp::L1Norm},
    {ExprOp::REDUCE_NORM2, CubReductionOp::L2Norm},
};

}  // namespace

TEST(ExpressionReductionPlan, EveryDenseValueReductionBuildsACachedPlan) {
    REQUIRE_CUDA_DEVICE();

    TensorPlacement gpu_placement(TensorPlacement::MemDevices::GPU, 0);
    Tensor input(gpu_placement, TensorDescriptor(DataType::FP32, {2, 3, 4}));

    for (const ReductionCase& test_case : valueReductionCases) {
        auto compiled = std::make_shared<CompiledReduction>(test_case.expression_op,
                                                            std::vector<uint64_t>{1},
                                                            std::vector<uint64_t>{},
                                                            DataType::FP32,
                                                            DataType::FP32,
                                                            DataType::FP32);

        std::shared_ptr<BuiltReduction> built = StampedEquation::buildReduction(compiled, input, 0);
        ASSERT_NE(built, nullptr);
        EXPECT_EQ(built->key.result_kind, ReductionResultKind::Value);
        ASSERT_TRUE(built->value_op.has_value());
        EXPECT_EQ(built->value_op.value(), test_case.cub_op);
        ASSERT_TRUE(built->geometry.has_value());
        EXPECT_EQ(built->geometry->path, CubReductionPath::StridedFixedSegment);

        std::shared_ptr<BuiltReduction> cached = StampedEquation::buildReduction(compiled, input, 0);
        EXPECT_EQ(cached.get(), built.get());
    }
}

TEST(ExpressionReductionPlan, ValueReductionPreservesBf16InputStorageAndFp32OutputContract) {
    REQUIRE_CUDA_DEVICE();

    TensorPlacement gpu_placement(TensorPlacement::MemDevices::GPU, 0);
    Tensor input(gpu_placement, TensorDescriptor(DataType::BF16, {3, 5}));
    auto compiled = std::make_shared<CompiledReduction>(ExprOp::REDUCE_SUM,
                                                        std::vector<uint64_t>{1},
                                                        std::vector<uint64_t>{1},
                                                        DataType::BF16,
                                                        DataType::FP32,
                                                        DataType::FP32);

    std::shared_ptr<BuiltReduction> built = StampedEquation::buildReduction(compiled, input, 0);
    ASSERT_NE(built, nullptr);
    EXPECT_EQ(built->key.result_kind, ReductionResultKind::Value);
    EXPECT_EQ(built->key.input_dtype, DataType::BF16);
    EXPECT_EQ(built->key.compute_dtype, DataType::FP32);
    EXPECT_EQ(built->key.output_dtype, DataType::FP32);
    ASSERT_TRUE(built->geometry.has_value());
    EXPECT_EQ(built->geometry->output_dimensions, (std::vector<uint64_t>{3, 1}));
}

TEST(ExpressionReductionPlan, EveryDenseArgReductionBuildsACachedIndexPlan) {
    REQUIRE_CUDA_DEVICE();

    TensorPlacement gpu_placement(TensorPlacement::MemDevices::GPU, 0);
    Tensor input(gpu_placement, TensorDescriptor(DataType::BF16, {2, 3, 4}));

    struct ArgCase {
        ExprOp expression_op;
        CubArgReductionOp cub_op;
    };
    const std::vector<ArgCase> cases = {
        {ExprOp::REDUCE_ARGMIN, CubArgReductionOp::ArgMin},
        {ExprOp::REDUCE_ARGMAX, CubArgReductionOp::ArgMax},
    };

    for (const ArgCase& test_case : cases) {
        std::shared_ptr<BuiltReduction> built = StampedEquation::buildReduction(test_case.expression_op,
                                                                                {1},
                                                                                {1},
                                                                                DataType::BF16,
                                                                                DataType::FP32,
                                                                                DataType::FP32,
                                                                                ReductionResultKind::Indices,
                                                                                input,
                                                                                0);
        ASSERT_NE(built, nullptr);
        EXPECT_EQ(built->key.result_kind, ReductionResultKind::Indices);
        EXPECT_FALSE(built->value_op.has_value());
        ASSERT_TRUE(built->arg_op.has_value());
        EXPECT_EQ(built->arg_op.value(), test_case.cub_op);
        ASSERT_TRUE(built->geometry.has_value());
        EXPECT_EQ(built->geometry->path, CubReductionPath::StridedFixedSegment);
        EXPECT_EQ(built->key.input_dtype, DataType::BF16);
        EXPECT_EQ(built->key.compute_dtype, DataType::FP32);

        std::shared_ptr<BuiltReduction> cached = StampedEquation::buildReduction(test_case.expression_op,
                                                                                 {1},
                                                                                 {1},
                                                                                 DataType::BF16,
                                                                                 DataType::FP32,
                                                                                 DataType::FP32,
                                                                                 ReductionResultKind::Indices,
                                                                                 input,
                                                                                 0);
        EXPECT_EQ(cached.get(), built.get());
    }
}

TEST(ExpressionReductionPlan, ResultKindSeparatesValueAndIndexCacheEntries) {
    REQUIRE_CUDA_DEVICE();

    TensorPlacement gpu_placement(TensorPlacement::MemDevices::GPU, 0);
    Tensor input(gpu_placement, TensorDescriptor(DataType::FP32, {2, 3}));

    auto compiled_value = std::make_shared<CompiledReduction>(ExprOp::REDUCE_MIN,
                                                              std::vector<uint64_t>{1},
                                                              std::vector<uint64_t>{1},
                                                              DataType::FP32,
                                                              DataType::FP32,
                                                              DataType::FP32);
    std::shared_ptr<BuiltReduction> value_plan = StampedEquation::buildReduction(compiled_value, input, 0);
    std::shared_ptr<BuiltReduction> index_plan = StampedEquation::buildReduction(ExprOp::REDUCE_MIN,
                                                                                 {1},
                                                                                 {1},
                                                                                 DataType::FP32,
                                                                                 DataType::FP32,
                                                                                 DataType::FP32,
                                                                                 ReductionResultKind::Indices,
                                                                                 input,
                                                                                 0);

    ASSERT_NE(value_plan, nullptr);
    ASSERT_NE(index_plan, nullptr);
    EXPECT_NE(value_plan.get(), index_plan.get());
    EXPECT_EQ(value_plan->key.result_kind, ReductionResultKind::Value);
    EXPECT_EQ(index_plan->key.result_kind, ReductionResultKind::Indices);
    EXPECT_TRUE(value_plan->value_op.has_value());
    EXPECT_FALSE(value_plan->arg_op.has_value());
    EXPECT_FALSE(index_plan->value_op.has_value());
    EXPECT_TRUE(index_plan->arg_op.has_value());
}

TEST(ExpressionReductionPlan, ReduceMinMaxBackwardWinnerPlansProduceIndices) {
    REQUIRE_CUDA_DEVICE();

    TensorPlacement gpu_placement(TensorPlacement::MemDevices::GPU, 0);
    Tensor input(gpu_placement, TensorDescriptor(DataType::FP8_E4M3, {2, 3, 4}));

    for (const ExprOp op : {ExprOp::REDUCE_MIN, ExprOp::REDUCE_MAX}) {
        std::shared_ptr<BuiltReduction> built = StampedEquation::buildReduction(op,
                                                                                {1, 2},
                                                                                {1, 2},
                                                                                DataType::FP8_E4M3,
                                                                                DataType::FP32,
                                                                                DataType::FP32,
                                                                                ReductionResultKind::Indices,
                                                                                input,
                                                                                0);
        ASSERT_NE(built, nullptr);
        EXPECT_EQ(built->key.result_kind, ReductionResultKind::Indices);
        ASSERT_TRUE(built->arg_op.has_value());
        EXPECT_EQ(built->arg_op.value(),
                  op == ExprOp::REDUCE_MIN ? CubArgReductionOp::ArgMin : CubArgReductionOp::ArgMax);
        ASSERT_TRUE(built->geometry.has_value());
        EXPECT_EQ(built->geometry->path, CubReductionPath::ContiguousFixedSegment);
        EXPECT_EQ(built->key.input_dtype, DataType::FP8_E4M3);
    }
}
