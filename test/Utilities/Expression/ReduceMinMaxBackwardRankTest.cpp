#include "Utilities/Expression/ReduceMinMaxBackwardKernel.h"
#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"

#include "gtest/gtest.h"

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionTestSupport;

TEST(ExpressionReduction, ReduceMinMaxBackwardSupportsRankBeyondEight) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<uint64_t> dimensions{2, 1, 2, 1, 2, 1, 2, 1, 2};
    Tensor grad_output = makeGpuTensor({3.0f, 5.0f}, {1, 1, 1, 1, 1, 1, 1, 1, 2}, stream);
    Tensor indices = makeGpuUnsignedTensor({15U, 15U}, {1, 1, 1, 1, 1, 1, 1, 1, 2}, stream);
    Tensor grad_input = makeGpuTensor(std::vector<float>(32, 0.0f), dimensions, stream);

    const ReduceMinMaxBackwardScatterPlan plan = prepareReduceMinMaxBackwardScatter(
        dimensions, {0, 2, 4, 6}, {}, grad_input.getPlacement(), stream);
    EXPECT_EQ(plan.input_rank, 9U);
    EXPECT_EQ(plan.reduction_rank, 4U);
    EXPECT_EQ(plan.output_numel, 2U);

    launchReduceMinMaxBackwardScatter(grad_output.getMemPtr(),
                                      static_cast<const uint32_t*>(indices.getMemPtr()),
                                      grad_input.getMemPtr(),
                                      plan,
                                      DataType::FP32,
                                      DataType::FP32,
                                      stream.getStream());
    stream.synchronize();

    std::vector<float> expected(32, 0.0f);
    expected[30] = 3.0f;
    expected[31] = 5.0f;
    expectFloatVectorNear(copyGpuTensorAsFloat(grad_input, stream), expected);
}
