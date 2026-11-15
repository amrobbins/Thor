// 1335: StampedCubReduction::runOnTensorsForBenchmark on stamps of non-dense views (the view audit,
// benchmarks/ViewReductionAudit.cu; plans/Thor_View_Reduction_Plan_2026-11-08.md phase 0). A view stamp takes a
// replacement view of the same dtype, device, dims and strides, computes from the replacement's address, and rejects a
// dense replacement or other strides. Dense stamps keep taking dense replacements (the other benchmark-hook tests).
// 1365: the permuted full reduction runs the dense planner on the view's bytes (phase 1), and its benchmark run reads
// the replacement view as dense. 1366: the strided rank-1 view runs StridedR (phase 2) on the replacement's address.
#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"

#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionTestSupport;

namespace {

std::vector<float> iota(size_t count, float first) {
    std::vector<float> values(count);
    std::iota(values.begin(), values.end(), first);
    return values;
}

}  // namespace

TEST(CubReductionViewBenchmarkHook, ViewStampsRunOnReplacementViewsOfTheSameStrides) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    // A: a transposed [4, 6] (a compact permutation), every axis reduced.
    const Tensor a_storage = makeGpuTensor(iota(24, 0.0f), {24}, stream);
    const Tensor a_other = makeGpuTensor(iota(24, 100.0f), {24}, stream);
    const StridedSpan a_view = StridedSpan(a_storage, 0, {6, 4}, {1, 6});
    const StridedSpan a_replacement = StridedSpan(a_other, 0, {6, 4}, {1, 6});
    // C: a rank-1 view of 5 elements at stride 3.
    const Tensor c_storage = makeGpuTensor(iota(13, 1.0f), {13}, stream);
    const Tensor c_other = makeGpuTensor(iota(13, 10.0f), {13}, stream);
    const StridedSpan c_view = StridedSpan(c_storage, 0, {5}, {3});
    const StridedSpan c_replacement = StridedSpan(c_other, 0, {5}, {3});
    stream.synchronize();

    struct Case {
        const char* name;
        StridedSpan view, replacement;
        std::vector<uint32_t> axes;
        float bound_sum, replacement_sum;
        bool view_physical;  // 1365: runs the retained dense plan on the view's bytes
    };
    const std::vector<Case> cases = {
        {"permuted full reduction", a_view, a_replacement, {0, 1}, 276.0f, 2676.0f, true},
        {"strided rank-1", c_view, c_replacement, {0}, 1 + 4 + 7 + 10 + 13, 10 + 13 + 16 + 19 + 22, false},
    };
    for (const Case& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        ASSERT_FALSE(test_case.view.isDenseContiguous());
        auto stamped = CubReduction(CubReductionOp::Sum, test_case.axes, DataType::FP32).stamp(test_case.view, stream);
        ASSERT_EQ(stamped->usesDenseExecutablePlan(), test_case.view_physical);
        ASSERT_EQ(stamped->getStridedRLaunch().has_value(), !test_case.view_physical);  // 1366
        Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, stamped->getOutputTensor().getDimensions()));
        stamped->runOnTensorsForBenchmark(test_case.replacement, output, stream);
        EXPECT_EQ(copyGpuTensorAsFloat(output, stream), std::vector<float>{test_case.replacement_sum});
        stamped->run();
        EXPECT_EQ(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), std::vector<float>{test_case.bound_sum});

        // A dense replacement of the same element count, or a view of other strides, is refused.
        const Tensor dense(gpuPlacement, TensorDescriptor(DataType::FP32, test_case.view.getDimensions()));
        EXPECT_THROW(stamped->runOnTensorsForBenchmark(dense, output, stream), std::invalid_argument);
    }
    auto stamped = CubReduction(CubReductionOp::Sum, {0}, DataType::FP32).stamp(c_view, stream);
    Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, stamped->getOutputTensor().getDimensions()));
    EXPECT_THROW(stamped->runOnTensorsForBenchmark(StridedSpan(c_other, 0, {5}, {2}), output, stream),
                 std::invalid_argument);
}
