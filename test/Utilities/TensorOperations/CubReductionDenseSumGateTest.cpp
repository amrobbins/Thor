#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"

#include <array>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionTestSupport;

namespace {

[[nodiscard]] bool isOrdainedDenseSumPath(CubReductionPath path) {
    return path == CubReductionPath::DeviceTransformReduce || path == CubReductionPath::ContiguousFixedSegment
           || path == CubReductionPath::TiledFixedSegment || path == CubReductionPath::ComposedDense;
}

[[nodiscard]] std::vector<uint32_t> axesFromMask(uint32_t rank, uint32_t mask) {
    std::vector<uint32_t> axes;
    axes.reserve(rank);
    for (uint32_t axis = 0; axis < rank; ++axis) {
        if ((mask & (1U << axis)) != 0) {
            axes.push_back(axis);
        }
    }
    return axes;
}

[[nodiscard]] std::string denseGateContext(const std::vector<uint64_t>& dimensions,
                                           const std::vector<uint32_t>& axes) {
    std::ostringstream context;
    context << "dimensions=[";
    for (size_t i = 0; i < dimensions.size(); ++i) {
        if (i != 0) {
            context << ',';
        }
        context << dimensions[i];
    }
    context << "] axes=[";
    for (size_t i = 0; i < axes.size(); ++i) {
        if (i != 0) {
            context << ',';
        }
        context << axes[i];
    }
    context << ']';
    return context.str();
}

void expectEveryDenseSumMaskUsesOrdainedPath(const std::vector<uint64_t>& dimensions) {
    ASSERT_FALSE(dimensions.empty());
    ASSERT_LT(dimensions.size(), 32U);

    const uint32_t rank = static_cast<uint32_t>(dimensions.size());
    const uint32_t mask_count = 1U << rank;
    for (uint32_t mask = 1; mask < mask_count; ++mask) {
        const std::vector<uint32_t> axes = axesFromMask(rank, mask);
        SCOPED_TRACE(denseGateContext(dimensions, axes));
        const CubReductionGeometry geometry =
            CubReduction::analyzeValueGeometry(CubReductionOp::Sum, dimensions, axes);
        EXPECT_TRUE(isOrdainedDenseSumPath(geometry.path));
        EXPECT_NE(geometry.path, CubReductionPath::StridedFixedSegment);
    }
}

}  // namespace

TEST(CubReductionDenseSumGate, EveryNonemptyMaskAcrossRepresentativeDenseRanksAvoidsStridedFallback) {
    // Exhaust every reduction mask rather than sampling named topologies. Increasing ranks naturally cover prefix,
    // suffix, middle, alternating R/K runs, leading/trailing retained runs, and many-run compositions.
    const std::vector<std::vector<uint64_t>> dense_shapes = {
        {2},
        {2, 3},
        {2, 3, 5},
        {2, 3, 2, 5},
        {2, 3, 2, 5, 3},
        {2, 3, 2, 5, 3, 2},
        {2, 3, 2, 5, 3, 2, 3},
        {2, 3, 2, 5, 3, 2, 3, 2},
        {2, 3, 2, 5, 3, 2, 3, 2, 5},
    };

    for (const std::vector<uint64_t>& dimensions : dense_shapes) {
        expectEveryDenseSumMaskUsesOrdainedPath(dimensions);
    }
}

TEST(CubReductionDenseSumGate, SingletonSeparatedDenseRanksAlsoAvoidStridedFallbackForEveryMask) {
    // Retained and reduced singleton dimensions must not manufacture an artificial strided geometry. This also covers
    // reductions made entirely of singleton axes, which still need one direct/composed pass for conversion/scaling.
    expectEveryDenseSumMaskUsesOrdainedPath({2, 1, 3, 1, 5, 1, 2, 1, 3});
    expectEveryDenseSumMaskUsesOrdainedPath({1, 2, 1, 3, 1, 5, 1, 2, 1});
}

TEST(CubReductionDenseSumGate, GenuineIrregularViewsRemainOutsideTheDenseGate) {
    // DENSE-SUM-GATE is specifically an ordinary-dense invariant. A gapped view still belongs to the arbitrary-view
    // fallback until VIEW-1 replaces that implementation.
    const CubReductionGeometry irregular = CubReduction::analyzeValueGeometry(
        CubReductionOp::Sum, {2, 3, 4}, {20, 4, 1}, std::vector<uint32_t>{0, 2});
    EXPECT_EQ(irregular.path, CubReductionPath::StridedFixedSegment);
    EXPECT_FALSE(irregular.dense_run_geometry.has_value());
}

TEST(CubReductionDenseSumGate, ComposedDenseExecutesForEveryFloatingInputDTypeWithPreallocatedScaledOutput) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    const std::vector<uint64_t> dimensions{2, 2, 2, 2, 2, 2, 2};
    const std::vector<uint32_t> axes{0, 2, 4, 6};
    const std::vector<float> values(128, 1.0f);
    const std::vector<float> expected(8, 32.0f);  // 16 reduced ones, then runtime scale 2.

    for (DataType dtype : std::array<DataType, 3>{DataType::FP16, DataType::BF16, DataType::FP32}) {
        SCOPED_TRACE(static_cast<int>(dtype));
        Tensor input = makeGpuTensor(values, dimensions, stream, dtype);

        // [2,1,2,2] is singleton-equivalent to the squeezed retained shape [2,2,2]. This simultaneously gates
        // preallocated output validation, squeezed-equivalent layout acceptance, dtype conversion, and runtime scale.
        Tensor output(gpuPlacement, TensorDescriptor(dtype, {2, 1, 2, 2}));
        CubReduction reduction(CubReductionOp::Sum, axes, dtype, 3.0f);
        const size_t queried_workspace = reduction.queryWorkspaceSizeInBytes(input.getDescriptor(), stream);
        auto stamped = reduction.stamp(input, output, stream);

        ASSERT_EQ(stamped->getPath(), CubReductionPath::ComposedDense);
        EXPECT_NE(stamped->getPath(), CubReductionPath::StridedFixedSegment);
        EXPECT_EQ(stamped->getWorkspaceSizeInBytes(), queried_workspace);
        EXPECT_GT(stamped->getWorkspaceSizeInBytes(), 0U);

        // The runtime scale overrides the configured scale and must be applied exactly once, after all FP32 partials.
        stamped->run(2.0f);
        stream.synchronize();
        expectFloatVectorNear(copyGpuTensorAsFloat(output, stream), expected);
    }
}
