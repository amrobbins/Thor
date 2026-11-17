#include "test/Utilities/TensorOperations/Reduction/ReductionTestSupport.h"
#include "Utilities/Exceptions.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::ReductionTestSupport;

namespace {

bool axisIsReduced(const std::vector<uint32_t>& axes, uint32_t axis) {
    return std::binary_search(axes.begin(), axes.end(), axis);
}

std::vector<float> referenceDenseValueReduction(const std::vector<float>& values,
                                                const std::vector<uint64_t>& dimensions,
                                                const std::vector<uint32_t>& axes,
                                                ReductionOp op,
                                                float output_scale = 1.0f) {
    uint64_t output_elements = 1;
    uint64_t reduction_size = 1;
    for (uint32_t axis = 0; axis < dimensions.size(); ++axis) {
        if (axisIsReduced(axes, axis)) {
            reduction_size *= dimensions[axis];
        } else {
            output_elements *= dimensions[axis];
        }
    }

    float init = 0.0f;
    if (op == ReductionOp::Product) {
        init = 1.0f;
    } else if (op == ReductionOp::Min) {
        init = std::numeric_limits<float>::infinity();
    } else if (op == ReductionOp::Max) {
        init = -std::numeric_limits<float>::infinity();
    }
    std::vector<float> output(static_cast<size_t>(output_elements), init);

    for (uint64_t linear = 0; linear < values.size(); ++linear) {
        uint64_t remaining = linear;
        uint64_t output_index = 0;
        uint64_t output_stride = 1;
        for (size_t axis = dimensions.size(); axis-- > 0;) {
            const uint64_t coordinate = remaining % dimensions[axis];
            remaining /= dimensions[axis];
            if (!axisIsReduced(axes, static_cast<uint32_t>(axis))) {
                output_index += coordinate * output_stride;
                output_stride *= dimensions[axis];
            }
        }

        float value = values[static_cast<size_t>(linear)];
        if (op == ReductionOp::L1Norm) {
            value = std::fabs(value);
        } else if (op == ReductionOp::L2Norm || op == ReductionOp::SumSquares) {
            value *= value;
        }

        float& aggregate = output[static_cast<size_t>(output_index)];
        switch (op) {
            case ReductionOp::Sum:
            case ReductionOp::Mean:
            case ReductionOp::L1Norm:
            case ReductionOp::L2Norm:
            case ReductionOp::SumSquares:
                aggregate += value;
                break;
            case ReductionOp::Product:
                aggregate *= value;
                break;
            case ReductionOp::Min:
                aggregate = (std::isnan(aggregate) || std::isnan(value))
                                ? std::numeric_limits<float>::quiet_NaN()
                                : std::min(aggregate, value);
                break;
            case ReductionOp::Max:
                aggregate = (std::isnan(aggregate) || std::isnan(value))
                                ? std::numeric_limits<float>::quiet_NaN()
                                : std::max(aggregate, value);
                break;
        }
    }

    for (float& value : output) {
        if (op == ReductionOp::Mean) {
            value /= static_cast<float>(reduction_size);
        } else if (op == ReductionOp::L2Norm) {
            value = std::sqrt(value);
        }
        value *= output_scale;
    }
    return output;
}

std::vector<float> makeExactlyRepresentableComposedValues(size_t count) {
    constexpr float pattern[] = {-2.0f, -1.5f, -1.0f, -0.5f, 0.25f, 0.5f, 1.0f, 1.5f, 2.0f};
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = pattern[(i * 5 + i / 3) % (sizeof(pattern) / sizeof(pattern[0]))];
    }
    return values;
}

}  // namespace

TEST(Reducer, ComposedDenseSupportsEveryValueOperationAcrossStorageDtypes) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    const std::vector<uint64_t> dimensions{2, 3, 2, 2, 2};
    const std::vector<uint32_t> axes{0, 2, 4};
    const std::vector<float> values = makeExactlyRepresentableComposedValues(48);
    constexpr float runtime_scale = 1.25f;

    for (DataType input_dtype : {DataType::FP16, DataType::BF16, DataType::FP32}) {
        Tensor input = makeGpuTensor(values, dimensions, stream, input_dtype);
        for (ReductionOp op : {ReductionOp::Sum,
                               ReductionOp::Min,
                               ReductionOp::Max,
                               ReductionOp::Product,
                               ReductionOp::Mean,
                               ReductionOp::L1Norm,
                               ReductionOp::L2Norm,
                               ReductionOp::SumSquares}) {
            SCOPED_TRACE(static_cast<int>(input_dtype));
            SCOPED_TRACE(static_cast<int>(op));

            Reducer reduction(op, axes, DataType::FP32, 3.0f);
            const size_t queried_workspace = reduction.queryWorkspaceSizeInBytes(input.getDescriptor(), stream);
            std::shared_ptr<StampedReducer> stamped = reduction.stamp(input, stream);
            ASSERT_EQ(stamped->getPath(), ReductionLayout::DisjointRuns);
            ASSERT_TRUE(stamped->usesDenseExecutablePlan());
            EXPECT_GT(stamped->getDenseExecutablePassCount(), 0U);
            EXPECT_EQ(stamped->getWorkspaceSizeInBytes(), queried_workspace);
            ASSERT_EQ(stamped->getComposedStageAxes().size(), 3U);

            stamped->run(runtime_scale);
            stream.synchronize();
            const float tolerance = op == ReductionOp::Product ? 2.0e-4f : 1.0e-5f;
            expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream),
                                  referenceDenseValueReduction(values, dimensions, axes, op, runtime_scale),
                                  tolerance);
        }
    }
}

TEST(Reducer, SingletonOnlyCompositionUsesCompleteOperationSemantics) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    const std::vector<uint64_t> dimensions{2, 1, 3, 1, 2};
    const std::vector<uint32_t> axes{1, 3};
    const std::vector<float> values = {-2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f,
                                       8.0f, -9.0f, 10.0f, -11.0f, 12.0f, -13.0f};
    Tensor input = makeGpuTensor(values, dimensions, stream);

    for (ReductionOp op : {ReductionOp::Sum,
                           ReductionOp::Min,
                           ReductionOp::Max,
                           ReductionOp::Product,
                           ReductionOp::Mean,
                           ReductionOp::L1Norm,
                           ReductionOp::L2Norm,
                           ReductionOp::SumSquares}) {
        std::shared_ptr<StampedReducer> stamped = Reducer(op, axes, DataType::FP32).stamp(input, stream);
        ASSERT_EQ(stamped->getPath(), ReductionLayout::DisjointRuns);
        ASSERT_TRUE(stamped->usesDenseExecutablePlan());
        EXPECT_GT(stamped->getDenseExecutablePassCount(), 0U);
        ASSERT_EQ(stamped->getComposedStageAxes().size(), 1U);
        stamped->run();
        stream.synchronize();
        expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream),
                              referenceDenseValueReduction(values, dimensions, axes, op),
                              1.0e-5f);
    }
}

TEST(Reducer, ComposedMeanAndL2FinalizeOnceBeforeLowPrecisionStorage) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    const std::vector<uint64_t> dimensions{2, 2, 2};
    const std::vector<uint32_t> axes{0, 2};
    const std::vector<float> values{300.0f, 400.0f, 3.0f, 4.0f, 0.0f, 0.0f, 5.0f, 12.0f};
    Tensor input = makeGpuTensor(values, dimensions, stream);

    for (DataType output_dtype : {DataType::FP16, DataType::BF16}) {
        for (ReductionOp op : {ReductionOp::Mean, ReductionOp::L2Norm}) {
            SCOPED_TRACE(static_cast<int>(output_dtype));
            SCOPED_TRACE(static_cast<int>(op));
            std::shared_ptr<StampedReducer> stamped = Reducer(op, axes, output_dtype).stamp(input, stream);
            ASSERT_EQ(stamped->getPath(), ReductionLayout::DisjointRuns);
            stamped->run();
            stream.synchronize();
            expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream),
                                  referenceDenseValueReduction(values, dimensions, axes, op),
                                  output_dtype == DataType::FP16 ? 0.5f : 2.0f);
        }
    }
}

TEST(Reducer, ComposedMinMaxPreserveNanPropagation) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<uint64_t> dimensions{2, 2, 2};
    const std::vector<uint32_t> axes{0, 2};
    const std::vector<float> values{1.0f, nan, -1.0f, -2.0f, 3.0f, 4.0f, -3.0f, -4.0f};
    Tensor input = makeGpuTensor(values, dimensions, stream);

    for (ReductionOp op : {ReductionOp::Min, ReductionOp::Max}) {
        std::shared_ptr<StampedReducer> stamped = Reducer(op, axes, DataType::FP32).stamp(input, stream);
        ASSERT_EQ(stamped->getPath(), ReductionLayout::DisjointRuns);
        stamped->run();
        stream.synchronize();
        expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream),
                              referenceDenseValueReduction(values, dimensions, axes, op));
    }
}

TEST(Reducer, ComposedProductPreservesZeroSignAndInfinitySemantics) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const float infinity = std::numeric_limits<float>::infinity();
    const std::vector<uint64_t> dimensions{2, 4, 2};
    const std::vector<uint32_t> axes{0, 2};
    std::vector<float> values(16, 1.0f);

    const float groups[4][4] = {
        {0.0f, 2.0f, 3.0f, 4.0f},
        {-1.0f, 2.0f, -3.0f, 4.0f},
        {infinity, 2.0f, 3.0f, 4.0f},
        {-infinity, 2.0f, 3.0f, 4.0f},
    };
    for (uint64_t keep = 0; keep < 4; ++keep) {
        size_t group_index = 0;
        for (uint64_t outer = 0; outer < 2; ++outer) {
            for (uint64_t inner = 0; inner < 2; ++inner) {
                values[(outer * 4 + keep) * 2 + inner] = groups[keep][group_index++];
            }
        }
    }

    Tensor input = makeGpuTensor(values, dimensions, stream);
    std::shared_ptr<StampedReducer> stamped =
        Reducer(ReductionOp::Product, axes, DataType::FP32).stamp(input, stream);
    ASSERT_EQ(stamped->getPath(), ReductionLayout::DisjointRuns);
    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), {0.0f, 24.0f, infinity, -infinity});
}

TEST(Reducer, IrregularViewValueOperationsAreRejectedInsteadOfUsingStridedFallback) {
    const std::vector<uint64_t> dimensions{2, 3, 4};
    const std::vector<uint64_t> strides{20, 4, 1};
    const std::vector<uint32_t> axes{0, 2};
    EXPECT_THROW((void)Reducer::analyzeGeometry(dimensions, strides, axes), NotImplementedException);
    for (ReductionOp op : {ReductionOp::Sum,
                           ReductionOp::Min,
                           ReductionOp::Max,
                           ReductionOp::Product,
                           ReductionOp::Mean,
                           ReductionOp::L1Norm,
                           ReductionOp::L2Norm,
                           ReductionOp::SumSquares}) {
        EXPECT_THROW((void)Reducer::analyzeValueGeometry(op, dimensions, strides, axes),
                     NotImplementedException);
    }
}
