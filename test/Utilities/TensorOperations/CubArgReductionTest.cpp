#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionTestSupport;

namespace {

void expectArgOutputs(const std::shared_ptr<StampedCubArgReduction>& stamped,
                      const std::vector<float>& expected_values,
                      const std::vector<uint64_t>& expected_indices,
                      Stream& stream,
                      float tolerance = 0.0f) {
    ASSERT_TRUE(stamped->getValueOutputTensor().has_value());
    ASSERT_TRUE(stamped->getIndexOutputTensor().has_value());
    expectFloatVectorNear(
        copyGpuTensorAsFloat(stamped->getValueOutputTensor().value(), stream), expected_values, tolerance);
    EXPECT_EQ(copyGpuTensorAsUnsigned(stamped->getIndexOutputTensor().value(), stream), expected_indices);
}

CubArgReductionOutputOptions fp32ValueAndUint32Index() {
    CubArgReductionOutputOptions outputs;
    outputs.value_output_dtype = DataType::FP32;
    return outputs;
}

struct DenseArgReference {
    std::vector<float> values;
    std::vector<uint64_t> indices;
};

DenseArgReference referenceDenseMiddleAxis(const std::vector<float>& input,
                                           uint64_t outer_size,
                                           uint64_t reduction_size,
                                           uint64_t inner_size,
                                           CubArgReductionOp op) {
    DenseArgReference reference;
    reference.values.resize(outer_size * inner_size);
    reference.indices.resize(outer_size * inner_size);

    for (uint64_t outer = 0; outer < outer_size; ++outer) {
        for (uint64_t component = 0; component < inner_size; ++component) {
            float best_value = op == CubArgReductionOp::ArgMin ? std::numeric_limits<float>::infinity()
                                                               : -std::numeric_limits<float>::infinity();
            uint64_t best_index = std::numeric_limits<uint64_t>::max();
            for (uint64_t row = 0; row < reduction_size; ++row) {
                const float candidate = input[(outer * reduction_size + row) * inner_size + component];
                const bool best_nan = std::isnan(best_value);
                const bool candidate_nan = std::isnan(candidate);
                bool take_candidate = false;
                if (best_nan || candidate_nan) {
                    if (best_nan && candidate_nan) {
                        take_candidate = row < best_index;
                    } else {
                        take_candidate = candidate_nan;
                    }
                } else if (op == CubArgReductionOp::ArgMin) {
                    take_candidate = candidate < best_value || (candidate == best_value && row < best_index);
                } else {
                    take_candidate = candidate > best_value || (candidate == best_value && row < best_index);
                }
                if (take_candidate) {
                    best_value = candidate;
                    best_index = row;
                }
            }
            const uint64_t output_index = outer * inner_size + component;
            reference.values[output_index] = best_value;
            reference.indices[output_index] = best_index;
        }
    }
    return reference;
}

}  // namespace


TEST(CubArgReduction, DefinesExplicitEmptyDomainSentinels) {
    EXPECT_EQ(CubArgReduction::getFp32EmptyReductionValue(CubArgReductionOp::ArgMin),
              std::numeric_limits<float>::infinity());
    EXPECT_EQ(CubArgReduction::getFp32EmptyReductionValue(CubArgReductionOp::ArgMax),
              -std::numeric_limits<float>::infinity());
    EXPECT_EQ(CubArgReduction::getEmptyReductionIndex(), std::numeric_limits<uint64_t>::max());
}

TEST(CubArgReduction, DeviceWideContiguousAndLogicalIndexPathsProduceLocalIndices) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor device_input = makeGpuTensor({5.0f, -2.0f, 7.0f, -2.0f, 4.0f, 7.0f}, {2, 3}, stream);
    std::shared_ptr<StampedCubArgReduction> device_min =
        CubArgReduction(CubArgReductionOp::ArgMin, std::vector<uint32_t>{0, 1}, fp32ValueAndUint32Index())
            .stamp(device_input, stream);
    std::shared_ptr<StampedCubArgReduction> device_max =
        CubArgReduction(CubArgReductionOp::ArgMax, std::vector<uint32_t>{0, 1}, fp32ValueAndUint32Index())
            .stamp(device_input, stream);
    EXPECT_EQ(device_min->getPath(), CubReductionPath::DeviceTransformReduce);
    device_min->run();
    device_max->run();
    stream.synchronize();
    expectArgOutputs(device_min, {-2.0f}, {1}, stream);
    expectArgOutputs(device_max, {7.0f}, {2}, stream);

    Tensor contiguous_input =
        makeGpuTensor({3.0f, 1.0f, 1.0f, 2.0f, -5.0f, -1.0f, -5.0f, -2.0f}, {2, 4}, stream);
    std::shared_ptr<StampedCubArgReduction> contiguous_min =
        CubArgReduction(CubArgReductionOp::ArgMin, 1, fp32ValueAndUint32Index()).stamp(contiguous_input, stream);
    std::shared_ptr<StampedCubArgReduction> contiguous_max =
        CubArgReduction(CubArgReductionOp::ArgMax, 1, fp32ValueAndUint32Index()).stamp(contiguous_input, stream);
    EXPECT_EQ(contiguous_min->getPath(), CubReductionPath::ContiguousFixedSegment);
    contiguous_min->run();
    contiguous_max->run();
    stream.synchronize();
    expectArgOutputs(contiguous_min, {1.0f, -5.0f}, {1, 0}, stream);
    expectArgOutputs(contiguous_max, {3.0f, -1.0f}, {0, 1}, stream);

    Tensor strided_input = makeGpuTensor({9.0f, 1.0f, 5.0f, 8.0f, -1.0f, 4.0f,
                                          0.0f, 7.0f, 6.0f, 2.0f, 3.0f, -2.0f},
                                         {2, 3, 2},
                                         stream);
    std::shared_ptr<StampedCubArgReduction> middle_min =
        CubArgReduction(CubArgReductionOp::ArgMin, 1, fp32ValueAndUint32Index()).stamp(strided_input, stream);
    std::shared_ptr<StampedCubArgReduction> middle_max =
        CubArgReduction(CubArgReductionOp::ArgMax, 1, fp32ValueAndUint32Index()).stamp(strided_input, stream);
    EXPECT_EQ(middle_min->getPath(), CubReductionPath::TiledFixedSegment);
    middle_min->run();
    middle_max->run();
    stream.synchronize();
    expectArgOutputs(middle_min, {-1.0f, 1.0f, 0.0f, -2.0f}, {2, 0, 0, 2}, stream);
    expectArgOutputs(middle_max, {9.0f, 8.0f, 6.0f, 7.0f}, {0, 1, 1, 0}, stream);

    std::shared_ptr<StampedCubArgReduction> strided_min =
        CubArgReduction(CubArgReductionOp::ArgMin, std::vector<uint32_t>{0, 2}, fp32ValueAndUint32Index())
            .stamp(strided_input, stream);
    std::shared_ptr<StampedCubArgReduction> strided_max =
        CubArgReduction(CubArgReductionOp::ArgMax, std::vector<uint32_t>{0, 2}, fp32ValueAndUint32Index())
            .stamp(strided_input, stream);
    EXPECT_EQ(strided_min->getPath(), CubReductionPath::StridedFixedSegment);
    strided_min->run();
    strided_max->run();
    stream.synchronize();
    expectArgOutputs(strided_min, {0.0f, 2.0f, -2.0f}, {2, 3, 3}, stream);
    expectArgOutputs(strided_max, {9.0f, 8.0f, 4.0f}, {0, 1, 1}, stream);
}

TEST(CubArgReduction, PropagatesNaNsAndChoosesLowestIndexForEveryTie) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    Tensor input = makeGpuTensor({3.0f, nan, 2.0f, nan, 5.0f, 5.0f, 1.0f, 1.0f}, {2, 4}, stream);

    std::shared_ptr<StampedCubArgReduction> minimum =
        CubArgReduction(CubArgReductionOp::ArgMin, 1, fp32ValueAndUint32Index()).stamp(input, stream);
    std::shared_ptr<StampedCubArgReduction> maximum =
        CubArgReduction(CubArgReductionOp::ArgMax, 1, fp32ValueAndUint32Index()).stamp(input, stream);
    minimum->run();
    maximum->run();
    stream.synchronize();

    expectArgOutputs(minimum, {nan, 1.0f}, {1, 2}, stream);
    expectArgOutputs(maximum, {nan, 5.0f}, {1, 0}, stream);
}

TEST(CubArgReduction, InfiniteExtremaPreferRealInputOverTheEmptySentinel) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const float infinity = std::numeric_limits<float>::infinity();
    Tensor input = makeGpuTensor({infinity, infinity, -infinity, -infinity}, {2, 2}, stream);

    std::shared_ptr<StampedCubArgReduction> minimum =
        CubArgReduction(CubArgReductionOp::ArgMin, 1, fp32ValueAndUint32Index()).stamp(input, stream);
    std::shared_ptr<StampedCubArgReduction> maximum =
        CubArgReduction(CubArgReductionOp::ArgMax, 1, fp32ValueAndUint32Index()).stamp(input, stream);
    minimum->run();
    maximum->run();
    stream.synchronize();

    expectArgOutputs(minimum, {infinity, -infinity}, {0, 0}, stream);
    expectArgOutputs(maximum, {infinity, -infinity}, {0, 0}, stream);
}

TEST(CubArgReduction, SupportsValueOnlyIndexOnlyAndCombinedOutputs) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, 4.0f, 2.0f, -3.0f, -1.0f, -2.0f}, {2, 3}, stream, DataType::BF16);

    CubArgReductionOutputOptions combined_options;
    std::shared_ptr<StampedCubArgReduction> combined =
        CubArgReduction(CubArgReductionOp::ArgMax, 1, combined_options).stamp(input, stream);
    EXPECT_EQ(combined->getValueAccumulatorDataType(), DataType::FP32);
    ASSERT_TRUE(combined->getValueOutputTensor().has_value());
    ASSERT_TRUE(combined->getIndexOutputTensor().has_value());
    EXPECT_EQ(combined->getValueOutputTensor()->getDataType(), DataType::BF16);
    EXPECT_EQ(combined->getIndexOutputTensor()->getDataType(), DataType::UINT32);

    CubArgReductionOutputOptions value_only_options;
    value_only_options.produce_index = false;
    value_only_options.value_output_dtype = DataType::FP32;
    std::shared_ptr<StampedCubArgReduction> value_only =
        CubArgReduction(CubArgReductionOp::ArgMin, 1, value_only_options).stamp(input, stream);

    CubArgReductionOutputOptions index_only_options;
    index_only_options.produce_value = false;
    index_only_options.index_output_dtype = DataType::UINT64;
    std::shared_ptr<StampedCubArgReduction> index_only =
        CubArgReduction(CubArgReductionOp::ArgMin, 1, index_only_options).stamp(input, stream);

    combined->run();
    value_only->run();
    index_only->run();
    stream.synchronize();

    expectFloatVectorNear(copyGpuTensorAsFloat(combined->getValueOutputTensor().value(), stream), {4.0f, -1.0f});
    EXPECT_EQ(copyGpuTensorAsUnsigned(combined->getIndexOutputTensor().value(), stream), (std::vector<uint64_t>{1, 1}));
    ASSERT_TRUE(value_only->getValueOutputTensor().has_value());
    EXPECT_FALSE(value_only->getIndexOutputTensor().has_value());
    expectFloatVectorNear(copyGpuTensorAsFloat(value_only->getValueOutputTensor().value(), stream), {1.0f, -3.0f});
    EXPECT_FALSE(index_only->getValueOutputTensor().has_value());
    ASSERT_TRUE(index_only->getIndexOutputTensor().has_value());
    EXPECT_EQ(index_only->getIndexOutputTensor()->getDataType(), DataType::UINT64);
    EXPECT_EQ(copyGpuTensorAsUnsigned(index_only->getIndexOutputTensor().value(), stream), (std::vector<uint64_t>{0, 0}));
}

TEST(CubArgReduction, EverySupportedFloatingInputDtypeUsesFp32CandidateValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    std::vector<DataType> input_dtypes = {DataType::FP16, DataType::BF16, DataType::FP32};
#if THOR_CUB_ENABLE_FP8_TYPES
    input_dtypes.push_back(DataType::FP8_E4M3);
    input_dtypes.push_back(DataType::FP8_E5M2);
#endif
#if THOR_CUB_ENABLE_64BIT_TYPES
    input_dtypes.push_back(DataType::FP64);
#endif

    for (DataType input_dtype : input_dtypes) {
        SCOPED_TRACE(static_cast<int>(input_dtype));
        Tensor input = makeGpuTensor({1.0f, 4.0f, 2.0f}, {1, 3}, stream, input_dtype);
        std::shared_ptr<StampedCubArgReduction> stamped =
            CubArgReduction(CubArgReductionOp::ArgMax, 1, fp32ValueAndUint32Index()).stamp(input, stream);
        EXPECT_EQ(stamped->getInputDataType(), input_dtype);
        EXPECT_EQ(stamped->getValueAccumulatorDataType(), DataType::FP32);
        stamped->run();
        stream.synchronize();
        expectArgOutputs(stamped, {4.0f}, {1}, stream);
    }
}

TEST(CubArgReduction, ReusesPreallocatedOutputsAndWorkspaceAcrossExecutions) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, 3.0f, 2.0f, 4.0f, 0.0f, 5.0f}, {2, 3}, stream);
    Tensor value_output(gpuPlacement, TensorDescriptor(DataType::FP32, {2}));
    Tensor index_output(gpuPlacement, TensorDescriptor(DataType::UINT32, {2, 1}));

    std::shared_ptr<StampedCubArgReduction> stamped =
        CubArgReduction(CubArgReductionOp::ArgMax, 1, fp32ValueAndUint32Index())
            .stamp(input, value_output, index_output, stream);
    const uint64_t value_id = value_output.getTensorId();
    const uint64_t index_id = index_output.getTensorId();
    const size_t workspace_bytes = stamped->getWorkspaceSizeInBytes();

    stamped->run();
    stream.synchronize();
    expectArgOutputs(stamped, {3.0f, 5.0f}, {1, 2}, stream);

    Tensor replacement = makeGpuTensor({9.0f, 8.0f, 7.0f, -1.0f, -2.0f, -3.0f}, {2, 3}, stream);
    input.copyFromAsync(replacement, stream);
    stamped->run();
    stream.synchronize();

    EXPECT_EQ(stamped->getValueOutputTensor()->getTensorId(), value_id);
    EXPECT_EQ(stamped->getIndexOutputTensor()->getTensorId(), index_id);
    EXPECT_EQ(stamped->getWorkspaceSizeInBytes(), workspace_bytes);
    expectArgOutputs(stamped, {9.0f, -1.0f}, {0, 0}, stream);
}


TEST(CubArgReduction, DenseTiledPathCoversNarrowExactGroupedAndShardedWidths) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    const std::vector<uint64_t> inner_sizes = {2,    17,   64,   128,  256,  512,  513,   1024,
                                                2048, 2049, 4096, 4097, 8192, 8193, 16385};
    constexpr uint64_t outer_size = 2;
    constexpr uint64_t reduction_size = 5;

    for (uint64_t inner_size : inner_sizes) {
        SCOPED_TRACE(inner_size);
        std::vector<float> values(outer_size * reduction_size * inner_size);
        for (uint64_t outer = 0; outer < outer_size; ++outer) {
            for (uint64_t row = 0; row < reduction_size; ++row) {
                for (uint64_t component = 0; component < inner_size; ++component) {
                    values[(outer * reduction_size + row) * inner_size + component] =
                        static_cast<float>((outer * 29 + row * 17 + component * 7 + row * component) % 113) - 56.0f;
                }
            }
        }

        Tensor input = makeGpuTensor(values, {outer_size, reduction_size, inner_size}, stream);
        for (CubArgReductionOp op : {CubArgReductionOp::ArgMin, CubArgReductionOp::ArgMax}) {
            CubArgReductionOutputOptions outputs;
            outputs.value_output_dtype = DataType::FP32;
            outputs.index_output_dtype = DataType::UINT64;
            std::shared_ptr<StampedCubArgReduction> stamped = CubArgReduction(op, 1, outputs).stamp(input, stream);
            EXPECT_EQ(stamped->getPath(), CubReductionPath::TiledFixedSegment);
            stamped->run();
            stream.synchronize();

            const DenseArgReference reference =
                referenceDenseMiddleAxis(values, outer_size, reduction_size, inner_size, op);
            expectArgOutputs(stamped, reference.values, reference.indices, stream);
        }
    }
}


TEST(CubArgReduction, DenseTiledContiguousMultiAxisUsesFlattenedLocalIndices) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t outer_size = 2;
    constexpr uint64_t reduction_size = 12;
    constexpr uint64_t inner_size = 5;
    std::vector<float> values(outer_size * reduction_size * inner_size);
    for (uint64_t outer = 0; outer < outer_size; ++outer) {
        for (uint64_t row = 0; row < reduction_size; ++row) {
            for (uint64_t component = 0; component < inner_size; ++component) {
                const int target = static_cast<int>((outer * 3 + component * 2) % reduction_size);
                values[(outer * reduction_size + row) * inner_size + component] =
                    static_cast<float>(std::abs(static_cast<int>(row) - target));
            }
        }
    }

    Tensor input = makeGpuTensor(values, {2, 3, 4, 5}, stream);
    for (CubArgReductionOp op : {CubArgReductionOp::ArgMin, CubArgReductionOp::ArgMax}) {
        auto stamped = CubArgReduction(op, std::vector<uint32_t>{1, 2}, fp32ValueAndUint32Index()).stamp(input, stream);
        EXPECT_EQ(stamped->getPath(), CubReductionPath::TiledFixedSegment);
        stamped->run();
        stream.synchronize();
        const DenseArgReference reference =
            referenceDenseMiddleAxis(values, outer_size, reduction_size, inner_size, op);
        expectArgOutputs(stamped, reference.values, reference.indices, stream);
    }
}

TEST(CubArgReduction, DenseTiledPathSupportsValueOnlyAndIndexOnlyOutputs) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, 9.0f, 4.0f, 2.0f,
                                  5.0f, 3.0f, 8.0f, 7.0f,
                                  6.0f, 0.0f, 2.0f, 10.0f},
                                 {1, 3, 4},
                                 stream);

    CubArgReductionOutputOptions value_only_options;
    value_only_options.produce_index = false;
    value_only_options.value_output_dtype = DataType::FP32;
    auto value_only = CubArgReduction(CubArgReductionOp::ArgMax, 1, value_only_options).stamp(input, stream);
    EXPECT_EQ(value_only->getPath(), CubReductionPath::TiledFixedSegment);

    CubArgReductionOutputOptions index_only_options;
    index_only_options.produce_value = false;
    index_only_options.index_output_dtype = DataType::UINT64;
    auto index_only = CubArgReduction(CubArgReductionOp::ArgMax, 1, index_only_options).stamp(input, stream);
    EXPECT_EQ(index_only->getPath(), CubReductionPath::TiledFixedSegment);

    value_only->run();
    index_only->run();
    stream.synchronize();

    ASSERT_TRUE(value_only->getValueOutputTensor().has_value());
    EXPECT_FALSE(value_only->getIndexOutputTensor().has_value());
    expectFloatVectorNear(copyGpuTensorAsFloat(value_only->getValueOutputTensor().value(), stream),
                          {6.0f, 9.0f, 8.0f, 10.0f});
    EXPECT_FALSE(index_only->getValueOutputTensor().has_value());
    ASSERT_TRUE(index_only->getIndexOutputTensor().has_value());
    EXPECT_EQ(copyGpuTensorAsUnsigned(index_only->getIndexOutputTensor().value(), stream),
              (std::vector<uint64_t>{2, 0, 1, 2}));
}

TEST(CubArgReduction, DenseTiledVectorizedPathsSupportEveryFloatingInputDtype) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    std::vector<DataType> input_dtypes = {DataType::FP16, DataType::BF16, DataType::FP32};
#if THOR_CUB_ENABLE_FP8_TYPES
    input_dtypes.push_back(DataType::FP8_E4M3);
    input_dtypes.push_back(DataType::FP8_E5M2);
#endif
#if THOR_CUB_ENABLE_64BIT_TYPES
    input_dtypes.push_back(DataType::FP64);
#endif

    for (uint64_t inner_size : {64ULL, 4097ULL}) {
        SCOPED_TRACE(inner_size);
        constexpr uint64_t reduction_size = 5;
        std::vector<float> values(reduction_size * inner_size);
        for (uint64_t row = 0; row < reduction_size; ++row) {
            for (uint64_t component = 0; component < inner_size; ++component) {
                values[row * inner_size + component] = static_cast<float>(row + 1);
            }
        }

        for (DataType input_dtype : input_dtypes) {
            SCOPED_TRACE(static_cast<int>(input_dtype));
            Tensor input = makeGpuTensor(values, {1, reduction_size, inner_size}, stream, input_dtype);
            auto stamped =
                CubArgReduction(CubArgReductionOp::ArgMax, 1, fp32ValueAndUint32Index()).stamp(input, stream);
            EXPECT_EQ(stamped->getPath(), CubReductionPath::TiledFixedSegment);
            stamped->run();
            stream.synchronize();
            expectArgOutputs(
                stamped,
                std::vector<float>(inner_size, static_cast<float>(reduction_size)),
                std::vector<uint64_t>(inner_size, reduction_size - 1),
                stream);
        }
    }
}

TEST(CubArgReduction, DenseTiledPathPreservesNaNPropagationAndLowestIndexTies) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    Tensor input = makeGpuTensor({3.0f, 7.0f,
                                  nan, 5.0f,
                                  2.0f, 5.0f,
                                  nan, 1.0f},
                                 {1, 4, 2},
                                 stream);

    auto minimum = CubArgReduction(CubArgReductionOp::ArgMin, 1, fp32ValueAndUint32Index()).stamp(input, stream);
    auto maximum = CubArgReduction(CubArgReductionOp::ArgMax, 1, fp32ValueAndUint32Index()).stamp(input, stream);
    EXPECT_EQ(minimum->getPath(), CubReductionPath::TiledFixedSegment);
    EXPECT_EQ(maximum->getPath(), CubReductionPath::TiledFixedSegment);
    minimum->run();
    maximum->run();
    stream.synchronize();

    expectArgOutputs(minimum, {nan, 1.0f}, {1, 3}, stream);
    expectArgOutputs(maximum, {nan, 7.0f}, {1, 0}, stream);
}

TEST(CubArgReduction, ValidatesOutputConfigurationAndPreallocatedContracts) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, stream);

    EXPECT_THROW(static_cast<void>(CubArgReduction(CubArgReductionOp::ArgMin, std::vector<uint32_t>{})),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(CubArgReduction(CubArgReductionOp::ArgMin, std::vector<uint32_t>{1, 1})),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(CubArgReduction(CubArgReductionOp::ArgMin, 2).stamp(input, stream)),
                 std::invalid_argument);

    CubArgReductionOutputOptions no_outputs;
    no_outputs.produce_value = false;
    no_outputs.produce_index = false;
    EXPECT_THROW(static_cast<void>(CubArgReduction(CubArgReductionOp::ArgMin, 1, no_outputs)),
                 std::invalid_argument);

    CubArgReductionOutputOptions disabled_value_with_dtype;
    disabled_value_with_dtype.produce_value = false;
    disabled_value_with_dtype.value_output_dtype = DataType::FP32;
    EXPECT_THROW(static_cast<void>(CubArgReduction(CubArgReductionOp::ArgMin, 1, disabled_value_with_dtype)),
                 std::invalid_argument);

    CubArgReductionOutputOptions bad_index_dtype;
    bad_index_dtype.index_output_dtype = DataType::FP32;
    EXPECT_THROW(static_cast<void>(CubArgReduction(CubArgReductionOp::ArgMin, 1, bad_index_dtype)),
                 std::invalid_argument);

    CubArgReduction reduction(CubArgReductionOp::ArgMin, 1, fp32ValueAndUint32Index());
    Tensor wrong_value_dtype(gpuPlacement, TensorDescriptor(DataType::BF16, {2, 1}));
    Tensor correct_index(gpuPlacement, TensorDescriptor(DataType::UINT32, {2, 1}));
    EXPECT_THROW(static_cast<void>(reduction.stamp(input, wrong_value_dtype, correct_index, stream)),
                 std::invalid_argument);

    Tensor correct_value(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 1}));
    Tensor wrong_index_dtype(gpuPlacement, TensorDescriptor(DataType::UINT64, {2, 1}));
    EXPECT_THROW(static_cast<void>(reduction.stamp(input, correct_value, wrong_index_dtype, stream)),
                 std::invalid_argument);

    Tensor wrong_shape(gpuPlacement, TensorDescriptor(DataType::UINT32, {3}));
    EXPECT_THROW(static_cast<void>(reduction.stamp(input, correct_value, wrong_shape, stream)),
                 std::invalid_argument);

    Tensor scalar = makeGpuTensor({1.0f}, {1}, stream);
    CubArgReductionOutputOptions value_only;
    value_only.produce_index = false;
    EXPECT_THROW(static_cast<void>(CubArgReduction(CubArgReductionOp::ArgMin, 0, value_only)
                                       .stamp(scalar, scalar, std::nullopt, stream)),
                 std::invalid_argument);
}

TEST(CubArgReduction, RankNineStridedArgmaxUsesStampedDynamicMetadata) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    std::vector<float> values(32);
    for (uint64_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i);
    }
    Tensor input = makeGpuTensor(values, {2, 1, 2, 1, 2, 1, 2, 1, 2}, stream);
    CubArgReductionOutputOptions outputs;
    outputs.produce_value = false;
    outputs.produce_index = true;
    outputs.index_output_dtype = DataType::UINT32;
    std::shared_ptr<StampedCubArgReduction> stamped =
        CubArgReduction(CubArgReductionOp::ArgMax, std::vector<uint32_t>{0, 2, 4, 6}, outputs).stamp(input, stream);
    EXPECT_EQ(stamped->getPath(), CubReductionPath::StridedFixedSegment);
    stamped->run();
    stream.synchronize();
    ASSERT_TRUE(stamped->getIndexOutputTensor().has_value());
    EXPECT_EQ(copyGpuTensorAsUnsigned(stamped->getIndexOutputTensor().value(), stream),
              (std::vector<uint64_t>{15U, 15U}));
}
