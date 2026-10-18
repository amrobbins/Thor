#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"
#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/Common/LowPrecisionFloat.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionTestSupport;

TEST(CubReduction, EverySupportedInputStorageDtypeAccumulatesInFp32) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    expectInputStorageAccumulatesInFp32(DataType::FP32, stream);
    expectInputStorageAccumulatesInFp32(DataType::FP16, stream);
    expectInputStorageAccumulatesInFp32(DataType::BF16, stream);
#if THOR_CUB_ENABLE_FP8_TYPES
    expectInputStorageAccumulatesInFp32(DataType::FP8_E4M3, stream);
    expectInputStorageAccumulatesInFp32(DataType::FP8_E5M2, stream);
#endif
#if THOR_CUB_ENABLE_64BIT_TYPES
    expectInputStorageAccumulatesInFp32(DataType::FP64, stream);
#endif
}

TEST(CubReduction, PermutationAwareTiledPathSupportsEveryInputStorageDtype) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    const std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                       7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    const auto run_dtype = [&](DataType dtype) {
        SCOPED_TRACE(static_cast<int>(dtype));
        Tensor storage = makeGpuTensor(values, {2, 3, 2}, stream, dtype);
        Tensor permuted = storage.aliasView({2, 2, 3}, {1, 6, 2}, 0);
        std::shared_ptr<StampedCubReduction> stamped =
            CubReduction(CubReductionOp::Sum, 2, DataType::FP32).stamp(permuted, stream);
        ASSERT_EQ(stamped->getPath(), CubReductionPath::TiledFixedSegment);
        ASSERT_TRUE(stamped->getGeometry().permutation_aware_tiled_geometry.has_value());
        EXPECT_EQ(stamped->getAccumulatorDataType(), DataType::FP32);
        stamped->run();
        stream.synchronize();
        expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream),
                              {9.0f, 27.0f, 12.0f, 30.0f});
    };

    run_dtype(DataType::FP32);
    run_dtype(DataType::FP16);
    run_dtype(DataType::BF16);
#if THOR_CUB_ENABLE_FP8_TYPES
    run_dtype(DataType::FP8_E4M3);
    run_dtype(DataType::FP8_E5M2);
#endif
#if THOR_CUB_ENABLE_64BIT_TYPES
    run_dtype(DataType::FP64);
#endif
}

TEST(CubReduction, AsyncTiledLowPrecisionAlignmentPeelCoversEveryFullRowFamily) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    struct ProbeGeometry {
        uint64_t reduction_size;
        uint64_t inner_size;
    };
    constexpr uint64_t outer_size = 3;
    const std::vector<ProbeGeometry> probes = {
        {127, 15},    // async_full_row_narrow, RowLanes=2
        {127, 193},   // async_full_row_x8
        {127, 383},   // async_full_row_x16
        {127, 769},   // async_full_row_group2_x16
        {127, 1537},  // async_full_row_group4_x16
        {127, 2813},  // async_full_row_group8_x16
        {512, 383},   // aligned async_full_row_x16 control
    };

    for (DataType dtype : {DataType::FP16, DataType::BF16}) {
        for (const ProbeGeometry probe : probes) {
            SCOPED_TRACE(static_cast<int>(dtype));
            SCOPED_TRACE(probe.reduction_size);
            SCOPED_TRACE(probe.inner_size);

            std::vector<float> values(outer_size * probe.reduction_size * probe.inner_size);
            std::vector<float> expected(outer_size * probe.inner_size, 0.0f);
            for (uint64_t outer = 0; outer < outer_size; ++outer) {
                for (uint64_t row = 0; row < probe.reduction_size; ++row) {
                    for (uint64_t component = 0; component < probe.inner_size; ++component) {
                        const float value = static_cast<float>(
                            static_cast<int>((outer * 11 + row * 3 + component * 5) % 7) - 3);
                        values[(outer * probe.reduction_size + row) * probe.inner_size + component] = value;
                        expected[outer * probe.inner_size + component] += value;
                    }
                }
            }

            Tensor input = makeGpuTensor(
                values, {outer_size, probe.reduction_size, probe.inner_size}, stream, dtype);
            std::shared_ptr<StampedCubReduction> stamped =
                CubReduction(CubReductionOp::Sum, 1, DataType::FP32).stamp(input, stream);
            ASSERT_EQ(stamped->getPath(), CubReductionPath::TiledFixedSegment);

            stamped->run();
            stream.synchronize();
            expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), expected);
        }
    }
}

TEST(CubReduction, AsyncTiledAlignmentHeadBulkTailShareOneTransformedAccumulator) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t outer_size = 3;
    constexpr uint64_t reduction_size = 127;
    constexpr uint64_t inner_size = 383;
    std::vector<float> values(outer_size * reduction_size * inner_size);
    std::vector<float> expected(outer_size * inner_size, 0.0f);
    for (uint64_t outer = 0; outer < outer_size; ++outer) {
        for (uint64_t row = 0; row < reduction_size; ++row) {
            for (uint64_t component = 0; component < inner_size; ++component) {
                const float value =
                    static_cast<float>(static_cast<int>((outer * 7 + row * 3 + component) % 5) - 2);
                values[(outer * reduction_size + row) * inner_size + component] = value;
                expected[outer * inner_size + component] += value * value;
            }
        }
    }
    constexpr float output_scale = 0.25f;
    for (float& value : expected) {
        value = std::sqrt(value) * output_scale;
    }

    const std::vector<uint64_t> dimensions = {outer_size, reduction_size, inner_size};
    const CubReductionGeometry geometry =
        CubReduction::analyzeValueGeometry(CubReductionOp::L2Norm, dimensions, std::vector<uint32_t>{1});
    ASSERT_EQ(geometry.path, CubReductionPath::TiledFixedSegment);

    for (DataType dtype : {DataType::FP16, DataType::BF16}) {
        SCOPED_TRACE(static_cast<int>(dtype));
        Tensor input = makeGpuTensor(values, dimensions, stream, dtype);
        Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, geometry.output_dimensions));
        std::shared_ptr<StampedCubReduction> stamped =
            CubReduction(CubReductionOp::L2Norm, 1, DataType::FP32, output_scale).stamp(input, output, stream);
        ASSERT_EQ(stamped->getPath(), CubReductionPath::TiledFixedSegment);
        EXPECT_EQ(stamped->getOutputTensor().getMemPtr<void>(), output.getMemPtr<void>());

        stamped->run();
        stream.synchronize();
        expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), expected, 1.0e-5f);
    }
}

TEST(CubReduction, AwkwardLargeVectorizedShardsHandleAlignmentAndPaddedInputTailForEveryStorageDtype) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t outer_size = 2;
    constexpr uint64_t reduction_size = 3;
    constexpr uint64_t inner_size = 4097;
    std::vector<float> values(outer_size * reduction_size * inner_size);
    std::vector<float> expected(outer_size * inner_size, 0.0f);
    for (uint64_t outer = 0; outer < outer_size; ++outer) {
        for (uint64_t row = 0; row < reduction_size; ++row) {
            for (uint64_t component = 0; component < inner_size; ++component) {
                const float value = static_cast<float>(static_cast<int>((outer + row + component) % 5) - 2);
                values[(outer * reduction_size + row) * inner_size + component] = value;
                expected[outer * inner_size + component] += value;
            }
        }
    }

    const auto run_dtype = [&](DataType dtype) {
        SCOPED_TRACE(static_cast<int>(dtype));
        Tensor input = makeGpuTensor(values, {outer_size, reduction_size, inner_size}, stream, dtype);
        std::shared_ptr<StampedCubReduction> stamped =
            CubReduction(CubReductionOp::Sum, 1, DataType::FP32).stamp(input, stream);
        EXPECT_EQ(stamped->getPath(), CubReductionPath::TiledFixedSegment);
        if (dtype == DataType::FP16 || dtype == DataType::BF16 || dtype == DataType::FP32) {
            EXPECT_TRUE(stamped->usesModernRKFamilyPlan());
        } else {
            EXPECT_FALSE(stamped->usesModernRKFamilyPlan());
        }
        stamped->run();
        stream.synchronize();
        expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), expected);
    };

    run_dtype(DataType::FP32);
    run_dtype(DataType::FP16);
    run_dtype(DataType::BF16);
#if THOR_CUB_ENABLE_FP8_TYPES
    run_dtype(DataType::FP8_E4M3);
    run_dtype(DataType::FP8_E5M2);
#endif
#if THOR_CUB_ENABLE_64BIT_TYPES
    run_dtype(DataType::FP64);
#endif
}

#if THOR_CUB_ENABLE_FP8_TYPES
TEST(CubReduction, UnderSuppliedPacketAlignedFp8OrdinaryDenseUsesNarrowModernKParallelValuePlan) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t outer_size = 2;
    constexpr uint64_t reduction_size = 256;
    // These K values successively make 16/8/4/2 bytes the widest naturally aligned packet. This is deliberately a
    // small, under-supplied reduction, however, so the final FP8 packet policy must maximize useful K-side work
    // rather than blindly choose that widest legal packet. Packet2 exposes the most useful launch warps in every
    // case and is therefore the expected production choice. The family-policy unit tests cover the selector rule;
    // this lifecycle test verifies that the stamped ordinary VALUE plan preserves it end to end.
    for (uint64_t inner_size : std::vector<uint64_t>{528, 264, 132, 66}) {
        SCOPED_TRACE(inner_size);
        std::vector<float> values(outer_size * reduction_size * inner_size);
        std::vector<float> expected(outer_size * inner_size, 0.0f);
        for (uint64_t outer = 0; outer < outer_size; ++outer) {
            for (uint64_t row = 0; row < reduction_size; ++row) {
                const float value = row < 128 ? 1.0f : 2.0f;
                for (uint64_t component = 0; component < inner_size; ++component) {
                    values[(outer * reduction_size + row) * inner_size + component] = value;
                    expected[outer * inner_size + component] += value;
                }
            }
        }

        for (DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
            SCOPED_TRACE(static_cast<int>(dtype));
            Tensor input = makeGpuTensor(
                values, {outer_size, reduction_size, inner_size}, stream, dtype);
            std::shared_ptr<StampedCubReduction> stamped =
                CubReduction(CubReductionOp::Sum, 1, DataType::FP32).stamp(input, stream);

            ASSERT_TRUE(stamped->usesModernRKFamilyPlan());
            EXPECT_NE(stamped->getModernRKStrategyChain().find("k_parallel:aligned"), std::string::npos);
            EXPECT_EQ(stamped->getModernRKStrategyChain().find("rk:generic"), std::string::npos);
            EXPECT_EQ(stamped->getModernRKFirstPacketBytes(), 2U);

            stamped->run();
            stream.synchronize();
            expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), expected, 0.0f);
        }
    }
}
#endif

#if THOR_CUB_ENABLE_64BIT_TYPES
TEST(CubReduction, WideFp64RowsRespectGroupedStageCapacityAndVectorDirectBoundaries) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    for (uint64_t inner_size : {511ULL, 512ULL, 513ULL, 1023ULL, 1024ULL, 1025ULL, 2047ULL, 2048ULL, 2049ULL, 4096ULL}) {
        SCOPED_TRACE(inner_size);
        constexpr uint64_t reduction_size = 3;
        std::vector<float> values(reduction_size * inner_size);
        for (uint64_t row = 0; row < reduction_size; ++row) {
            for (uint64_t component = 0; component < inner_size; ++component) {
                values[row * inner_size + component] = static_cast<float>(row + 1);
            }
        }

        Tensor input =
            makeGpuTensor(values, {1, reduction_size, inner_size}, stream, DataType::FP64);
        std::shared_ptr<StampedCubReduction> stamped =
            CubReduction(CubReductionOp::Sum, 1, DataType::FP32).stamp(input, stream);
        EXPECT_EQ(stamped->getPath(), CubReductionPath::TiledFixedSegment);

        stamped->run();
        stream.synchronize();
        expectFloatVectorNear(
            copyGpuTensorAsFloat(stamped->getOutputTensor(), stream),
            std::vector<float>(inner_size, 6.0f));
    }
}
#endif

TEST(CubReduction, WholeTensorBf16SumAccumulatesInFp32AndDefaultsOutputToInputDtype) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    std::vector<float> values(257, 1.0f);
    values[0] = 256.0f;
    Tensor input = makeGpuTensor(values, {257}, stream, DataType::BF16);

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

    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), {512.0f});
}

TEST(CubReduction, ReusesStampedOutputAndWorkspaceAcrossExecutions) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, stream, DataType::BF16);
    std::shared_ptr<StampedCubReduction> stamped =
        CubReduction(CubReductionOp::L1Norm, 1, DataType::FP32).stamp(input, stream);
    const uint64_t output_tensor_id = stamped->getOutputTensor().getTensorId();
    const size_t workspace_bytes = stamped->getWorkspaceSizeInBytes();

    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), {3.0f, 7.0f});

    Tensor replacement = makeGpuTensor({-10.0f, 20.0f, -30.0f, 40.0f}, {2, 2}, stream, DataType::BF16);
    input.copyFromAsync(replacement, stream);
    stamped->run();
    stream.synchronize();

    EXPECT_EQ(stamped->getOutputTensor().getTensorId(), output_tensor_id);
    EXPECT_EQ(stamped->getWorkspaceSizeInBytes(), workspace_bytes);
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), {30.0f, 70.0f});
}

TEST(CubReduction, ValidatesPreallocatedOutputContract) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, stream, DataType::BF16);
    CubReduction reduction(CubReductionOp::Mean, 1, DataType::FP32);

    Tensor correct_output(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 1}));
    std::shared_ptr<StampedCubReduction> stamped = reduction.stamp(input, correct_output, stream);
    EXPECT_EQ(stamped->getOutputTensor().getTensorId(), correct_output.getTensorId());

    Tensor wrong_dtype(gpuPlacement, TensorDescriptor(DataType::BF16, {2, 1}));
    EXPECT_THROW(static_cast<void>(reduction.stamp(input, wrong_dtype, stream)), std::invalid_argument);

    Tensor wrong_shape(gpuPlacement, TensorDescriptor(DataType::FP32, {1, 3}));
    EXPECT_THROW(static_cast<void>(reduction.stamp(input, wrong_shape, stream)), std::invalid_argument);

    Tensor scalar = makeGpuTensor({1.0f}, {1}, stream, DataType::BF16);
    EXPECT_THROW(static_cast<void>(CubReduction(CubReductionOp::Sum, 0).stamp(scalar, scalar, stream)),
                 std::invalid_argument);
}

TEST(CubReduction, PreallocatedOutputOverlapUsesReachableViewAddressSpan) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    // A rank-1 affine broadcast is an ordained DeviceTransformReduce path. Its logical view contains 1024 FP32
    // elements but can only address storage element 0. Put a valid dense scalar output later in the same allocation.
    // The old logical-byte overlap check incorrectly rejected this even though the reachable input/output address spans
    // are disjoint.
    Tensor storage = Tensor::zeros(gpuPlacement, TensorDescriptor(DataType::FP32, {128}), stream);
    Tensor input = storage.aliasView({1024}, {0}, 0);
    Tensor disjoint_output = storage.aliasView({1}, {1}, 64);

    CubReduction reduction(CubReductionOp::Sum, 0, DataType::FP32);
    std::shared_ptr<StampedCubReduction> stamped = reduction.stamp(input, disjoint_output, stream);
    EXPECT_EQ(stamped->getPath(), CubReductionPath::DeviceTransformReduce);
    EXPECT_TRUE(stamped->getGeometry().device_transform_uses_affine_stride);
    EXPECT_EQ(stamped->getGeometry().affine_input_stride, 0U);
    EXPECT_EQ(stamped->getOutputTensor().getMemPtr<void>(), disjoint_output.getMemPtr<void>());

    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(disjoint_output, stream), {0.0f});

    // Moving the output onto an address the broadcast view can actually read must still be rejected.
    Tensor overlapping_output = storage.aliasView({1}, {1}, 0);
    EXPECT_THROW(static_cast<void>(reduction.stamp(input, overlapping_output, stream)), std::invalid_argument);

    // Dense aliases are valid preallocated outputs, but genuinely strided multi-element output views remain
    // unsupported. Use an independent ordained dense reduction because a one-element output has no meaningful stride.
    Tensor dense_input = storage.aliasView({2, 2}, {2, 1}, 8);
    Tensor strided_output = storage.aliasView({2}, {2}, 64);
    EXPECT_THROW(static_cast<void>(CubReduction(CubReductionOp::Sum, 0, DataType::FP32)
                                       .stamp(dense_input, strided_output, stream)),
                 std::invalid_argument);
}

#if THOR_CUB_ENABLE_FP8_TYPES
TEST(CubReduction, RuntimeKCompleteRoundsFp8AndConsumesDeepRTail) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    // R=1200 exposed a false benchmark failure: 1200/512=2.34375 narrows to
    // E4M3 2.25. A second outer row with a different final 48 rows ensures the
    // same rounded output cannot mask a Complete kernel that drops the R tail.
    constexpr uint64_t outer = 2;
    constexpr uint64_t reduction = 1200;
    constexpr uint64_t inner = 513;  // multiple K tiles and an odd trailing component
    constexpr uint64_t tail_begin = 1152;
    constexpr float base_value = 1.0f / 512.0f;
    std::vector<float> values(outer * reduction * inner, base_value);
    for (uint64_t row = tail_begin; row < reduction; ++row) {
        for (uint64_t k = 0; k < inner; ++k) {
            values[(reduction + row) * inner + k] =
                (k % 3 == 0) ? 0.125f : (k % 3 == 1) ? 0.25f : 0.5f;
        }
    }

    const auto geometry = CubReduction::analyzeValueGeometry(
        CubReductionOp::Sum, {outer, reduction, inner}, {1});
    ASSERT_EQ(geometry.path, CubReductionPath::TiledFixedSegment);

    for (DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        SCOPED_TRACE(static_cast<int>(dtype));
        Tensor input = makeGpuTensor(values, {outer, reduction, inner}, stream, dtype);
        Tensor output(gpuPlacement, TensorDescriptor(dtype, geometry.output_dimensions));
        for (uint32_t warps : {1u, 4u}) {
            SCOPED_TRACE(warps);
            CubReductionInternal::launchFp8GeneralRParallelKTileRCooperativeCalibrationComplete(
                input, output, geometry, warps, 496, stream);
            stream.synchronize();
            const std::vector<float> actual = copyGpuTensorAsFloat(output, stream);
            ASSERT_EQ(actual.size(), outer * inner);
            for (uint64_t k = 0; k < inner; ++k) {
                const float tail_value =
                    (k % 3 == 0) ? 0.125f : (k % 3 == 1) ? 0.25f : 0.5f;
                for (uint64_t o = 0; o < outer; ++o) {
                    const float sum = o == 0
                        ? static_cast<float>(reduction) * base_value
                        : static_cast<float>(tail_begin) * base_value +
                              static_cast<float>(reduction - tail_begin) * tail_value;
                    const float stored = dtype == DataType::FP8_E4M3
                        ? static_cast<float>(ThorLowPrecision::toFp8E4M3Satfinite(sum))
                        : static_cast<float>(ThorLowPrecision::toFp8E5M2Satfinite(sum));
                    EXPECT_FLOAT_EQ(actual[o * inner + k], stored)
                        << "outer=" << o << " k=" << k;
                }
            }
        }
    }
}

TEST(CubReduction, Fp8GeneralCompleteRuntimeOutputStorage) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    struct Shape { uint64_t outer, reduction, inner; uint32_t warps, outputs_per_warp, packet, tile; };
    // Cover OneWindow subgroup/warp and multi-component ownership, plus
    // RuntimeK K-tile tails and its multiwarp FP32 fold.
    const std::vector<Shape> one_window = {
        {9, 97, 3, 1, 1, 16, 0}, {9, 97, 3, 4, 1, 16, 0},
        {17, 67, 3, 1, 4, 8, 0}, {5, 73, 127, 4, 1, 16, 0}
    };
    const std::vector<Shape> runtime_k = {
        {2, 1200, 513, 1, 1, 16, 496}, {2, 1200, 513, 4, 1, 16, 496},
        {7, 79, 255, 2, 1, 16, 128}
    };

    for (DataType input_dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        std::vector<DataType> outputs = {DataType::FP8_E4M3, DataType::FP8_E5M2,
                                         DataType::FP16, DataType::BF16, DataType::FP32};
#if THOR_CUB_ENABLE_64BIT_TYPES
        outputs.push_back(DataType::FP64);
#endif
        for (bool tiled : {false, true}) {
            const auto& shapes = tiled ? runtime_k : one_window;
            for (const Shape& shape : shapes) {
                SCOPED_TRACE(static_cast<int>(input_dtype));
                SCOPED_TRACE(tiled);
                SCOPED_TRACE(shape.inner);
                SCOPED_TRACE(shape.warps);
                SCOPED_TRACE(shape.outputs_per_warp);
                std::vector<float> values(shape.outer * shape.reduction * shape.inner);
                std::vector<float> expected(shape.outer * shape.inner, 0.0f);
                // All contributions are exactly representable in both FP8 formats.
                // A nonuniform final row and alternating signs expose skipped R tails.
                for (uint64_t o = 0; o < shape.outer; ++o) {
                    for (uint64_t r = 0; r < shape.reduction; ++r) {
                        for (uint64_t k = 0; k < shape.inner; ++k) {
                            const float value = (r + k + o) % 4 == 0 ? 0.125f :
                                                (r + k + o) % 4 == 1 ? -0.0625f :
                                                (r + k + o) % 4 == 2 ? 0.03125f : 0.015625f;
                            values[(o * shape.reduction + r) * shape.inner + k] = value;
                            expected[o * shape.inner + k] += value;
                        }
                    }
                }
                const auto geometry = CubReduction::analyzeValueGeometry(
                    CubReductionOp::Sum, {shape.outer, shape.reduction, shape.inner}, {1});
                ASSERT_EQ(geometry.path, CubReductionPath::TiledFixedSegment);
                Tensor input = makeGpuTensor(values, {shape.outer, shape.reduction, shape.inner},
                                             stream, input_dtype);
                for (DataType output_dtype : outputs) {
                    SCOPED_TRACE(static_cast<int>(output_dtype));
                    Tensor output(gpuPlacement, TensorDescriptor(output_dtype, geometry.output_dimensions));
                    if (tiled) {
                        CubReductionInternal::launchFp8GeneralRParallelKTileRCooperativeCalibrationComplete(
                            input, output, geometry, shape.warps, shape.tile, stream);
                    } else {
                        CubReductionInternal::launchFp8GeneralRParallelRCooperativeCalibrationComplete(
                            input, output, geometry, shape.outputs_per_warp, shape.warps, shape.packet, stream);
                    }
                    stream.synchronize();
                    const auto actual = copyGpuTensorAsFloat(output, stream);
                    ASSERT_EQ(actual.size(), expected.size());
                    for (size_t i = 0; i < expected.size(); ++i) {
                        const float sum = expected[i];
                        float rounded = sum;
                        switch (output_dtype) {
                            case DataType::FP8_E4M3:
                                rounded = static_cast<float>(ThorLowPrecision::toFp8E4M3Satfinite(sum)); break;
                            case DataType::FP8_E5M2:
                                rounded = static_cast<float>(ThorLowPrecision::toFp8E5M2Satfinite(sum)); break;
                            case DataType::FP16:
                                rounded = __half2float(__float2half_rn(sum)); break;
                            case DataType::BF16:
                                rounded = __bfloat162float(__float2bfloat16_rn(sum)); break;
                            default: break;
                        }
                        EXPECT_FLOAT_EQ(actual[i], rounded) << "component=" << i;
                    }
                }
                // Fail closed on an output format that runtime store does not implement.
                Tensor bad_output(gpuPlacement,
                    TensorDescriptor(DataType::UINT32, geometry.output_dimensions));
                if (tiled) {
                    EXPECT_THROW(CubReductionInternal::launchFp8GeneralRParallelKTileRCooperativeCalibrationComplete(
                        input, bad_output, geometry, shape.warps, shape.tile, stream), std::logic_error);
                } else {
                    EXPECT_THROW(CubReductionInternal::launchFp8GeneralRParallelRCooperativeCalibrationComplete(
                        input, bad_output, geometry, shape.outputs_per_warp, shape.warps, shape.packet, stream),
                        std::logic_error);
                }
            }
        }
    }
}

TEST(CubReduction, Fp8CompleteRuntimeFinalizationMatchesFp32ThenStorage) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    using namespace CubReductionInternal;
    struct Shape {
        bool tiled;
        uint64_t outer, reduction, inner;
        uint32_t warps, outputs_per_warp, packet, tile;
    };
    const std::vector<Shape> shapes = {
        {false, 5, 97, 3, 1, 1, 16, 0},
        {false, 9, 79, 3, 1, 4, 8, 0},
        {false, 5, 73, 127, 4, 1, 16, 0},
        {true, 2, 79, 513, 1, 1, 16, 496},
        {true, 2, 97, 513, 4, 1, 16, 496}
    };
    for (DataType input_dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        for (const Shape& shape : shapes) {
            SCOPED_TRACE(static_cast<int>(input_dtype));
            SCOPED_TRACE(shape.tiled);
            SCOPED_TRACE(shape.inner);
            SCOPED_TRACE(shape.warps);
            std::vector<float> values(shape.outer * shape.reduction * shape.inner);
            std::array<std::vector<float>, 3> transform_sums = {
                std::vector<float>(shape.outer * shape.inner, 0.0f),
                std::vector<float>(shape.outer * shape.inner, 0.0f),
                std::vector<float>(shape.outer * shape.inner, 0.0f)};
            for (uint64_t o = 0; o < shape.outer; ++o) {
                for (uint64_t r = 0; r < shape.reduction; ++r) {
                    for (uint64_t k = 0; k < shape.inner; ++k) {
                        // Exactly representable FP8 fractions, including negative inputs
                        // and irregular R/K tails. Every final sum remains positive.
                        const int pattern = static_cast<int>((o + r + k) % 4);
                        const float value = pattern == 0 ? -0.015625f :
                                            pattern == 1 ? 0.0625f :
                                            pattern == 2 ? 0.03125f : 0.125f;
                        values[(o * shape.reduction + r) * shape.inner + k] = value;
                        transform_sums[0][o * shape.inner + k] += value;
                        transform_sums[1][o * shape.inner + k] += std::fabs(value);
                        transform_sums[2][o * shape.inner + k] += value * value;
                    }
                }
            }
            const auto geometry = CubReduction::analyzeValueGeometry(
                CubReductionOp::Sum, {shape.outer, shape.reduction, shape.inner}, {1});
            Tensor input = makeGpuTensor(values, {shape.outer, shape.reduction, shape.inner},
                                         stream, input_dtype);
            std::vector<DataType> output_types = {DataType::FP8_E4M3, DataType::FP8_E5M2,
                                                  DataType::FP16, DataType::BF16, DataType::FP32};
#if THOR_CUB_ENABLE_64BIT_TYPES
            output_types.push_back(DataType::FP64);
#endif
            for (CubReductionStageInputTransform transform :
                 {CubReductionStageInputTransform::Identity,
                  CubReductionStageInputTransform::AbsoluteValue,
                  CubReductionStageInputTransform::Square}) {
                const size_t transform_index = static_cast<size_t>(transform);
                const auto& sums = transform_sums[transform_index];
                SCOPED_TRACE(static_cast<int>(transform));
                for (DataType output_dtype : output_types) {
                    for (CubReductionStageFinalize mode : {CubReductionStageFinalize::Identity,
                                                           CubReductionStageFinalize::Divide,
                                                           CubReductionStageFinalize::SquareRoot}) {
                        SCOPED_TRACE(static_cast<int>(output_dtype));
                        SCOPED_TRACE(static_cast<int>(mode));
                        CubReductionStageSemantics semantics;
                        semantics.input_transform = transform;
                        semantics.finalize = mode;
                        semantics.finalize_divisor = shape.reduction;
                        Tensor output(gpuPlacement, TensorDescriptor(output_dtype, geometry.output_dimensions));
                        if (shape.tiled) {
                            launchFp8GeneralRParallelKTileRCooperativeCalibrationComplete(
                                input, output, geometry, shape.warps, shape.tile, stream, semantics);
                        } else {
                            launchFp8GeneralRParallelRCooperativeCalibrationComplete(
                                input, output, geometry, shape.outputs_per_warp, shape.warps,
                                shape.packet, stream, false, semantics);
                        }
                        stream.synchronize();
                        const auto actual = copyGpuTensorAsFloat(output, stream);
                        ASSERT_EQ(actual.size(), sums.size());
                        for (size_t i = 0; i < sums.size(); ++i) {
                            float result = sums[i];
                            if (mode == CubReductionStageFinalize::Divide)
                                result /= static_cast<float>(shape.reduction);
                            else if (mode == CubReductionStageFinalize::SquareRoot)
                                result = std::sqrt(result);
                            switch (output_dtype) {
                                case DataType::FP8_E4M3:
                                    result = static_cast<float>(ThorLowPrecision::toFp8E4M3Satfinite(result));
                                    break;
                                case DataType::FP8_E5M2:
                                    result = static_cast<float>(ThorLowPrecision::toFp8E5M2Satfinite(result));
                                    break;
                                case DataType::FP16:
                                    result = __half2float(__float2half_rn(result));
                                    break;
                                case DataType::BF16:
                                    result = __bfloat162float(__float2bfloat16_rn(result));
                                    break;
                                default: break;
                            }
                            EXPECT_NEAR(actual[i], result, output_dtype == DataType::FP32 ? 1.0e-6f : 0.0f)
                                << "output index=" << i;
                        }
                    }
                }
            } // runtime input transform
        }
    }
}

// With negative FP8 values the three transforms must produce DISTINCT aggregates;
// this catches applying ABS/SQUARE after reduction (or silently retaining identity).
TEST(CubReduction, Fp8CompleteRuntimeInputTransformBeforeReduction) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    for (DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        for (bool tiled : {false, true}) {
            const uint64_t k = tiled ? 513 : 3;
            constexpr uint64_t R = 79;
            const auto geometry = CubReduction::analyzeValueGeometry(
                CubReductionOp::Sum, {3, R, k}, {1});
            std::vector<float> values(3 * R * k);
            for (uint64_t o = 0; o < 3; ++o) {
                for (uint64_t r = 0; r < R; ++r) {
                    for (uint64_t j = 0; j < k; ++j) {
                        values[(o * R + r) * k + j] = ((o + r + j) % 2 == 0) ? -0.25f : 0.5f;
                    }
                }
            }
            Tensor input = makeGpuTensor(values, {3, R, k}, stream, dtype);
            for (auto transform : {CubReductionStageInputTransform::Identity,
                                   CubReductionStageInputTransform::AbsoluteValue,
                                   CubReductionStageInputTransform::Square}) {
                CubReductionStageSemantics semantics;
                semantics.input_transform = transform;
                Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, geometry.output_dimensions));
                if (tiled) {
                    launchFp8GeneralRParallelKTileRCooperativeCalibrationComplete(
                        input, output, geometry, 2, 496, stream, semantics);
                } else {
                    launchFp8GeneralRParallelRCooperativeCalibrationComplete(
                        input, output, geometry, 1, 2, 16, stream, false, semantics);
                }
                stream.synchronize();
                const auto actual = copyGpuTensorAsFloat(output, stream);
                ASSERT_EQ(actual.size(), 3 * k);
                for (uint64_t o = 0; o < 3; ++o) {
                    for (uint64_t j = 0; j < k; ++j) {
                        float expected = 0.0f;
                        for (uint64_t r = 0; r < R; ++r) {
                            const float value = values[(o * R + r) * k + j];
                            expected += transform == CubReductionStageInputTransform::AbsoluteValue
                                ? std::fabs(value)
                                : (transform == CubReductionStageInputTransform::Square
                                       ? value * value : value);
                        }
                        EXPECT_FLOAT_EQ(actual[o * k + j], expected)
                            << "tiled=" << tiled << " o=" << o << " k=" << j
                            << " transform=" << static_cast<int>(transform);
                    }
                }
            }
        }
    }
}

// RuntimeK PacketStaged and both p16 rotated layouts must transform decoded FP8
// values BEFORE producing per-shard FP32 sums. This extends the existing Complete
// coverage without re-testing the independent FP32 continuation/output conversion.
TEST(CubReduction, Fp8RuntimeKStagedCompiledInputTransformBeforePartials) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    constexpr uint64_t Outer = 2;
    constexpr uint64_t R = 257;
    constexpr uint64_t K = 513; // awkward p16 tails, two K tiles in both rotated layouts
    const auto geometry = CubReduction::analyzeValueGeometry(
        CubReductionOp::Sum, {Outer, R, K}, {1});
    std::vector<float> values(Outer * R * K);
    for (uint64_t o = 0; o < Outer; ++o) {
        for (uint64_t r = 0; r < R; ++r) {
            for (uint64_t k = 0; k < K; ++k) {
                const auto residue = (o + 3 * r + k) % 4;
                values[(o * R + r) * K + k] = residue == 0 ? -0.25f :
                    residue == 1 ? 0.5f : residue == 2 ? -0.125f : 0.25f;
            }
        }
    }
    for (DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        Tensor input = makeGpuTensor(values, {Outer, R, K}, stream, dtype);
        for (auto transform : {CubReductionStageInputTransform::Identity,
                               CubReductionStageInputTransform::AbsoluteValue,
                               CubReductionStageInputTransform::Square}) {
            for (int mode = 0; mode < 3; ++mode) {
                const bool packet_staged = mode == 0;
                const uint64_t rows = 64;
                const uint64_t shards = packet_staged ? 3 : (R + rows - 1) / rows;
                Tensor partials(gpuPlacement, TensorDescriptor(
                    DataType::FP32, {Outer, shards, K}));
                SCOPED_TRACE(static_cast<int>(dtype));
                SCOPED_TRACE(static_cast<int>(transform));
                SCOPED_TRACE(mode);
                const auto occupancy = packet_staged
                    ? queryFp8GeneralRParallelKTileRCooperativeCalibrationFirstStageOccupancy(dtype, transform)
                    : queryFp8GeneralRParallelKTileRotatedCalibrationOccupancy(dtype, mode == 2, rows, transform);
                EXPECT_GT(occupancy.max_active_blocks_per_sm, 0);
                if (packet_staged) {
                    launchFp8GeneralRParallelKTileRCooperativeCalibrationFirstStage(
                        input, partials, geometry, 2, static_cast<uint32_t>(shards),
                        496, stream, transform);
                } else {
                    launchFp8GeneralRParallelKTileRotatedCalibrationFirstStage(
                        input, partials, geometry, mode == 2, rows, shards, stream, transform);
                }
                stream.synchronize();
                const auto actual = copyGpuTensorAsFloat(partials, stream);
                ASSERT_EQ(actual.size(), Outer * shards * K);
                for (uint64_t o = 0; o < Outer; ++o) {
                    for (uint64_t shard = 0; shard < shards; ++shard) {
                        const uint64_t base = R / shards;
                        const uint64_t extra = R % shards;
                        const uint64_t row_begin = packet_staged
                            ? shard * base + std::min(shard, extra) : shard * rows;
                        const uint64_t row_end = packet_staged
                            ? row_begin + base + static_cast<uint64_t>(shard < extra)
                            : std::min(R, row_begin + rows);
                        for (uint64_t k = 0; k < K; ++k) {
                            float expected = 0.0f;
                            for (uint64_t r = row_begin; r < row_end; ++r) {
                                const float value = values[(o * R + r) * K + k];
                                expected += transform == CubReductionStageInputTransform::AbsoluteValue
                                    ? std::fabs(value)
                                    : transform == CubReductionStageInputTransform::Square
                                        ? value * value : value;
                            }
                            EXPECT_FLOAT_EQ(actual[(o * shards + shard) * K + k], expected)
                                << "outer=" << o << " shard=" << shard << " k=" << k;
                        }
                    }
                }
            }
        }
    }
}

TEST(CubReduction, Fp8CompleteRuntimeFinalizationRejectsUnsupportedSemantics) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    for (bool tiled : {false, true}) {
        const uint64_t k = tiled ? 513 : 3;
        const auto geometry = CubReduction::analyzeValueGeometry(CubReductionOp::Sum, {1, 79, k}, {1});
        Tensor input = makeGpuTensor(std::vector<float>(79 * k, 0.125f),
                                     {1, 79, k}, stream, DataType::FP8_E4M3);
        Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, geometry.output_dimensions));
        const auto launch = [&](const CubReductionStageSemantics& semantics) {
            if (tiled) {
                launchFp8GeneralRParallelKTileRCooperativeCalibrationComplete(
                    input, output, geometry, 1, 496, stream, semantics);
            } else {
                launchFp8GeneralRParallelRCooperativeCalibrationComplete(
                    input, output, geometry, 1, 1, 16, stream, false, semantics);
            }
        };
        CubReductionStageSemantics semantics;
        semantics.input_transform = static_cast<CubReductionStageInputTransform>(255);
        EXPECT_THROW(launch(semantics), std::logic_error);
        semantics = {};
        semantics.input_transform = CubReductionStageInputTransform::AbsoluteValue;
        EXPECT_NO_THROW(launch(semantics));
        semantics.input_transform = CubReductionStageInputTransform::Square;
        EXPECT_NO_THROW(launch(semantics));
        semantics = {};
        semantics.combine = CubReductionStageCombine::Product;
        EXPECT_THROW(launch(semantics), std::logic_error);
        semantics = {};
        semantics.finalize = CubReductionStageFinalize::Divide;
        semantics.finalize_divisor = 0;
        EXPECT_THROW(launch(semantics), std::logic_error);
        semantics = {};
        semantics.finalize = static_cast<CubReductionStageFinalize>(255);
        EXPECT_THROW(launch(semantics), std::logic_error);
    }
}

TEST(CubReduction, Fp8CompleteRuntimeStorageSaturatesOnlyAtFinalOutput) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    for (bool tiled : {false, true}) {
        const uint64_t inner = tiled ? 513 : 3;
        constexpr uint64_t reduction = 256;
        const auto geometry = CubReduction::analyzeValueGeometry(
            CubReductionOp::Sum, {1, reduction, inner}, {1});
        Tensor input = makeGpuTensor(std::vector<float>(reduction * inner, 2.0f),
                                     {1, reduction, inner}, stream, DataType::FP8_E4M3);
        for (DataType output_dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2, DataType::FP32}) {
            SCOPED_TRACE(tiled);
            SCOPED_TRACE(static_cast<int>(output_dtype));
            Tensor output(gpuPlacement, TensorDescriptor(output_dtype, geometry.output_dimensions));
            if (tiled) {
                CubReductionInternal::launchFp8GeneralRParallelKTileRCooperativeCalibrationComplete(
                    input, output, geometry, 2, 496, stream);
            } else {
                CubReductionInternal::launchFp8GeneralRParallelRCooperativeCalibrationComplete(
                    input, output, geometry, 1, 2, 16, stream);
            }
            stream.synchronize();
            const auto actual = copyGpuTensorAsFloat(output, stream);
            const float expected = output_dtype == DataType::FP8_E4M3 ? 448.0f : 512.0f;
            for (float value : actual) EXPECT_FLOAT_EQ(value, expected);
        }
    }
}

TEST(CubReduction, Fp8OutputsUseCanonicalSatfiniteOverflowSemantics) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuTensor({40000.0f, 40000.0f, -40000.0f, -40000.0f}, {2, 2}, stream);

    std::shared_ptr<StampedCubReduction> e4m3 =
        CubReduction(CubReductionOp::Sum, 1, DataType::FP8_E4M3).stamp(input, stream);
    std::shared_ptr<StampedCubReduction> e5m2 =
        CubReduction(CubReductionOp::Sum, 1, DataType::FP8_E5M2).stamp(input, stream);

    e4m3->run();
    e5m2->run();
    stream.synchronize();

    expectFloatVectorNear(copyGpuTensorAsFloat(e4m3->getOutputTensor(), stream), {448.0f, -448.0f});
    expectFloatVectorNear(copyGpuTensorAsFloat(e5m2->getOutputTensor(), stream), {57344.0f, -57344.0f});
}
#endif

TEST(CubReduction, RuntimeOutputIteratorSupportsEveryConfiguredOutputDtype) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f}, {1, 3}, stream, DataType::FP32);

    std::vector<DataType> output_dtypes = {DataType::FP16, DataType::BF16, DataType::FP32};
#if THOR_CUB_ENABLE_FP8_TYPES
    output_dtypes.push_back(DataType::FP8_E4M3);
    output_dtypes.push_back(DataType::FP8_E5M2);
#endif
#if THOR_CUB_ENABLE_64BIT_TYPES
    output_dtypes.push_back(DataType::FP64);
#endif

    for (DataType output_dtype : output_dtypes) {
        SCOPED_TRACE(static_cast<int>(output_dtype));
        std::shared_ptr<StampedCubReduction> stamped =
            CubReduction(CubReductionOp::Sum, 1, output_dtype).stamp(input, stream);
        stamped->run();
        stream.synchronize();
        EXPECT_EQ(stamped->getOutputDataType(), output_dtype);
        expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), {6.0f});
    }
}

TEST(CubReduction, AppliesRuntimeOutputScaleInFp32) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, stream);
    CubReduction reduction(CubReductionOp::Sum, 0, DataType::FP32, 0.25f);
    std::shared_ptr<StampedCubReduction> stamped = reduction.stamp(input, stream);

    EXPECT_FLOAT_EQ(reduction.getOutputScale(), 0.25f);
    EXPECT_FLOAT_EQ(stamped->getOutputScale(), 0.25f);
    stamped->run();
    stream.synchronize();

    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), {1.25f, 1.75f, 2.25f});
}

TEST(CubReduction, QueryOnlyWorkspaceMatchesStampedWorkspace) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, stream, DataType::BF16);
    CubReduction reduction(CubReductionOp::Sum, 0, DataType::FP32, 0.5f);

    const size_t queried = reduction.queryWorkspaceSizeInBytes(input.getDescriptor(), stream);
    std::shared_ptr<StampedCubReduction> stamped = reduction.stamp(input, stream);
    EXPECT_EQ(queried, stamped->getWorkspaceSizeInBytes());
}

TEST(CubReduction, RejectsNonFiniteOutputScale) {
    EXPECT_THROW(static_cast<void>(CubReduction(CubReductionOp::Sum,
                                                0,
                                                DataType::FP32,
                                                std::numeric_limits<float>::infinity())),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(CubReduction(CubReductionOp::Sum,
                                                0,
                                                DataType::FP32,
                                                std::numeric_limits<float>::quiet_NaN())),
                 std::invalid_argument);
}
