#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"
#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8OneWindowSelector.h"
#include "Utilities/Common/LowPrecisionFloat.h"

#include <algorithm>
#include <array>
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
        if (dtype == DataType::FP16 || dtype == DataType::BF16 || dtype == DataType::FP32 ||
            dtype == DataType::FP8_E4M3 || dtype == DataType::FP8_E5M2) {
            // Original FP8 RK now uses the retained physical selector rather
            // than the old generic legacy path.
            EXPECT_TRUE(stamped->usesModernRKFamilyPlan());
            if (dtype == DataType::FP8_E4M3 || dtype == DataType::FP8_E5M2) {
                const auto passes = stamped->getDensePhysicalPassDiagnostics();
                ASSERT_FALSE(passes.empty());
                EXPECT_EQ(passes.front().family.find("retained_fp8_rk:"), 0U);
            }
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
}

#if THOR_CUB_ENABLE_FP8_TYPES
TEST(CubReduction, UnderSuppliedFp8OrdinaryDenseUsesRetainedFirstStageWithoutLegacyFallback) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t outer_size = 2;
    constexpr uint64_t reduction_size = 256;
    // These K values exercise differing natural packet alignments on an under-supplied FP8 RK reduction.
    // The retained selector chooses the first-stage physical kernel; no legacy packet-width policy is fixed here.
    // Verify that the stamped production path starts with a retained FP8 kernel and produces correct values.
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
            EXPECT_EQ(stamped->getModernRKStrategyChain().find("rk:generic"), std::string::npos);
            const auto physical_passes = stamped->getDensePhysicalPassDiagnostics();
            ASSERT_FALSE(physical_passes.empty());
            EXPECT_EQ(physical_passes.front().family.find("retained_fp8_rk:"), 0U);
            EXPECT_EQ(physical_passes.front().input_dtype, dtype);
            EXPECT_EQ(physical_passes.front().output_dtype, DataType::FP32);

            stamped->run();
            stream.synchronize();
            expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), expected, 0.0f);
        }
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
    // 1256: the retained FP16 / BF16 R pass (an R shape whose scratch, if any, is its own; the CUB pass it replaced
    // always had some), and the stamp owns exactly what the query reports.
    const auto passes = stamped->getDensePhysicalPassDiagnostics();
    ASSERT_EQ(passes.size(), 1U);
    EXPECT_EQ(passes.front().family.rfind("retained_f16_kr:", 0), 0U) << passes.front().family;
    EXPECT_EQ(stamped->getWorkspaceSizeInBytes(), reduction.queryWorkspaceSizeInBytes(input.getDescriptor(), stream));

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
        {17, 67, 9, 1, 4, 16, 0}, {5, 73, 127, 4, 1, 16, 0}
    };
    const std::vector<Shape> runtime_k = {
        {2, 1200, 513, 1, 1, 16, 496}, {2, 1200, 513, 4, 1, 16, 496},
        {7, 79, 255, 2, 1, 16, 128}
    };

    for (DataType input_dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        std::vector<DataType> outputs = {DataType::FP8_E4M3, DataType::FP8_E5M2,
                                         DataType::FP16, DataType::BF16, DataType::FP32};
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
        {false, 9, 79, 9, 1, 4, 16, 0},
        {false, 5, 73, 127, 4, 1, 16, 0},
        {false, 9, 67, 127, 1, 2, 16, 0},  // O2 subgroups, 8 store slots per lane.
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
                                shape.packet, stream, semantics);
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
                        input, output, geometry, 1, 2, 16, stream, semantics);
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

namespace {
float quantizeFp32ToStorage(float value, DataType dtype) {
    switch (dtype) {
        case DataType::FP8_E4M3: return static_cast<float>(ThorLowPrecision::toFp8E4M3Satfinite(value));
        case DataType::FP8_E5M2: return static_cast<float>(ThorLowPrecision::toFp8E5M2Satfinite(value));
        case DataType::FP16: return __half2float(__float2half_rn(value));
        case DataType::BF16: return __bfloat162float(__float2bfloat16_rn(value));
        default: return value;
    }
}

std::vector<DataType> fp8CompleteOutputDTypes() {
    std::vector<DataType> outputs = {DataType::FP8_E4M3, DataType::FP8_E5M2,
                                     DataType::FP16, DataType::BF16, DataType::FP32};
    return outputs;
}
}  // namespace

// P5-A moved the typed AlignedK Complete finalizer ahead of the output-dtype switch and
// out of line for divide/sqrt. Cover the transformed additive operations (L1/L2/SumSquares)
// and MEAN at every retained Complete configuration, through direct and transposed stores.
TEST(CubReduction, Fp8AlignedKCompleteTransformedFinalizationMatchesFp32ThenStorage) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    constexpr uint64_t outer_size = 3;
    constexpr uint64_t reduction_size = 79;
    constexpr uint64_t inner_size = 520;  // Several warp tiles and a partial last packet tile.
    constexpr float output_scale = -0.5f;
    std::vector<float> values(outer_size * reduction_size * inner_size);
    for (uint64_t o = 0; o < outer_size; ++o)
        for (uint64_t r = 0; r < reduction_size; ++r)
            for (uint64_t k = 0; k < inner_size; ++k) {
                const int pattern = static_cast<int>((o + r + k) % 4);
                values[(o * reduction_size + r) * inner_size + k] = pattern == 0 ? -0.015625f :
                                                                    pattern == 1 ? 0.0625f :
                                                                    pattern == 2 ? 0.03125f : -0.125f;
            }
    const std::vector<CubReductionOp> operations = {CubReductionOp::Mean, CubReductionOp::L1Norm,
                                                    CubReductionOp::L2Norm, CubReductionOp::SumSquares};
    const std::vector<std::pair<size_t, uint32_t>> physical = {{2, 32}, {8, 32}, {8, 64}};
    for (DataType input_dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        SCOPED_TRACE(static_cast<int>(input_dtype));
        Tensor input = makeGpuTensor(values, {outer_size, reduction_size, inner_size}, stream, input_dtype);
        const auto geometry = CubReduction::analyzeGeometry(input.getDimensions(), 1);
        ASSERT_EQ(geometry.path, CubReductionPath::TiledFixedSegment);
        for (const auto& [packet, cta] : physical) {
            SCOPED_TRACE(packet);
            SCOPED_TRACE(cta);
            const KParallelTiledStagePlan plan{TiledRKStageTopology::Complete, packet, cta, 1};
            for (CubReductionOp op : operations) {
                SCOPED_TRACE(static_cast<int>(op));
                const auto semantics = makeValueReductionStageSemantics(
                    op, CubReductionStageRole::Complete, reduction_size);
                for (DataType output_dtype : fp8CompleteOutputDTypes()) {
                    SCOPED_TRACE(static_cast<int>(output_dtype));
                    Tensor output(gpuPlacement, TensorDescriptor(output_dtype, {outer_size, inner_size}));
                    launchKParallelTiledStage(semantics, input, output, geometry, plan, output_scale, stream);
                    stream.synchronize();
                    const auto actual = copyGpuTensorAsFloat(output, stream);
                    ASSERT_EQ(actual.size(), outer_size * inner_size);
                    for (uint64_t o = 0; o < outer_size; ++o) {
                        for (uint64_t k = 0; k < inner_size; ++k) {
                            // Every partial sum is exact in FP32, so accumulation order cannot matter.
                            float sum = 0.0f;
                            for (uint64_t r = 0; r < reduction_size; ++r) {
                                const float x = values[(o * reduction_size + r) * inner_size + k];
                                sum += op == CubReductionOp::Mean ? x
                                     : op == CubReductionOp::L1Norm ? std::fabs(x) : x * x;
                            }
                            if (op == CubReductionOp::Mean) sum /= static_cast<float>(reduction_size);
                            if (op == CubReductionOp::L2Norm) sum = std::sqrt(sum);
                            const float expected = quantizeFp32ToStorage(sum * output_scale, output_dtype);
                            EXPECT_NEAR(actual[o * inner_size + k], expected,
                                        output_dtype == DataType::FP32 ? 1.0e-6f : 0.0f)
                                << "outer=" << o << " k=" << k;
                        }
                    }
                }
            }
        }
    }
}

// P5-A compiles no finalizer into OneWindow / RuntimeK PRODUCT, MIN and MAX kernels.
// Cover those operations through O1 (W1/W4, one and several components per lane),
// O2/O4 subgroups and RuntimeK K tiles, with a runtime scale and every output dtype.
// K=127 at O1 (4 slots) and O2 (8 slots) and RuntimeK (16 slots) cover multi-slot stores.
TEST(CubReduction, Fp8CompleteNonadditiveOperationsSkipFinalizerAndApplyScale) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    struct Shape { bool tiled; uint64_t outer, reduction, inner; uint32_t warps, outputs_per_warp, packet, tile; };
    const std::vector<Shape> shapes = {
        {false, 9, 97, 3, 1, 1, 16, 0}, {false, 9, 97, 3, 4, 1, 16, 0},
        {false, 17, 67, 9, 1, 4, 16, 0}, {false, 5, 73, 127, 1, 1, 16, 0},
        {false, 5, 73, 127, 4, 1, 16, 0}, {false, 9, 67, 127, 1, 2, 16, 0},
        {true, 2, 120, 513, 1, 1, 16, 496}, {true, 7, 79, 255, 2, 1, 16, 128}};
    constexpr float output_scale = -0.5f;
    for (DataType input_dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        for (const Shape& shape : shapes) {
            SCOPED_TRACE(static_cast<int>(input_dtype));
            SCOPED_TRACE(shape.tiled);
            SCOPED_TRACE(shape.inner);
            SCOPED_TRACE(shape.warps);
            SCOPED_TRACE(shape.outputs_per_warp);
            // Rows are 1 except one negative row (component dependent) and one row of 2, all
            // exactly representable in both FP8 formats: PRODUCT = 2 v, MIN = v, MAX = 2.
            std::vector<float> values(shape.outer * shape.reduction * shape.inner, 1.0f);
            std::vector<float> negative(shape.outer * shape.inner);
            for (uint64_t o = 0; o < shape.outer; ++o) {
                for (uint64_t k = 0; k < shape.inner; ++k) {
                    const float v = -0.25f * static_cast<float>(1 + (k + o) % 3);
                    negative[o * shape.inner + k] = v;
                    values[(o * shape.reduction + k % shape.reduction) * shape.inner + k] = v;
                    values[(o * shape.reduction + (k + 1) % shape.reduction) * shape.inner + k] = 2.0f;
                }
            }
            const auto geometry = CubReduction::analyzeValueGeometry(
                CubReductionOp::Sum, {shape.outer, shape.reduction, shape.inner}, {1});
            ASSERT_EQ(geometry.path, CubReductionPath::TiledFixedSegment);
            Tensor input = makeGpuTensor(values, {shape.outer, shape.reduction, shape.inner}, stream, input_dtype);
            for (CubReductionStageCombine combine : {CubReductionStageCombine::Product,
                                                     CubReductionStageCombine::Minimum,
                                                     CubReductionStageCombine::Maximum}) {
                SCOPED_TRACE(static_cast<int>(combine));
                CubReductionStageSemantics semantics;
                semantics.combine = combine;
                for (DataType output_dtype : fp8CompleteOutputDTypes()) {
                    SCOPED_TRACE(static_cast<int>(output_dtype));
                    Tensor output(gpuPlacement, TensorDescriptor(output_dtype, geometry.output_dimensions));
                    if (shape.tiled) {
                        launchFp8GeneralRParallelKTileRCooperativeCalibrationComplete(
                            input, output, geometry, shape.warps, shape.tile, stream, semantics, output_scale);
                    } else {
                        launchFp8GeneralRParallelRCooperativeCalibrationComplete(
                            input, output, geometry, shape.outputs_per_warp, shape.warps, shape.packet, stream,
                            semantics, output_scale);
                    }
                    stream.synchronize();
                    const auto actual = copyGpuTensorAsFloat(output, stream);
                    ASSERT_EQ(actual.size(), negative.size());
                    for (size_t i = 0; i < negative.size(); ++i) {
                        const float reduced = combine == CubReductionStageCombine::Product ? 2.0f * negative[i]
                                            : combine == CubReductionStageCombine::Minimum ? negative[i] : 2.0f;
                        EXPECT_EQ(actual[i], quantizeFp32ToStorage(reduced * output_scale, output_dtype))
                            << "output index=" << i;
                    }
                }
            }
        }
    }
}

// P5-F: OneWindow MAX runs the MIN kernel on sign-flipped FP8 bytes; ABS and SQUARE are folded into the same byte
// mask (SQUARE squares the |x| extremum once per output). Cover every retained ownership (O1 W1/W2/W8, O2/O4/O8,
// P4/P8/P16, one and several components per lane), both FP8 formats, every output dtype and a negative scale.
// Output 0 holds only signed zeros and output 1 holds a NaN: MIN/MAX follow IEEE 754-2019 minimum/maximum
// (NaN propagates, -0 < +0). The reference uses the FP8 values read back from the input tensor. K=1 is not a
// TiledFixedSegment geometry, so the W8 shape uses K=2.
TEST(CubReduction, Fp8OneWindowFoldedMinMaxMatchesSeparateOperations) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    struct Shape { uint64_t outer, reduction, inner; uint32_t outputs_per_warp, warps, packet; };
    const std::vector<Shape> shapes = {
        {5, 97, 3, 1, 1, 16},  {5, 131, 3, 1, 2, 16}, {3, 300, 2, 1, 8, 16}, {5, 67, 17, 2, 1, 16},
        {5, 53, 11, 4, 1, 16}, {9, 41, 5, 8, 1, 16},  {9, 29, 9, 8, 1, 16},  {3, 73, 127, 1, 1, 16},
        {5, 37, 63, 2, 1, 16}};
    const float pattern[8] = {-2.0f, -0.75f, -0.25f, -0.0f, 0.0f, 0.5f, 1.5f, 3.0f};
    const auto ieee_min = [](float a, float b) {
        if (std::isnan(a) || std::isnan(b)) return std::numeric_limits<float>::quiet_NaN();
        if (a == b) return std::signbit(a) ? a : b;
        return a < b ? a : b;
    };
    const auto ieee_max = [](float a, float b) {
        if (std::isnan(a) || std::isnan(b)) return std::numeric_limits<float>::quiet_NaN();
        if (a == b) return std::signbit(a) ? b : a;
        return a > b ? a : b;
    };
    for (DataType input_dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        for (const Shape& shape : shapes) {
            SCOPED_TRACE(static_cast<int>(input_dtype));
            SCOPED_TRACE(shape.inner);
            SCOPED_TRACE(shape.outputs_per_warp);
            SCOPED_TRACE(shape.warps);
            SCOPED_TRACE(shape.packet);
            const uint64_t R = shape.reduction;
            const uint64_t K = shape.inner;
            std::vector<float> values(shape.outer * R * K);
            for (uint64_t o = 0; o < shape.outer; ++o) {
                for (uint64_t r = 0; r < R; ++r) {
                    for (uint64_t k = 0; k < K; ++k) {
                        float v = pattern[(o * 7 + r * 3 + k * 5) % 8];
                        if (o == 0 && k == 0) v = (r % 2 == 0) ? 0.0f : -0.0f;
                        if (o == 1 && k == 0 && r == R / 2) v = std::numeric_limits<float>::quiet_NaN();
                        values[(o * R + r) * K + k] = v;
                    }
                }
            }
            const auto geometry = CubReduction::analyzeValueGeometry(CubReductionOp::Sum, {shape.outer, R, K}, {1});
            ASSERT_EQ(geometry.path, CubReductionPath::TiledFixedSegment);
            Tensor input = makeGpuTensor(values, {shape.outer, R, K}, stream, input_dtype);
            const std::vector<float> stored = copyGpuTensorAsFloat(input, stream);
            ASSERT_EQ(stored.size(), values.size());
            for (CubReductionStageCombine combine :
                 {CubReductionStageCombine::Minimum, CubReductionStageCombine::Maximum}) {
                for (CubReductionStageInputTransform transform : {CubReductionStageInputTransform::Identity,
                                                                  CubReductionStageInputTransform::AbsoluteValue,
                                                                  CubReductionStageInputTransform::Square}) {
                    SCOPED_TRACE(static_cast<int>(combine));
                    SCOPED_TRACE(static_cast<int>(transform));
                    const bool maximum = combine == CubReductionStageCombine::Maximum;
                    std::vector<float> reference(shape.outer * K);
                    for (uint64_t o = 0; o < shape.outer; ++o) {
                        for (uint64_t k = 0; k < K; ++k) {
                            float acc = maximum ? -std::numeric_limits<float>::infinity()
                                                : std::numeric_limits<float>::infinity();
                            for (uint64_t r = 0; r < R; ++r) {
                                float x = stored[(o * R + r) * K + k];
                                if (transform == CubReductionStageInputTransform::AbsoluteValue) x = std::fabs(x);
                                if (transform == CubReductionStageInputTransform::Square) x = x * x;
                                acc = maximum ? ieee_max(acc, x) : ieee_min(acc, x);
                            }
                            reference[o * K + k] = acc;
                        }
                    }
                    CubReductionStageSemantics semantics;
                    semantics.combine = combine;
                    semantics.input_transform = transform;
                    for (float output_scale : {1.0f, -0.5f}) {
                        for (DataType output_dtype : fp8CompleteOutputDTypes()) {
                            SCOPED_TRACE(output_scale);
                            SCOPED_TRACE(static_cast<int>(output_dtype));
                            Tensor output(gpuPlacement, TensorDescriptor(output_dtype, geometry.output_dimensions));
                            launchFp8GeneralRParallelRCooperativeCalibrationComplete(
                                input, output, geometry, shape.outputs_per_warp, shape.warps, shape.packet, stream,
                                semantics, output_scale);
                            stream.synchronize();
                            const auto actual = copyGpuTensorAsFloat(output, stream);
                            ASSERT_EQ(actual.size(), reference.size());
                            for (size_t i = 0; i < reference.size(); ++i) {
                                if (std::isnan(reference[i])) {
                                    EXPECT_TRUE(std::isnan(actual[i])) << "output index=" << i;
                                    continue;
                                }
                                const float expected = quantizeFp32ToStorage(reference[i] * output_scale, output_dtype);
                                EXPECT_EQ(actual[i], expected) << "output index=" << i;
                                if (output_dtype == DataType::FP32 && expected == 0.0f) {
                                    EXPECT_EQ(std::signbit(actual[i]), std::signbit(expected))
                                        << "signed zero, output index=" << i;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Every FP8 encoding of both formats through the O1 and the multi-output OneWindow decode sites (and the MIN/MAX sign
// fold). Two identical rows give SUM 0 + v + v, PRODUCT v*v and MIN/MAX v, all exact in FP32, so each encoding must
// decode exactly (NaN as NaN, signed zeros kept; SUM starts from +0, so -0 sums to +0). The reference is the input
// tensor read back by the host.
TEST(CubReduction, Fp8OneWindowDecodeCoversEveryEncoding) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    const auto decode = [](uint32_t bits, bool e4m3) {
        const float sign = (bits & 0x80U) != 0 ? -1.0f : 1.0f;
        const uint32_t magnitude = bits & 0x7fU;
        if (e4m3) {
            if (magnitude == 0x7fU) return std::numeric_limits<float>::quiet_NaN();
            const uint32_t exponent = magnitude >> 3, mantissa = magnitude & 7U;
            return sign * (exponent == 0 ? std::ldexp(static_cast<float>(mantissa), -9)
                                         : std::ldexp(static_cast<float>(8 + mantissa), static_cast<int>(exponent) - 10));
        }
        const uint32_t exponent = magnitude >> 2, mantissa = magnitude & 3U;
        if (exponent == 31) return mantissa == 0 ? sign * std::numeric_limits<float>::infinity()
                                                 : std::numeric_limits<float>::quiet_NaN();
        return sign * (exponent == 0 ? std::ldexp(static_cast<float>(mantissa), -16)
                                     : std::ldexp(static_cast<float>(4 + mantissa), static_cast<int>(exponent) - 17));
    };
    struct Shape { uint32_t outputs_per_warp, packet; };
    for (DataType input_dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        SCOPED_TRACE(static_cast<int>(input_dtype));
        // Two outputs of K=128 hold all 256 encodings; both rows of each output are identical.
        std::vector<float> values(2 * 2 * 128);
        for (uint32_t bits = 0; bits < 256; ++bits) {
            const uint32_t o = bits / 128, k = bits % 128;
            for (uint32_t r = 0; r < 2; ++r) {
                values[(o * 2 + r) * 128 + k] = decode(bits, input_dtype == DataType::FP8_E4M3);
            }
        }
        const auto geometry = CubReduction::analyzeValueGeometry(CubReductionOp::Sum, {2, 2, 128}, {1});
        ASSERT_EQ(geometry.path, CubReductionPath::TiledFixedSegment);
        Tensor input = makeGpuTensor(values, {2, 2, 128}, stream, input_dtype);
        const std::vector<float> readback = copyGpuTensorAsFloat(input, stream);
        ASSERT_EQ(readback.size(), values.size());
        std::vector<float> stored(256);
        for (uint32_t i = 0; i < 256; ++i) {
            stored[i] = readback[((i / 128) * 2) * 128 + i % 128];
            ASSERT_TRUE((std::isnan(stored[i]) && std::isnan(readback[((i / 128) * 2 + 1) * 128 + i % 128])) ||
                        stored[i] == readback[((i / 128) * 2 + 1) * 128 + i % 128]);
        }
        for (const Shape& shape : {Shape{1, 16}, Shape{2, 16}}) {
            SCOPED_TRACE(shape.outputs_per_warp);
            for (CubReductionStageCombine combine : {CubReductionStageCombine::Sum, CubReductionStageCombine::Product,
                                                     CubReductionStageCombine::Minimum,
                                                     CubReductionStageCombine::Maximum}) {
                SCOPED_TRACE(static_cast<int>(combine));
                CubReductionStageSemantics semantics;
                semantics.combine = combine;
                Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, geometry.output_dimensions));
                launchFp8GeneralRParallelRCooperativeCalibrationComplete(
                    input, output, geometry, shape.outputs_per_warp, 1, shape.packet, stream, semantics, 1.0f);
                stream.synchronize();
                const auto actual = copyGpuTensorAsFloat(output, stream);
                ASSERT_EQ(actual.size(), stored.size());
                for (size_t i = 0; i < stored.size(); ++i) {
                    const float v = stored[i];
                    const float expected = combine == CubReductionStageCombine::Sum       ? (0.0f + v) + v
                                         : combine == CubReductionStageCombine::Product ? v * v
                                                                                         : v;
                    if (std::isnan(expected)) {
                        EXPECT_TRUE(std::isnan(actual[i])) << "encoding index=" << i;
                    } else {
                        EXPECT_EQ(actual[i], expected) << "encoding index=" << i;
                        EXPECT_EQ(std::signbit(actual[i]), std::signbit(expected)) << "encoding index=" << i;
                    }
                }
            }
        }
    }
}

// OneWindow units (bucketed, runtime K; P5-J2). Every unit must match a CPU reference exactly: SUM/MIN/MAX use
// quarters with small sums, PRODUCT mostly ones, so every result is exact in FP32 regardless of order. (1092-1097
// also checked bit identity against the former compile-time-K kernels.) The K list covers every compiled unit with
// W1/W2/W4, all operations and transforms, and the selector-facing launcher maps each P16 variant to its unit.
TEST(CubReduction, Fp8OneWindowUnitsMatchReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    constexpr uint64_t Outer = 5;
    constexpr uint64_t R = 37;
    const std::array<float, 8> pattern = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f};
    const std::array<uint64_t, 17> ks = {2, 3, 7, 17, 63, 127, 129, 497, 6, 10, 20, 24, 33, 65, 111, 200, 300};
    EXPECT_TRUE(fp8OneWindowRuntimeKUnits(600).empty());
    EXPECT_TRUE(fp8OneWindowRuntimeKUnits(505).empty());  // P16 O1 holds 497 bytes of a K that is not a multiple of 16
    EXPECT_FALSE(fp8OneWindowRuntimeKUnits(512).empty());
    for (uint64_t k = 1; k <= 520; ++k) {
        EXPECT_EQ(!fp8OneWindowRuntimeKUnits(k).empty(), Fp8RK::Fp8OneWindowPolicy::oneWindowCoversK(k)) << k;
    }
    {
        const auto units = fp8OneWindowRuntimeKUnits(65);
        const Fp8OneWindowRuntimeKUnit o2_c8_la{2, 16, 8, true};
        EXPECT_NE(std::find(units.begin(), units.end(), o2_c8_la), units.end());
        // O4/O8 one-component buckets and P4/P8 variants have no unit: the selector-facing query throws.
        EXPECT_THROW(static_cast<void>(queryFp8GeneralRParallelRCooperativeCalibrationOccupancy(
                         DataType::FP8_E4M3, 6, 4, 1, 16)), std::logic_error);
        EXPECT_THROW(static_cast<void>(queryFp8GeneralRParallelRCooperativeCalibrationOccupancy(
                         DataType::FP8_E4M3, 9, 8, 1, 8)), std::logic_error);
    }
    for (DataType input_dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        SCOPED_TRACE(static_cast<int>(input_dtype));
        for (uint64_t K : ks) {
            SCOPED_TRACE(K);
            std::vector<float> additive(Outer * R * K);
            std::vector<float> multiplicative(Outer * R * K, 1.0f);
            for (uint64_t i = 0; i < additive.size(); ++i) {
                additive[i] = pattern[(i * 5 + i / K) % pattern.size()];
                if (i % 7 == 3) multiplicative[i] = (i % 3 == 0) ? -1.0f : ((i % 3 == 1) ? 2.0f : 0.5f);
            }
            const auto geometry = CubReduction::analyzeValueGeometry(CubReductionOp::Sum, {Outer, R, K}, {1});
            ASSERT_EQ(geometry.path, CubReductionPath::TiledFixedSegment);
            Tensor additive_input = makeGpuTensor(additive, {Outer, R, K}, stream, input_dtype);
            Tensor product_input = makeGpuTensor(multiplicative, {Outer, R, K}, stream, input_dtype);
            const std::vector<float> additive_stored = copyGpuTensorAsFloat(additive_input, stream);
            const std::vector<float> product_stored = copyGpuTensorAsFloat(product_input, stream);
            const auto units = fp8OneWindowRuntimeKUnits(K);
            ASSERT_FALSE(units.empty());
            for (const Fp8OneWindowRuntimeKUnit& unit : units) {
                const uint32_t o = unit.outputs_per_warp;
                for (uint32_t w : {1u, 2u, 4u}) {
                    if (o > 1 && w != 1) continue;
                    {
                        SCOPED_TRACE(o);
                        SCOPED_TRACE(w);
                        SCOPED_TRACE(unit.packet_bytes);
                        SCOPED_TRACE(unit.max_components_per_lane);
                        SCOPED_TRACE(unit.load_ahead);
                        for (CubReductionStageCombine combine :
                             {CubReductionStageCombine::Sum, CubReductionStageCombine::Product,
                              CubReductionStageCombine::Minimum, CubReductionStageCombine::Maximum}) {
                            for (CubReductionStageInputTransform transform :
                                 {CubReductionStageInputTransform::Identity,
                                  CubReductionStageInputTransform::AbsoluteValue,
                                  CubReductionStageInputTransform::Square}) {
                                SCOPED_TRACE(static_cast<int>(combine));
                                SCOPED_TRACE(static_cast<int>(transform));
                                const bool product = combine == CubReductionStageCombine::Product;
                                const Tensor& input = product ? product_input : additive_input;
                                const std::vector<float>& stored = product ? product_stored : additive_stored;
                                CubReductionStageSemantics semantics;
                                semantics.combine = combine;
                                semantics.input_transform = transform;
                                constexpr float output_scale = -0.5f;
                                const auto occupancy = queryFp8OneWindowRuntimeKOccupancy(
                                    input_dtype, K, unit, w, combine);
                                ASSERT_GT(occupancy.max_active_blocks_per_sm, 0);
                                Tensor output(gpuPlacement,
                                              TensorDescriptor(DataType::FP32, geometry.output_dimensions));
                                launchFp8OneWindowRuntimeKComplete(input, output, geometry, unit, w, stream, semantics,
                                                                   output_scale);
                                stream.synchronize();
                                const auto actual = copyGpuTensorAsFloat(output, stream);
                                ASSERT_EQ(actual.size(), Outer * K);
                                {
                                    // The selector-facing launcher runs this same unit for variant (O, W, P16).
                                    Tensor via_variant(gpuPlacement,
                                                       TensorDescriptor(DataType::FP32, geometry.output_dimensions));
                                    launchFp8GeneralRParallelRCooperativeCalibrationComplete(
                                        input, via_variant, geometry, o, w, unit.packet_bytes, stream, semantics,
                                        output_scale);
                                    stream.synchronize();
                                    const auto variant_result = copyGpuTensorAsFloat(via_variant, stream);
                                    for (size_t i = 0; i < actual.size(); ++i) {
                                        ASSERT_EQ(actual[i], variant_result[i]) << "variant launcher, index=" << i;
                                    }
                                }
                                for (uint64_t outer = 0; outer < Outer; ++outer) {
                                    for (uint64_t k = 0; k < K; ++k) {
                                        float acc = combine == CubReductionStageCombine::Sum       ? 0.0f
                                                  : product                                       ? 1.0f
                                                  : combine == CubReductionStageCombine::Minimum
                                                      ? std::numeric_limits<float>::infinity()
                                                      : -std::numeric_limits<float>::infinity();
                                        for (uint64_t r = 0; r < R; ++r) {
                                            float x = stored[(outer * R + r) * K + k];
                                            if (transform == CubReductionStageInputTransform::AbsoluteValue) {
                                                x = std::fabs(x);
                                            } else if (transform == CubReductionStageInputTransform::Square) {
                                                x = x * x;
                                            }
                                            acc = combine == CubReductionStageCombine::Sum ? acc + x
                                                : product                                 ? acc * x
                                                : combine == CubReductionStageCombine::Minimum ? std::min(acc, x)
                                                                                                : std::max(acc, x);
                                        }
                                        ASSERT_EQ(actual[outer * K + k], acc * output_scale)
                                            << "outer=" << outer << " k=" << k;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// P5-F (RuntimeK): rotated MIN/MAX fold ABS/SQUARE into the FP8 bytes (PrefixReserved also folds the MAX sign into
// its MIN kernel; Exact keeps MIN and MAX). The rotated first stage unfolds each FP32 partial before storing it, so its
// partials must equal the requested operation's partials exactly. Packet-complete (not folded) is covered as a control.
// Outer 0 / component 0 holds only signed zeros, outer 1 / component 1 holds one NaN.
TEST(CubReduction, Fp8RuntimeKFoldedMinMaxMatchesSeparateOperations) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    const float pattern[8] = {-2.0f, -0.75f, -0.25f, -0.0f, 0.0f, 0.5f, 1.5f, 3.0f};
    const auto ieee_min = [](float a, float b) {
        if (std::isnan(a) || std::isnan(b)) return std::numeric_limits<float>::quiet_NaN();
        if (a == b) return std::signbit(a) ? a : b;
        return a < b ? a : b;
    };
    const auto ieee_max = [](float a, float b) {
        if (std::isnan(a) || std::isnan(b)) return std::numeric_limits<float>::quiet_NaN();
        if (a == b) return std::signbit(a) ? b : a;
        return a > b ? a : b;
    };
    const auto make_values = [&](uint64_t outer, uint64_t R, uint64_t K) {
        std::vector<float> values(outer * R * K);
        for (uint64_t o = 0; o < outer; ++o) {
            for (uint64_t r = 0; r < R; ++r) {
                for (uint64_t k = 0; k < K; ++k) {
                    float v = pattern[(o * 7 + r * 3 + k * 5) % 8];
                    if (o == 0 && k == 0) v = (r % 2 == 0) ? 0.0f : -0.0f;
                    if (o == 1 && k == 1 && r == R / 2) v = std::numeric_limits<float>::quiet_NaN();
                    values[(o * R + r) * K + k] = v;
                }
            }
        }
        return values;
    };
    // Reduce rows [row_begin, row_end) of one (outer, component) with the requested operation.
    const auto reference = [&](const std::vector<float>& stored, uint64_t R, uint64_t K, uint64_t o, uint64_t k,
                               uint64_t row_begin, uint64_t row_end, bool maximum,
                               CubReductionStageInputTransform transform) {
        float acc = maximum ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
        for (uint64_t r = row_begin; r < row_end; ++r) {
            float x = stored[(o * R + r) * K + k];
            if (transform == CubReductionStageInputTransform::AbsoluteValue) x = std::fabs(x);
            if (transform == CubReductionStageInputTransform::Square) x = x * x;
            acc = maximum ? ieee_max(acc, x) : ieee_min(acc, x);
        }
        return acc;
    };
    const auto expect_same = [](float actual, float expected, const char* what, size_t index) {
        if (std::isnan(expected)) {
            EXPECT_TRUE(std::isnan(actual)) << what << " index=" << index;
            return;
        }
        EXPECT_EQ(actual, expected) << what << " index=" << index;
        if (expected == 0.0f) {
            EXPECT_EQ(std::signbit(actual), std::signbit(expected)) << what << " signed zero, index=" << index;
        }
    };
    const std::vector<CubReductionStageInputTransform> transforms = {
        CubReductionStageInputTransform::Identity, CubReductionStageInputTransform::AbsoluteValue,
        CubReductionStageInputTransform::Square};

    for (DataType input_dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        SCOPED_TRACE(static_cast<int>(input_dtype));
        // Packet-complete: W1/W2/W4, several K tiles, ragged last tile.
        struct PacketShape { uint64_t outer, reduction, inner; uint32_t warps, tile; };
        for (const PacketShape& shape : std::vector<PacketShape>{
                 {2, 120, 513, 1, 496}, {7, 79, 255, 2, 128}, {3, 64, 1000, 4, 496}}) {
            SCOPED_TRACE(shape.inner);
            SCOPED_TRACE(shape.warps);
            const uint64_t R = shape.reduction;
            const uint64_t K = shape.inner;
            const auto values = make_values(shape.outer, R, K);
            const auto geometry = CubReduction::analyzeValueGeometry(CubReductionOp::Sum, {shape.outer, R, K}, {1});
            Tensor input = makeGpuTensor(values, {shape.outer, R, K}, stream, input_dtype);
            const std::vector<float> stored = copyGpuTensorAsFloat(input, stream);
            for (CubReductionStageCombine combine :
                 {CubReductionStageCombine::Minimum, CubReductionStageCombine::Maximum}) {
                for (CubReductionStageInputTransform transform : transforms) {
                    SCOPED_TRACE(static_cast<int>(combine));
                    SCOPED_TRACE(static_cast<int>(transform));
                    const bool maximum = combine == CubReductionStageCombine::Maximum;
                    const auto occupancy =
                        queryFp8GeneralRParallelKTileRCooperativeCalibrationOccupancy(input_dtype, transform, combine);
                    ASSERT_GT(occupancy.max_active_blocks_per_sm, 0);
                    CubReductionStageSemantics semantics;
                    semantics.combine = combine;
                    semantics.input_transform = transform;
                    for (float output_scale : {1.0f, -0.5f}) {
                        for (DataType output_dtype : fp8CompleteOutputDTypes()) {
                            SCOPED_TRACE(output_scale);
                            SCOPED_TRACE(static_cast<int>(output_dtype));
                            Tensor output(gpuPlacement, TensorDescriptor(output_dtype, geometry.output_dimensions));
                            launchFp8GeneralRParallelKTileRCooperativeCalibrationComplete(
                                input, output, geometry, shape.warps, shape.tile, stream, semantics, output_scale);
                            stream.synchronize();
                            const auto actual = copyGpuTensorAsFloat(output, stream);
                            ASSERT_EQ(actual.size(), shape.outer * K);
                            for (uint64_t o = 0; o < shape.outer; ++o) {
                                for (uint64_t k = 0; k < K; ++k) {
                                    const float reduced = reference(stored, R, K, o, k, 0, R, maximum, transform);
                                    const size_t i = o * K + k;
                                    if (std::isnan(reduced)) {
                                        EXPECT_TRUE(std::isnan(actual[i])) << "packet index=" << i;
                                    } else if (output_dtype == DataType::FP32) {
                                        expect_same(actual[i], reduced * output_scale, "packet", i);
                                    } else {
                                        EXPECT_EQ(actual[i], quantizeFp32ToStorage(reduced * output_scale, output_dtype))
                                            << "packet index=" << i;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Rotated first stage: both p16 layouts (Exact needs its distributed tail at K=513), FP32 partials.
        constexpr uint64_t Outer = 2;
        constexpr uint64_t R = 257;
        for (uint64_t K : {uint64_t{513}, uint64_t{200}}) {
            SCOPED_TRACE(K);
            const auto values = make_values(Outer, R, K);
            const auto geometry = CubReduction::analyzeValueGeometry(CubReductionOp::Sum, {Outer, R, K}, {1});
            Tensor input = makeGpuTensor(values, {Outer, R, K}, stream, input_dtype);
            const std::vector<float> stored = copyGpuTensorAsFloat(input, stream);
            for (bool exact_tile : {false, true}) {
                for (uint64_t rows : {uint64_t{64}, uint64_t{128}}) {
                    const uint64_t shards = (R + rows - 1) / rows;
                    for (CubReductionStageCombine combine :
                         {CubReductionStageCombine::Minimum, CubReductionStageCombine::Maximum}) {
                        for (CubReductionStageInputTransform transform : transforms) {
                            SCOPED_TRACE(exact_tile);
                            SCOPED_TRACE(rows);
                            SCOPED_TRACE(static_cast<int>(combine));
                            SCOPED_TRACE(static_cast<int>(transform));
                            const bool maximum = combine == CubReductionStageCombine::Maximum;
                            const auto occupancy = queryFp8GeneralRParallelKTileRotatedCalibrationOccupancy(
                                input_dtype, exact_tile, rows, transform, combine);
                            ASSERT_GT(occupancy.max_active_blocks_per_sm, 0);
                            Tensor partials(gpuPlacement, TensorDescriptor(DataType::FP32, {Outer, shards, K}));
                            launchFp8GeneralRParallelKTileRotatedCalibrationFirstStage(
                                input, partials, geometry, exact_tile, rows, shards, stream, transform, combine);
                            stream.synchronize();
                            const auto actual = copyGpuTensorAsFloat(partials, stream);
                            ASSERT_EQ(actual.size(), Outer * shards * K);
                            for (uint64_t o = 0; o < Outer; ++o) {
                                for (uint64_t shard = 0; shard < shards; ++shard) {
                                    const uint64_t row_begin = shard * rows;
                                    const uint64_t row_end = std::min(R, row_begin + rows);
                                    for (uint64_t k = 0; k < K; ++k) {
                                        const size_t i = (o * shards + shard) * K + k;
                                        expect_same(actual[i],
                                                    reference(stored, R, K, o, k, row_begin, row_end, maximum, transform),
                                                    "rotated", i);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Both retained RuntimeK p16 rotated layouts must transform decoded FP8 values
// BEFORE producing per-shard FP32 sums. PacketStaged was removed from the
// inventory and must not be exercised by this test.
TEST(CubReduction, Fp8RuntimeKRotatedStagedCompiledInputTransformBeforePartials) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    constexpr uint64_t Outer = 2;
    constexpr uint64_t R = 257;
    constexpr uint64_t K = 513; // awkward p16 tails in both rotated layouts
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
            for (bool exact_tile : {false, true}) {
                constexpr uint64_t rows = 64;
                const uint64_t shards = (R + rows - 1) / rows;
                Tensor partials(gpuPlacement, TensorDescriptor(
                    DataType::FP32, {Outer, shards, K}));
                SCOPED_TRACE(static_cast<int>(dtype));
                SCOPED_TRACE(static_cast<int>(transform));
                SCOPED_TRACE(exact_tile);
                const auto occupancy = queryFp8GeneralRParallelKTileRotatedCalibrationOccupancy(
                    dtype, exact_tile, rows, transform);
                ASSERT_GT(occupancy.max_active_blocks_per_sm, 0);
                launchFp8GeneralRParallelKTileRotatedCalibrationFirstStage(
                    input, partials, geometry, exact_tile, rows, shards, stream, transform);
                stream.synchronize();
                const auto actual = copyGpuTensorAsFloat(partials, stream);
                ASSERT_EQ(actual.size(), Outer * shards * K);
                for (uint64_t o = 0; o < Outer; ++o) {
                    for (uint64_t shard = 0; shard < shards; ++shard) {
                        const uint64_t row_begin = shard * rows;
                        const uint64_t row_end = std::min(R, row_begin + rows);
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

TEST(CubReduction, Fp8CompleteRuntimeFinalizationValidatesSemantics) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    for (bool tiled : {false, true}) {
        const uint64_t k = tiled ? 513 : 3;
        const auto geometry = CubReduction::analyzeValueGeometry(CubReductionOp::Sum, {1, 79, k}, {1});
        Tensor input = makeGpuTensor(std::vector<float>(79 * k, 0.5f),
                                     {1, 79, k}, stream, DataType::FP8_E4M3);
        Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, geometry.output_dimensions));
        const auto launch = [&](const CubReductionStageSemantics& semantics) {
            if (tiled) {
                launchFp8GeneralRParallelKTileRCooperativeCalibrationComplete(
                    input, output, geometry, 1, 496, stream, semantics);
            } else {
                launchFp8GeneralRParallelRCooperativeCalibrationComplete(
                    input, output, geometry, 1, 1, 16, stream, semantics);
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
        // Product is a retained compile-time complete-kernel operation for both
        // OneWindow and RuntimeK. Its FP32 result must be correct, not rejected.
        semantics = {};
        semantics.combine = CubReductionStageCombine::Product;
        ASSERT_NO_THROW(launch(semantics));
        const auto products = copyGpuTensorAsFloat(output, stream);
        ASSERT_EQ(products.size(), k);
        for (float value : products) {
            EXPECT_FLOAT_EQ(value, std::ldexp(1.0f, -79));
        }
        // Nonadditive combines still cannot request an additive finalizer.
        semantics.finalize = CubReductionStageFinalize::Divide;
        semantics.finalize_divisor = 79;
        EXPECT_THROW(launch(semantics), std::logic_error);
        semantics = {};
        semantics.combine = static_cast<CubReductionStageCombine>(255);
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
