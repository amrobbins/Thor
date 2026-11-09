// 1314 (phase 5, plans/Thor_Ragged_Reduction_Plan_2026-11-06.md): the ragged RK ARG host launcher, rows of R = D x
// element bytes >= 16, FP8 first (1320: FP16 / BF16; 1321: FP32). It resolves the launch as the RK VALUE launcher does
// (CubRaggedRK.cu: the K tile cap, the geometry, the path ranges and the grid; here with the ARG records), validates,
// launches the main kernel (CubRaggedRKArgFp8.cu, 1320 CubRaggedRKArgF16.cu, 1321 CubRaggedRKArgFp32.cu; the kernel
// RaggedRKArgKernels.cuh) and then the ARG finisher (launchRaggedFinishArg: the block finisher up to D 15, the wide one
// above; 1321 FP32's pairs: launchRaggedFinishArgPair, the same two forms).
#include "Utilities/TensorOperations/Cub/F16ArgKey.h"
#include "Utilities/TensorOperations/Cub/RaggedRKArgKernels.cuh"

#include "Utilities/Common/ScopedGpu.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ThorImplementation::CubReductionInternal {
namespace {

bool raggedRKArgIsFp8(DataType dtype) { return dtype == DataType::FP8_E4M3 || dtype == DataType::FP8_E5M2; }
bool raggedRKArgIsF16(DataType dtype) { return dtype == DataType::FP16 || dtype == DataType::BF16; }  // 1320
bool raggedRKArgHasKernels(DataType dtype) {
    return raggedRKArgIsFp8(dtype) || raggedRKArgIsF16(dtype) || dtype == DataType::FP32;  // 1321: FP32
}

// A column's head and tail record bytes per range (two 64-bit words), for the auto K tile cap; 1321: FP32's pairs (a
// 4-byte key and an 8-byte row each).
constexpr uint32_t RAGGED_RK_ARG_RECORD_BYTES = 16;
constexpr uint32_t RAGGED_RK_ARG_PAIR_RECORD_BYTES = 24;

}  // namespace

bool raggedRKArgSupported(DataType input_dtype, uint64_t components) {
    // Rows of 16 bytes and more: FP8 D >= 16 (1314), FP16 / BF16 D >= 8 (1320), FP32 D >= 4 (1321).
    if (!raggedRKArgHasKernels(input_dtype)) return false;
    const uint64_t bytes = raggedKRInputBytes(input_dtype);
    return components <= RAGGED_FINISH_MAX_COMPONENTS &&
           components <= RaggedMergePath::RAGGED_RK_MAX_ROW_BYTES / bytes &&
           components * bytes >= RaggedMergePath::RAGGED_RK_MIN_ROW_BYTES;
}

uint32_t raggedRKArgCtas(const RaggedRKShape& shape, DataType input_dtype, int gpu) {
    if (!raggedRKShapeLegal(shape))
        throw std::invalid_argument("raggedRKArgCtas: illegal shape " + raggedRKShapeName(shape));
    if (!raggedRKArgHasKernels(input_dtype))
        throw std::invalid_argument("raggedRKArgCtas: ragged RK ARG takes FP8, FP16, BF16 or FP32 input");
    ScopedGpu scoped_gpu(gpu);
    int sms = 0;
    RaggedKR::raggedKRCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, gpu), "SM count");
    const int resident = raggedRKArgIsFp8(input_dtype)   ? RaggedKR::raggedRKArgResidentFp8()
                         : raggedRKArgIsF16(input_dtype) ? RaggedKR::raggedRKArgResidentF16()
                                                         : RaggedKR::raggedRKArgResidentFp32();  // 1321
    const uint32_t per_sm = std::max<uint32_t>(
        1, shape.ctas_per_sm == 0 ? static_cast<uint32_t>(resident)
                                  : std::min<uint32_t>(shape.ctas_per_sm, static_cast<uint32_t>(resident)));
    return static_cast<uint32_t>(sms) * per_sm;
}

RaggedRKLaunch raggedRKArgLaunch(const RaggedRKShape& shape, uint32_t ctas, DataType input_dtype, uint64_t components,
                                 uint64_t capacity_rows) {
    const auto fail = [](const std::string& what) { throw std::invalid_argument("raggedRKArgLaunch: " + what); };
    if (!raggedRKShapeLegal(shape)) fail("illegal shape " + raggedRKShapeName(shape));
    if (ctas == 0) fail("the CTA budget must be at least 1");
    if (!raggedRKArgSupported(input_dtype, components))
        fail("ragged RK ARG takes FP8 (D >= 16), FP16 / BF16 (D >= 8) or FP32 (D >= 4) input: rows of 16 bytes and "
             "more");
    const uint32_t bytes = raggedKRInputBytes(input_dtype);
    const uint64_t row_bytes = components * bytes;
    const bool pairs = input_dtype == DataType::FP32;  // 1321: FP32's pair records
    RaggedRKLaunch launch;
    launch.tile_cap = shape.tile_bytes != 0
                          ? shape.tile_bytes
                          : RaggedMergePath::raggedRKAutoTileCap(
                                capacity_rows * row_bytes, ctas, bytes,
                                pairs ? RAGGED_RK_ARG_PAIR_RECORD_BYTES : RAGGED_RK_ARG_RECORD_BYTES);
    launch.geometry = RaggedMergePath::raggedRKGeometry(row_bytes, launch.tile_cap);
    launch.ranges = RaggedMergePath::raggedRKRanges(ctas, launch.geometry.ktiles, RAGGED_FINISH_MAX_GRID);
    const uint64_t grid = uint64_t{launch.ranges} * launch.geometry.ktiles;
    if (grid > 0x7fffffffULL) fail("too many K tiles");
    launch.grid = static_cast<uint32_t>(grid);
    launch.workspace_bytes =
        pairs ? RaggedMergePath::raggedArgPairRecordLayout(launch.ranges, static_cast<uint32_t>(components)).bytes
              : RaggedMergePath::raggedArgRecordLayout(launch.ranges, static_cast<uint32_t>(components)).bytes;
    return launch;
}

RaggedMergePath::PathWeights raggedRKArgWeights(const RaggedRKShape& shape, DataType input_dtype,
                                                DataType offsets_dtype, DataType index_dtype, uint64_t components) {
    if (!raggedRKArgSupported(input_dtype, components))
        throw std::invalid_argument(
            "raggedRKArgWeights: ragged RK ARG takes FP8 (D >= 16), FP16 / BF16 (D >= 8) or FP32 (D >= 4) input");
    if (index_dtype != DataType::UINT32 && index_dtype != DataType::UINT64)
        throw std::invalid_argument("raggedRKArgWeights: the index dtype must be UINT32 or UINT64");
    if (!shape.weighted) return RaggedMergePath::PathWeights{};
    const uint64_t offset_bytes = offsets_dtype == DataType::UINT64 ? 8 : 4;
    const uint64_t index_bytes = index_dtype == DataType::UINT64 ? 8 : 4;
    return RaggedMergePath::PathWeights{components * raggedKRInputBytes(input_dtype),
                                        offset_bytes + components * index_bytes + raggedRKEndExtra(shape)};
}

void launchRaggedRKArg(CubArgReductionOp op,
                       const RaggedRKShape& shape,
                       uint32_t ctas,
                       const Tensor& input,
                       const Tensor& offsets,
                       Tensor& index_output,
                       Tensor& workspace,
                       Stream& stream,
                       RaggedLaunchParts parts) {
    const auto fail = [](const std::string& what) { throw std::invalid_argument("launchRaggedRKArg: " + what); };
    if (op != CubArgReductionOp::ArgMin && op != CubArgReductionOp::ArgMax) fail("the operation must be ARGMIN or ARGMAX");
    if (!raggedRKShapeLegal(shape)) fail("illegal shape " + raggedRKShapeName(shape));
    if (!input.isInitialized()) fail("the input must be an FP8, FP16, BF16 or FP32 tensor");
    const DataType input_dtype = input.getDataType();
#if !THOR_CUB_ENABLE_FP8_TYPES
    if (raggedRKArgIsFp8(input_dtype)) fail("FP8 types are disabled in this build");
#endif
    // Each tensor's dimensions are copied once (GCC 14's -Warray-bounds false positive, as launchRaggedRKValue).
    const std::vector<uint64_t> input_dims = input.getDimensions();
    const std::vector<uint64_t> offsets_dims = offsets.isInitialized() ? offsets.getDimensions() : std::vector<uint64_t>{};
    if (input_dims.size() < 2) fail("the input must be [capacity, D...]");
    if (!input.isDenseContiguous()) fail("the input must be dense");
    const uint64_t capacity = input_dims.front();
    uint64_t d = 1;
    for (size_t i = 1; i < input_dims.size(); ++i) d *= input_dims[i];
    if (!raggedRKArgSupported(input_dtype, d))
        fail("the input must be [capacity, D...]: FP8 with D >= 16, FP16 / BF16 with D >= 8 or FP32 with D >= 4");
    if (offsets_dims.size() != 1 || offsets_dims.front() == 0) fail("the offsets must be [segments + 1]");
    if (offsets.getDataType() != DataType::UINT32 && offsets.getDataType() != DataType::UINT64)
        fail("the offsets must be UINT32 or UINT64");
    const uint64_t segments = offsets_dims.front() - 1;
    if (!index_output.isInitialized() || index_output.getTotalNumElements() != segments * d ||
        (index_output.getDataType() != DataType::UINT32 && index_output.getDataType() != DataType::UINT64))
        fail("the index output must hold segments x D UINT32 or UINT64 indices");
    const uint32_t index_bytes = index_output.getDataType() == DataType::UINT64 ? 8 : 4;
    const bool fp8 = raggedRKArgIsFp8(input_dtype);
    const bool fp32 = input_dtype == DataType::FP32;  // 1321: CTA-local rows in the kernel, global rows in pair records
    const uint32_t row_bits = fp8 ? RaggedKR::RaggedArgKeyFp8::RowBits : RaggedKR::RaggedArgKeyF16::RowBits;
    // The packed indices stay below the sentinel, and the rows within the candidate word's row field.
    if (index_bytes == 4 && capacity * d > uint64_t{0xFFFFFFFFU}) fail("a UINT32 index output needs capacity x D <= 2^32 - 1");
    if (!fp32 && capacity >= (uint64_t{1} << row_bits) - 1)
        fail("the capacity must stay below 2^" + std::to_string(row_bits) + " - 1 rows");
    if (index_bytes == 8 && capacity > ~uint64_t{0} / d) fail("capacity x D must fit 64 bits");
    const RaggedRKLaunch launch = raggedRKArgLaunch(shape, ctas, input_dtype, d, capacity);
    if (!workspace.isInitialized() || workspace.getArraySizeInBytes() < launch.workspace_bytes)
        fail("the workspace is too small");
    if (input.getPlacement() != offsets.getPlacement() || input.getPlacement() != index_output.getPlacement() ||
        input.getPlacement() != workspace.getPlacement() ||
        input.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU)
        fail("every tensor must be on the same GPU");
    if (input.getPlacement().getDeviceNum() != stream.getGpuNum()) fail("the stream must be on the tensors' GPU");
    const uintptr_t address = reinterpret_cast<uintptr_t>(input.getMemPtr<void>());
    if (address % raggedKRInputBytes(input_dtype) != 0) fail("the input must be aligned to its element size");
    const RaggedMergePath::PathWeights weights =
        raggedRKArgWeights(shape, input_dtype, offsets.getDataType(), index_output.getDataType(), d);
    // A range's events stay below 2^31 (its relative 32-bit coordinates), as launchRaggedRKValue.
    const uint64_t cost_bound = weights.end * segments + weights.row * capacity;
    const uint64_t cheaper = std::min(weights.row, weights.end);
    if ((cost_bound / launch.ranges + 2 * weights.end + 2) / cheaper >= (uint64_t{1} << 31) ||
        (capacity + segments) / launch.ranges + 2 >= (uint64_t{1} << 31))
        fail("a CTA's range must stay below 2^31 events");
    if (segments == 0) return;

    ScopedGpu scoped_gpu(stream.getGpuNum());
    RaggedKR::RaggedRKArgParams params{};
    params.bytes = reinterpret_cast<const uint8_t*>(address & ~uintptr_t{15});
    params.head = static_cast<uint32_t>(address & 15U);
    params.offsets = RaggedMergePath::Offsets{offsets.getMemPtr<void>(),
                                              offsets.getDataType() == DataType::UINT64 ? 8U : 4U};
    params.segments = segments;
    params.capacity_rows = capacity;
    params.weights = weights;
    if (fp8) {
        params.fold = fp8ArgFold(op == CubArgReductionOp::ArgMax, input_dtype == DataType::FP8_E5M2);
    } else if (fp32) {  // 1321: the sign flip, no NaN bias (raggedFp32ArgKey), as launchRaggedKRArg
        params.fold = Fp8ArgFold{op == CubArgReductionOp::ArgMax ? 0x80000000U : 0U, 0U};
    } else {  // 1320: the F16 fold's values in the same two fields, as launchRaggedKRArg
        const F16ArgFold fold = f16ArgFold(op == CubArgReductionOp::ArgMax, input_dtype == DataType::BF16);
        params.fold = Fp8ArgFold{fold.xor_mask, fold.nan_bias};
    }
    params.output = index_output.getMemPtr<void>();
    params.index_bytes = index_bytes;
    params.workspace = static_cast<uint8_t*>(workspace.getMemPtr<void>());
    params.components = d;
    params.row_bytes = static_cast<uint32_t>(d * raggedKRInputBytes(input_dtype));
    params.ranges = launch.ranges;
    params.ktiles = launch.geometry.ktiles;
    params.tile_bytes = launch.geometry.tile_bytes;
    params.chunks = launch.geometry.chunks;
    params.workers = launch.geometry.workers;
    const cudaStream_t s = stream.getStream();
    if (parts != RaggedLaunchParts::FinishOnly) {
        const uint32_t rows = RaggedMergePath::raggedRKRowsPerWorker(launch.geometry);
        if (fp8) {
            RaggedKR::raggedRKArgLaunchMainFp8(rows, launch.grid, params, s);
        } else if (fp32) {
            RaggedKR::raggedRKArgLaunchMainFp32(rows, launch.grid, params, s);
        } else {
            RaggedKR::raggedRKArgLaunchMainF16(rows, launch.grid, params, s);
        }
    }
    if (parts != RaggedLaunchParts::MainOnly) {
        if (fp32) {  // 1321: the pair finisher (block up to D 15, wide above)
            launchRaggedFinishArgPair(launch.ranges, static_cast<uint32_t>(d), params.workspace, params.output,
                                      index_bytes, s);
        } else {
            launchRaggedFinishArg(launch.ranges, static_cast<uint32_t>(d), params.workspace, row_bits, params.output,
                                  index_bytes, s);
        }
    }
}

}  // namespace ThorImplementation::CubReductionInternal
