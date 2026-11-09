// 1285 (phase 4, plans/Thor_Ragged_Reduction_Plan_2026-11-06.md): the ragged RK VALUE host launcher, rows of
// R = D x element bytes >= 16 (FP8 in 1285; 1295: FP16, BF16 and FP32 too). It resolves the launch (the K tile cap,
// the geometry, the path ranges and the grid: raggedRKLaunch), validates, launches the main kernel of the input's
// dtype object (CubRaggedRKFp8.cu, CubRaggedRKF16.cu, CubRaggedRKFp32.cu, each forwarding narrow rows to its Narrow
// object; the kernel: RaggedRKKernels.cuh) and then the finisher (CubRaggedFinish.cu) over the ranges' records.
#include "Utilities/TensorOperations/Cub/RaggedRKKernels.cuh"

#include "Utilities/Common/ScopedGpu.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ThorImplementation::CubReductionInternal {
namespace {

using RaggedKR::RaggedRKParams;

[[nodiscard]] bool raggedRKOutputSupported(DataType dtype) {
    switch (dtype) {
#if THOR_CUB_ENABLE_FP8_TYPES
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
#endif
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
            return true;
        default:
            return false;
    }
}

// 1295: every input KR takes (1285-1294: FP8 only); 0 for any other dtype.
[[nodiscard]] uint32_t raggedRKInputBytes(DataType input_dtype) { return raggedKRInputBytes(input_dtype); }

// The resident CTAs per SM of the input's RK kernels (the smaller of its two row counts').
[[nodiscard]] int raggedRKResident(uint32_t bytes) {
    if (bytes == 1) return RaggedKR::raggedRKResidentFp8();
    if (bytes == 2) return RaggedKR::raggedRKResidentF16();
    return RaggedKR::raggedRKResidentFp32();
}

[[nodiscard]] bool isPowerOfTwo(uint32_t x) { return x != 0 && (x & (x - 1)) == 0; }

}  // namespace

bool raggedRKShapeLegal(const RaggedRKShape& shape) {
    using RaggedMergePath::RAGGED_RK_MAX_TILE_BYTES;
    using RaggedMergePath::RAGGED_RK_MIN_TILE_BYTES;
    const bool tile_ok = shape.tile_bytes == 0 || (isPowerOfTwo(shape.tile_bytes) &&
                                                   shape.tile_bytes >= RAGGED_RK_MIN_TILE_BYTES &&
                                                   shape.tile_bytes <= RAGGED_RK_MAX_TILE_BYTES);
    return tile_ok && shape.ctas_per_sm <= 32 && shape.end_extra >= -1 && shape.end_extra <= 255;
}

uint32_t raggedRKEndExtra(const RaggedRKShape& shape) {
    if (!shape.weighted) return 0;
    if (shape.end_extra >= 0) return static_cast<uint32_t>(shape.end_extra);
    return 4;  // KR's default (1273)
}

std::string raggedRKShapeName(const RaggedRKShape& shape) {
    return "ragged_rk:t" + (shape.tile_bytes == 0 ? std::string("a") : std::to_string(shape.tile_bytes)) + "_c" +
           std::to_string(shape.ctas_per_sm) +
           (shape.weighted ? "_e" + std::to_string(raggedRKEndExtra(shape)) : std::string("_u"));
}

uint32_t raggedRKCtas(const RaggedRKShape& shape, DataType input_dtype, int gpu) {
    if (!raggedRKShapeLegal(shape))
        throw std::invalid_argument("raggedRKCtas: illegal shape " + raggedRKShapeName(shape));
    const uint32_t bytes = raggedRKInputBytes(input_dtype);
    if (bytes == 0) throw std::invalid_argument("raggedRKCtas: the input must be FP8, FP16, BF16 or FP32");
    ScopedGpu scoped_gpu(gpu);
    int sms = 0;
    RaggedKR::raggedKRCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, gpu), "SM count");
    const int resident = raggedRKResident(bytes);
    const uint32_t per_sm = std::max<uint32_t>(
        1, shape.ctas_per_sm == 0 ? static_cast<uint32_t>(resident)
                                  : std::min<uint32_t>(shape.ctas_per_sm, static_cast<uint32_t>(resident)));
    return static_cast<uint32_t>(sms) * per_sm;
}

RaggedRKLaunch raggedRKLaunch(const RaggedRKShape& shape, uint32_t ctas, DataType input_dtype, uint64_t components,
                              uint64_t capacity_rows) {
    const auto fail = [](const std::string& what) { throw std::invalid_argument("raggedRKLaunch: " + what); };
    if (!raggedRKShapeLegal(shape)) fail("illegal shape " + raggedRKShapeName(shape));
    if (ctas == 0) fail("the CTA budget must be at least 1");
    const uint32_t bytes = raggedRKInputBytes(input_dtype);
    if (bytes == 0) fail("the input must be FP8, FP16, BF16 or FP32");
    if (components == 0 || components > RAGGED_FINISH_MAX_COMPONENTS) fail("D out of range");
    const uint64_t row_bytes = components * bytes;
    if (row_bytes < RaggedMergePath::RAGGED_RK_MIN_ROW_BYTES) fail("rows under 16 bytes are ragged KR's");
    if (row_bytes > RaggedMergePath::RAGGED_RK_MAX_ROW_BYTES) fail("rows over 2 GiB");
    RaggedRKLaunch launch;
    launch.tile_cap = shape.tile_bytes != 0
                          ? shape.tile_bytes
                          : RaggedMergePath::raggedRKAutoTileCap(capacity_rows * row_bytes, ctas, bytes);
    launch.geometry = RaggedMergePath::raggedRKGeometry(row_bytes, launch.tile_cap);
    launch.ranges = RaggedMergePath::raggedRKRanges(ctas, launch.geometry.ktiles, RAGGED_FINISH_MAX_GRID);
    const uint64_t grid = uint64_t{launch.ranges} * launch.geometry.ktiles;
    if (grid > 0x7fffffffULL) fail("too many K tiles");
    launch.grid = static_cast<uint32_t>(grid);
    launch.workspace_bytes =
        RaggedMergePath::raggedRecordLayout(launch.ranges, static_cast<uint32_t>(components)).bytes;
    return launch;
}

RaggedMergePath::PathWeights raggedRKWeights(const RaggedRKShape& shape, DataType input_dtype, DataType offsets_dtype,
                                             DataType output_dtype, uint64_t components) {
    const uint32_t bytes = raggedRKInputBytes(input_dtype);
    if (bytes == 0) throw std::invalid_argument("raggedRKWeights: the input must be FP8, FP16, BF16 or FP32");
    if (components * bytes < RaggedMergePath::RAGGED_RK_MIN_ROW_BYTES)
        throw std::invalid_argument("raggedRKWeights: rows under 16 bytes are ragged KR's");
    if (!shape.weighted) return RaggedMergePath::PathWeights{};
    const uint64_t offset_bytes = offsets_dtype == DataType::UINT64 ? 8 : 4;
    const uint64_t output_bytes = output_dtype == DataType::FP32 ? 4 : (output_dtype == DataType::FP16 ||
                                                                        output_dtype == DataType::BF16 ? 2 : 1);
    return RaggedMergePath::PathWeights{components * bytes,
                                        offset_bytes + components * output_bytes + raggedRKEndExtra(shape)};
}

#if THOR_RAGGED_RK_PROBE
namespace {
unsigned long long* g_ragged_rk_probe = nullptr;
}
void raggedRKSetProbe(unsigned long long* probe) { g_ragged_rk_probe = probe; }
#endif

void launchRaggedRKValue(CubReductionOp op,
                         const RaggedRKShape& shape,
                         uint32_t ctas,
                         const Tensor& input,
                         const Tensor& offsets,
                         Tensor& output,
                         Tensor& workspace,
                         Stream& stream,
                         RaggedLaunchParts parts) {
    const auto fail = [](const std::string& what) { throw std::invalid_argument("launchRaggedRKValue: " + what); };
    if (op != CubReductionOp::Sum && op != CubReductionOp::Mean && op != CubReductionOp::Min &&
        op != CubReductionOp::Max)
        fail("the operation must be SUM, MEAN, MIN or MAX");
    if (!raggedRKShapeLegal(shape)) fail("illegal shape " + raggedRKShapeName(shape));
    if (!input.isInitialized() || raggedRKInputBytes(input.getDataType()) == 0)
        fail("the input must be FP8, FP16, BF16 or FP32");
    const DataType input_dtype = input.getDataType();
    const uint32_t bytes = raggedRKInputBytes(input_dtype);
#if !THOR_CUB_ENABLE_FP8_TYPES
    if (bytes == 1) fail("FP8 types are disabled in this build");
#endif
    // Each tensor's dimensions are copied once (GCC 14's -Warray-bounds false positive on repeated copies).
    const std::vector<uint64_t> input_dims = input.getDimensions();
    const std::vector<uint64_t> offsets_dims =
        offsets.isInitialized() ? offsets.getDimensions() : std::vector<uint64_t>{};
    if (input_dims.size() < 2) fail("the input must be [capacity, D...]");
    if (!input.isDenseContiguous()) fail("the input must be dense");
    const uint64_t capacity = input_dims.front();
    uint64_t d = 1;
    for (size_t i = 1; i < input_dims.size(); ++i) d *= input_dims[i];
    if (d * bytes < RaggedMergePath::RAGGED_RK_MIN_ROW_BYTES) fail("rows under 16 bytes are ragged KR's");
    if (offsets_dims.size() != 1 || offsets_dims.front() == 0) fail("the offsets must be [segments + 1]");
    if (offsets.getDataType() != DataType::UINT32 && offsets.getDataType() != DataType::UINT64)
        fail("the offsets must be UINT32 or UINT64");
    const uint64_t segments = offsets_dims.front() - 1;
    if (!output.isInitialized() || output.getTotalNumElements() != segments * d ||
        !raggedRKOutputSupported(output.getDataType()))
        fail("the output must hold segments x D values of FP8, FP16, BF16 or FP32");
    const RaggedRKLaunch launch = raggedRKLaunch(shape, ctas, input_dtype, d, capacity);
    if (!workspace.isInitialized() || workspace.getArraySizeInBytes() < launch.workspace_bytes)
        fail("the workspace is too small");
    if (input.getPlacement() != offsets.getPlacement() || input.getPlacement() != output.getPlacement() ||
        input.getPlacement() != workspace.getPlacement() ||
        input.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU)
        fail("every tensor must be on the same GPU");
    if (input.getPlacement().getDeviceNum() != stream.getGpuNum()) fail("the stream must be on the tensors' GPU");
    const uintptr_t address = reinterpret_cast<uintptr_t>(input.getMemPtr<void>());
    if (address % bytes != 0) fail("the input must be aligned to its element size");
    const RaggedMergePath::PathWeights weights =
        raggedRKWeights(shape, input_dtype, offsets.getDataType(), output.getDataType(), d);
    // A range's events stay below 2^31 (its relative 32-bit coordinates).
    const uint64_t cost_bound = weights.end * segments + weights.row * capacity;
    const uint64_t cheaper = std::min(weights.row, weights.end);
    if ((cost_bound / launch.ranges + 2 * weights.end + 2) / cheaper >= (uint64_t{1} << 31) ||
        (capacity + segments) / launch.ranges + 2 >= (uint64_t{1} << 31))
        fail("a CTA's range must stay below 2^31 events");
    if (segments == 0) return;

    ScopedGpu scoped_gpu(stream.getGpuNum());
    const int fixed_op = (op == CubReductionOp::Min || op == CubReductionOp::Max) ? 2 : 0;
    // MAX runs as MIN of the sign-flipped input (every element of a word) and negates the result (as KR).
    const uint32_t max_flip = bytes == 1 ? 0x80808080U : (bytes == 2 ? 0x80008000U : 0x80000000U);
    RaggedRKParams params{};
    params.bytes = reinterpret_cast<const uint8_t*>(address & ~uintptr_t{15});
    params.head = static_cast<uint32_t>(address & 15U);
    params.offsets = RaggedMergePath::Offsets{offsets.getMemPtr<void>(),
                                              offsets.getDataType() == DataType::UINT64 ? 8U : 4U};
    params.segments = segments;
    params.capacity_rows = capacity;
    params.weights = weights;
    params.fold = op == CubReductionOp::Max ? Fp8ExtremumFold{0xffffffffU, max_flip, false} : FP8_EXTREMUM_FOLD_NONE;
    params.finalize = RaggedFinalize{op == CubReductionOp::Mean, op == CubReductionOp::Max ? -1.0f : 1.0f};
    params.output = output.getMemPtr<void>();
    params.output_dtype = output.getDataType();
    params.workspace = static_cast<uint8_t*>(workspace.getMemPtr<void>());
    params.components = d;
    params.row_bytes = static_cast<uint32_t>(d * bytes);
    params.ranges = launch.ranges;
    params.ktiles = launch.geometry.ktiles;
    params.tile_bytes = launch.geometry.tile_bytes;
    params.chunks = launch.geometry.chunks;
    params.workers = launch.geometry.workers;
#if THOR_RAGGED_RK_PROBE
    params.probe = g_ragged_rk_probe;
#endif
    const cudaStream_t s = stream.getStream();
    if (parts != RaggedLaunchParts::FinishOnly) {
        const uint32_t rows = RaggedMergePath::raggedRKRowsPerWorker(launch.geometry);  // 1288: 4 on narrow rows
        if (bytes == 1) {
            RaggedKR::raggedRKLaunchMainFp8(input_dtype, fixed_op, rows, launch.grid, params, s);
        } else if (bytes == 2) {
            RaggedKR::raggedRKLaunchMainF16(input_dtype, fixed_op, rows, launch.grid, params, s);
        } else {
            RaggedKR::raggedRKLaunchMainFp32(fixed_op, rows, launch.grid, params, s);
        }
    }
    if (parts != RaggedLaunchParts::MainOnly)
        launchRaggedFinishValue(fixed_op, launch.ranges, static_cast<uint32_t>(d), params.workspace, params.offsets,
                                segments, capacity, params.finalize, params.output, params.output_dtype, s);
}

}  // namespace ThorImplementation::CubReductionInternal
