// 1284 (phase 3b, plans/Thor_Ragged_Reduction_Plan_2026-11-06.md): the ragged KR VALUE host launcher for every input
// dtype (1265-1283: FP8 only, in CubRaggedKRFp8.cu). It validates, builds the main kernels' parameters, and launches
// the main kernel of the input's dtype object (CubRaggedKRFp8.cu, CubRaggedKRF16.cu, CubRaggedKRFp32.cu; the kernels
// are RaggedKRKernels.cuh) and then the finisher (CubRaggedFinish.cu). A row is D elements under 16 bytes: D 1..15 for FP8,
// 1..7 for FP16 / BF16, 1..3 for FP32.
#include "Utilities/TensorOperations/Cub/RaggedKRKernels.cuh"

#include "Utilities/Common/ScopedGpu.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ThorImplementation::CubReductionInternal {
namespace {

using RaggedKR::RaggedKRParams;

[[nodiscard]] bool raggedOutputSupported(DataType dtype) {
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

[[nodiscard]] uint32_t checkedComponents(const char* who, DataType input_dtype, uint32_t components) {
    const uint32_t bytes = raggedKRInputBytes(input_dtype);
    if (bytes == 0) throw std::invalid_argument(std::string(who) + ": the input must be FP8, FP16, BF16 or FP32");
    const uint32_t max_components = RaggedMergePath::raggedKRMaxComponents(bytes);
    if (components == 0 || components > max_components)
        throw std::invalid_argument(std::string(who) + ": D must be 1 to " + std::to_string(max_components) +
                                    " for this input (rows under 16 bytes)");
    return bytes;
}

}  // namespace

uint32_t raggedKRInputBytes(DataType input_dtype) {
    switch (input_dtype) {
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
            return 1;
        case DataType::FP16:
        case DataType::BF16:
            return 2;
        case DataType::FP32:
            return 4;
        default:
            return 0;
    }
}

bool raggedKRShapeLegal(const RaggedKRShape& shape) {
    return shape.items_per_thread == 16 && shape.ctas_per_sm <= 32 &&  // 1280: 16 rows per lane only
           shape.end_extra >= -1 && shape.end_extra <= 255;
}

uint32_t raggedKREndExtra(const RaggedKRShape& shape) {
    if (!shape.weighted) return 0;
    if (shape.end_extra >= 0) return static_cast<uint32_t>(shape.end_extra);
    return 4;  // 1273: the 1272 fit (i16: 3.5-3.7 rows beyond an end's bytes; i32: about 2)
}

std::string raggedKRShapeName(const RaggedKRShape& shape) {
    return "ragged_kr:i" + std::to_string(shape.items_per_thread) + "_c" + std::to_string(shape.ctas_per_sm) +
           (shape.weighted ? "_e" + std::to_string(raggedKREndExtra(shape)) : std::string("_u"));
}

RaggedMergePath::PathWeights raggedKRWeights(const RaggedKRShape& shape, DataType input_dtype, DataType offsets_dtype,
                                             DataType output_dtype, uint32_t components) {
    const uint32_t bytes = checkedComponents("raggedKRWeights", input_dtype, components);
    if (!shape.weighted) return RaggedMergePath::PathWeights{};
    const uint64_t offset_bytes = offsets_dtype == DataType::UINT64 ? 8 : 4;
    const uint64_t output_bytes = output_dtype == DataType::FP32 ? 4 : (output_dtype == DataType::FP16 ||
                                                                        output_dtype == DataType::BF16 ? 2 : 1);
    // A row: its D elements' bytes. An end: its offset, its D outputs and its work (1272).
    return RaggedMergePath::PathWeights{uint64_t{components} * bytes,
                                        offset_bytes + components * output_bytes + raggedKREndExtra(shape)};
}

uint64_t raggedValueWorkspaceBytes(uint32_t grid, uint32_t components) {
    return RaggedMergePath::raggedRecordLayout(grid, components).bytes;
}

uint32_t raggedKRGrid(const RaggedKRShape& shape, DataType input_dtype, int gpu, uint32_t components) {
    if (!raggedKRShapeLegal(shape)) throw std::invalid_argument("raggedKRGrid: illegal shape " + raggedKRShapeName(shape));
    const uint32_t bytes = checkedComponents("raggedKRGrid", input_dtype, components);
    ScopedGpu scoped_gpu(gpu);
    int sms = 0;
    RaggedKR::raggedKRCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, gpu), "SM count");
    // Resident CTAs depend on the dtype object and the kernel (D = 1 or the components kernel), not on the format
    // within the object or the operation.
    const int resident = bytes == 1   ? RaggedKR::raggedKRResidentFp8(components)
                         : bytes == 2 ? RaggedKR::raggedKRResidentF16(components)
                                      : RaggedKR::raggedKRResidentFp32(components);
    const uint32_t per_sm = std::max<uint32_t>(
        1, shape.ctas_per_sm == 0 ? static_cast<uint32_t>(resident)
                                  : std::min<uint32_t>(shape.ctas_per_sm, static_cast<uint32_t>(resident)));
    return std::clamp<uint32_t>(static_cast<uint32_t>(sms) * per_sm, 1, RAGGED_FINISH_MAX_GRID);
}

void launchRaggedKRValue(CubReductionOp op,
                         const RaggedKRShape& shape,
                         uint32_t grid,
                         const Tensor& input,
                         const Tensor& offsets,
                         Tensor& output,
                         Tensor& workspace,
                         Stream& stream,
                         RaggedLaunchParts parts) {
    const auto fail = [](const std::string& what) { throw std::invalid_argument("launchRaggedKRValue: " + what); };
    if (op != CubReductionOp::Sum && op != CubReductionOp::Mean && op != CubReductionOp::Min &&
        op != CubReductionOp::Max)
        fail("the operation must be SUM, MEAN, MIN or MAX");
    if (!raggedKRShapeLegal(shape)) fail("illegal shape " + raggedKRShapeName(shape));
    if (grid == 0 || grid > RAGGED_FINISH_MAX_GRID) fail("the grid must be 1 to " + std::to_string(RAGGED_FINISH_MAX_GRID));
    if (!input.isInitialized()) fail("the input must be an FP8, FP16, BF16 or FP32 tensor");
    const DataType input_dtype = input.getDataType();
    const uint32_t bytes = raggedKRInputBytes(input_dtype);
    if (bytes == 0) fail("the input must be an FP8, FP16, BF16 or FP32 tensor");
#if !THOR_CUB_ENABLE_FP8_TYPES
    if (bytes == 1) fail("FP8 types are disabled in this build");
#endif
    // Each tensor's dimensions are copied once: repeated getDimensions() copies trip GCC 14's -Warray-bounds (a false
    // positive on the inlined vector copy after a size() check), which this build treats as an error.
    const std::vector<uint64_t> input_dims = input.getDimensions();
    const std::vector<uint64_t> offsets_dims = offsets.isInitialized() ? offsets.getDimensions() : std::vector<uint64_t>{};
    if (input_dims.empty()) fail("the input must be [capacity] or [capacity, D]");
    if (!input.isDenseContiguous()) fail("the input must be dense");
    // 1280: D is the product of the dimensions after the first; 1284: rows under 16 bytes (D x element bytes <= 15).
    const uint64_t capacity = input_dims.front();
    const uint32_t max_components = RaggedMergePath::raggedKRMaxComponents(bytes);
    uint64_t d = 1;
    for (size_t i = 1; i < input_dims.size(); ++i) d *= input_dims[i];
    if (d == 0 || d > max_components)
        fail("the input must be [capacity] or [capacity, D] with D 1 to " + std::to_string(max_components) +
             " (rows under 16 bytes)");
    const uint32_t components = static_cast<uint32_t>(d);
    if (offsets_dims.size() != 1 || offsets_dims.front() == 0) fail("the offsets must be [segments + 1]");
    if (offsets.getDataType() != DataType::UINT32 && offsets.getDataType() != DataType::UINT64)
        fail("the offsets must be UINT32 or UINT64");
    const uint64_t segments = offsets_dims.front() - 1;
    if (!output.isInitialized() || output.getTotalNumElements() != segments * components ||
        !raggedOutputSupported(output.getDataType()))
        fail("the output must hold segments x D values of FP8, FP16, BF16 or FP32");
    if (!workspace.isInitialized() || workspace.getArraySizeInBytes() < raggedValueWorkspaceBytes(grid, components))
        fail("the workspace is too small");
    if (input.getPlacement() != offsets.getPlacement() || input.getPlacement() != output.getPlacement() ||
        input.getPlacement() != workspace.getPlacement() ||
        input.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU)
        fail("every tensor must be on the same GPU");
    if (input.getPlacement().getDeviceNum() != stream.getGpuNum()) fail("the stream must be on the tensors' GPU");
    const uintptr_t address = reinterpret_cast<uintptr_t>(input.getMemPtr<void>());
    // 1284: the kernels read whole elements from shared memory (and FP32 words in place).
    if (address % bytes != 0) fail("the input must be aligned to its element size");
    const RaggedMergePath::PathWeights weights =
        raggedKRWeights(shape, input_dtype, offsets.getDataType(), output.getDataType(), components);
    // A CTA's events (rows and ends, relative 32-bit coordinates in the kernel) stay below 2^31: its cost share, plus
    // the event that straddles each edge, over the cheaper event's weight.
    const uint64_t cost_bound = weights.end * segments + weights.row * capacity;
    if ((cost_bound / grid + 2 * weights.end + 2) / std::min(weights.row, weights.end) >= (uint64_t{1} << 31) ||
        (capacity + segments) / grid + 2 >= (uint64_t{1} << 31))
        fail("a CTA's range must stay below 2^31 events");
    if (segments == 0) return;

    ScopedGpu scoped_gpu(stream.getGpuNum());
    const int fixed_op = (op == CubReductionOp::Min || op == CubReductionOp::Max) ? 2 : 0;
    // MAX runs as MIN of the sign-flipped input (every element of a word) and negates the result.
    const uint32_t max_flip = bytes == 1 ? 0x80808080U : (bytes == 2 ? 0x80008000U : 0x80000000U);
    RaggedKRParams params{};
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
    params.components = components;
    params.workers = RaggedMergePath::raggedComponentWorkers(components);
    params.ends_cap = RaggedMergePath::raggedComponentEndsCap(components, RAGGED_KR_RING);
    const cudaStream_t s = stream.getStream();
    if (parts != RaggedLaunchParts::FinishOnly) {
        if (bytes == 1) {
            RaggedKR::raggedKRLaunchMainFp8(input_dtype, fixed_op, grid, params, s);
        } else if (bytes == 2) {
            RaggedKR::raggedKRLaunchMainF16(input_dtype, fixed_op, grid, params, s);
        } else {
            RaggedKR::raggedKRLaunchMainFp32(fixed_op, grid, params, s);
        }
    }
    if (parts != RaggedLaunchParts::MainOnly)
        launchRaggedFinishValue(fixed_op, grid, components, params.workspace, params.offsets, segments, capacity,
                                params.finalize, params.output, params.output_dtype, s);
}

}  // namespace ThorImplementation::CubReductionInternal
