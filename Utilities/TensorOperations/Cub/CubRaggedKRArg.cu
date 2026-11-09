// 1298 (phase 5, plans/Thor_Ragged_Reduction_Plan_2026-11-06.md): the ragged KR ARG host launcher. It validates, builds
// the ARG main kernel's parameters (RaggedKRArgKernels.cuh), launches the kernel of the input's dtype object
// (CubRaggedKRArgFp8.cu: FP8 at D = 1; 1300: D 1 to 15, rows under 16 bytes; 1311: CubRaggedKRArgF16.cu, FP16 / BF16
// at D 1 to 7; 1313: CubRaggedKRArgFp32.cu, FP32 at D 1 to 3) and then the ARG finisher (CubRaggedFinishArg.cu; 1313:
// FP32's pair records, CubRaggedFinishArgFp32.cu).
#include "Utilities/TensorOperations/Cub/F16ArgKey.h"
#include "Utilities/TensorOperations/Cub/RaggedKRArgKernels.cuh"

#include "Utilities/Common/ScopedGpu.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ThorImplementation::CubReductionInternal {

namespace {

bool raggedArgIsFp8(DataType dtype) { return dtype == DataType::FP8_E4M3 || dtype == DataType::FP8_E5M2; }
bool raggedArgIsF16(DataType dtype) { return dtype == DataType::FP16 || dtype == DataType::BF16; }  // 1311

constexpr const char* RAGGED_KR_ARG_DOMAIN = "FP8 (D 1 to 15), FP16 / BF16 (D 1 to 7) or FP32 (D 1 to 3) input";

}  // namespace

bool raggedKRArgSupported(DataType input_dtype, uint64_t components) {
    // Rows under 16 bytes: FP8 D 1 to 15 (1300; 1298-1299 D = 1), FP16 / BF16 D 1 to 7 (1311), FP32 D 1 to 3 (1313).
    if (components == 0) return false;
    if (raggedArgIsFp8(input_dtype)) return components <= RaggedMergePath::RAGGED_KR_MAX_COMPONENTS;
    if (raggedArgIsF16(input_dtype)) return components <= 7;
    if (input_dtype == DataType::FP32) return components <= 3;
    return false;
}

uint64_t raggedArgWorkspaceBytes(DataType input_dtype, uint32_t grid, uint32_t components) {
    // 1313: FP32's pair records.
    if (input_dtype == DataType::FP32) return RaggedMergePath::raggedArgPairRecordLayout(grid, components).bytes;
    return RaggedMergePath::raggedArgRecordLayout(grid, components).bytes;
}

RaggedMergePath::PathWeights raggedKRArgWeights(const RaggedKRShape& shape, DataType input_dtype,
                                                DataType offsets_dtype, DataType index_dtype, uint32_t components) {
    if (!raggedKRArgSupported(input_dtype, components))
        throw std::invalid_argument(std::string("raggedKRArgWeights: ragged KR ARG takes ") + RAGGED_KR_ARG_DOMAIN);
    if (index_dtype != DataType::UINT32 && index_dtype != DataType::UINT64)
        throw std::invalid_argument("raggedKRArgWeights: the index dtype must be UINT32 or UINT64");
    if (!shape.weighted) return RaggedMergePath::PathWeights{};
    const uint64_t offset_bytes = offsets_dtype == DataType::UINT64 ? 8 : 4;
    const uint64_t index_bytes = index_dtype == DataType::UINT64 ? 8 : 4;
    const uint64_t element_bytes = raggedKRInputBytes(input_dtype);
    return RaggedMergePath::PathWeights{uint64_t{components} * element_bytes,
                                        offset_bytes + components * index_bytes + raggedKREndExtra(shape)};
}

uint32_t raggedKRArgGrid(const RaggedKRShape& shape, DataType input_dtype, int gpu, uint32_t components) {
    if (!raggedKRShapeLegal(shape))
        throw std::invalid_argument("raggedKRArgGrid: illegal shape " + raggedKRShapeName(shape));
    if (!raggedKRArgSupported(input_dtype, components))
        throw std::invalid_argument(std::string("raggedKRArgGrid: ragged KR ARG takes ") + RAGGED_KR_ARG_DOMAIN);
    ScopedGpu scoped_gpu(gpu);
    int sms = 0;
    RaggedKR::raggedKRCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, gpu), "SM count");
    const int resident = raggedArgIsFp8(input_dtype)   ? RaggedKR::raggedKRArgResidentFp8(components)
                         : raggedArgIsF16(input_dtype) ? RaggedKR::raggedKRArgResidentF16(components)
                                                       : RaggedKR::raggedKRArgResidentFp32(components);  // 1313
    const uint32_t per_sm = std::max<uint32_t>(
        1, shape.ctas_per_sm == 0 ? static_cast<uint32_t>(resident)
                                  : std::min<uint32_t>(shape.ctas_per_sm, static_cast<uint32_t>(resident)));
    return std::clamp<uint32_t>(static_cast<uint32_t>(sms) * per_sm, 1, RAGGED_FINISH_MAX_GRID);
}

void launchRaggedKRArg(CubArgReductionOp op,
                       const RaggedKRShape& shape,
                       uint32_t grid,
                       const Tensor& input,
                       const Tensor& offsets,
                       Tensor& index_output,
                       Tensor& workspace,
                       Stream& stream,
                       RaggedLaunchParts parts) {
    const auto fail = [](const std::string& what) { throw std::invalid_argument("launchRaggedKRArg: " + what); };
    if (op != CubArgReductionOp::ArgMin && op != CubArgReductionOp::ArgMax) fail("the operation must be ARGMIN or ARGMAX");
    if (!raggedKRShapeLegal(shape)) fail("illegal shape " + raggedKRShapeName(shape));
    if (grid == 0 || grid > RAGGED_FINISH_MAX_GRID) fail("the grid must be 1 to " + std::to_string(RAGGED_FINISH_MAX_GRID));
    if (!input.isInitialized()) fail("the input must be an FP8, FP16, BF16 or FP32 tensor");
    const DataType input_dtype = input.getDataType();
#if !THOR_CUB_ENABLE_FP8_TYPES
    if (input_dtype == DataType::FP8_E4M3 || input_dtype == DataType::FP8_E5M2) fail("FP8 types are disabled in this build");
#endif
    // Each tensor's dimensions are copied once (GCC 14's -Warray-bounds false positive, as launchRaggedKRValue).
    const std::vector<uint64_t> input_dims = input.getDimensions();
    const std::vector<uint64_t> offsets_dims = offsets.isInitialized() ? offsets.getDimensions() : std::vector<uint64_t>{};
    if (input_dims.empty()) fail("the input must be [capacity] or [capacity, D]");
    if (!input.isDenseContiguous()) fail("the input must be dense");
    const uint64_t capacity = input_dims.front();
    uint64_t d = 1;
    for (size_t i = 1; i < input_dims.size(); ++i) d *= input_dims[i];
    if (!raggedKRArgSupported(input_dtype, d))
        fail("the input must be [capacity] or [capacity, D]: FP8 with D 1 to 15, FP16 / BF16 with D 1 to 7, FP32 with D 1 "
             "to 3");
    const bool fp8 = raggedArgIsFp8(input_dtype);
    const bool fp32 = input_dtype == DataType::FP32;  // 1313: CTA-local rows in the kernel, global rows in pair records
    const uint32_t row_bits = fp8 ? RaggedKR::RaggedArgKeyFp8::RowBits : RaggedKR::RaggedArgKeyF16::RowBits;
    const uint32_t components = static_cast<uint32_t>(d);
    if (offsets_dims.size() != 1 || offsets_dims.front() == 0) fail("the offsets must be [segments + 1]");
    if (offsets.getDataType() != DataType::UINT32 && offsets.getDataType() != DataType::UINT64)
        fail("the offsets must be UINT32 or UINT64");
    const uint64_t segments = offsets_dims.front() - 1;
    if (!index_output.isInitialized() || index_output.getTotalNumElements() != segments * components ||
        (index_output.getDataType() != DataType::UINT32 && index_output.getDataType() != DataType::UINT64))
        fail("the index output must hold segments x D UINT32 or UINT64 indices");
    const uint32_t index_bytes = index_output.getDataType() == DataType::UINT64 ? 8 : 4;
    // The packed indices stay below the sentinel, and the rows within the candidate word's row field.
    if (index_bytes == 4 && capacity * components > uint64_t{0xFFFFFFFFU})
        fail("a UINT32 index output needs capacity x D <= 2^32 - 1");
    if (!fp32 && capacity >= (uint64_t{1} << row_bits) - 1)
        fail("the capacity must stay below 2^" + std::to_string(row_bits) + " - 1 rows");
    if (!workspace.isInitialized() ||
        workspace.getArraySizeInBytes() < raggedArgWorkspaceBytes(input_dtype, grid, components))
        fail("the workspace is too small");
    if (input.getPlacement() != offsets.getPlacement() || input.getPlacement() != index_output.getPlacement() ||
        input.getPlacement() != workspace.getPlacement() ||
        input.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU)
        fail("every tensor must be on the same GPU");
    if (input.getPlacement().getDeviceNum() != stream.getGpuNum()) fail("the stream must be on the tensors' GPU");
    const RaggedMergePath::PathWeights weights =
        raggedKRArgWeights(shape, input_dtype, offsets.getDataType(), index_output.getDataType(), components);
    // A CTA's events stay below 2^31 (as launchRaggedKRValue; 1313: so FP32's CTA-local rows fit 31 bits).
    const uint64_t cost_bound = weights.end * segments + weights.row * capacity;
    if ((cost_bound / grid + 2 * weights.end + 2) / std::min(weights.row, weights.end) >= (uint64_t{1} << 31) ||
        (capacity + segments) / grid + 2 >= (uint64_t{1} << 31))
        fail("a CTA's range must stay below 2^31 events");

    const uintptr_t address = reinterpret_cast<uintptr_t>(input.getMemPtr<void>());
    if (address % raggedKRInputBytes(input_dtype) != 0) fail("the input must be aligned to its element size");
    if (segments == 0) return;

    ScopedGpu scoped_gpu(stream.getGpuNum());
    RaggedKR::RaggedKRArgParams params{};
    params.bytes = reinterpret_cast<const uint8_t*>(address & ~uintptr_t{15});
    params.head = static_cast<uint32_t>(address & 15U);
    params.offsets = RaggedMergePath::Offsets{offsets.getMemPtr<void>(),
                                              offsets.getDataType() == DataType::UINT64 ? 8U : 4U};
    params.segments = segments;
    params.capacity_rows = capacity;
    params.weights = weights;
    if (fp8) {
        params.fold = fp8ArgFold(op == CubArgReductionOp::ArgMax, input_dtype == DataType::FP8_E5M2);
    } else if (fp32) {  // 1313: the sign flip only (the FP32 key finds NaN by its magnitude)
        params.fold = Fp8ArgFold{op == CubArgReductionOp::ArgMax ? 0x80000000U : 0U, 0U};
    } else {  // 1311: the F16 fold's values in the same two fields
        const F16ArgFold fold = f16ArgFold(op == CubArgReductionOp::ArgMax, input_dtype == DataType::BF16);
        params.fold = Fp8ArgFold{fold.xor_mask, fold.nan_bias};
    }
    params.output = index_output.getMemPtr<void>();
    params.index_bytes = index_bytes;
    params.workspace = static_cast<uint8_t*>(workspace.getMemPtr<void>());
    params.components = components;
    params.workers = RaggedMergePath::raggedComponentWorkers(components);
    params.ends_cap = RaggedMergePath::raggedComponentEndsCap(components, RAGGED_KR_RING);
    const cudaStream_t s = stream.getStream();
    if (parts != RaggedLaunchParts::FinishOnly) {
        if (fp8) {
            RaggedKR::raggedKRArgLaunchMainFp8(grid, params, s);
        } else if (fp32) {
            RaggedKR::raggedKRArgLaunchMainFp32(grid, params, s);
        } else {
            RaggedKR::raggedKRArgLaunchMainF16(grid, params, s);
        }
    }
    if (parts != RaggedLaunchParts::MainOnly) {
        if (fp32) {
            launchRaggedFinishArgPair(grid, components, params.workspace, params.output, index_bytes, s);
        } else {
            launchRaggedFinishArg(grid, components, params.workspace, row_bits, params.output, index_bytes, s);
        }
    }
}

}  // namespace ThorImplementation::CubReductionInternal
