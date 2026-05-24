#include "Utilities/TensorOperations/Embedding/EmbeddingKernels.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#ifndef THOR_CUDA_INCLUDE_DIR
#define THOR_CUDA_INCLUDE_DIR ""
#endif

#ifndef THOR_CUDA_CCCL_INCLUDE_DIR
#define THOR_CUDA_CCCL_INCLUDE_DIR ""
#endif

namespace ThorImplementation {

struct GeneratedEmbeddingKernel {
    std::string cache_key;
    CUmodule module = nullptr;
    CUfunction kernel = nullptr;
    int device_num = 0;

    GeneratedEmbeddingKernel() = default;
    GeneratedEmbeddingKernel(const GeneratedEmbeddingKernel&) = delete;
    GeneratedEmbeddingKernel& operator=(const GeneratedEmbeddingKernel&) = delete;

    ~GeneratedEmbeddingKernel() {
        if (module != nullptr) {
            try {
                CU_CHECK(cuModuleUnload(module));
            } catch (...) {
            }
        }
    }
};

struct PreparedEmbeddingForward {
    using LaunchFn = void (*)(const PreparedEmbeddingForward&, const Tensor&, const Tensor&, Tensor&, Stream, const std::vector<Tensor>&);

    LaunchFn launch = nullptr;
    std::shared_ptr<GeneratedEmbeddingKernel> generated_kernel;
    uint64_t num_indices = 0;
    uint64_t vocabulary_size = 0;
    uint64_t embedding_dim = 0;
    uint64_t padding_index = 0;
    bool has_padding_index = false;
    uint32_t elements_per_lane = 0;
    uint32_t grid_blocks = 1;
    int device_num = 0;
    TensorDescriptor::DataType index_dtype = TensorDescriptor::DataType::UINT32;
    TensorDescriptor::DataType weights_dtype = TensorDescriptor::DataType::FP32;
    EmbeddingForwardEpilogue epilogue;
};

namespace {

using DataType = TensorDescriptor::DataType;

constexpr uint32_t WARP_SIZE_EMBEDDING = 32;
constexpr uint32_t WARPS_PER_BLOCK = 8;
constexpr uint32_t THREADS_PER_BLOCK = WARP_SIZE_EMBEDDING * WARPS_PER_BLOCK;
constexpr uint32_t MAX_GRID_BLOCKS = 131072;
// A lane could hold substantially more in registers, but using a wide per-lane
// struct makes each warp instruction touch sparse memory sectors for wide rows.
// Keep each lane copy to one 16-byte vector so every substep is a contiguous
// 512-byte warp transaction: lane 0 copies bytes 0..15, lane 1 copies 16..31,
// etc. Wider embedding rows iterate over additional coalesced warp substeps.
//
// Tensor allocations include padding at the end, but embedding rows are packed
// back-to-back. The exact fast path therefore only removes tail logic when the
// row width is an exact multiple of the warp chunk; otherwise it uses the simple
// scalar fallback below instead of writing past a row boundary.
constexpr uint32_t MAX_COALESCED_BYTES_PER_LANE = 16;

static_assert(THREADS_PER_BLOCK == 256);
static_assert(MAX_COALESCED_BYTES_PER_LANE == 16);

constexpr bool PRINT_GENERATED_EMBEDDING_KERNELS = true;

template <typename T, uint32_t Elements>
struct alignas((Elements * sizeof(T) >= MAX_COALESCED_BYTES_PER_LANE) ? MAX_COALESCED_BYTES_PER_LANE : Elements * sizeof(T)) LaneVector {
    T value[Elements];
};

template <typename IndexT, typename ValueT, bool HasPaddingIndex>
__global__ void embeddingForwardWarpScalarFallbackKernel(const IndexT* __restrict__ indices,
                                                         const ValueT* __restrict__ weights,
                                                         ValueT* __restrict__ output,
                                                         uint64_t numIndices,
                                                         uint64_t vocabularySize,
                                                         uint64_t embeddingDim,
                                                         uint64_t paddingIndex) {
    static_assert(!std::is_signed_v<IndexT>, "Embedding indices are unsigned-only.");

    const uint32_t lane = threadIdx.x & (WARP_SIZE_EMBEDDING - 1);
    const uint32_t warpInBlock = threadIdx.x >> 5;
    const uint64_t globalWarp = static_cast<uint64_t>(blockIdx.x) * WARPS_PER_BLOCK + warpInBlock;
    const uint64_t totalWarps = static_cast<uint64_t>(gridDim.x) * WARPS_PER_BLOCK;

    for (uint64_t token = globalWarp; token < numIndices; token += totalWarps) {
        const uint64_t row = static_cast<uint64_t>(indices[token]);
        ValueT* __restrict__ outBase = output + token * embeddingDim;
        const bool zeroRow = row >= vocabularySize || (HasPaddingIndex && row == paddingIndex);

        if (zeroRow) {
            for (uint64_t dim = lane; dim < embeddingDim; dim += WARP_SIZE_EMBEDDING) {
                outBase[dim] = ValueT{};
            }
        } else {
            const ValueT* __restrict__ rowBase = weights + row * embeddingDim;
            for (uint64_t dim = lane; dim < embeddingDim; dim += WARP_SIZE_EMBEDDING) {
                outBase[dim] = rowBase[dim];
            }
        }
    }
}

uint32_t gridForWarps(uint64_t tokens) {
    const uint64_t blocks = (tokens + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    return static_cast<uint32_t>(std::min<uint64_t>(std::max<uint64_t>(blocks, 1), MAX_GRID_BLOCKS));
}

uint32_t gridForTinyEmbedding(uint64_t tokens, uint32_t groupSize) {
    if (groupSize == 0 || groupSize > WARP_SIZE_EMBEDDING || (WARP_SIZE_EMBEDDING % groupSize) != 0) {
        throw std::invalid_argument("Tiny EmbeddingLookup group size must divide the warp size.");
    }
    const uint32_t groupsPerWarp = WARP_SIZE_EMBEDDING / groupSize;
    const uint64_t groupsPerBlock = static_cast<uint64_t>(WARPS_PER_BLOCK) * groupsPerWarp;
    const uint64_t blocks = (tokens + groupsPerBlock - 1) / groupsPerBlock;
    return static_cast<uint32_t>(std::min<uint64_t>(std::max<uint64_t>(blocks, 1), MAX_GRID_BLOCKS));
}

bool isSupportedEmbeddingIndexDtype(DataType dtype) {
    switch (dtype) {
        case DataType::UINT32:
        case DataType::UINT64:
            return true;
        default:
            return false;
    }
}

bool isSupportedEmbeddingValueDtype(DataType dtype) {
    switch (dtype) {
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
            return true;
        default:
            return false;
    }
}

std::string dataTypeName(DataType dtype) { return TensorDescriptor::getElementTypeName(dtype); }

void validateDenseContiguous(const Tensor& tensor, const std::string& name) {
    if (tensor.hasCustomStrides() || !tensor.isDenseContiguous()) {
        throw std::invalid_argument("Embedding " + name + " tensor must be dense contiguous in this backend slice.");
    }
}

const char* generatedEmbeddingIndexTypeName(DataType dtype) {
    switch (dtype) {
        case DataType::UINT32:
            return "unsigned int";
        case DataType::UINT64:
            return "unsigned long long";
        default:
            throw std::runtime_error("Generated EmbeddingLookup kernel received unsupported index dtype: " + dataTypeName(dtype));
    }
}

const char* generatedEmbeddingValueTypeName(DataType dtype) {
    switch (dtype) {
        case DataType::FP16:
            return "__half";
        case DataType::BF16:
            return "__nv_bfloat16";
        case DataType::FP32:
            return "float";
        default:
            throw std::runtime_error("Generated EmbeddingLookup kernel received unsupported value dtype: " + dataTypeName(dtype));
    }
}

std::string generatedEmbeddingValueIncludes(DataType dtype) {
    switch (dtype) {
        case DataType::FP16:
            return "#include <cuda_fp16.h>\n";
        case DataType::BF16:
            return "#include <cuda_bf16.h>\n";
        case DataType::FP32:
            return "";
        default:
            throw std::runtime_error("Generated EmbeddingLookup kernel received unsupported value dtype: " + dataTypeName(dtype));
    }
}

std::string generatedEmbeddingExtraArgs(const EmbeddingForwardEpilogue& epilogue, DataType valueDtype) {
    std::ostringstream src;
    for (size_t i = 0; i < epilogue.extra_input_dtypes.size(); ++i) {
        if (epilogue.extra_input_dtypes[i] != valueDtype) {
            throw std::runtime_error("EmbeddingLookup epilogue tensor inputs must match the embedding output dtype in this backend slice.");
        }
        src << ",\n    const " << generatedEmbeddingValueTypeName(valueDtype) << "* __restrict__ arg" << i;
    }
    return src.str();
}

std::string generatedEmbeddingEpilogueValue(const EmbeddingForwardEpilogue& epilogue) {
    return epilogue.enabled() ? epilogue.expression : std::string("v");
}

std::string generatedEmbeddingExactCommonPrefix(DataType indexDtype,
                                                DataType valueDtype,
                                                uint32_t elementsPerLane,
                                                uint64_t numIndices,
                                                uint64_t embeddingDim,
                                                uint64_t elementsPerWarpIteration,
                                                const EmbeddingForwardEpilogue& epilogue) {
    std::ostringstream src;
    src << generatedEmbeddingValueIncludes(valueDtype);
    src << R"cuda(
extern "C" __global__ void embedding_lookup(
    const )cuda"
        << generatedEmbeddingIndexTypeName(indexDtype) << R"cuda(* __restrict__ indices,
    const )cuda"
        << generatedEmbeddingValueTypeName(valueDtype) << R"cuda(* __restrict__ weights,
    )cuda"
        << generatedEmbeddingValueTypeName(valueDtype) << R"cuda(* __restrict__ output)cuda"
        << generatedEmbeddingExtraArgs(epilogue, valueDtype) << R"cuda() {
    using ValueT = )cuda"
        << generatedEmbeddingValueTypeName(valueDtype) << R"cuda(;
    constexpr unsigned int WARP_SIZE_EMBEDDING = 32u;
    constexpr unsigned int WARPS_PER_BLOCK = 8u;
    constexpr unsigned int ELEMENTS_PER_LANE = )cuda"
        << elementsPerLane << R"cuda(u;
    constexpr unsigned long long NUM_INDICES = )cuda"
        << numIndices << R"cuda(ull;
    constexpr unsigned long long EMBEDDING_DIM = )cuda"
        << embeddingDim << R"cuda(ull;
    constexpr unsigned long long ELEMENTS_PER_WARP_ITERATION = )cuda"
        << elementsPerWarpIteration << R"cuda(ull;

    struct alignas((ELEMENTS_PER_LANE * sizeof(ValueT) >= 16u) ? 16u : ELEMENTS_PER_LANE * sizeof(ValueT)) LaneVector {
        ValueT value[ELEMENTS_PER_LANE];
    };

    const unsigned int lane = threadIdx.x & (WARP_SIZE_EMBEDDING - 1u);
    const unsigned int warpInBlock = threadIdx.x >> 5;
    const unsigned long long globalWarp = static_cast<unsigned long long>(blockIdx.x) * WARPS_PER_BLOCK + warpInBlock;
    const unsigned long long totalWarps = static_cast<unsigned long long>(gridDim.x) * WARPS_PER_BLOCK;
)cuda";
    return src.str();
}

std::string generatedEmbeddingExactUncheckedNoPaddingSource(DataType indexDtype,
                                                            DataType valueDtype,
                                                            uint32_t elementsPerLane,
                                                            uint64_t numIndices,
                                                            uint64_t embeddingDim,
                                                            const EmbeddingForwardEpilogue& epilogue) {
    const uint64_t elementsPerWarpIteration = static_cast<uint64_t>(elementsPerLane) * WARP_SIZE_EMBEDDING;
    if (embeddingDim % elementsPerWarpIteration != 0) {
        throw std::runtime_error(
            "Generated EmbeddingLookup exact kernel requires embedding_dim to be a multiple of the warp iteration width.");
    }

    std::ostringstream src;
    src << generatedEmbeddingExactCommonPrefix(
        indexDtype, valueDtype, elementsPerLane, numIndices, embeddingDim, elementsPerWarpIteration, epilogue);
    if (!epilogue.enabled()) {
        src << R"cuda(
    for (unsigned long long token = globalWarp; token < NUM_INDICES; token += totalWarps) {
        const unsigned long long row = static_cast<unsigned long long>(indices[token]);
        const ValueT* __restrict__ rowBase = weights + row * EMBEDDING_DIM;
        ValueT* __restrict__ outBase = output + token * EMBEDDING_DIM;

        for (unsigned long long dimBase = 0; dimBase < EMBEDDING_DIM; dimBase += ELEMENTS_PER_WARP_ITERATION) {
            const unsigned long long laneBase = dimBase + static_cast<unsigned long long>(lane) * ELEMENTS_PER_LANE;
            const ValueT* __restrict__ srcPtr = rowBase + laneBase;
            ValueT* __restrict__ outPtr = outBase + laneBase;
            const LaneVector tmp = *reinterpret_cast<const LaneVector*>(srcPtr);
            *reinterpret_cast<LaneVector*>(outPtr) = tmp;
        }
    }
}
)cuda";
    } else {
        src << R"cuda(
    for (unsigned long long token = globalWarp; token < NUM_INDICES; token += totalWarps) {
        const unsigned long long row = static_cast<unsigned long long>(indices[token]);
        const ValueT* __restrict__ rowBase = weights + row * EMBEDDING_DIM;
        ValueT* __restrict__ outBase = output + token * EMBEDDING_DIM;

        for (unsigned long long dimBase = 0; dimBase < EMBEDDING_DIM; dimBase += ELEMENTS_PER_WARP_ITERATION) {
            const unsigned long long laneBase = dimBase + static_cast<unsigned long long>(lane) * ELEMENTS_PER_LANE;
            const unsigned long long linearBase = token * EMBEDDING_DIM + laneBase;
            const LaneVector tmp = *reinterpret_cast<const LaneVector*>(rowBase + laneBase);
            LaneVector out{};
            #pragma unroll
            for (unsigned int i = 0; i < ELEMENTS_PER_LANE; ++i) {
                const unsigned long long linear = linearBase + i;
                ValueT v = tmp.value[i];
                out.value[i] = static_cast<ValueT>()cuda"
            << generatedEmbeddingEpilogueValue(epilogue) << R"cuda();
            }
            *reinterpret_cast<LaneVector*>(outBase + laneBase) = out;
        }
    }
}
)cuda";
    }
    return src.str();
}

std::string generatedEmbeddingExactPaddingSource(DataType indexDtype,
                                                 DataType valueDtype,
                                                 uint32_t elementsPerLane,
                                                 uint64_t numIndices,
                                                 uint64_t vocabularySize,
                                                 uint64_t embeddingDim,
                                                 uint64_t paddingIndex,
                                                 const EmbeddingForwardEpilogue& epilogue) {
    const uint64_t elementsPerWarpIteration = static_cast<uint64_t>(elementsPerLane) * WARP_SIZE_EMBEDDING;
    if (embeddingDim % elementsPerWarpIteration != 0) {
        throw std::runtime_error(
            "Generated EmbeddingLookup exact kernel requires embedding_dim to be a multiple of the warp iteration width.");
    }

    std::ostringstream src;
    src << generatedEmbeddingExactCommonPrefix(
        indexDtype, valueDtype, elementsPerLane, numIndices, embeddingDim, elementsPerWarpIteration, epilogue);
    src << R"cuda(
    constexpr unsigned long long VOCABULARY_SIZE = )cuda"
        << vocabularySize << R"cuda(ull;
    constexpr unsigned long long PADDING_INDEX = )cuda"
        << paddingIndex << R"cuda(ull;
)cuda";
    if (!epilogue.enabled()) {
        src << R"cuda(
    for (unsigned long long token = globalWarp; token < NUM_INDICES; token += totalWarps) {
        const unsigned long long row = static_cast<unsigned long long>(indices[token]);
        ValueT* __restrict__ outBase = output + token * EMBEDDING_DIM;

        if (row == PADDING_INDEX || row >= VOCABULARY_SIZE) {
            const LaneVector zero{};
            for (unsigned long long dimBase = 0; dimBase < EMBEDDING_DIM; dimBase += ELEMENTS_PER_WARP_ITERATION) {
                ValueT* __restrict__ outPtr = outBase + dimBase + static_cast<unsigned long long>(lane) * ELEMENTS_PER_LANE;
                *reinterpret_cast<LaneVector*>(outPtr) = zero;
            }
            continue;
        }

        const ValueT* __restrict__ rowBase = weights + row * EMBEDDING_DIM;
        for (unsigned long long dimBase = 0; dimBase < EMBEDDING_DIM; dimBase += ELEMENTS_PER_WARP_ITERATION) {
            const unsigned long long laneBase = dimBase + static_cast<unsigned long long>(lane) * ELEMENTS_PER_LANE;
            const ValueT* __restrict__ srcPtr = rowBase + laneBase;
            ValueT* __restrict__ outPtr = outBase + laneBase;
            const LaneVector tmp = *reinterpret_cast<const LaneVector*>(srcPtr);
            *reinterpret_cast<LaneVector*>(outPtr) = tmp;
        }
    }
}
)cuda";
    } else {
        src << R"cuda(
    for (unsigned long long token = globalWarp; token < NUM_INDICES; token += totalWarps) {
        const unsigned long long row = static_cast<unsigned long long>(indices[token]);
        const bool zeroRow = (row == PADDING_INDEX || row >= VOCABULARY_SIZE);
        ValueT* __restrict__ outBase = output + token * EMBEDDING_DIM;

        for (unsigned long long dimBase = 0; dimBase < EMBEDDING_DIM; dimBase += ELEMENTS_PER_WARP_ITERATION) {
            const unsigned long long laneBase = dimBase + static_cast<unsigned long long>(lane) * ELEMENTS_PER_LANE;
            const unsigned long long linearBase = token * EMBEDDING_DIM + laneBase;
            LaneVector out{};
            #pragma unroll
            for (unsigned int i = 0; i < ELEMENTS_PER_LANE; ++i) {
                const unsigned long long linear = linearBase + i;
                ValueT v = zeroRow ? ValueT{} : weights[row * EMBEDDING_DIM + laneBase + i];
                out.value[i] = static_cast<ValueT>()cuda"
            << generatedEmbeddingEpilogueValue(epilogue) << R"cuda();
            }
            *reinterpret_cast<LaneVector*>(outBase + laneBase) = out;
        }
    }
}
)cuda";
    }
    return src.str();
}

std::string generatedEmbeddingExactSource(DataType indexDtype,
                                          DataType valueDtype,
                                          uint32_t elementsPerLane,
                                          bool hasPaddingIndex,
                                          uint64_t numIndices,
                                          uint64_t vocabularySize,
                                          uint64_t embeddingDim,
                                          uint64_t paddingIndex,
                                          const EmbeddingForwardEpilogue& epilogue) {
    if (!hasPaddingIndex) {
        return generatedEmbeddingExactUncheckedNoPaddingSource(indexDtype, valueDtype, elementsPerLane, numIndices, embeddingDim, epilogue);
    }
    return generatedEmbeddingExactPaddingSource(
        indexDtype, valueDtype, elementsPerLane, numIndices, vocabularySize, embeddingDim, paddingIndex, epilogue);
}

uint32_t tinyEmbeddingGroupSize(uint64_t embeddingDim) {
    if (embeddingDim <= 1)
        return 1;
    if (embeddingDim <= 2)
        return 2;
    if (embeddingDim <= 4)
        return 4;
    if (embeddingDim <= 8)
        return 8;
    // For 16- and 32-wide rows, keep the tiny-D path packed by letting each
    // participating lane copy two contiguous elements.  This doubles the number
    // of independent rows handled by each physical warp compared with the
    // scalar-per-lane variant while preserving contiguous per-row memory access.
    if (embeddingDim <= 16)
        return 8;
    if (embeddingDim <= 32)
        return 16;
    return 0;
}

uint32_t tinyEmbeddingElementsPerLane(uint64_t embeddingDim, uint32_t groupSize) {
    if (groupSize == 0) {
        return 0;
    }
    const uint64_t elementsPerLane = (embeddingDim + groupSize - 1) / groupSize;
    if (elementsPerLane == 0 || elementsPerLane > 2) {
        return 0;
    }
    return static_cast<uint32_t>(elementsPerLane);
}

std::string generatedEmbeddingTinyCommonPrefix(DataType indexDtype,
                                               DataType valueDtype,
                                               uint64_t numIndices,
                                               uint64_t embeddingDim,
                                               uint32_t groupSize,
                                               uint32_t elementsPerLane,
                                               const EmbeddingForwardEpilogue& epilogue) {
    if (groupSize == 0 || groupSize > WARP_SIZE_EMBEDDING || (WARP_SIZE_EMBEDDING % groupSize) != 0) {
        throw std::runtime_error("Generated tiny EmbeddingLookup kernel requires a power-of-two group size that divides warp size.");
    }
    if (elementsPerLane == 0 || elementsPerLane > 2) {
        throw std::runtime_error("Generated tiny EmbeddingLookup kernel requires one or two elements per lane.");
    }
    if (embeddingDim == 0 || embeddingDim > static_cast<uint64_t>(groupSize) * elementsPerLane) {
        throw std::runtime_error("Generated tiny EmbeddingLookup kernel requires 0 < embedding_dim <= group_size * elements_per_lane.");
    }

    std::ostringstream src;
    src << generatedEmbeddingValueIncludes(valueDtype);
    src << R"cuda(
extern "C" __global__ void embedding_lookup(
    const )cuda"
        << generatedEmbeddingIndexTypeName(indexDtype) << R"cuda(* __restrict__ indices,
    const )cuda"
        << generatedEmbeddingValueTypeName(valueDtype) << R"cuda(* __restrict__ weights,
    )cuda"
        << generatedEmbeddingValueTypeName(valueDtype) << R"cuda(* __restrict__ output)cuda"
        << generatedEmbeddingExtraArgs(epilogue, valueDtype) << R"cuda() {
    using ValueT = )cuda"
        << generatedEmbeddingValueTypeName(valueDtype) << R"cuda(;
    constexpr unsigned int WARP_SIZE_EMBEDDING = 32u;
    constexpr unsigned int WARPS_PER_BLOCK = 8u;
    constexpr unsigned int GROUP_SIZE = )cuda"
        << groupSize << R"cuda(u;
    constexpr unsigned int GROUPS_PER_WARP = WARP_SIZE_EMBEDDING / GROUP_SIZE;
    constexpr unsigned int ELEMENTS_PER_LANE = )cuda"
        << elementsPerLane << R"cuda(u;
    constexpr unsigned long long NUM_INDICES = )cuda"
        << numIndices << R"cuda(ull;
    constexpr unsigned long long EMBEDDING_DIM = )cuda"
        << embeddingDim << R"cuda(ull;

    struct alignas((ELEMENTS_PER_LANE * sizeof(ValueT) >= 16u) ? 16u : ELEMENTS_PER_LANE * sizeof(ValueT)) LaneVector {
        ValueT value[ELEMENTS_PER_LANE];
    };

    const unsigned int lane = threadIdx.x & (WARP_SIZE_EMBEDDING - 1u);
    const unsigned int warpInBlock = threadIdx.x >> 5;
    const unsigned int logicalLane = lane % GROUP_SIZE;
    const unsigned int groupInWarp = lane / GROUP_SIZE;
    const unsigned long long globalGroup = static_cast<unsigned long long>(blockIdx.x) * WARPS_PER_BLOCK * GROUPS_PER_WARP +
                                           static_cast<unsigned long long>(warpInBlock) * GROUPS_PER_WARP + groupInWarp;
    const unsigned long long totalGroups = static_cast<unsigned long long>(gridDim.x) * WARPS_PER_BLOCK * GROUPS_PER_WARP;
)cuda";
    if (groupSize == WARP_SIZE_EMBEDDING) {
        src << R"cuda(    const unsigned int groupMask = 0xffffffffu;
)cuda";
    } else {
        src << R"cuda(    const unsigned int groupMask = ((1u << GROUP_SIZE) - 1u) << (groupInWarp * GROUP_SIZE);
)cuda";
    }
    return src.str();
}

std::string generatedEmbeddingTinyUncheckedNoPaddingSource(
    DataType indexDtype, DataType valueDtype, uint64_t numIndices, uint64_t embeddingDim, const EmbeddingForwardEpilogue& epilogue) {
    const uint32_t groupSize = tinyEmbeddingGroupSize(embeddingDim);
    const uint32_t elementsPerLane = tinyEmbeddingElementsPerLane(embeddingDim, groupSize);
    if (groupSize == 0 || elementsPerLane == 0) {
        throw std::runtime_error("Generated tiny EmbeddingLookup kernel requires embedding_dim <= 32.");
    }

    std::ostringstream src;
    src << generatedEmbeddingTinyCommonPrefix(indexDtype, valueDtype, numIndices, embeddingDim, groupSize, elementsPerLane, epilogue);
    if (!epilogue.enabled()) {
        src << R"cuda(
    for (unsigned long long token = globalGroup; token < NUM_INDICES; token += totalGroups) {
        unsigned long long row = 0ull;
        if (logicalLane == 0u) {
            row = static_cast<unsigned long long>(indices[token]);
        }
        row = __shfl_sync(groupMask, row, groupInWarp * GROUP_SIZE);

        const unsigned long long laneBase = static_cast<unsigned long long>(logicalLane) * ELEMENTS_PER_LANE;
        if constexpr (EMBEDDING_DIM == static_cast<unsigned long long>(GROUP_SIZE) * ELEMENTS_PER_LANE) {
            const LaneVector tmp = *reinterpret_cast<const LaneVector*>(weights + row * EMBEDDING_DIM + laneBase);
            *reinterpret_cast<LaneVector*>(output + token * EMBEDDING_DIM + laneBase) = tmp;
        } else {
            for (unsigned int i = 0; i < ELEMENTS_PER_LANE; ++i) {
                const unsigned long long dim = laneBase + i;
                if (dim < EMBEDDING_DIM) {
                    output[token * EMBEDDING_DIM + dim] = weights[row * EMBEDDING_DIM + dim];
                }
            }
        }
    }
}
)cuda";
    } else {
        src << R"cuda(
    for (unsigned long long token = globalGroup; token < NUM_INDICES; token += totalGroups) {
        unsigned long long row = 0ull;
        if (logicalLane == 0u) {
            row = static_cast<unsigned long long>(indices[token]);
        }
        row = __shfl_sync(groupMask, row, groupInWarp * GROUP_SIZE);

        const unsigned long long laneBase = static_cast<unsigned long long>(logicalLane) * ELEMENTS_PER_LANE;
        const unsigned long long linearBase = token * EMBEDDING_DIM + laneBase;
        if constexpr (EMBEDDING_DIM == static_cast<unsigned long long>(GROUP_SIZE) * ELEMENTS_PER_LANE) {
            const LaneVector tmp = *reinterpret_cast<const LaneVector*>(weights + row * EMBEDDING_DIM + laneBase);
            LaneVector out{};
            #pragma unroll
            for (unsigned int i = 0; i < ELEMENTS_PER_LANE; ++i) {
                const unsigned long long linear = linearBase + i;
                ValueT v = tmp.value[i];
                out.value[i] = static_cast<ValueT>()cuda"
            << generatedEmbeddingEpilogueValue(epilogue) << R"cuda();
            }
            *reinterpret_cast<LaneVector*>(output + token * EMBEDDING_DIM + laneBase) = out;
        } else {
            for (unsigned int i = 0; i < ELEMENTS_PER_LANE; ++i) {
                const unsigned long long dim = laneBase + i;
                if (dim < EMBEDDING_DIM) {
                    const unsigned long long linear = token * EMBEDDING_DIM + dim;
                    ValueT v = weights[row * EMBEDDING_DIM + dim];
                    output[linear] = static_cast<ValueT>()cuda"
            << generatedEmbeddingEpilogueValue(epilogue) << R"cuda();
                }
            }
        }
    }
}
)cuda";
    }
    return src.str();
}

std::string generatedEmbeddingTinyPaddingSource(DataType indexDtype,
                                                DataType valueDtype,
                                                uint64_t numIndices,
                                                uint64_t vocabularySize,
                                                uint64_t embeddingDim,
                                                uint64_t paddingIndex,
                                                const EmbeddingForwardEpilogue& epilogue) {
    const uint32_t groupSize = tinyEmbeddingGroupSize(embeddingDim);
    const uint32_t elementsPerLane = tinyEmbeddingElementsPerLane(embeddingDim, groupSize);
    if (groupSize == 0 || elementsPerLane == 0) {
        throw std::runtime_error("Generated tiny EmbeddingLookup kernel requires embedding_dim <= 32.");
    }

    std::ostringstream src;
    src << generatedEmbeddingTinyCommonPrefix(indexDtype, valueDtype, numIndices, embeddingDim, groupSize, elementsPerLane, epilogue);
    src << R"cuda(
    constexpr unsigned long long VOCABULARY_SIZE = )cuda"
        << vocabularySize << R"cuda(ull;
    constexpr unsigned long long PADDING_INDEX = )cuda"
        << paddingIndex << R"cuda(ull;
)cuda";
    if (!epilogue.enabled()) {
        src << R"cuda(
    for (unsigned long long token = globalGroup; token < NUM_INDICES; token += totalGroups) {
        unsigned long long row = 0ull;
        if (logicalLane == 0u) {
            row = static_cast<unsigned long long>(indices[token]);
        }
        row = __shfl_sync(groupMask, row, groupInWarp * GROUP_SIZE);

        const unsigned long long laneBase = static_cast<unsigned long long>(logicalLane) * ELEMENTS_PER_LANE;
        ValueT* __restrict__ outBase = output + token * EMBEDDING_DIM;
        if (row == PADDING_INDEX || row >= VOCABULARY_SIZE) {
            const LaneVector zero{};
            if constexpr (EMBEDDING_DIM == static_cast<unsigned long long>(GROUP_SIZE) * ELEMENTS_PER_LANE) {
                *reinterpret_cast<LaneVector*>(outBase + laneBase) = zero;
            } else {
                for (unsigned int i = 0; i < ELEMENTS_PER_LANE; ++i) {
                    const unsigned long long dim = laneBase + i;
                    if (dim < EMBEDDING_DIM) {
                        outBase[dim] = ValueT{};
                    }
                }
            }
        } else if constexpr (EMBEDDING_DIM == static_cast<unsigned long long>(GROUP_SIZE) * ELEMENTS_PER_LANE) {
            const LaneVector tmp = *reinterpret_cast<const LaneVector*>(weights + row * EMBEDDING_DIM + laneBase);
            *reinterpret_cast<LaneVector*>(outBase + laneBase) = tmp;
        } else {
            for (unsigned int i = 0; i < ELEMENTS_PER_LANE; ++i) {
                const unsigned long long dim = laneBase + i;
                if (dim < EMBEDDING_DIM) {
                    outBase[dim] = weights[row * EMBEDDING_DIM + dim];
                }
            }
        }
    }
}
)cuda";
    } else {
        src << R"cuda(
    for (unsigned long long token = globalGroup; token < NUM_INDICES; token += totalGroups) {
        unsigned long long row = 0ull;
        if (logicalLane == 0u) {
            row = static_cast<unsigned long long>(indices[token]);
        }
        row = __shfl_sync(groupMask, row, groupInWarp * GROUP_SIZE);

        const bool zeroRow = (row == PADDING_INDEX || row >= VOCABULARY_SIZE);
        const unsigned long long laneBase = static_cast<unsigned long long>(logicalLane) * ELEMENTS_PER_LANE;
        const unsigned long long linearBase = token * EMBEDDING_DIM + laneBase;
        ValueT* __restrict__ outBase = output + token * EMBEDDING_DIM;
        if constexpr (EMBEDDING_DIM == static_cast<unsigned long long>(GROUP_SIZE) * ELEMENTS_PER_LANE) {
            LaneVector out{};
            #pragma unroll
            for (unsigned int i = 0; i < ELEMENTS_PER_LANE; ++i) {
                const unsigned long long linear = linearBase + i;
                ValueT v = zeroRow ? ValueT{} : weights[row * EMBEDDING_DIM + laneBase + i];
                out.value[i] = static_cast<ValueT>()cuda"
            << generatedEmbeddingEpilogueValue(epilogue) << R"cuda();
            }
            *reinterpret_cast<LaneVector*>(outBase + laneBase) = out;
        } else {
            for (unsigned int i = 0; i < ELEMENTS_PER_LANE; ++i) {
                const unsigned long long dim = laneBase + i;
                if (dim < EMBEDDING_DIM) {
                    const unsigned long long linear = token * EMBEDDING_DIM + dim;
                    ValueT v = zeroRow ? ValueT{} : weights[row * EMBEDDING_DIM + dim];
                    outBase[dim] = static_cast<ValueT>()cuda"
            << generatedEmbeddingEpilogueValue(epilogue) << R"cuda();
                }
            }
        }
    }
}
)cuda";
    }
    return src.str();
}

std::string generatedEmbeddingTinySource(DataType indexDtype,
                                         DataType valueDtype,
                                         bool hasPaddingIndex,
                                         uint64_t numIndices,
                                         uint64_t vocabularySize,
                                         uint64_t embeddingDim,
                                         uint64_t paddingIndex,
                                         const EmbeddingForwardEpilogue& epilogue) {
    if (!hasPaddingIndex) {
        return generatedEmbeddingTinyUncheckedNoPaddingSource(indexDtype, valueDtype, numIndices, embeddingDim, epilogue);
    }
    return generatedEmbeddingTinyPaddingSource(indexDtype, valueDtype, numIndices, vocabularySize, embeddingDim, paddingIndex, epilogue);
}

void checkNvrtc(nvrtcResult status, const char* call) {
    if (status != NVRTC_SUCCESS) {
        throw std::runtime_error(std::string(call) + " failed with " + nvrtcGetErrorString(status));
    }
}

void checkNvrtcCompile(nvrtcProgram prog, const std::vector<const char*>& options) {
    const nvrtcResult status = nvrtcCompileProgram(prog, static_cast<int>(options.size()), options.data());
    if (status == NVRTC_SUCCESS) {
        return;
    }

    size_t logSize = 0;
    (void)nvrtcGetProgramLogSize(prog, &logSize);
    std::string log;
    if (logSize > 1) {
        log.resize(logSize);
        (void)nvrtcGetProgramLog(prog, log.data());
    }
    throw std::runtime_error(std::string("Generated EmbeddingLookup NVRTC compile failed with ") + nvrtcGetErrorString(status) +
                             (log.empty() ? std::string{} : std::string("\n") + log));
}

void addUniqueIncludeDir(std::vector<std::string>& dirs, const std::string& dir) {
    if (dir.empty()) {
        return;
    }
    if (std::find(dirs.begin(), dirs.end(), dir) == dirs.end()) {
        dirs.push_back(dir);
    }
}

std::vector<std::string> generatedEmbeddingCudaIncludeDirs() {
    std::vector<std::string> dirs;
    if (const char* p = std::getenv("THOR_CUDA_INCLUDE_DIR")) {
        addUniqueIncludeDir(dirs, p);
    }
    addUniqueIncludeDir(dirs, THOR_CUDA_INCLUDE_DIR);
    if (const char* p = std::getenv("THOR_CUDA_CCCL_INCLUDE_DIR")) {
        addUniqueIncludeDir(dirs, p);
    }
    addUniqueIncludeDir(dirs, THOR_CUDA_CCCL_INCLUDE_DIR);
    return dirs;
}

void ensureCudaContextCurrentForGeneratedEmbedding(int deviceNum) {
    CU_CHECK(cuInit(0));

    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, deviceNum));

    CUcontext ctx = nullptr;
    CU_CHECK(cuCtxGetCurrent(&ctx));

    if (ctx == nullptr) {
        CUcontext primary;
        CU_CHECK(cuDevicePrimaryCtxRetain(&primary, device));
        CU_CHECK(cuCtxSetCurrent(primary));
        return;
    }

    CUdevice currentDevice;
    CU_CHECK(cuCtxGetDevice(&currentDevice));
    if (static_cast<int>(currentDevice) != deviceNum) {
        CUcontext primary;
        CU_CHECK(cuDevicePrimaryCtxRetain(&primary, device));
        CU_CHECK(cuCtxSetCurrent(primary));
    }
}

std::vector<char> compileGeneratedEmbeddingSourceToCubin(const std::string& source, int deviceNum) {
    if constexpr (PRINT_GENERATED_EMBEDDING_KERNELS) {
        std::fprintf(
            stdout,
            "\n===== Generated EmbeddingLookup CUDA source begin =====\n%s\n===== Generated EmbeddingLookup CUDA source end =====\n",
            source.c_str());
        std::fflush(stdout);
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceNum));

    nvrtcProgram prog = nullptr;
    checkNvrtc(nvrtcCreateProgram(&prog, source.c_str(), "embedding_lookup.cu", 0, nullptr, nullptr), "nvrtcCreateProgram");

    const std::string arch = "--gpu-architecture=sm_" + std::to_string(prop.major) + std::to_string(prop.minor);
    std::vector<std::string> includeArgs;
    for (const std::string& dir : generatedEmbeddingCudaIncludeDirs()) {
        includeArgs.emplace_back("--include-path=" + dir);
    }

    std::vector<const char*> options;
    options.reserve(3 + includeArgs.size());
    options.push_back(arch.c_str());
    options.push_back("--std=c++17");
    options.push_back("-fmad=true");
    for (const std::string& includeArg : includeArgs) {
        options.push_back(includeArg.c_str());
    }

    try {
        checkNvrtcCompile(prog, options);

        size_t cubinSize = 0;
        checkNvrtc(nvrtcGetCUBINSize(prog, &cubinSize), "nvrtcGetCUBINSize");
        std::vector<char> cubin(cubinSize);
        checkNvrtc(nvrtcGetCUBIN(prog, cubin.data()), "nvrtcGetCUBIN");
        checkNvrtc(nvrtcDestroyProgram(&prog), "nvrtcDestroyProgram");
        return cubin;
    } catch (...) {
        if (prog != nullptr) {
            (void)nvrtcDestroyProgram(&prog);
        }
        throw;
    }
}

std::mutex generated_embedding_kernel_cache_mutex;
std::unordered_map<std::string, std::shared_ptr<GeneratedEmbeddingKernel>> generated_embedding_kernel_cache;

std::shared_ptr<GeneratedEmbeddingKernel> lookupGeneratedEmbeddingKernel(const std::string& key) {
    std::lock_guard<std::mutex> lock(generated_embedding_kernel_cache_mutex);
    auto it = generated_embedding_kernel_cache.find(key);
    if (it == generated_embedding_kernel_cache.end()) {
        return nullptr;
    }
    return it->second;
}

void insertGeneratedEmbeddingKernel(const std::string& key, const std::shared_ptr<GeneratedEmbeddingKernel>& compiled) {
    std::lock_guard<std::mutex> lock(generated_embedding_kernel_cache_mutex);
    generated_embedding_kernel_cache[key] = compiled;
}

std::shared_ptr<GeneratedEmbeddingKernel> compileGeneratedEmbeddingExactKernel(DataType indexDtype,
                                                                               DataType valueDtype,
                                                                               uint32_t elementsPerLane,
                                                                               bool hasPaddingIndex,
                                                                               uint64_t numIndices,
                                                                               uint64_t vocabularySize,
                                                                               uint64_t embeddingDim,
                                                                               uint64_t paddingIndex,
                                                                               int deviceNum,
                                                                               const EmbeddingForwardEpilogue& epilogue) {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceNum));

    const std::string source = generatedEmbeddingExactSource(
        indexDtype, valueDtype, elementsPerLane, hasPaddingIndex, numIndices, vocabularySize, embeddingDim, paddingIndex, epilogue);

    std::ostringstream key;
    key << "embedding_lookup_generated_exact:v5\n";
    key << "sm=" << prop.major << prop.minor << "\n";
    key << "device=" << deviceNum << "\n";
    key << source;

    const std::string cacheKey = key.str();
    if (auto hit = lookupGeneratedEmbeddingKernel(cacheKey)) {
        return hit;
    }

    ScopedGpu scopedGpu(deviceNum);
    ensureCudaContextCurrentForGeneratedEmbedding(deviceNum);

    std::vector<char> cubin = compileGeneratedEmbeddingSourceToCubin(source, deviceNum);

    auto compiled = std::make_shared<GeneratedEmbeddingKernel>();
    compiled->cache_key = cacheKey;
    compiled->device_num = deviceNum;
    CU_CHECK(cuModuleLoadData(&compiled->module, cubin.data()));
    CU_CHECK(cuModuleGetFunction(&compiled->kernel, compiled->module, "embedding_lookup"));

    insertGeneratedEmbeddingKernel(cacheKey, compiled);
    return compiled;
}

std::shared_ptr<GeneratedEmbeddingKernel> compileGeneratedEmbeddingTinyKernel(DataType indexDtype,
                                                                              DataType valueDtype,
                                                                              bool hasPaddingIndex,
                                                                              uint64_t numIndices,
                                                                              uint64_t vocabularySize,
                                                                              uint64_t embeddingDim,
                                                                              uint64_t paddingIndex,
                                                                              int deviceNum,
                                                                              const EmbeddingForwardEpilogue& epilogue) {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceNum));

    const std::string source = generatedEmbeddingTinySource(
        indexDtype, valueDtype, hasPaddingIndex, numIndices, vocabularySize, embeddingDim, paddingIndex, epilogue);

    std::ostringstream key;
    key << "embedding_lookup_generated_tiny:v3\n";
    key << "sm=" << prop.major << prop.minor << "\n";
    key << "device=" << deviceNum << "\n";
    key << source;

    const std::string cacheKey = key.str();
    if (auto hit = lookupGeneratedEmbeddingKernel(cacheKey)) {
        return hit;
    }

    ScopedGpu scopedGpu(deviceNum);
    ensureCudaContextCurrentForGeneratedEmbedding(deviceNum);

    std::vector<char> cubin = compileGeneratedEmbeddingSourceToCubin(source, deviceNum);

    auto compiled = std::make_shared<GeneratedEmbeddingKernel>();
    compiled->cache_key = cacheKey;
    compiled->device_num = deviceNum;
    CU_CHECK(cuModuleLoadData(&compiled->module, cubin.data()));
    CU_CHECK(cuModuleGetFunction(&compiled->kernel, compiled->module, "embedding_lookup"));

    insertGeneratedEmbeddingKernel(cacheKey, compiled);
    return compiled;
}

void launchPreparedGeneratedEmbeddingForward(const PreparedEmbeddingForward& prepared,
                                             const Tensor& indices,
                                             const Tensor& weights,
                                             Tensor& output,
                                             Stream stream,
                                             const std::vector<Tensor>& epilogue_inputs) {
    if (epilogue_inputs.size() != prepared.epilogue.extra_input_dtypes.size()) {
        throw std::runtime_error("Prepared EmbeddingLookup epilogue received the wrong number of tensor inputs.");
    }
    if (epilogue_inputs.size() > 16) {
        throw std::runtime_error("Prepared EmbeddingLookup epilogue supports at most 16 extra tensor inputs.");
    }
    void* indicesPtr = const_cast<void*>(indices.getMemPtr<void>());
    void* weightsPtr = const_cast<void*>(weights.getMemPtr<void>());
    void* outputPtr = output.getMemPtr<void>();
    std::array<void*, 16> epiloguePtrs{};
    std::array<void*, 19> kernelArgs{};
    kernelArgs[0] = &indicesPtr;
    kernelArgs[1] = &weightsPtr;
    kernelArgs[2] = &outputPtr;
    for (size_t i = 0; i < epilogue_inputs.size(); ++i) {
        epiloguePtrs[i] = const_cast<void*>(epilogue_inputs[i].getMemPtr<void>());
        kernelArgs[3 + i] = &epiloguePtrs[i];
    }

    ScopedGpu scopedGpu(prepared.device_num);
    CU_CHECK(cuLaunchKernel(prepared.generated_kernel->kernel,
                            prepared.grid_blocks,
                            1,
                            1,
                            THREADS_PER_BLOCK,
                            1,
                            1,
                            0,
                            reinterpret_cast<CUstream>(stream.getStream()),
                            kernelArgs.data(),
                            nullptr));
}

template <typename IndexT, typename ValueT, bool HasPaddingIndex>
void launchPreparedScalarEmbeddingForward(const PreparedEmbeddingForward& prepared,
                                          const Tensor& indices,
                                          const Tensor& weights,
                                          Tensor& output,
                                          Stream stream,
                                          const std::vector<Tensor>&) {
    embeddingForwardWarpScalarFallbackKernel<IndexT, ValueT, HasPaddingIndex>
        <<<prepared.grid_blocks, THREADS_PER_BLOCK, 0, stream.getStream()>>>(indices.getMemPtr<IndexT>(),
                                                                             weights.getMemPtr<ValueT>(),
                                                                             output.getMemPtr<ValueT>(),
                                                                             prepared.num_indices,
                                                                             prepared.vocabulary_size,
                                                                             prepared.embedding_dim,
                                                                             prepared.padding_index);
    CUDA_CHECK(cudaPeekAtLastError());
}

template <typename IndexT, typename ValueT, bool HasPaddingIndex>
std::shared_ptr<PreparedEmbeddingForward> prepareEmbeddingForwardTinyTyped(const Tensor& indices,
                                                                           const Tensor& weights,
                                                                           const Tensor& output,
                                                                           std::optional<uint64_t> paddingIndex,
                                                                           const EmbeddingForwardEpilogue& epilogue) {
    const uint64_t numIndices = indices.getTotalNumElements();
    const std::vector<uint64_t> weightDims = weights.getDimensions();
    const uint64_t vocabularySize = weightDims[0];
    const uint64_t embeddingDim = weightDims[1];
    const uint32_t groupSize = tinyEmbeddingGroupSize(embeddingDim);
    if (groupSize == 0) {
        throw std::invalid_argument("Tiny EmbeddingLookup generated path requires embedding_dim <= 32.");
    }

    static_assert(!std::is_signed_v<IndexT>, "Embedding indices are unsigned-only.");

    auto prepared = std::make_shared<PreparedEmbeddingForward>();
    prepared->launch = &launchPreparedGeneratedEmbeddingForward;
    prepared->num_indices = numIndices;
    prepared->vocabulary_size = vocabularySize;
    prepared->embedding_dim = embeddingDim;
    prepared->padding_index = paddingIndex.value_or(0);
    prepared->has_padding_index = HasPaddingIndex;
    prepared->elements_per_lane = 0;
    prepared->grid_blocks = gridForTinyEmbedding(numIndices, groupSize);
    prepared->device_num = output.getPlacement().getDeviceNum();
    prepared->index_dtype = indices.getDataType();
    prepared->weights_dtype = weights.getDataType();
    prepared->epilogue = epilogue;

    if (numIndices != 0) {
        prepared->generated_kernel = compileGeneratedEmbeddingTinyKernel(indices.getDataType(),
                                                                         weights.getDataType(),
                                                                         HasPaddingIndex,
                                                                         numIndices,
                                                                         vocabularySize,
                                                                         embeddingDim,
                                                                         prepared->padding_index,
                                                                         prepared->device_num,
                                                                         epilogue);
    }
    return prepared;
}

template <typename IndexT, typename ValueT, uint32_t ElementsPerLane, bool HasPaddingIndex>
std::shared_ptr<PreparedEmbeddingForward> prepareEmbeddingForwardWarpExactTyped(const Tensor& indices,
                                                                                const Tensor& weights,
                                                                                const Tensor& output,
                                                                                std::optional<uint64_t> paddingIndex,
                                                                                const EmbeddingForwardEpilogue& epilogue) {
    const uint64_t numIndices = indices.getTotalNumElements();
    const std::vector<uint64_t> weightDims = weights.getDimensions();
    const uint64_t vocabularySize = weightDims[0];
    const uint64_t embeddingDim = weightDims[1];

    static_assert(!std::is_signed_v<IndexT>, "Embedding indices are unsigned-only.");
    static_assert(ElementsPerLane > 0, "ElementsPerLane must be non-zero.");
    static_assert(ElementsPerLane * sizeof(ValueT) <= MAX_COALESCED_BYTES_PER_LANE,
                  "Embedding lane load must fit in one coalesced 16-byte vector step.");

    auto prepared = std::make_shared<PreparedEmbeddingForward>();
    prepared->launch = &launchPreparedGeneratedEmbeddingForward;
    prepared->num_indices = numIndices;
    prepared->vocabulary_size = vocabularySize;
    prepared->embedding_dim = embeddingDim;
    prepared->padding_index = paddingIndex.value_or(0);
    prepared->has_padding_index = HasPaddingIndex;
    prepared->elements_per_lane = ElementsPerLane;
    prepared->grid_blocks = gridForWarps(numIndices);
    prepared->device_num = output.getPlacement().getDeviceNum();
    prepared->index_dtype = indices.getDataType();
    prepared->weights_dtype = weights.getDataType();
    prepared->epilogue = epilogue;

    if (numIndices != 0) {
        prepared->generated_kernel = compileGeneratedEmbeddingExactKernel(indices.getDataType(),
                                                                          weights.getDataType(),
                                                                          ElementsPerLane,
                                                                          HasPaddingIndex,
                                                                          numIndices,
                                                                          vocabularySize,
                                                                          embeddingDim,
                                                                          prepared->padding_index,
                                                                          prepared->device_num,
                                                                          epilogue);
    }
    return prepared;
}

template <typename IndexT, typename ValueT, bool HasPaddingIndex>
std::shared_ptr<PreparedEmbeddingForward> prepareEmbeddingForwardWarpScalarFallbackTyped(const Tensor& indices,
                                                                                         const Tensor& weights,
                                                                                         const Tensor& output,
                                                                                         std::optional<uint64_t> paddingIndex) {
    const uint64_t numIndices = indices.getTotalNumElements();
    const std::vector<uint64_t> weightDims = weights.getDimensions();

    auto prepared = std::make_shared<PreparedEmbeddingForward>();
    prepared->launch = &launchPreparedScalarEmbeddingForward<IndexT, ValueT, HasPaddingIndex>;
    prepared->num_indices = numIndices;
    prepared->vocabulary_size = weightDims[0];
    prepared->embedding_dim = weightDims[1];
    prepared->padding_index = paddingIndex.value_or(0);
    prepared->has_padding_index = HasPaddingIndex;
    prepared->elements_per_lane = 0;
    prepared->grid_blocks = gridForWarps(numIndices);
    prepared->device_num = output.getPlacement().getDeviceNum();
    prepared->index_dtype = indices.getDataType();
    prepared->weights_dtype = weights.getDataType();
    return prepared;
}

uint32_t selectExactElementsPerLane(uint64_t embeddingDim, uint32_t maxElementsPerLane) {
    if (maxElementsPerLane >= 8 && embeddingDim % (8u * WARP_SIZE_EMBEDDING) == 0)
        return 8;
    if (maxElementsPerLane >= 4 && embeddingDim % (4u * WARP_SIZE_EMBEDDING) == 0)
        return 4;
    if (maxElementsPerLane >= 2 && embeddingDim % (2u * WARP_SIZE_EMBEDDING) == 0)
        return 2;
    if (embeddingDim % WARP_SIZE_EMBEDDING == 0)
        return 1;
    return 0;
}

template <typename IndexT, typename ValueT, bool HasPaddingIndex>
std::shared_ptr<PreparedEmbeddingForward> prepareEmbeddingForwardTypedWithPaddingMode(const Tensor& indices,
                                                                                      const Tensor& weights,
                                                                                      const Tensor& output,
                                                                                      std::optional<uint64_t> paddingIndex,
                                                                                      const EmbeddingForwardEpilogue& epilogue) {
    const uint64_t embeddingDim = weights.getDimensions()[1];
    constexpr uint32_t maxElementsPerLane = MAX_COALESCED_BYTES_PER_LANE / sizeof(ValueT);
    static_assert(maxElementsPerLane >= 1, "Embedding value dtype is too wide for the coalesced lane copy size.");
    static_assert(maxElementsPerLane <= 8, "Embedding forward intentionally caps lane copies to 16 bytes.");

    if (embeddingDim <= WARP_SIZE_EMBEDDING) {
        return prepareEmbeddingForwardTinyTyped<IndexT, ValueT, HasPaddingIndex>(indices, weights, output, paddingIndex, epilogue);
    }

    switch (selectExactElementsPerLane(embeddingDim, maxElementsPerLane)) {
        case 8:
            if constexpr (maxElementsPerLane >= 8) {
                return prepareEmbeddingForwardWarpExactTyped<IndexT, ValueT, 8, HasPaddingIndex>(
                    indices, weights, output, paddingIndex, epilogue);
            } else {
                THOR_UNREACHABLE();
            }
        case 4:
            if constexpr (maxElementsPerLane >= 4) {
                return prepareEmbeddingForwardWarpExactTyped<IndexT, ValueT, 4, HasPaddingIndex>(
                    indices, weights, output, paddingIndex, epilogue);
            } else {
                THOR_UNREACHABLE();
            }
        case 2:
            if constexpr (maxElementsPerLane >= 2) {
                return prepareEmbeddingForwardWarpExactTyped<IndexT, ValueT, 2, HasPaddingIndex>(
                    indices, weights, output, paddingIndex, epilogue);
            } else {
                THOR_UNREACHABLE();
            }
        case 1:
            return prepareEmbeddingForwardWarpExactTyped<IndexT, ValueT, 1, HasPaddingIndex>(
                indices, weights, output, paddingIndex, epilogue);
        case 0:
            if (epilogue.enabled()) {
                throw std::runtime_error("EmbeddingLookup root fusion currently requires an exact generated embedding row width.");
            }
            return prepareEmbeddingForwardWarpScalarFallbackTyped<IndexT, ValueT, HasPaddingIndex>(indices, weights, output, paddingIndex);
        default:
            THOR_UNREACHABLE();
    }
}

template <typename IndexT, typename ValueT>
std::shared_ptr<PreparedEmbeddingForward> prepareEmbeddingForwardTyped(const Tensor& indices,
                                                                       const Tensor& weights,
                                                                       const Tensor& output,
                                                                       std::optional<uint64_t> paddingIndex,
                                                                       const EmbeddingForwardEpilogue& epilogue) {
    if (paddingIndex.has_value()) {
        return prepareEmbeddingForwardTypedWithPaddingMode<IndexT, ValueT, true>(indices, weights, output, paddingIndex, epilogue);
    }
    return prepareEmbeddingForwardTypedWithPaddingMode<IndexT, ValueT, false>(indices, weights, output, paddingIndex, epilogue);
}

template <typename IndexT>
std::shared_ptr<PreparedEmbeddingForward> prepareEmbeddingForwardValueDtype(const Tensor& indices,
                                                                            const Tensor& weights,
                                                                            const Tensor& output,
                                                                            std::optional<uint64_t> paddingIndex,
                                                                            const EmbeddingForwardEpilogue& epilogue) {
    switch (weights.getDataType()) {
        case DataType::FP16:
            return prepareEmbeddingForwardTyped<IndexT, __half>(indices, weights, output, paddingIndex, epilogue);
        case DataType::BF16:
            return prepareEmbeddingForwardTyped<IndexT, __nv_bfloat16>(indices, weights, output, paddingIndex, epilogue);
        case DataType::FP32:
            return prepareEmbeddingForwardTyped<IndexT, float>(indices, weights, output, paddingIndex, epilogue);
        default:
            throw std::invalid_argument("Embedding forward weights dtype must be fp16, bf16, or fp32. Got " +
                                        dataTypeName(weights.getDataType()) + ".");
    }
}

void validateEmbeddingForwardInputs(const Tensor& indices, const Tensor& weights, const Tensor& output) {
    if (indices.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU ||
        weights.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU ||
        output.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::invalid_argument("Embedding forward tensors must all live on GPU.");
    }
    if (indices.getPlacement() != weights.getPlacement() || indices.getPlacement() != output.getPlacement()) {
        throw std::invalid_argument("Embedding forward tensors must be on the same GPU placement.");
    }
    if (!isSupportedEmbeddingIndexDtype(indices.getDataType())) {
        throw std::invalid_argument("Embedding indices dtype must be uint32 or uint64. Got " + dataTypeName(indices.getDataType()) + ".");
    }
    if (!isSupportedEmbeddingValueDtype(weights.getDataType())) {
        throw std::invalid_argument("Embedding weights dtype must be fp16, bf16, or fp32. Got " + dataTypeName(weights.getDataType()) +
                                    ".");
    }
    if (output.getDataType() != weights.getDataType()) {
        throw std::invalid_argument("Embedding output dtype must match weights dtype in this backend slice.");
    }
    const std::vector<uint64_t> weightDims = weights.getDimensions();
    if (weightDims.size() != 2 || weightDims[0] == 0 || weightDims[1] == 0) {
        throw std::invalid_argument("Embedding weights tensor must have shape [vocabulary_size, embedding_dim].");
    }
    std::vector<uint64_t> expectedOutputDims = indices.getDimensions();
    expectedOutputDims.push_back(weightDims[1]);
    if (output.getDimensions() != expectedOutputDims) {
        throw std::invalid_argument("Embedding output dimensions must equal indices dimensions with embedding_dim appended.");
    }
    validateDenseContiguous(indices, "indices");
    validateDenseContiguous(weights, "weights");
    validateDenseContiguous(output, "output");
}

void validateEmbeddingForwardEpilogue(const EmbeddingForwardEpilogue& epilogue, const Tensor& output) {
    if (!epilogue.enabled()) {
        if (!epilogue.extra_input_dtypes.empty()) {
            throw std::invalid_argument("Embedding epilogue input dtype list must be empty when no epilogue expression is provided.");
        }
        return;
    }
    if (epilogue.extra_input_dtypes.size() > 16) {
        throw std::invalid_argument("Embedding epilogue supports at most 16 extra tensor inputs.");
    }
    for (DataType dtype : epilogue.extra_input_dtypes) {
        if (dtype != output.getDataType()) {
            throw std::invalid_argument("Embedding epilogue tensor input dtypes must match the output dtype.");
        }
    }
}

}  // namespace

std::shared_ptr<PreparedEmbeddingForward> prepareEmbeddingForward(const Tensor& indices,
                                                                  const Tensor& weights,
                                                                  const Tensor& output,
                                                                  std::optional<uint64_t> paddingIndex,
                                                                  const EmbeddingForwardEpilogue& epilogue) {
    validateEmbeddingForwardInputs(indices, weights, output);
    validateEmbeddingForwardEpilogue(epilogue, output);

    switch (indices.getDataType()) {
        case DataType::UINT32:
            return prepareEmbeddingForwardValueDtype<uint32_t>(indices, weights, output, paddingIndex, epilogue);
        case DataType::UINT64:
            return prepareEmbeddingForwardValueDtype<uint64_t>(indices, weights, output, paddingIndex, epilogue);
        default:
            THOR_UNREACHABLE();
    }
}

void launchPreparedEmbeddingForward(const PreparedEmbeddingForward& prepared,
                                    const Tensor& indices,
                                    const Tensor& weights,
                                    Tensor& output,
                                    Stream stream,
                                    const std::vector<Tensor>& epilogue_inputs) {
    if (prepared.num_indices == 0) {
        return;
    }
    if (prepared.launch == nullptr) {
        throw std::runtime_error("PreparedEmbeddingForward is missing its launch function.");
    }
    prepared.launch(prepared, indices, weights, output, stream, epilogue_inputs);
}

void launchEmbeddingForward(
    const Tensor& indices, const Tensor& weights, Tensor& output, std::optional<uint64_t> paddingIndex, Stream stream) {
    std::shared_ptr<PreparedEmbeddingForward> prepared = prepareEmbeddingForward(indices, weights, output, paddingIndex);
    launchPreparedEmbeddingForward(*prepared, indices, weights, output, stream);
}

}  // namespace ThorImplementation
