#include "Utilities/TensorOperations/Embedding/EmbeddingSparseGradient.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/Event.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/Embedding/EmbeddingSparseGradientCudaCompile.h"
#include "Utilities/TensorOperations/Embedding/ReduceStageController.h"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ThorImplementation {
namespace {

#ifndef THOR_EMBEDDING_SPARSE_GRADIENT_LOW_RUN_MAX
#define THOR_EMBEDDING_SPARSE_GRADIENT_LOW_RUN_MAX 16U
#endif

#ifndef THOR_EMBEDDING_SPARSE_GRADIENT_ULTRA_HIGH_RUN_MIN
#define THOR_EMBEDDING_SPARSE_GRADIENT_ULTRA_HIGH_RUN_MIN 512U
#endif

#ifndef THOR_EMBEDDING_SPARSE_GRADIENT_ULTRA_HIGH_TOKENS_PER_PARTIAL
#define THOR_EMBEDDING_SPARSE_GRADIENT_ULTRA_HIGH_TOKENS_PER_PARTIAL 1024U
#endif

constexpr uint32_t EMBEDDING_SPARSE_GRADIENT_LOW_RUN_MAX = THOR_EMBEDDING_SPARSE_GRADIENT_LOW_RUN_MAX;
constexpr uint32_t EMBEDDING_SPARSE_GRADIENT_ULTRA_HIGH_RUN_MIN = THOR_EMBEDDING_SPARSE_GRADIENT_ULTRA_HIGH_RUN_MIN;
constexpr uint32_t EMBEDDING_SPARSE_GRADIENT_ULTRA_HIGH_TOKENS_PER_PARTIAL = THOR_EMBEDDING_SPARSE_GRADIENT_ULTRA_HIGH_TOKENS_PER_PARTIAL;

static_assert(EMBEDDING_SPARSE_GRADIENT_LOW_RUN_MAX >= 1U);
static_assert(EMBEDDING_SPARSE_GRADIENT_ULTRA_HIGH_RUN_MIN > EMBEDDING_SPARSE_GRADIENT_LOW_RUN_MAX);
static_assert(EMBEDDING_SPARSE_GRADIENT_ULTRA_HIGH_TOKENS_PER_PARTIAL >= 1U);

constexpr uint32_t THREADS_PER_BLOCK = 256;
constexpr bool PRINT_GENERATED_EMBEDDING_SPARSE_GRADIENT_KERNELS = false;
constexpr uint32_t WARP_SIZE_EMBEDDING = 32;
constexpr uint32_t WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE_EMBEDDING;
constexpr uint32_t DEFAULT_EMBEDDING_SPARSE_GRADIENT_ULTRA_HIGH_TOKENS_PER_PARTIAL = 512U;
// Keep static fixed-D reducer instantiations bounded so the CUDA TU does not explode in compile time/code size.
// Larger dimensions are handled by generated vectorized JIT reducers.
constexpr uint32_t MAX_STATIC_FIXED_EMBEDDING_SPARSE_GRADIENT_DIM = 4096U;
// Keep the generated vectorized JIT path available well past 256 KiB rows.  This is a
// capability/policy cap, not a geometry cap: for large aligned dimensions the generated
// kernels still process one 4-element vector per participating thread and scale by adding
// grid.y vector tiles.  1,048,576 FP32 dims is a 4 MiB row, so memory pressure should be
// the practical limiter long before the scheduler geometry is.
constexpr uint64_t MAX_VECTORIZED_JIT_EMBEDDING_SPARSE_GRADIENT_DIM = 1048576ULL;
static_assert(MAX_VECTORIZED_JIT_EMBEDDING_SPARSE_GRADIENT_DIM <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));
static_assert(THREADS_PER_BLOCK == 256);
static_assert(WARPS_PER_BLOCK == 8);

std::mutex gEmbeddingSparseGradientRunBucketConfigMutex;
std::optional<EmbeddingSparseGradientRunBucketConfig> gEmbeddingSparseGradientRunBucketConfigOverride;

void validateEmbeddingSparseGradientRunBucketConfig(const EmbeddingSparseGradientRunBucketConfig& config) {
    if (config.lowRunMax + 1U >= config.ultraHighRunMin) {
        throw std::invalid_argument("Embedding sparse-gradient bucket config requires low_run_max + 1 < ultra_high_run_min.");
    }
    if (config.ultraHighTokensPerPartial == 0U) {
        throw std::invalid_argument("Embedding sparse-gradient bucket config requires ultra_high_tokens_per_partial > 0.");
    }
}

EmbeddingSparseGradientRunBucketConfig currentEmbeddingSparseGradientRunBucketConfig() {
    std::lock_guard<std::mutex> lock(gEmbeddingSparseGradientRunBucketConfigMutex);
    return gEmbeddingSparseGradientRunBucketConfigOverride.value_or(
        EmbeddingSparseGradientRunBucketConfig{EMBEDDING_SPARSE_GRADIENT_LOW_RUN_MAX,
                                               EMBEDDING_SPARSE_GRADIENT_ULTRA_HIGH_RUN_MIN,
                                               DEFAULT_EMBEDDING_SPARSE_GRADIENT_ULTRA_HIGH_TOKENS_PER_PARTIAL});
}

std::string dtypeName(DataType dtype) { return TensorDescriptor::getElementTypeName(dtype); }

bool isSupportedIndexType(DataType dtype) { return dtype == DataType::UINT32 || dtype == DataType::UINT64; }

bool isSupportedRowType(DataType dtype) { return dtype == DataType::UINT16 || dtype == DataType::UINT32 || dtype == DataType::UINT64; }

bool isSupportedGradientType(DataType dtype) {
    switch (dtype) {
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
            return true;
        default:
            return false;
    }
}

void validateDenseContiguous(const Tensor& tensor, const std::string& name) {
    if (tensor.hasCustomStrides() || !tensor.isDenseContiguous()) {
        throw std::invalid_argument("Embedding sparse-gradient " + name + " tensor must be dense contiguous.");
    }
}

void validateSamePlacement(const Tensor& tensor, const TensorPlacement& placement, const std::string& name) {
    if (!tensor.isInitialized()) {
        throw std::invalid_argument("Embedding sparse-gradient " + name + " tensor is not initialized.");
    }
    if (tensor.getPlacement() != placement) {
        throw std::invalid_argument("Embedding sparse-gradient " + name + " tensor must live on the same placement as the indices tensor.");
    }
}

uint64_t checkedProduct(uint64_t a, uint64_t b, const std::string& label) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        throw std::overflow_error("Embedding sparse-gradient " + label + " exceeds uint64_t range.");
    }
    return a * b;
}

int checkedCubItems(uint64_t n, const std::string& label) {
    if (n > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("Embedding sparse-gradient " + label + " exceeds CUB's int item-count limit in this backend slice.");
    }
    return static_cast<int>(n);
}

uint32_t downloadUint32Scalar(const Tensor& tensor, Stream stream) {
    uint32_t value = 0;
    CUDA_CHECK(cudaMemcpyAsync(&value, tensor.getMemPtr<uint32_t>(), sizeof(value), cudaMemcpyDeviceToHost, stream.getStream()));
    stream.synchronize();
    return value;
}

uint64_t downloadSparseRowCountScalar(const Tensor& tensor, DataType dtype, Stream stream) {
    uint64_t value = 0;
    switch (dtype) {
        case DataType::UINT16: {
            uint16_t typed = 0;
            CUDA_CHECK(cudaMemcpyAsync(&typed, tensor.getMemPtr<uint16_t>(), sizeof(typed), cudaMemcpyDeviceToHost, stream.getStream()));
            stream.synchronize();
            value = static_cast<uint64_t>(typed);
            break;
        }
        case DataType::UINT32: {
            uint32_t typed = 0;
            CUDA_CHECK(cudaMemcpyAsync(&typed, tensor.getMemPtr<uint32_t>(), sizeof(typed), cudaMemcpyDeviceToHost, stream.getStream()));
            stream.synchronize();
            value = static_cast<uint64_t>(typed);
            break;
        }
        case DataType::UINT64: {
            uint64_t typed = 0;
            CUDA_CHECK(cudaMemcpyAsync(&typed, tensor.getMemPtr<uint64_t>(), sizeof(typed), cudaMemcpyDeviceToHost, stream.getStream()));
            stream.synchronize();
            value = typed;
            break;
        }
        default:
            throw std::invalid_argument("Embedding sparse-gradient profile row count tensor must be uint16, uint32, or uint64.");
    }
    return value;
}

template <typename T>
DataType rowDTypeForCppType();

template <>
DataType rowDTypeForCppType<uint16_t>() {
    return DataType::UINT16;
}

template <>
DataType rowDTypeForCppType<uint32_t>() {
    return DataType::UINT32;
}

template <>
DataType rowDTypeForCppType<uint64_t>() {
    return DataType::UINT64;
}

template <typename RowT>
bool rowTypeCanRepresentVocabularySentinel(uint64_t vocabularySize) {
    return vocabularySize <= static_cast<uint64_t>(std::numeric_limits<RowT>::max());
}

template <typename T>
__device__ __forceinline__ float thor_embedding_grad_to_float(T v) {
    return static_cast<float>(v);
}

template <>
__device__ __forceinline__ float thor_embedding_grad_to_float<__half>(__half v) {
    return __half2float(v);
}

template <>
__device__ __forceinline__ float thor_embedding_grad_to_float<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

__device__ __forceinline__ float2 thor_embedding_half2_bits_to_float2(uint32_t bits) {
    const __half2 packed = *reinterpret_cast<const __half2*>(&bits);
    return __half22float2(packed);
}

__device__ __forceinline__ float2 thor_embedding_bfloat162_bits_to_float2(uint32_t bits) {
    const __nv_bfloat162 packed = *reinterpret_cast<const __nv_bfloat162*>(&bits);
    return __bfloat1622float2(packed);
}

template <typename GradT, uint32_t EmbeddingDim, uint32_t BlockThreads>
struct EmbeddingSparseGradientVectorOps {
    static_assert(EmbeddingDim % 4U == 0U, "Fixed embedding sparse-gradient vector reducer requires D divisible by 4.");

    __device__ __forceinline__ static float4 load(uint64_t token, uint32_t vectorIndex, const GradT* __restrict__ upstreamGradient) {
        const uint64_t base = token * static_cast<uint64_t>(EmbeddingDim) + static_cast<uint64_t>(vectorIndex) * 4ULL;
        return make_float4(thor_embedding_grad_to_float(upstreamGradient[base + 0ULL]),
                           thor_embedding_grad_to_float(upstreamGradient[base + 1ULL]),
                           thor_embedding_grad_to_float(upstreamGradient[base + 2ULL]),
                           thor_embedding_grad_to_float(upstreamGradient[base + 3ULL]));
    }

    __device__ __forceinline__ static void store(uint64_t row, uint32_t vectorIndex, float4 value, float* __restrict__ outputValues) {
        float4* __restrict__ output = reinterpret_cast<float4*>(outputValues + row * static_cast<uint64_t>(EmbeddingDim));
        output[vectorIndex] = value;
    }
};

template <uint32_t EmbeddingDim, uint32_t BlockThreads>
struct EmbeddingSparseGradientVectorOps<float, EmbeddingDim, BlockThreads> {
    static_assert(EmbeddingDim % 4U == 0U, "FP32 vector reducer requires D divisible by 4.");

    __device__ __forceinline__ static float4 load(uint64_t token, uint32_t vectorIndex, const float* __restrict__ upstreamGradient) {
        const float4* __restrict__ input = reinterpret_cast<const float4*>(upstreamGradient + token * static_cast<uint64_t>(EmbeddingDim));
        return input[vectorIndex];
    }

    __device__ __forceinline__ static void store(uint64_t row, uint32_t vectorIndex, float4 value, float* __restrict__ outputValues) {
        float4* __restrict__ output = reinterpret_cast<float4*>(outputValues + row * static_cast<uint64_t>(EmbeddingDim));
        output[vectorIndex] = value;
    }
};

template <uint32_t EmbeddingDim, uint32_t BlockThreads>
struct EmbeddingSparseGradientVectorOps<__half, EmbeddingDim, BlockThreads> {
    static_assert(EmbeddingDim % 4U == 0U, "FP16 vector reducer requires D divisible by 4.");

    __device__ __forceinline__ static float4 load(uint64_t token, uint32_t vectorIndex, const __half* __restrict__ upstreamGradient) {
        const unsigned long long* __restrict__ input =
            reinterpret_cast<const unsigned long long*>(upstreamGradient + token * static_cast<uint64_t>(EmbeddingDim));
        const unsigned long long packed = input[vectorIndex];
        const float2 lo = thor_embedding_half2_bits_to_float2(static_cast<uint32_t>(packed));
        const float2 hi = thor_embedding_half2_bits_to_float2(static_cast<uint32_t>(packed >> 32U));
        return make_float4(lo.x, lo.y, hi.x, hi.y);
    }

    __device__ __forceinline__ static void store(uint64_t row, uint32_t vectorIndex, float4 value, float* __restrict__ outputValues) {
        float4* __restrict__ output = reinterpret_cast<float4*>(outputValues + row * static_cast<uint64_t>(EmbeddingDim));
        output[vectorIndex] = value;
    }
};

template <uint32_t EmbeddingDim, uint32_t BlockThreads>
struct EmbeddingSparseGradientVectorOps<__nv_bfloat16, EmbeddingDim, BlockThreads> {
    static_assert(EmbeddingDim % 4U == 0U, "BF16 vector reducer requires D divisible by 4.");

    __device__ __forceinline__ static float4 load(uint64_t token,
                                                  uint32_t vectorIndex,
                                                  const __nv_bfloat16* __restrict__ upstreamGradient) {
        const unsigned long long* __restrict__ input =
            reinterpret_cast<const unsigned long long*>(upstreamGradient + token * static_cast<uint64_t>(EmbeddingDim));
        const unsigned long long packed = input[vectorIndex];
        const float2 lo = thor_embedding_bfloat162_bits_to_float2(static_cast<uint32_t>(packed));
        const float2 hi = thor_embedding_bfloat162_bits_to_float2(static_cast<uint32_t>(packed >> 32U));
        return make_float4(lo.x, lo.y, hi.x, hi.y);
    }

    __device__ __forceinline__ static void store(uint64_t row, uint32_t vectorIndex, float4 value, float* __restrict__ outputValues) {
        float4* __restrict__ output = reinterpret_cast<float4*>(outputValues + row * static_cast<uint64_t>(EmbeddingDim));
        output[vectorIndex] = value;
    }
};

__device__ __forceinline__ float4 thor_embedding_add_float4(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

template <typename IndexT, typename RowT>
__global__ void materializeEmbeddingSparseGradientSortPairsKernel(const IndexT* __restrict__ indices,
                                                                  RowT* __restrict__ rowKeys,
                                                                  uint32_t* __restrict__ tokenIds,
                                                                  uint64_t numTokens,
                                                                  uint64_t vocabularySize,
                                                                  uint64_t paddingIndex,
                                                                  bool hasPaddingIndex) {
    static_assert(!std::is_signed_v<IndexT>, "Embedding sparse-gradient indices are unsigned-only.");
    static_assert(!std::is_signed_v<RowT>, "Embedding sparse-gradient row keys are unsigned-only.");

    const uint64_t token = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (token >= numTokens) {
        return;
    }

    const uint64_t row = static_cast<uint64_t>(indices[token]);
    const bool valid = row < vocabularySize && (!hasPaddingIndex || row != paddingIndex);
    rowKeys[token] = static_cast<RowT>(valid ? row : vocabularySize);
    tokenIds[token] = static_cast<uint32_t>(token);
}

template <uint32_t EmbeddingDim>
struct IsFixedEmbeddingReducerDim {
    static constexpr bool value =
        EmbeddingDim == 4U || EmbeddingDim == 8U || EmbeddingDim == 16U || EmbeddingDim == 32U || EmbeddingDim == 64U ||
        EmbeddingDim == 128U || EmbeddingDim == 768U ||
        (EmbeddingDim >= 256U && EmbeddingDim <= MAX_STATIC_FIXED_EMBEDDING_SPARSE_GRADIENT_DIM && EmbeddingDim % 256U == 0U);
};

template <uint32_t EmbeddingDim>
struct FixedEmbeddingReducerBlockThreads {
    static_assert(IsFixedEmbeddingReducerDim<EmbeddingDim>::value, "Unsupported fixed embedding sparse-gradient reducer dimension.");
    static constexpr uint32_t value = EmbeddingDim <= THREADS_PER_BLOCK ? THREADS_PER_BLOCK : EmbeddingDim <= 1024U ? EmbeddingDim : 1024U;
    static_assert(value <= 1024U, "Fixed embedding sparse-gradient reducer block size exceeds CUDA's portable block limit.");
};

template <uint32_t EmbeddingDim>
struct FixedEmbeddingReducerGeometry {
    static_assert(IsFixedEmbeddingReducerDim<EmbeddingDim>::value, "Unsupported fixed embedding sparse-gradient reducer dimension.");
    static_assert(EmbeddingDim % 4U == 0U, "Fixed embedding sparse-gradient vector reducer requires D divisible by 4.");
    static constexpr uint32_t BLOCK_THREADS = FixedEmbeddingReducerBlockThreads<EmbeddingDim>::value;
    static constexpr uint32_t NUM_VECTORS = EmbeddingDim / 4U;
    static constexpr uint32_t VECTORS_PER_TILE = NUM_VECTORS < (BLOCK_THREADS / 4U) ? NUM_VECTORS : (BLOCK_THREADS / 4U);
    static constexpr uint32_t ROWS_PER_LOW_BLOCK = BLOCK_THREADS / VECTORS_PER_TILE;
    static constexpr uint32_t DUPLICATE_LANES = BLOCK_THREADS / VECTORS_PER_TILE;
    static_assert(VECTORS_PER_TILE >= 1U, "Fixed embedding sparse-gradient reducer requires at least one vector per tile.");
    static_assert(BLOCK_THREADS % VECTORS_PER_TILE == 0U, "Fixed embedding sparse-gradient reducer tile must divide block size.");
    static_assert(ROWS_PER_LOW_BLOCK >= 1U, "Fixed embedding sparse-gradient reducer requires at least one row per low-run block.");
    static_assert(DUPLICATE_LANES >= 1U, "Fixed embedding sparse-gradient reducer requires at least one duplicate lane.");
};

template <typename RowT, typename GradT, uint32_t EmbeddingDim>
__global__ void reduceEmbeddingSparseGradientValuesFixedDimKernel(const uint32_t* __restrict__ sortedTokenIds,
                                                                  const uint32_t* __restrict__ runOffsets,
                                                                  const uint32_t* __restrict__ runCounts,
                                                                  const uint32_t* __restrict__ numRunRows,
                                                                  const uint32_t* __restrict__ runRowIndices,
                                                                  const GradT* __restrict__ upstreamGradient,
                                                                  float* __restrict__ outputValues) {
    static_assert(IsFixedEmbeddingReducerDim<EmbeddingDim>::value, "Unsupported fixed embedding sparse-gradient reducer dimension.");
    static_assert(EmbeddingDim % 4U == 0U, "Fixed embedding sparse-gradient vector reducer requires D divisible by 4.");

    using Geometry = FixedEmbeddingReducerGeometry<EmbeddingDim>;
    using VecOps = EmbeddingSparseGradientVectorOps<GradT, EmbeddingDim, Geometry::BLOCK_THREADS>;
    constexpr uint32_t VECTORS_PER_TILE = Geometry::VECTORS_PER_TILE;
    constexpr uint32_t ROWS_PER_BLOCK = Geometry::ROWS_PER_LOW_BLOCK;

    const uint32_t rowInBlock = threadIdx.x / VECTORS_PER_TILE;
    const uint32_t vectorInTile = threadIdx.x - rowInBlock * VECTORS_PER_TILE;
    const uint32_t globalVectorIndex = blockIdx.y * VECTORS_PER_TILE + vectorInTile;
    const uint64_t bucketRow = static_cast<uint64_t>(blockIdx.x) * ROWS_PER_BLOCK + rowInBlock;
    if (bucketRow >= static_cast<uint64_t>(numRunRows[0]) || globalVectorIndex >= Geometry::NUM_VECTORS) {
        return;
    }
    const uint64_t row = static_cast<uint64_t>(runRowIndices[bucketRow]);

    const uint64_t begin = static_cast<uint64_t>(runOffsets[row]);
    const uint64_t count = static_cast<uint64_t>(runCounts[row]);
    const uint64_t firstToken = static_cast<uint64_t>(sortedTokenIds[begin]);

    // Each active thread handles one 16-byte FP32 output vector. For D>1024 the row is split across grid.y D-tiles,
    // keeping the block size within CUDA's 1024-thread limit while preserving coalesced vector loads/stores.
    float4 sum = VecOps::load(firstToken, globalVectorIndex, upstreamGradient);
    for (uint64_t i = 1ULL; i < count; ++i) {
        const uint64_t token = static_cast<uint64_t>(sortedTokenIds[begin + i]);
        sum = thor_embedding_add_float4(sum, VecOps::load(token, globalVectorIndex, upstreamGradient));
    }
    VecOps::store(row, globalVectorIndex, sum, outputValues);
}

template <typename RowT, typename GradT, uint32_t EmbeddingDim>
__global__ void reduceEmbeddingSparseGradientValuesHighRunFixedDimKernel(const uint32_t* __restrict__ sortedTokenIds,
                                                                         const uint32_t* __restrict__ runOffsets,
                                                                         const uint32_t* __restrict__ runCounts,
                                                                         const uint32_t* __restrict__ numRunRows,
                                                                         const uint32_t* __restrict__ runRowIndices,
                                                                         const GradT* __restrict__ upstreamGradient,
                                                                         float* __restrict__ outputValues) {
    static_assert(IsFixedEmbeddingReducerDim<EmbeddingDim>::value, "Unsupported fixed embedding sparse-gradient reducer dimension.");
    static_assert(EmbeddingDim % 4U == 0U, "High-run fixed embedding sparse-gradient reducer requires D divisible by 4.");

    using Geometry = FixedEmbeddingReducerGeometry<EmbeddingDim>;
    using VecOps = EmbeddingSparseGradientVectorOps<GradT, EmbeddingDim, Geometry::BLOCK_THREADS>;
    constexpr uint32_t VECTORS_PER_TILE = Geometry::VECTORS_PER_TILE;
    constexpr uint32_t DUPLICATE_LANES = Geometry::DUPLICATE_LANES;

    __shared__ float4 partialSums[Geometry::BLOCK_THREADS];

    const uint32_t vectorInTile = threadIdx.x % VECTORS_PER_TILE;
    const uint32_t duplicateLane = threadIdx.x / VECTORS_PER_TILE;
    const uint32_t globalVectorIndex = blockIdx.y * VECTORS_PER_TILE + vectorInTile;
    const uint64_t bucketRow = static_cast<uint64_t>(blockIdx.x);
    if (bucketRow >= static_cast<uint64_t>(numRunRows[0])) {
        return;
    }
    const bool validVector = globalVectorIndex < Geometry::NUM_VECTORS;
    const uint64_t row = static_cast<uint64_t>(runRowIndices[bucketRow]);

    const uint64_t begin = static_cast<uint64_t>(runOffsets[row]);
    const uint64_t count = static_cast<uint64_t>(runCounts[row]);
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (validVector) {
        for (uint64_t i = static_cast<uint64_t>(duplicateLane); i < count; i += DUPLICATE_LANES) {
            const uint64_t token = static_cast<uint64_t>(sortedTokenIds[begin + i]);
            sum = thor_embedding_add_float4(sum, VecOps::load(token, globalVectorIndex, upstreamGradient));
        }
    }

    partialSums[threadIdx.x] = sum;
    __syncthreads();

    if (duplicateLane != 0U || !validVector) {
        return;
    }

    for (uint32_t lane = 1U; lane < DUPLICATE_LANES; ++lane) {
        sum = thor_embedding_add_float4(sum, partialSums[lane * VECTORS_PER_TILE + vectorInTile]);
    }
    VecOps::store(row, globalVectorIndex, sum, outputValues);
}

template <typename RowT, typename GradT>
__global__ void reduceEmbeddingSparseGradientValuesTiledKernel(const uint32_t* __restrict__ sortedTokenIds,
                                                               const uint32_t* __restrict__ runOffsets,
                                                               const uint32_t* __restrict__ runCounts,
                                                               const uint32_t* __restrict__ numRunRows,
                                                               const uint32_t* __restrict__ runRowIndices,
                                                               const GradT* __restrict__ upstreamGradient,
                                                               float* __restrict__ outputValues,
                                                               uint64_t embeddingDim) {
    const uint64_t bucketRow = static_cast<uint64_t>(blockIdx.x);
    if (bucketRow >= static_cast<uint64_t>(numRunRows[0])) {
        return;
    }
    const uint64_t row = static_cast<uint64_t>(runRowIndices[bucketRow]);

    const uint64_t dim = static_cast<uint64_t>(blockIdx.y) * THREADS_PER_BLOCK + threadIdx.x;
    if (dim >= embeddingDim) {
        return;
    }

    const uint64_t begin = static_cast<uint64_t>(runOffsets[row]);
    const uint64_t count = static_cast<uint64_t>(runCounts[row]);
    if (count == 1ULL) {
        const uint64_t token = static_cast<uint64_t>(sortedTokenIds[begin]);
        outputValues[row * embeddingDim + dim] = thor_embedding_grad_to_float(upstreamGradient[token * embeddingDim + dim]);
        return;
    }

    float sum = 0.0f;
    for (uint64_t i = 0; i < count; ++i) {
        const uint64_t token = static_cast<uint64_t>(sortedTokenIds[begin + i]);
        sum += thor_embedding_grad_to_float(upstreamGradient[token * embeddingDim + dim]);
    }
    outputValues[row * embeddingDim + dim] = sum;
}

bool hasVectorizedJitEmbeddingSparseGradientReducer(uint64_t embeddingDim) {
    return embeddingDim != 0ULL && embeddingDim <= MAX_VECTORIZED_JIT_EMBEDDING_SPARSE_GRADIENT_DIM;
}

void validateVectorizedJitEmbeddingSparseGradientDim(uint64_t embeddingDim) {
    if (!hasVectorizedJitEmbeddingSparseGradientReducer(embeddingDim)) {
        throw std::invalid_argument(
            "Embedding sparse-gradient reducer requires 1 <= embedding_dim <= " +
            std::to_string(MAX_VECTORIZED_JIT_EMBEDDING_SPARSE_GRADIENT_DIM) +
            "; above that cap the implementation intentionally refuses to fall back to the old scalar tiled reducer.");
    }
}

uint64_t gcdUint64(uint64_t a, uint64_t b) {
    while (b != 0ULL) {
        const uint64_t r = a % b;
        a = b;
        b = r;
    }
    return a;
}

uint64_t lcmUint64(uint64_t a, uint64_t b) { return (a / gcdUint64(a, b)) * b; }

uint32_t vectorizedReducerVectorCountForEmbeddingSparseGradient(uint64_t embeddingDim) {
    return static_cast<uint32_t>((embeddingDim + 3ULL) / 4ULL);
}

uint32_t vectorizedReducerVectorsPerTileForEmbeddingSparseGradient(uint64_t embeddingDim) {
    const uint64_t vectorCount = vectorizedReducerVectorCountForEmbeddingSparseGradient(embeddingDim);
    if (vectorCount == 0ULL) {
        return 1U;
    }

    const uint64_t maxVectorsPerTile = std::min<uint64_t>(vectorCount, 256ULL);
    for (uint64_t candidate = maxVectorsPerTile; candidate >= 1ULL; --candidate) {
        if (lcmUint64(candidate, static_cast<uint64_t>(WARP_SIZE_EMBEDDING)) <= 1024ULL) {
            return static_cast<uint32_t>(candidate);
        }
    }
    return 1U;
}

uint32_t vectorizedReducerBlockThreadsForEmbeddingSparseGradient(uint64_t embeddingDim) {
    const uint64_t vectorsPerTile = vectorizedReducerVectorsPerTileForEmbeddingSparseGradient(embeddingDim);
    const uint64_t quantum = lcmUint64(vectorsPerTile, static_cast<uint64_t>(WARP_SIZE_EMBEDDING));
    return static_cast<uint32_t>((1024ULL / quantum) * quantum);
}

uint32_t reduceBlockThreadsForEmbeddingSparseGradient(uint64_t embeddingDim) {
    if (hasVectorizedJitEmbeddingSparseGradientReducer(embeddingDim)) {
        return vectorizedReducerBlockThreadsForEmbeddingSparseGradient(embeddingDim);
    }
    return THREADS_PER_BLOCK;
}

uint32_t fixedReducerVectorsPerTileForEmbeddingSparseGradient(uint64_t embeddingDim) {
    if (hasVectorizedJitEmbeddingSparseGradientReducer(embeddingDim)) {
        return vectorizedReducerVectorsPerTileForEmbeddingSparseGradient(embeddingDim);
    }
    const uint64_t numVectors = (embeddingDim + 3ULL) / 4ULL;
    return static_cast<uint32_t>(std::max<uint64_t>(1ULL, std::min<uint64_t>(numVectors, THREADS_PER_BLOCK / 4ULL)));
}

uint32_t fixedReducerVectorTileCountForEmbeddingSparseGradient(uint64_t embeddingDim) {
    if (!hasVectorizedJitEmbeddingSparseGradientReducer(embeddingDim)) {
        return static_cast<uint32_t>((embeddingDim + THREADS_PER_BLOCK - 1ULL) / THREADS_PER_BLOCK);
    }
    const uint64_t numVectors = vectorizedReducerVectorCountForEmbeddingSparseGradient(embeddingDim);
    const uint32_t vectorsPerTile = fixedReducerVectorsPerTileForEmbeddingSparseGradient(embeddingDim);
    return static_cast<uint32_t>((numVectors + vectorsPerTile - 1ULL) / vectorsPerTile);
}

uint32_t reduceRowsPerGridXForEmbeddingSparseGradient(uint64_t embeddingDim) {
    if (!hasVectorizedJitEmbeddingSparseGradientReducer(embeddingDim)) {
        return 1U;
    }
    // JIT reducers use one 4-value vector lane per thread. Choose the vector tile so blockDim.x is both a full-warp
    // multiple and a full-row multiple; otherwise awkward dimensions like D=96 leave stray lanes that touch the next row.
    return static_cast<uint32_t>(reduceBlockThreadsForEmbeddingSparseGradient(embeddingDim) /
                                 fixedReducerVectorsPerTileForEmbeddingSparseGradient(embeddingDim));
}

uint32_t reduceGridDimXForEmbeddingSparseGradient(uint64_t numRows, uint64_t embeddingDim) {
    const uint32_t rowsPerGridX = reduceRowsPerGridXForEmbeddingSparseGradient(embeddingDim);
    return static_cast<uint32_t>((numRows + rowsPerGridX - 1ULL) / rowsPerGridX);
}

uint32_t reduceGridDimYForEmbeddingSparseGradient(uint64_t embeddingDim) {
    if (hasVectorizedJitEmbeddingSparseGradientReducer(embeddingDim)) {
        return fixedReducerVectorTileCountForEmbeddingSparseGradient(embeddingDim);
    }
    return static_cast<uint32_t>((embeddingDim + THREADS_PER_BLOCK - 1ULL) / THREADS_PER_BLOCK);
}

bool hasHighRunCtaEmbeddingSparseGradientReducer(uint64_t embeddingDim) {
    return hasVectorizedJitEmbeddingSparseGradientReducer(embeddingDim);
}

uint32_t highRunReduceRowsPerGridXForEmbeddingSparseGradient(uint64_t embeddingDim) {
    if (!hasHighRunCtaEmbeddingSparseGradientReducer(embeddingDim)) {
        return reduceRowsPerGridXForEmbeddingSparseGradient(embeddingDim);
    }
    return 1U;
}

uint32_t highRunReduceGridDimXForEmbeddingSparseGradient(uint64_t numRows, uint64_t embeddingDim) {
    const uint32_t rowsPerGridX = highRunReduceRowsPerGridXForEmbeddingSparseGradient(embeddingDim);
    return static_cast<uint32_t>((numRows + rowsPerGridX - 1ULL) / rowsPerGridX);
}

uint32_t highRunReduceGridDimYForEmbeddingSparseGradient(uint64_t embeddingDim) {
    return reduceGridDimYForEmbeddingSparseGradient(embeddingDim);
}

uint32_t highRunReduceBlockThreadsForEmbeddingSparseGradient(uint64_t embeddingDim) {
    return reduceBlockThreadsForEmbeddingSparseGradient(embeddingDim);
}

bool hasUltraHighTwoStageEmbeddingSparseGradientReducer(uint64_t embeddingDim) {
    return hasVectorizedJitEmbeddingSparseGradientReducer(embeddingDim);
}

uint32_t ultraHighReduceBlockThreadsForEmbeddingSparseGradient(uint64_t embeddingDim) {
    return reduceBlockThreadsForEmbeddingSparseGradient(embeddingDim);
}

uint32_t ultraHighFinalReduceRowsPerGridXForEmbeddingSparseGradient(uint64_t embeddingDim) {
    if (!hasUltraHighTwoStageEmbeddingSparseGradientReducer(embeddingDim)) {
        return reduceRowsPerGridXForEmbeddingSparseGradient(embeddingDim);
    }
    return 1U;
}

uint32_t ultraHighFinalReduceGridDimXForEmbeddingSparseGradient(uint64_t numRows, uint64_t embeddingDim) {
    const uint32_t rowsPerGridX = ultraHighFinalReduceRowsPerGridXForEmbeddingSparseGradient(embeddingDim);
    return static_cast<uint32_t>((numRows + rowsPerGridX - 1ULL) / rowsPerGridX);
}

uint32_t ultraHighFinalReduceGridDimYForEmbeddingSparseGradient(uint64_t embeddingDim) {
    return reduceGridDimYForEmbeddingSparseGradient(embeddingDim);
}

uint32_t ultraHighPartialReduceGridDimYForEmbeddingSparseGradient(uint64_t embeddingDim) {
    return reduceGridDimYForEmbeddingSparseGradient(embeddingDim);
}

uint64_t maxRowsWithAtLeastTokensPerRun(uint64_t capacity, uint64_t numTokens, uint32_t minTokensPerRun) {
    if (minTokensPerRun == 0U) {
        throw std::invalid_argument("Embedding sparse-gradient bucket max-row estimate requires min_tokens_per_run > 0.");
    }
    const uint64_t rows = numTokens / static_cast<uint64_t>(minTokensPerRun);
    return std::max<uint64_t>(1ULL, std::min<uint64_t>(capacity, rows));
}

uint64_t maxHighRunRowsForEmbeddingSparseGradient(uint64_t capacity,
                                                  uint64_t numTokens,
                                                  const EmbeddingSparseGradientRunBucketConfig& config) {
    // Every high-run row has at least lowRunMax + 1 contributing tokens. This is a tight, data-independent upper bound that
    // keeps the non-graph/raw launch path from launching the high-run reducer over the full sparse-gradient capacity.
    return maxRowsWithAtLeastTokensPerRun(capacity, numTokens, config.lowRunMax + 1U);
}

uint64_t maxUltraHighRunRowsForEmbeddingSparseGradient(uint64_t capacity,
                                                       uint64_t numTokens,
                                                       const EmbeddingSparseGradientRunBucketConfig& config) {
    // Every ultra-high row has at least ultraHighRunMin contributing tokens. Graph execution is updated to the exact runtime
    // row count by the finalizer; raw launches use this conservative upper bound instead of outputGradient.capacity.
    return maxRowsWithAtLeastTokensPerRun(capacity, numTokens, config.ultraHighRunMin);
}

uint64_t maxUltraHighPartialsForEmbeddingSparseGradient(uint64_t numTokens, const EmbeddingSparseGradientRunBucketConfig& config) {
    const uint64_t byTokenChunks = (numTokens + config.ultraHighTokensPerPartial - 1ULL) / config.ultraHighTokensPerPartial;
    const uint64_t byUltraRows = (numTokens + config.ultraHighRunMin - 1ULL) / config.ultraHighRunMin;
    return std::max<uint64_t>(1ULL, byTokenChunks + byUltraRows + 1ULL);
}

uint32_t ultraHighPartialReduceGridDimXForEmbeddingSparseGradient(uint64_t maxUltraHighPartials) {
    if (maxUltraHighPartials > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::invalid_argument("Embedding sparse-gradient ultra-high partial count exceeds CUDA gridDim.x range.");
    }
    return static_cast<uint32_t>(std::max<uint64_t>(1ULL, maxUltraHighPartials));
}

EmbeddingSparseGradientReduceGridUpdateConfig ultraHighPartialReduceGridUpdateConfigForEmbeddingSparseGradient(
    uint64_t maxUltraHighPartials, uint64_t embeddingDim) {
    EmbeddingSparseGradientReduceGridUpdateConfig config;
    config.reduceRowsPerGridX = 1U;
    config.reduceGridDimY = ultraHighPartialReduceGridDimYForEmbeddingSparseGradient(embeddingDim);
    config.minReduceGridDimX = 1U;
    config.maxReduceGridDimX = ultraHighPartialReduceGridDimXForEmbeddingSparseGradient(maxUltraHighPartials);
    config.maxReduceGridDimY = config.reduceGridDimY;
    return config;
}

EmbeddingSparseGradientReduceGridUpdateConfig reduceGridUpdateConfigForEmbeddingSparseGradient(uint64_t maxRows, uint64_t embeddingDim) {
    EmbeddingSparseGradientReduceGridUpdateConfig config;
    config.reduceRowsPerGridX = reduceRowsPerGridXForEmbeddingSparseGradient(embeddingDim);
    config.reduceGridDimY = reduceGridDimYForEmbeddingSparseGradient(embeddingDim);
    config.minReduceGridDimX = 1U;
    config.maxReduceGridDimX = reduceGridDimXForEmbeddingSparseGradient(maxRows, embeddingDim);
    config.maxReduceGridDimY = config.reduceGridDimY;
    return config;
}

EmbeddingSparseGradientReduceGridUpdateConfig highRunReduceGridUpdateConfigForEmbeddingSparseGradient(uint64_t maxRows,
                                                                                                      uint64_t embeddingDim) {
    EmbeddingSparseGradientReduceGridUpdateConfig config;
    config.reduceRowsPerGridX = highRunReduceRowsPerGridXForEmbeddingSparseGradient(embeddingDim);
    config.reduceGridDimY = highRunReduceGridDimYForEmbeddingSparseGradient(embeddingDim);
    config.minReduceGridDimX = 1U;
    config.maxReduceGridDimX = highRunReduceGridDimXForEmbeddingSparseGradient(maxRows, embeddingDim);
    config.maxReduceGridDimY = config.reduceGridDimY;
    return config;
}

EmbeddingSparseGradientReduceGridUpdateConfig ultraHighFinalReduceGridUpdateConfigForEmbeddingSparseGradient(uint64_t maxRows,
                                                                                                             uint64_t embeddingDim) {
    EmbeddingSparseGradientReduceGridUpdateConfig config;
    config.reduceRowsPerGridX = ultraHighFinalReduceRowsPerGridXForEmbeddingSparseGradient(embeddingDim);
    config.reduceGridDimY = ultraHighFinalReduceGridDimYForEmbeddingSparseGradient(embeddingDim);
    config.minReduceGridDimX = 1U;
    config.maxReduceGridDimX = ultraHighFinalReduceGridDimXForEmbeddingSparseGradient(maxRows, embeddingDim);
    config.maxReduceGridDimY = config.reduceGridDimY;
    return config;
}

template <typename RowT>
size_t querySortTempBytes(RowT* keysIn, RowT* keysOut, uint32_t* valuesIn, uint32_t* valuesOut, int numItems) {
    size_t bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, bytes, keysIn, keysOut, valuesIn, valuesOut, numItems, 0, 8 * sizeof(RowT)));
    return bytes;
}

template <typename RowT>
size_t queryRleTempBytes(RowT* in, RowT* uniqueOut, uint32_t* countsOut, uint32_t* numRunsOut, int numItems) {
    size_t bytes = 0;
    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(nullptr, bytes, in, uniqueOut, countsOut, numRunsOut, numItems));
    return bytes;
}

size_t queryScanTempBytes(uint32_t* countsIn, uint32_t* offsetsOut, int numItems) {
    size_t bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, bytes, countsIn, offsetsOut, numItems));
    return bytes;
}

Tensor allocateByteScratch(const TensorPlacement& placement, size_t bytes) {
    return Tensor(placement, TensorDescriptor(DataType::UINT8, {std::max<uint64_t>(static_cast<uint64_t>(bytes), 1ULL)}));
}

std::string sparseGradientScalarType(DataType dtype) {
    switch (dtype) {
        case DataType::FP16:
            return "__half";
        case DataType::BF16:
            return "__nv_bfloat16";
        case DataType::FP32:
            return "float";
        case DataType::UINT16:
            return "unsigned short";
        case DataType::UINT32:
            return "unsigned int";
        case DataType::UINT64:
            return "unsigned long long";
        default:
            throw std::runtime_error("Unsupported sparse-gradient graph dtype " + dtypeName(dtype) + ".");
    }
}

std::string sparseGradientReduceKernelName(DataType rowDType, DataType gradDType, uint64_t embeddingDim, bool hasSparseRowUpdate) {
    std::ostringstream ss;
    ss << "thor_embedding_sparse_reduce_r" << static_cast<uint64_t>(rowDType) << "_g" << static_cast<uint64_t>(gradDType) << "_d"
       << embeddingDim;
    if (hasSparseRowUpdate) {
        ss << "_sru";
    }
    return ss.str();
}

std::string sparseGradientHighRunReduceKernelName(DataType rowDType, DataType gradDType, uint64_t embeddingDim, bool hasSparseRowUpdate) {
    std::ostringstream ss;
    ss << "thor_embedding_sparse_high_run_reduce_r" << static_cast<uint64_t>(rowDType) << "_g" << static_cast<uint64_t>(gradDType) << "_d"
       << embeddingDim;
    if (hasSparseRowUpdate) {
        ss << "_sru";
    }
    return ss.str();
}

std::string sparseGradientUltraHighPartialReduceKernelName(DataType rowDType, DataType gradDType, uint64_t embeddingDim) {
    std::ostringstream ss;
    ss << "thor_embedding_sparse_ultra_high_partial_reduce_r" << static_cast<uint64_t>(rowDType) << "_g" << static_cast<uint64_t>(gradDType)
       << "_d" << embeddingDim;
    return ss.str();
}

std::string sparseGradientUltraHighFinalReduceKernelName(DataType rowDType,
                                                         DataType gradDType,
                                                         uint64_t embeddingDim,
                                                         bool hasSparseRowUpdate) {
    std::ostringstream ss;
    ss << "thor_embedding_sparse_ultra_high_final_reduce_r" << static_cast<uint64_t>(rowDType) << "_g" << static_cast<uint64_t>(gradDType)
       << "_d" << embeddingDim;
    if (hasSparseRowUpdate) {
        ss << "_sru";
    }
    return ss.str();
}


std::string emitEmbeddingSparseGradientFloat4LoadHelperSource() {
    // For odd D, row bases rotate through 16-byte alignments. Do not repair every unaligned
    // load with two aligned float4 loads plus a shift; that turns D=16K+1 into effectively
    // D=32K worth of load instructions. Use vector loads only when actually aligned.
    std::ostringstream ss;
    ss << "__device__ __forceinline__ float4 thor_embedding_load_float4_aligned(const float* p, unsigned long long i) {\n";
    ss << "  return *reinterpret_cast<const float4*>(p + i);\n";
    ss << "}\n";
    ss << "__device__ __forceinline__ float4 thor_embedding_load_float4_masked(const float* p, unsigned long long i, unsigned int lanes) {\n";
    ss << "  if (lanes >= 4U && ((i & 3ULL) == 0ULL)) return thor_embedding_load_float4_aligned(p, i);\n";
    ss << "  float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);\n";
    ss << "  if (lanes > 0U) v.x = p[i];\n";
    ss << "  if (lanes > 1U) v.y = p[i + 1ULL];\n";
    ss << "  if (lanes > 2U) v.z = p[i + 2ULL];\n";
    ss << "  if (lanes > 3U) v.w = p[i + 3ULL];\n";
    ss << "  return v;\n";
    ss << "}\n\n";
    return ss.str();
}

std::string emitSparseGradientReduceKernelSource(DataType rowDType,
                                                 DataType gradDType,
                                                 uint64_t embeddingDim,
                                                 const std::string& kernelName,
                                                 const SparseRowUpdateFusionSource* sparseRowUpdate) {
    if (sparseRowUpdate != nullptr && !hasVectorizedJitEmbeddingSparseGradientReducer(embeddingDim)) {
        throw std::invalid_argument(
            "Embedding sparse-gradient fused sparse-row update currently requires a vectorized JIT reducer with embedding_dim <= " +
            std::to_string(MAX_VECTORIZED_JIT_EMBEDDING_SPARSE_GRADIENT_DIM) + ".");
    }

    const std::string rowType = sparseGradientScalarType(rowDType);
    const std::string gradType = sparseGradientScalarType(gradDType);

    std::ostringstream ss;
    ss << "#include <cuda_fp16.h>\n";
    ss << "#include <cuda_bf16.h>\n";
    ss << "#include <math_functions.h>\n\n";
    if (sparseRowUpdate != nullptr) {
        ss << sparseRowUpdate->helperSource;
    }
    ss << "__device__ __forceinline__ float thor_embedding_grad_to_float(float v) { return v; }\n";
    ss << "__device__ __forceinline__ float thor_embedding_grad_to_float(__half v) { return __half2float(v); }\n";
    ss << "__device__ __forceinline__ float thor_embedding_grad_to_float(__nv_bfloat16 v) { return __bfloat162float(v); }\n";
    ss << "__device__ __forceinline__ float2 thor_embedding_half2_bits_to_float2(unsigned int bits) {\n";
    ss << "  const __half2 packed = *reinterpret_cast<const __half2*>(&bits);\n";
    ss << "  return __half22float2(packed);\n";
    ss << "}\n";
    ss << "__device__ __forceinline__ float2 thor_embedding_bfloat162_bits_to_float2(unsigned int bits) {\n";
    ss << "  const __nv_bfloat162 packed = *reinterpret_cast<const __nv_bfloat162*>(&bits);\n";
    ss << "  return __bfloat1622float2(packed);\n";
    ss << "}\n\n";
    ss << "__device__ __forceinline__ float4 thor_embedding_add_float4(float4 a, float4 b) {\n";
    ss << "  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);\n";
    ss << "}\n\n";
    ss << emitEmbeddingSparseGradientFloat4LoadHelperSource();
    ss << "extern \"C\" __global__\n";
    ss << "void " << kernelName << "(const unsigned int* sortedTokenIds, const unsigned int* runOffsets, const unsigned int* runCounts, ";
    ss << "const unsigned int* numRunRows, const unsigned int* runRowIndices, const " << gradType << "* upstreamGradient";
    if (sparseRowUpdate != nullptr) {
        ss << ", const " << rowType << "* outputRows" << sparseRowUpdate->parameterSource;
    } else {
        ss << ", float* outputValues";
    }
    ss << ") {\n";

    if (hasVectorizedJitEmbeddingSparseGradientReducer(embeddingDim)) {
        const uint64_t threadsPerBlock = reduceBlockThreadsForEmbeddingSparseGradient(embeddingDim);
        const uint64_t numVectors = (embeddingDim + 3ULL) / 4ULL;
        const uint64_t vectorsPerTile = fixedReducerVectorsPerTileForEmbeddingSparseGradient(embeddingDim);
        const uint64_t rowsPerBlock = threadsPerBlock / vectorsPerTile;
        ss << "  constexpr unsigned int THREADS_PER_BLOCK_LOCAL = " << threadsPerBlock << "U;\n";
        ss << "  constexpr unsigned int embeddingDim = " << embeddingDim << "U;\n";
        ss << "  constexpr unsigned int dimsPerThread = 4U;\n";
        ss << "  constexpr unsigned int numVectors = " << numVectors << "U;\n";
        ss << "  constexpr unsigned int vectorsPerTile = " << vectorsPerTile << "U;\n";
        ss << "  constexpr unsigned int rowsPerBlock = " << rowsPerBlock << "U;\n";
        ss << "  const unsigned int rowInBlock = threadIdx.x / vectorsPerTile;\n";
        ss << "  const unsigned int vectorInTile = threadIdx.x - rowInBlock * vectorsPerTile;\n";
        ss << "  const unsigned int globalVectorIndex = blockIdx.y * vectorsPerTile + vectorInTile;\n";
        ss << "  const unsigned long long bucketRow = static_cast<unsigned long long>(blockIdx.x) * rowsPerBlock + rowInBlock;\n";
        ss << "  if (bucketRow >= static_cast<unsigned long long>(numRunRows[0]) || globalVectorIndex >= numVectors) return;\n";
        ss << "  const unsigned long long row = static_cast<unsigned long long>(runRowIndices[bucketRow]);\n";
        ss << "  const unsigned long long begin = static_cast<unsigned long long>(runOffsets[row]);\n";
        ss << "  const unsigned long long count = static_cast<unsigned long long>(runCounts[row]);\n";
        ss << "  const unsigned long long firstToken = static_cast<unsigned long long>(sortedTokenIds[begin]);\n";
        ss << "  const unsigned int dimBase = globalVectorIndex * dimsPerThread;\n";
        if (embeddingDim % 4ULL == 0ULL) {
            ss << "  constexpr unsigned int vectorValidLanes = dimsPerThread;\n";
        } else {
            ss << "  const unsigned int vectorValidLanes = (dimBase + dimsPerThread <= embeddingDim) ? dimsPerThread : (embeddingDim - "
                  "dimBase);\n";
        }

        auto emitLoad = [&](const std::string& tokenExpr, const std::string& valueName) {
            if (embeddingDim % 4ULL != 0ULL) {
                ss << "  const unsigned long long base_" << valueName << " = (" << tokenExpr
                   << ") * static_cast<unsigned long long>(embeddingDim) + static_cast<unsigned long long>(dimBase);\n";
                if (gradDType == DataType::FP32) {
                    ss << "  " << (valueName == "sum" ? "float4 " : "const float4 ") << valueName
                       << " = thor_embedding_load_float4_masked(upstreamGradient, base_" << valueName << ", vectorValidLanes);\n";
                } else {
                    ss << "  float4 " << valueName << " = make_float4(0.0f, 0.0f, 0.0f, 0.0f);\n";
                    ss << "  if (vectorValidLanes > 0U) " << valueName << ".x = thor_embedding_grad_to_float(upstreamGradient[base_"
                       << valueName << "]);\n";
                    ss << "  if (vectorValidLanes > 1U) " << valueName << ".y = thor_embedding_grad_to_float(upstreamGradient[base_"
                       << valueName << " + 1ULL]);\n";
                    ss << "  if (vectorValidLanes > 2U) " << valueName << ".z = thor_embedding_grad_to_float(upstreamGradient[base_"
                       << valueName << " + 2ULL]);\n";
                    ss << "  if (vectorValidLanes > 3U) " << valueName << ".w = thor_embedding_grad_to_float(upstreamGradient[base_"
                       << valueName << " + 3ULL]);\n";
                }
                return;
            }
            if (gradDType == DataType::FP32) {
                ss << "  const float4* __restrict__ input_" << valueName << " = reinterpret_cast<const float4*>(upstreamGradient + ("
                   << tokenExpr << ") * embeddingDim);\n";
                ss << "  " << (valueName == "sum" ? "float4 " : "const float4 ") << valueName << " = input_" << valueName
                   << "[globalVectorIndex];\n";
            } else if (gradDType == DataType::FP16) {
                ss << "  const unsigned long long* __restrict__ input_" << valueName
                   << " = reinterpret_cast<const unsigned long long*>(upstreamGradient + (" << tokenExpr << ") * embeddingDim);\n";
                ss << "  const unsigned long long packed_" << valueName << " = input_" << valueName << "[globalVectorIndex];\n";
                ss << "  const float2 lo_" << valueName << " = thor_embedding_half2_bits_to_float2(static_cast<unsigned int>(packed_"
                   << valueName << "));\n";
                ss << "  const float2 hi_" << valueName << " = thor_embedding_half2_bits_to_float2(static_cast<unsigned int>(packed_"
                   << valueName << " >> 32U));\n";
                ss << "  " << (valueName == "sum" ? "float4 " : "const float4 ") << valueName << " = make_float4(lo_" << valueName
                   << ".x, lo_" << valueName << ".y, hi_" << valueName << ".x, hi_" << valueName << ".y);\n";
            } else if (gradDType == DataType::BF16) {
                ss << "  const unsigned long long* __restrict__ input_" << valueName
                   << " = reinterpret_cast<const unsigned long long*>(upstreamGradient + (" << tokenExpr << ") * embeddingDim);\n";
                ss << "  const unsigned long long packed_" << valueName << " = input_" << valueName << "[globalVectorIndex];\n";
                ss << "  const float2 lo_" << valueName << " = thor_embedding_bfloat162_bits_to_float2(static_cast<unsigned int>(packed_"
                   << valueName << "));\n";
                ss << "  const float2 hi_" << valueName << " = thor_embedding_bfloat162_bits_to_float2(static_cast<unsigned int>(packed_"
                   << valueName << " >> 32U));\n";
                ss << "  " << (valueName == "sum" ? "float4 " : "const float4 ") << valueName << " = make_float4(lo_" << valueName
                   << ".x, lo_" << valueName << ".y, hi_" << valueName << ".x, hi_" << valueName << ".y);\n";
            } else {
                ss << "  const unsigned long long base_" << valueName << " = (" << tokenExpr
                   << ") * static_cast<unsigned long long>(embeddingDim) + static_cast<unsigned long long>(globalVectorIndex) * "
                      "dimsPerThread;\n";
                ss << "  " << (valueName == "sum" ? "float4 " : "const float4 ") << valueName
                   << " = make_float4(thor_embedding_grad_to_float(upstreamGradient[base_" << valueName
                   << " + 0ULL]), thor_embedding_grad_to_float(upstreamGradient[base_" << valueName
                   << " + 1ULL]), thor_embedding_grad_to_float(upstreamGradient[base_" << valueName
                   << " + 2ULL]), thor_embedding_grad_to_float(upstreamGradient[base_" << valueName << " + 3ULL]));\n";
            }
        };

        emitLoad("firstToken", "sum");
        ss << "  for (unsigned long long i = 1ULL; i < count; ++i) {\n";
        ss << "    const unsigned long long token = static_cast<unsigned long long>(sortedTokenIds[begin + i]);\n";
        emitLoad("token", "v");
        ss << "    sum = thor_embedding_add_float4(sum, v);\n";
        ss << "  }\n";
        if (sparseRowUpdate != nullptr) {
            ss << "  const unsigned long long sru_logical_row = row;\n";
            ss << "  const unsigned long long sru_indexed_row = static_cast<unsigned long long>(outputRows[row]);\n";
            ss << "  const unsigned long long sru_vector_index = static_cast<unsigned long long>(globalVectorIndex);\n";
            ss << "  const unsigned int sru_vector_valid_lanes = vectorValidLanes;\n";
            ss << sparseRowUpdate->bodySource;
        } else {
            if (embeddingDim % 4ULL == 0ULL) {
                ss << "  float4* __restrict__ output = reinterpret_cast<float4*>(outputValues + row * embeddingDim);\n";
                ss << "  output[globalVectorIndex] = sum;\n";
            } else {
                ss << "  float* __restrict__ output = outputValues + row * static_cast<unsigned long long>(embeddingDim) + dimBase;\n";
                ss << "  if (vectorValidLanes > 0U) output[0] = sum.x;\n";
                ss << "  if (vectorValidLanes > 1U) output[1] = sum.y;\n";
                ss << "  if (vectorValidLanes > 2U) output[2] = sum.z;\n";
                ss << "  if (vectorValidLanes > 3U) output[3] = sum.w;\n";
            }
        }
    } else {
        ss << "  constexpr unsigned int THREADS_PER_BLOCK_LOCAL = " << THREADS_PER_BLOCK << "U;\n";
        ss << "  const unsigned long long bucketRow = static_cast<unsigned long long>(blockIdx.x);\n";
        ss << "  if (bucketRow >= static_cast<unsigned long long>(numRunRows[0])) return;\n";
        ss << "  const unsigned long long row = static_cast<unsigned long long>(runRowIndices[bucketRow]);\n";
        ss << "  const unsigned long long embeddingDim = " << embeddingDim << "ULL;\n";
        ss << "  const unsigned long long dim = static_cast<unsigned long long>(blockIdx.y) * THREADS_PER_BLOCK_LOCAL + threadIdx.x;\n";
        ss << "  if (dim >= embeddingDim) return;\n";
        ss << "  const unsigned long long begin = static_cast<unsigned long long>(runOffsets[row]);\n";
        ss << "  const unsigned long long count = static_cast<unsigned long long>(runCounts[row]);\n";
        ss << "  const unsigned long long firstToken = static_cast<unsigned long long>(sortedTokenIds[begin]);\n";
        ss << "  float sum = thor_embedding_grad_to_float(upstreamGradient[firstToken * embeddingDim + dim]);\n";
        ss << "  for (unsigned long long i = 1ULL; i < count; ++i) {\n";
        ss << "    const unsigned long long token = static_cast<unsigned long long>(sortedTokenIds[begin + i]);\n";
        ss << "    sum += thor_embedding_grad_to_float(upstreamGradient[token * embeddingDim + dim]);\n";
        ss << "  }\n";
        ss << "  outputValues[row * embeddingDim + dim] = sum;\n";
    }

    ss << "}\n";
    return ss.str();
}

std::string emitSparseGradientHighRunReduceKernelSource(DataType rowDType,
                                                        DataType gradDType,
                                                        uint64_t embeddingDim,
                                                        const std::string& kernelName,
                                                        const SparseRowUpdateFusionSource* sparseRowUpdate) {
    if (!hasHighRunCtaEmbeddingSparseGradientReducer(embeddingDim)) {
        throw std::invalid_argument(
            "Embedding sparse-gradient high-run reducer currently requires a vectorized JIT reducer with embedding_dim <= " +
            std::to_string(MAX_VECTORIZED_JIT_EMBEDDING_SPARSE_GRADIENT_DIM) + ".");
    }

    const std::string rowType = sparseGradientScalarType(rowDType);
    const std::string gradType = sparseGradientScalarType(gradDType);
    const uint64_t threadsPerBlock = highRunReduceBlockThreadsForEmbeddingSparseGradient(embeddingDim);
    const uint64_t numVectors = (embeddingDim + 3ULL) / 4ULL;
    const uint64_t vectorsPerTile = fixedReducerVectorsPerTileForEmbeddingSparseGradient(embeddingDim);
    const uint64_t duplicateLanes = threadsPerBlock / vectorsPerTile;

    std::ostringstream ss;
    ss << "#include <cuda_fp16.h>\n";
    ss << "#include <cuda_bf16.h>\n";
    ss << "#include <math_functions.h>\n\n";
    if (sparseRowUpdate != nullptr) {
        ss << sparseRowUpdate->helperSource;
    }
    ss << "__device__ __forceinline__ float thor_embedding_grad_to_float(float v) { return v; }\n";
    ss << "__device__ __forceinline__ float thor_embedding_grad_to_float(__half v) { return __half2float(v); }\n";
    ss << "__device__ __forceinline__ float thor_embedding_grad_to_float(__nv_bfloat16 v) { return __bfloat162float(v); }\n";
    ss << "__device__ __forceinline__ float2 thor_embedding_half2_bits_to_float2(unsigned int bits) {\n";
    ss << "  const __half2 packed = *reinterpret_cast<const __half2*>(&bits);\n";
    ss << "  return __half22float2(packed);\n";
    ss << "}\n";
    ss << "__device__ __forceinline__ float2 thor_embedding_bfloat162_bits_to_float2(unsigned int bits) {\n";
    ss << "  const __nv_bfloat162 packed = *reinterpret_cast<const __nv_bfloat162*>(&bits);\n";
    ss << "  return __bfloat1622float2(packed);\n";
    ss << "}\n\n";
    ss << "__device__ __forceinline__ float4 thor_embedding_add_float4(float4 a, float4 b) {\n";
    ss << "  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);\n";
    ss << "}\n\n";
    ss << emitEmbeddingSparseGradientFloat4LoadHelperSource();
    ss << "extern \"C\" __global__\n";
    ss << "void " << kernelName << "(const unsigned int* sortedTokenIds, const unsigned int* runOffsets, const unsigned int* runCounts, ";
    ss << "const unsigned int* numRunRows, const unsigned int* runRowIndices, const " << gradType << "* upstreamGradient";
    if (sparseRowUpdate != nullptr) {
        ss << ", const " << rowType << "* outputRows" << sparseRowUpdate->parameterSource;
    } else {
        ss << ", float* outputValues";
    }
    ss << ") {\n";
    ss << "  constexpr unsigned int THREADS_PER_BLOCK_LOCAL = " << threadsPerBlock << "U;\n";
    ss << "  constexpr unsigned int embeddingDim = " << embeddingDim << "U;\n";
    ss << "  constexpr unsigned int dimsPerThread = 4U;\n";
    ss << "  constexpr unsigned int numVectors = " << numVectors << "U;\n";
    ss << "  constexpr unsigned int vectorsPerTile = " << vectorsPerTile << "U;\n";
    ss << "  constexpr unsigned int duplicateLanes = " << duplicateLanes << "U;\n";
    ss << "  __shared__ float4 partialSums[THREADS_PER_BLOCK_LOCAL];\n";
    ss << "  const unsigned int vectorInTile = threadIdx.x % vectorsPerTile;\n";
    ss << "  const unsigned int duplicateLane = threadIdx.x / vectorsPerTile;\n";
    ss << "  const unsigned int globalVectorIndex = blockIdx.y * vectorsPerTile + vectorInTile;\n";
    ss << "  const unsigned long long bucketRow = static_cast<unsigned long long>(blockIdx.x);\n";
    ss << "  if (bucketRow >= static_cast<unsigned long long>(numRunRows[0])) return;\n";
    ss << "  const bool validVector = globalVectorIndex < numVectors;\n";
    ss << "  const unsigned int dimBase = globalVectorIndex * dimsPerThread;\n";
    if (embeddingDim % 4ULL == 0ULL) {
        ss << "  const unsigned int vectorValidLanes = validVector ? dimsPerThread : 0U;\n";
    } else {
        ss << "  const unsigned int vectorValidLanes = validVector ? ((dimBase + dimsPerThread <= embeddingDim) ? dimsPerThread : "
              "(embeddingDim - dimBase)) : 0U;\n";
    }
    ss << "  const unsigned long long row = static_cast<unsigned long long>(runRowIndices[bucketRow]);\n";
    ss << "  const unsigned long long begin = static_cast<unsigned long long>(runOffsets[row]);\n";
    ss << "  const unsigned long long count = static_cast<unsigned long long>(runCounts[row]);\n";
    ss << "  float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);\n";
    ss << "  if (validVector) {\n";
    ss << "  for (unsigned long long i = static_cast<unsigned long long>(duplicateLane); i < count; i += duplicateLanes) {\n";
    ss << "    const unsigned long long token = static_cast<unsigned long long>(sortedTokenIds[begin + i]);\n";
    if (embeddingDim % 4ULL != 0ULL) {
        ss << "    const unsigned long long base_v = token * static_cast<unsigned long long>(embeddingDim) + static_cast<unsigned long "
              "long>(dimBase);\n";
        if (gradDType == DataType::FP32) {
            ss << "    const float4 v = thor_embedding_load_float4_masked(upstreamGradient, base_v, vectorValidLanes);\n";
        } else {
            ss << "    float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);\n";
            ss << "    if (vectorValidLanes > 0U) v.x = thor_embedding_grad_to_float(upstreamGradient[base_v]);\n";
            ss << "    if (vectorValidLanes > 1U) v.y = thor_embedding_grad_to_float(upstreamGradient[base_v + 1ULL]);\n";
            ss << "    if (vectorValidLanes > 2U) v.z = thor_embedding_grad_to_float(upstreamGradient[base_v + 2ULL]);\n";
            ss << "    if (vectorValidLanes > 3U) v.w = thor_embedding_grad_to_float(upstreamGradient[base_v + 3ULL]);\n";
        }
    } else if (gradDType == DataType::FP32) {
        ss << "    const float4* __restrict__ input_v = reinterpret_cast<const float4*>(upstreamGradient + token * embeddingDim);\n";
        ss << "    const float4 v = input_v[globalVectorIndex];\n";
    } else if (gradDType == DataType::FP16) {
        ss << "    const unsigned long long* __restrict__ input_v = reinterpret_cast<const unsigned long long*>(upstreamGradient + token * "
              "embeddingDim);\n";
        ss << "    const unsigned long long packed_v = input_v[globalVectorIndex];\n";
        ss << "    const float2 lo_v = thor_embedding_half2_bits_to_float2(static_cast<unsigned int>(packed_v));\n";
        ss << "    const float2 hi_v = thor_embedding_half2_bits_to_float2(static_cast<unsigned int>(packed_v >> 32U));\n";
        ss << "    const float4 v = make_float4(lo_v.x, lo_v.y, hi_v.x, hi_v.y);\n";
    } else if (gradDType == DataType::BF16) {
        ss << "    const unsigned long long* __restrict__ input_v = reinterpret_cast<const unsigned long long*>(upstreamGradient + token * "
              "embeddingDim);\n";
        ss << "    const unsigned long long packed_v = input_v[globalVectorIndex];\n";
        ss << "    const float2 lo_v = thor_embedding_bfloat162_bits_to_float2(static_cast<unsigned int>(packed_v));\n";
        ss << "    const float2 hi_v = thor_embedding_bfloat162_bits_to_float2(static_cast<unsigned int>(packed_v >> 32U));\n";
        ss << "    const float4 v = make_float4(lo_v.x, lo_v.y, hi_v.x, hi_v.y);\n";
    } else {
        ss << "    const unsigned long long base_v = token * static_cast<unsigned long long>(embeddingDim) + static_cast<unsigned long "
              "long>(globalVectorIndex) * dimsPerThread;\n";
        ss << "    const float4 v = make_float4(thor_embedding_grad_to_float(upstreamGradient[base_v + 0ULL]), "
              "thor_embedding_grad_to_float(upstreamGradient[base_v + 1ULL]), thor_embedding_grad_to_float(upstreamGradient[base_v + "
              "2ULL]), "
              "thor_embedding_grad_to_float(upstreamGradient[base_v + 3ULL]));\n";
    }
    ss << "    sum = thor_embedding_add_float4(sum, v);\n";
    ss << "  }\n";
    ss << "  }\n";
    ss << "  partialSums[threadIdx.x] = sum;\n";
    ss << "  __syncthreads();\n";
    ss << "  if (duplicateLane != 0U || !validVector) return;\n";
    ss << "  for (unsigned int lane = 1U; lane < duplicateLanes; ++lane) {\n";
    ss << "    sum = thor_embedding_add_float4(sum, partialSums[lane * vectorsPerTile + vectorInTile]);\n";
    ss << "  }\n";
    if (sparseRowUpdate != nullptr) {
        ss << "  const unsigned long long sru_logical_row = row;\n";
        ss << "  const unsigned long long sru_indexed_row = static_cast<unsigned long long>(outputRows[row]);\n";
        ss << "  const unsigned long long sru_vector_index = static_cast<unsigned long long>(globalVectorIndex);\n";
        ss << "  const unsigned int sru_vector_valid_lanes = vectorValidLanes;\n";
        ss << sparseRowUpdate->bodySource;
    } else {
        if (embeddingDim % 4ULL == 0ULL) {
            ss << "  float4* __restrict__ output = reinterpret_cast<float4*>(outputValues + row * embeddingDim);\n";
            ss << "  output[globalVectorIndex] = sum;\n";
        } else {
            ss << "  float* __restrict__ output = outputValues + row * static_cast<unsigned long long>(embeddingDim) + dimBase;\n";
            ss << "  if (vectorValidLanes > 0U) output[0] = sum.x;\n";
            ss << "  if (vectorValidLanes > 1U) output[1] = sum.y;\n";
            ss << "  if (vectorValidLanes > 2U) output[2] = sum.z;\n";
            ss << "  if (vectorValidLanes > 3U) output[3] = sum.w;\n";
        }
    }
    ss << "}\n";
    return ss.str();
}

std::string emitSparseGradientUltraHighPartialReduceKernelSource(
    DataType rowDType, DataType gradDType, uint64_t embeddingDim, const std::string& kernelName, uint32_t tokensPerPartial) {
    (void)rowDType;
    if (!hasUltraHighTwoStageEmbeddingSparseGradientReducer(embeddingDim)) {
        throw std::invalid_argument(
            "Embedding sparse-gradient ultra-high partial reducer currently requires a vectorized JIT reducer with embedding_dim <= " +
            std::to_string(MAX_VECTORIZED_JIT_EMBEDDING_SPARSE_GRADIENT_DIM) + ".");
    }

    const std::string gradType = sparseGradientScalarType(gradDType);
    const uint64_t threadsPerBlock = ultraHighReduceBlockThreadsForEmbeddingSparseGradient(embeddingDim);
    const uint64_t numVectors = (embeddingDim + 3ULL) / 4ULL;
    const uint64_t vectorsPerTile = fixedReducerVectorsPerTileForEmbeddingSparseGradient(embeddingDim);
    const uint64_t duplicateLanes = threadsPerBlock / vectorsPerTile;

    std::ostringstream ss;
    ss << "#include <cuda_fp16.h>\n";
    ss << "#include <cuda_bf16.h>\n";
    ss << "#include <math_functions.h>\n\n";
    ss << "__device__ __forceinline__ float thor_embedding_grad_to_float(float v) { return v; }\n";
    ss << "__device__ __forceinline__ float thor_embedding_grad_to_float(__half v) { return __half2float(v); }\n";
    ss << "__device__ __forceinline__ float thor_embedding_grad_to_float(__nv_bfloat16 v) { return __bfloat162float(v); }\n";
    ss << "__device__ __forceinline__ float2 thor_embedding_half2_bits_to_float2(unsigned int bits) {\n";
    ss << "  const __half2 packed = *reinterpret_cast<const __half2*>(&bits);\n";
    ss << "  return __half22float2(packed);\n";
    ss << "}\n";
    ss << "__device__ __forceinline__ float2 thor_embedding_bfloat162_bits_to_float2(unsigned int bits) {\n";
    ss << "  const __nv_bfloat162 packed = *reinterpret_cast<const __nv_bfloat162*>(&bits);\n";
    ss << "  return __bfloat1622float2(packed);\n";
    ss << "}\n\n";
    ss << "__device__ __forceinline__ float4 thor_embedding_add_float4(float4 a, float4 b) {\n";
    ss << "  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);\n";
    ss << "}\n\n";
    ss << emitEmbeddingSparseGradientFloat4LoadHelperSource();
    ss << "extern \"C\" __global__\n";
    ss << "void " << kernelName << "(const unsigned int* sortedTokenIds, const unsigned int* runOffsets, const unsigned int* runCounts, ";
    ss << "const unsigned int* numUltraHighPartials, const unsigned int* partialRunRows, ";
    ss << "const unsigned int* partialTokenOffsets, const " << gradType << "* upstreamGradient, float* partialSums) {\n";
    ss << "  constexpr unsigned int THREADS_PER_BLOCK_LOCAL = " << threadsPerBlock << "U;\n";
    ss << "  constexpr unsigned int embeddingDim = " << embeddingDim << "U;\n";
    ss << "  constexpr unsigned int dimsPerThread = 4U;\n";
    ss << "  constexpr unsigned int numVectors = " << numVectors << "U;\n";
    ss << "  constexpr unsigned int vectorsPerTile = " << vectorsPerTile << "U;\n";
    ss << "  constexpr unsigned int duplicateLanes = " << duplicateLanes << "U;\n";
    ss << "  constexpr unsigned int tokensPerPartial = " << tokensPerPartial << "U;\n";
    ss << "  __shared__ float4 partialSumsShared[THREADS_PER_BLOCK_LOCAL];\n";
    ss << "  const unsigned int vectorInTile = threadIdx.x % vectorsPerTile;\n";
    ss << "  const unsigned int duplicateLane = threadIdx.x / vectorsPerTile;\n";
    ss << "  const unsigned int globalVectorIndex = blockIdx.y * vectorsPerTile + vectorInTile;\n";
    ss << "  const unsigned long long partialIndex = static_cast<unsigned long long>(blockIdx.x);\n";
    ss << "  if (partialIndex >= static_cast<unsigned long long>(numUltraHighPartials[0])) return;\n";
    ss << "  const bool validVector = globalVectorIndex < numVectors;\n";
    ss << "  const unsigned int dimBase = globalVectorIndex * dimsPerThread;\n";
    if (embeddingDim % 4ULL == 0ULL) {
        ss << "  const unsigned int vectorValidLanes = validVector ? dimsPerThread : 0U;\n";
    } else {
        ss << "  const unsigned int vectorValidLanes = validVector ? ((dimBase + dimsPerThread <= embeddingDim) ? dimsPerThread : "
              "(embeddingDim - dimBase)) : 0U;\n";
    }
    ss << "  const unsigned long long row = static_cast<unsigned long long>(partialRunRows[partialIndex]);\n";
    ss << "  const unsigned long long tokenOffset = static_cast<unsigned long long>(partialTokenOffsets[partialIndex]);\n";
    ss << "  const unsigned long long begin = static_cast<unsigned long long>(runOffsets[row]) + tokenOffset;\n";
    ss << "  const unsigned long long totalCount = static_cast<unsigned long long>(runCounts[row]);\n";
    ss << "  const unsigned long long remaining = totalCount - tokenOffset;\n";
    ss << "  const unsigned long long count = remaining < static_cast<unsigned long long>(tokensPerPartial) ? remaining : "
          "static_cast<unsigned long long>(tokensPerPartial);\n";
    ss << "  float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);\n";
    ss << "  if (validVector) {\n";
    ss << "  for (unsigned long long i = static_cast<unsigned long long>(duplicateLane); i < count; i += duplicateLanes) {\n";
    ss << "    const unsigned long long token = static_cast<unsigned long long>(sortedTokenIds[begin + i]);\n";
    if (embeddingDim % 4ULL != 0ULL) {
        ss << "    const unsigned long long base_v = token * static_cast<unsigned long long>(embeddingDim) + static_cast<unsigned long "
              "long>(dimBase);\n";
        if (gradDType == DataType::FP32) {
            ss << "    const float4 v = thor_embedding_load_float4_masked(upstreamGradient, base_v, vectorValidLanes);\n";
        } else {
            ss << "    float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);\n";
            ss << "    if (vectorValidLanes > 0U) v.x = thor_embedding_grad_to_float(upstreamGradient[base_v]);\n";
            ss << "    if (vectorValidLanes > 1U) v.y = thor_embedding_grad_to_float(upstreamGradient[base_v + 1ULL]);\n";
            ss << "    if (vectorValidLanes > 2U) v.z = thor_embedding_grad_to_float(upstreamGradient[base_v + 2ULL]);\n";
            ss << "    if (vectorValidLanes > 3U) v.w = thor_embedding_grad_to_float(upstreamGradient[base_v + 3ULL]);\n";
        }
    } else if (gradDType == DataType::FP32) {
        ss << "    const float4* __restrict__ input_v = reinterpret_cast<const float4*>(upstreamGradient + token * embeddingDim);\n";
        ss << "    const float4 v = input_v[globalVectorIndex];\n";
    } else if (gradDType == DataType::FP16) {
        ss << "    const unsigned long long* __restrict__ input_v = reinterpret_cast<const unsigned long long*>(upstreamGradient + token * "
              "embeddingDim);\n";
        ss << "    const unsigned long long packed_v = input_v[globalVectorIndex];\n";
        ss << "    const float2 lo_v = thor_embedding_half2_bits_to_float2(static_cast<unsigned int>(packed_v));\n";
        ss << "    const float2 hi_v = thor_embedding_half2_bits_to_float2(static_cast<unsigned int>(packed_v >> 32U));\n";
        ss << "    const float4 v = make_float4(lo_v.x, lo_v.y, hi_v.x, hi_v.y);\n";
    } else if (gradDType == DataType::BF16) {
        ss << "    const unsigned long long* __restrict__ input_v = reinterpret_cast<const unsigned long long*>(upstreamGradient + token * "
              "embeddingDim);\n";
        ss << "    const unsigned long long packed_v = input_v[globalVectorIndex];\n";
        ss << "    const float2 lo_v = thor_embedding_bfloat162_bits_to_float2(static_cast<unsigned int>(packed_v));\n";
        ss << "    const float2 hi_v = thor_embedding_bfloat162_bits_to_float2(static_cast<unsigned int>(packed_v >> 32U));\n";
        ss << "    const float4 v = make_float4(lo_v.x, lo_v.y, hi_v.x, hi_v.y);\n";
    } else {
        ss << "    const unsigned long long base_v = token * static_cast<unsigned long long>(embeddingDim) + static_cast<unsigned long "
              "long>(globalVectorIndex) * dimsPerThread;\n";
        ss << "    const float4 v = make_float4(thor_embedding_grad_to_float(upstreamGradient[base_v + 0ULL]), "
              "thor_embedding_grad_to_float(upstreamGradient[base_v + 1ULL]), thor_embedding_grad_to_float(upstreamGradient[base_v + "
              "2ULL]), "
              "thor_embedding_grad_to_float(upstreamGradient[base_v + 3ULL]));\n";
    }
    ss << "    sum = thor_embedding_add_float4(sum, v);\n";
    ss << "  }\n";
    ss << "  }\n";
    ss << "  partialSumsShared[threadIdx.x] = sum;\n";
    ss << "  __syncthreads();\n";
    ss << "  if (duplicateLane != 0U || !validVector) return;\n";
    ss << "  for (unsigned int lane = 1U; lane < duplicateLanes; ++lane) {\n";
    ss << "    sum = thor_embedding_add_float4(sum, partialSumsShared[lane * vectorsPerTile + vectorInTile]);\n";
    ss << "  }\n";
    if (embeddingDim % 4ULL == 0ULL) {
        ss << "  float4* __restrict__ output = reinterpret_cast<float4*>(partialSums + partialIndex * static_cast<unsigned long "
              "long>(embeddingDim));\n";
        ss << "  output[globalVectorIndex] = sum;\n";
    } else {
        ss << "  float* __restrict__ output = partialSums + partialIndex * static_cast<unsigned long long>(embeddingDim) + dimBase;\n";
        ss << "  if (vectorValidLanes > 0U) output[0] = sum.x;\n";
        ss << "  if (vectorValidLanes > 1U) output[1] = sum.y;\n";
        ss << "  if (vectorValidLanes > 2U) output[2] = sum.z;\n";
        ss << "  if (vectorValidLanes > 3U) output[3] = sum.w;\n";
    }
    ss << "}\n";
    return ss.str();
}

std::string emitSparseGradientUltraHighFinalReduceKernelSource(DataType rowDType,
                                                               DataType gradDType,
                                                               uint64_t embeddingDim,
                                                               const std::string& kernelName,
                                                               const SparseRowUpdateFusionSource* sparseRowUpdate) {
    (void)gradDType;
    if (!hasUltraHighTwoStageEmbeddingSparseGradientReducer(embeddingDim)) {
        throw std::invalid_argument(
            "Embedding sparse-gradient ultra-high final reducer currently requires a vectorized JIT reducer with embedding_dim <= " +
            std::to_string(MAX_VECTORIZED_JIT_EMBEDDING_SPARSE_GRADIENT_DIM) + ".");
    }

    const std::string rowType = sparseGradientScalarType(rowDType);
    const uint64_t threadsPerBlock = ultraHighReduceBlockThreadsForEmbeddingSparseGradient(embeddingDim);
    const uint64_t numVectors = (embeddingDim + 3ULL) / 4ULL;
    const uint64_t vectorsPerTile = fixedReducerVectorsPerTileForEmbeddingSparseGradient(embeddingDim);
    const uint64_t duplicateLanes = threadsPerBlock / vectorsPerTile;

    std::ostringstream ss;
    ss << "#include <cuda_fp16.h>\n";
    ss << "#include <cuda_bf16.h>\n";
    ss << "#include <math_functions.h>\n\n";
    if (sparseRowUpdate != nullptr) {
        ss << sparseRowUpdate->helperSource;
    }
    ss << "__device__ __forceinline__ float4 thor_embedding_add_float4(float4 a, float4 b) {\n";
    ss << "  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);\n";
    ss << "}\n\n";
    ss << emitEmbeddingSparseGradientFloat4LoadHelperSource();
    ss << "extern \"C\" __global__\n";
    ss << "void " << kernelName
       << "(const unsigned int* numRunRows, const unsigned int* runRowIndices, const unsigned int* ultraHighRunPartialCounts, ";
    ss << "const unsigned int* ultraHighRunPartialOffsets, const float* partialSums";
    if (sparseRowUpdate != nullptr) {
        ss << ", const " << rowType << "* outputRows" << sparseRowUpdate->parameterSource;
    } else {
        ss << ", float* outputValues";
    }
    ss << ") {\n";
    ss << "  constexpr unsigned int THREADS_PER_BLOCK_LOCAL = " << threadsPerBlock << "U;\n";
    ss << "  constexpr unsigned int embeddingDim = " << embeddingDim << "U;\n";
    ss << "  constexpr unsigned int numVectors = " << numVectors << "U;\n";
    ss << "  constexpr unsigned int dimsPerThread = 4U;\n";
    ss << "  constexpr unsigned int vectorsPerTile = " << vectorsPerTile << "U;\n";
    ss << "  constexpr unsigned int duplicateLanes = " << duplicateLanes << "U;\n";
    ss << "  __shared__ float4 partialSumsShared[THREADS_PER_BLOCK_LOCAL];\n";
    ss << "  const unsigned int vectorInTile = threadIdx.x % vectorsPerTile;\n";
    ss << "  const unsigned int duplicateLane = threadIdx.x / vectorsPerTile;\n";
    ss << "  const unsigned int globalVectorIndex = blockIdx.y * vectorsPerTile + vectorInTile;\n";
    ss << "  const unsigned long long bucketRow = static_cast<unsigned long long>(blockIdx.x);\n";
    ss << "  if (bucketRow >= static_cast<unsigned long long>(numRunRows[0])) return;\n";
    ss << "  const bool validVector = globalVectorIndex < numVectors;\n";
    ss << "  const unsigned int dimBase = globalVectorIndex * dimsPerThread;\n";
    if (embeddingDim % 4ULL == 0ULL) {
        ss << "  const unsigned int vectorValidLanes = validVector ? dimsPerThread : 0U;\n";
    } else {
        ss << "  const unsigned int vectorValidLanes = validVector ? ((dimBase + dimsPerThread <= embeddingDim) ? dimsPerThread : "
              "(embeddingDim - dimBase)) : 0U;\n";
    }
    ss << "  const unsigned long long row = static_cast<unsigned long long>(runRowIndices[bucketRow]);\n";
    ss << "  const unsigned long long partialBegin = static_cast<unsigned long long>(ultraHighRunPartialOffsets[bucketRow]);\n";
    ss << "  const unsigned long long partialCount = static_cast<unsigned long long>(ultraHighRunPartialCounts[bucketRow]);\n";
    ss << "  float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);\n";
    ss << "  if (validVector) {\n";
    ss << "  for (unsigned long long p = static_cast<unsigned long long>(duplicateLane); p < partialCount; p += duplicateLanes) {\n";
    if (embeddingDim % 4ULL == 0ULL) {
        ss << "    const float4* __restrict__ input = reinterpret_cast<const float4*>(partialSums + (partialBegin + p) * "
              "static_cast<unsigned "
              "long long>(embeddingDim));\n";
        ss << "    const float4 v = input[globalVectorIndex];\n";
    } else {
        ss << "    const unsigned long long base_v = (partialBegin + p) * static_cast<unsigned long long>(embeddingDim) + "
              "static_cast<unsigned long long>(dimBase);\n";
        ss << "    const float4 v = thor_embedding_load_float4_masked(partialSums, base_v, vectorValidLanes);\n";
    }
    ss << "    sum = thor_embedding_add_float4(sum, v);\n";
    ss << "  }\n";
    ss << "  }\n";
    ss << "  partialSumsShared[threadIdx.x] = sum;\n";
    ss << "  __syncthreads();\n";
    ss << "  if (duplicateLane != 0U || !validVector) return;\n";
    ss << "  for (unsigned int lane = 1U; lane < duplicateLanes; ++lane) {\n";
    ss << "    sum = thor_embedding_add_float4(sum, partialSumsShared[lane * vectorsPerTile + vectorInTile]);\n";
    ss << "  }\n";
    if (sparseRowUpdate != nullptr) {
        ss << "  const unsigned long long sru_logical_row = row;\n";
        ss << "  const unsigned long long sru_indexed_row = static_cast<unsigned long long>(outputRows[row]);\n";
        ss << "  const unsigned long long sru_vector_index = static_cast<unsigned long long>(globalVectorIndex);\n";
        ss << "  const unsigned int sru_vector_valid_lanes = vectorValidLanes;\n";
        ss << sparseRowUpdate->bodySource;
    } else {
        if (embeddingDim % 4ULL == 0ULL) {
            ss << "  float4* __restrict__ output = reinterpret_cast<float4*>(outputValues + row * static_cast<unsigned long "
                  "long>(embeddingDim));\n";
            ss << "  output[globalVectorIndex] = sum;\n";
        } else {
            ss << "  float* __restrict__ output = outputValues + row * static_cast<unsigned long long>(embeddingDim) + dimBase;\n";
            ss << "  if (vectorValidLanes > 0U) output[0] = sum.x;\n";
            ss << "  if (vectorValidLanes > 1U) output[1] = sum.y;\n";
            ss << "  if (vectorValidLanes > 2U) output[2] = sum.z;\n";
            ss << "  if (vectorValidLanes > 3U) output[3] = sum.w;\n";
        }
    }
    ss << "}\n";
    return ss.str();
}

}  // namespace

struct PreparedEmbeddingSparseGradient {
    ~PreparedEmbeddingSparseGradient();

    uint64_t numTokens = 0;
    uint64_t vocabularySize = 0;
    uint64_t embeddingDim = 0;
    uint64_t paddingIndex = 0;
    bool hasPaddingIndex = false;
    int deviceNum = 0;
    DataType indexDataType = DataType::UINT32;
    DataType gradientDataType = DataType::FP32;
    DataType rowDataType = DataType::UINT64;
    EmbeddingSparseGradientRunBucketConfig runBucketConfig;

    Tensor rowKeys;
    Tensor tokenIds;
    Tensor sortedRowKeys;
    Tensor sortedTokenIds;
    Tensor runCounts;
    Tensor runOffsets;
    Tensor numRuns;
    Tensor lowRunRows;
    Tensor highRunRows;
    Tensor ultraHighRunRows;
    Tensor ultraHighRunPartialCounts;
    Tensor ultraHighRunPartialOffsets;
    Tensor ultraHighPartialRunRows;
    Tensor ultraHighPartialTokenOffsets;
    Tensor ultraHighPartialSums;
    Tensor numLowRunRows;
    Tensor numHighRunRows;
    Tensor numUltraHighRunRows;
    Tensor numUltraHighPartials;
    Tensor twoStageLowRunRowsScratch;
    Tensor twoStageHighRunRowsScratch;
    Tensor twoStageUltraHighRunRowsScratch;
    Tensor twoStageUltraHighRunPartialCountsScratch;
    Tensor twoStageUltraHighRunPartialOffsetsScratch;
    Tensor twoStageLowRunRowCounts;
    Tensor twoStageHighRunRowCounts;
    Tensor twoStageUltraHighRunRowCounts;
    Tensor twoStageUltraHighPartialCounts;
    Tensor sortTempStorage;
    Tensor rleTempStorage;
    Tensor scanTempStorage;
    uint64_t maxUltraHighPartials = 1;
    size_t sortTempBytes = 0;
    size_t rleTempBytes = 0;
    size_t scanTempBytes = 0;

    CUmodule reduceModule = nullptr;
    CUfunction reduceKernel = nullptr;
    CUmodule highRunReduceModule = nullptr;
    CUfunction highRunReduceKernel = nullptr;
    CUmodule ultraHighPartialReduceModule = nullptr;
    CUfunction ultraHighPartialReduceKernel = nullptr;
    CUmodule ultraHighFinalReduceModule = nullptr;
    CUfunction ultraHighFinalReduceKernel = nullptr;
    std::string reduceKernelName;
    std::string highRunReduceKernelName;
    std::string ultraHighPartialReduceKernelName;
    std::string ultraHighFinalReduceKernelName;
    std::optional<SparseRowUpdateFusionSource> sparseRowUpdate;
};

PreparedEmbeddingSparseGradient::~PreparedEmbeddingSparseGradient() {
    if (reduceModule != nullptr) {
        try {
            CU_CHECK(cuModuleUnload(reduceModule));
        } catch (...) {
        }
    }
    if (highRunReduceModule != nullptr) {
        try {
            CU_CHECK(cuModuleUnload(highRunReduceModule));
        } catch (...) {
        }
    }
    if (ultraHighPartialReduceModule != nullptr) {
        try {
            CU_CHECK(cuModuleUnload(ultraHighPartialReduceModule));
        } catch (...) {
        }
    }
    if (ultraHighFinalReduceModule != nullptr) {
        try {
            CU_CHECK(cuModuleUnload(ultraHighFinalReduceModule));
        } catch (...) {
        }
    }
}

__global__ void expandEmbeddingSparseGradientUltraHighPartialMetadataKernel(const uint32_t* __restrict__ numUltraHighRunRows,
                                                                            const uint32_t* __restrict__ ultraHighRunRows,
                                                                            const uint32_t* __restrict__ ultraHighRunPartialCounts,
                                                                            const uint32_t* __restrict__ ultraHighRunPartialOffsets,
                                                                            uint32_t* __restrict__ ultraHighPartialRunRows,
                                                                            uint32_t* __restrict__ ultraHighPartialTokenOffsets,
                                                                            uint32_t ultraHighTokensPerPartial) {
    const uint32_t rows = numUltraHighRunRows[0];

    for (uint32_t bucketRow = blockIdx.x * blockDim.x + threadIdx.x; bucketRow < rows; bucketRow += blockDim.x * gridDim.x) {
        const uint32_t runRow = ultraHighRunRows[bucketRow];
        const uint32_t partialCount = ultraHighRunPartialCounts[bucketRow];
        const uint32_t partialOffset = ultraHighRunPartialOffsets[bucketRow];
        for (uint32_t p = 0U; p < partialCount; ++p) {
            const uint32_t out = partialOffset + p;
            ultraHighPartialRunRows[out] = runRow;
            ultraHighPartialTokenOffsets[out] = p * ultraHighTokensPerPartial;
        }
    }
}

void populateRunCountProfileStats(EmbeddingSparseGradientProfileResult& result,
                                  const PreparedEmbeddingSparseGradient& prepared,
                                  const SparseRowGradient& outputGradient,
                                  Stream stream) {
    result.activeRows = downloadSparseRowCountScalar(outputGradient.numRows, outputGradient.rowDataType, stream);
    if (result.activeRows == 0) {
        result.singletonRows = 0;
        result.duplicateRows = 0;
        result.lowRunRows = downloadUint32Scalar(prepared.numLowRunRows, stream);
        result.highRunRows = downloadUint32Scalar(prepared.numHighRunRows, stream);
        result.ultraHighRunRows = downloadUint32Scalar(prepared.numUltraHighRunRows, stream);
        result.lowRunTokens = 0;
        result.highRunTokens = 0;
        result.ultraHighRunTokens = 0;
        result.maxRunCount = 0;
        return;
    }
    if (result.activeRows > outputGradient.capacity) {
        throw std::runtime_error("Embedding sparse-gradient profiler observed active row count larger than sparse-gradient capacity.");
    }

    std::vector<uint32_t> runCounts(static_cast<size_t>(result.activeRows));
    CUDA_CHECK(cudaMemcpyAsync(runCounts.data(),
                               prepared.runCounts.getMemPtr<uint32_t>(),
                               runCounts.size() * sizeof(uint32_t),
                               cudaMemcpyDeviceToHost,
                               stream.getStream()));
    stream.synchronize();

    uint64_t singletonRows = 0;
    uint64_t duplicateRows = 0;
    uint64_t lowRunTokens = 0;
    uint64_t highRunTokens = 0;
    uint64_t ultraHighRunTokens = 0;
    uint32_t maxRunCount = 0;
    for (uint32_t count : runCounts) {
        if (count == 1U) {
            ++singletonRows;
        } else if (count > 1U) {
            ++duplicateRows;
        }
        if (count <= prepared.runBucketConfig.lowRunMax) {
            lowRunTokens += count;
        } else if (count < prepared.runBucketConfig.ultraHighRunMin) {
            highRunTokens += count;
        } else {
            ultraHighRunTokens += count;
        }
        maxRunCount = std::max(maxRunCount, count);
    }
    result.singletonRows = singletonRows;
    result.duplicateRows = duplicateRows;
    result.lowRunRows = downloadUint32Scalar(prepared.numLowRunRows, stream);
    result.highRunRows = downloadUint32Scalar(prepared.numHighRunRows, stream);
    result.ultraHighRunRows = downloadUint32Scalar(prepared.numUltraHighRunRows, stream);
    result.lowRunTokens = lowRunTokens;
    result.highRunTokens = highRunTokens;
    result.ultraHighRunTokens = ultraHighRunTokens;
    result.maxRunCount = maxRunCount;
}

void compileGraphReduceKernel(PreparedEmbeddingSparseGradient& prepared) {
    if (prepared.reduceKernel != nullptr) {
        return;
    }

    prepared.reduceKernelName = sparseGradientReduceKernelName(
        prepared.rowDataType, prepared.gradientDataType, prepared.embeddingDim, prepared.sparseRowUpdate.has_value());
    const std::string src =
        emitSparseGradientReduceKernelSource(prepared.rowDataType,
                                             prepared.gradientDataType,
                                             prepared.embeddingDim,
                                             prepared.reduceKernelName,
                                             prepared.sparseRowUpdate.has_value() ? &prepared.sparseRowUpdate.value() : nullptr);
    if constexpr (PRINT_GENERATED_EMBEDDING_SPARSE_GRADIENT_KERNELS) {
        std::fprintf(stderr,
                     "\n===== Generated Embedding sparse-gradient reducer CUDA source begin: %s =====\n%s\n===== Generated Embedding "
                     "sparse-gradient reducer CUDA source end: %s =====\n",
                     prepared.reduceKernelName.c_str(),
                     src.c_str(),
                     prepared.reduceKernelName.c_str());
        std::fflush(stderr);
    }
    CUDA_CHECK(cudaFree(nullptr));
    CompiledEmbeddingSparseGradientCudaKernel compiled =
        compileEmbeddingSparseGradientCudaKernel(src, prepared.reduceKernelName, prepared.deviceNum);
    prepared.reduceModule = compiled.module;
    prepared.reduceKernel = compiled.function;

    if (hasHighRunCtaEmbeddingSparseGradientReducer(prepared.embeddingDim)) {
        prepared.highRunReduceKernelName = sparseGradientHighRunReduceKernelName(
            prepared.rowDataType, prepared.gradientDataType, prepared.embeddingDim, prepared.sparseRowUpdate.has_value());
        const std::string highSrc =
            emitSparseGradientHighRunReduceKernelSource(prepared.rowDataType,
                                                        prepared.gradientDataType,
                                                        prepared.embeddingDim,
                                                        prepared.highRunReduceKernelName,
                                                        prepared.sparseRowUpdate.has_value() ? &prepared.sparseRowUpdate.value() : nullptr);
        if constexpr (PRINT_GENERATED_EMBEDDING_SPARSE_GRADIENT_KERNELS) {
            std::fprintf(stderr,
                         "\n===== Generated Embedding sparse-gradient high-run reducer CUDA source begin: %s =====\n%s\n===== "
                         "Generated Embedding sparse-gradient high-run reducer CUDA source end: %s =====\n",
                         prepared.highRunReduceKernelName.c_str(),
                         highSrc.c_str(),
                         prepared.highRunReduceKernelName.c_str());
            std::fflush(stderr);
        }
        CompiledEmbeddingSparseGradientCudaKernel compiledHigh =
            compileEmbeddingSparseGradientCudaKernel(highSrc, prepared.highRunReduceKernelName, prepared.deviceNum);
        prepared.highRunReduceModule = compiledHigh.module;
        prepared.highRunReduceKernel = compiledHigh.function;
    }

    if (hasUltraHighTwoStageEmbeddingSparseGradientReducer(prepared.embeddingDim)) {
        prepared.ultraHighPartialReduceKernelName =
            sparseGradientUltraHighPartialReduceKernelName(prepared.rowDataType, prepared.gradientDataType, prepared.embeddingDim);
        const std::string ultraPartialSrc =
            emitSparseGradientUltraHighPartialReduceKernelSource(prepared.rowDataType,
                                                                 prepared.gradientDataType,
                                                                 prepared.embeddingDim,
                                                                 prepared.ultraHighPartialReduceKernelName,
                                                                 prepared.runBucketConfig.ultraHighTokensPerPartial);
        if constexpr (PRINT_GENERATED_EMBEDDING_SPARSE_GRADIENT_KERNELS) {
            std::fprintf(stderr,
                         "\n===== Generated Embedding sparse-gradient ultra-high partial reducer CUDA source begin: %s =====\n%s\n===== "
                         "Generated Embedding sparse-gradient ultra-high partial reducer CUDA source end: %s =====\n",
                         prepared.ultraHighPartialReduceKernelName.c_str(),
                         ultraPartialSrc.c_str(),
                         prepared.ultraHighPartialReduceKernelName.c_str());
            std::fflush(stderr);
        }
        CompiledEmbeddingSparseGradientCudaKernel compiledUltraPartial =
            compileEmbeddingSparseGradientCudaKernel(ultraPartialSrc, prepared.ultraHighPartialReduceKernelName, prepared.deviceNum);
        prepared.ultraHighPartialReduceModule = compiledUltraPartial.module;
        prepared.ultraHighPartialReduceKernel = compiledUltraPartial.function;

        prepared.ultraHighFinalReduceKernelName = sparseGradientUltraHighFinalReduceKernelName(
            prepared.rowDataType, prepared.gradientDataType, prepared.embeddingDim, prepared.sparseRowUpdate.has_value());
        const std::string ultraFinalSrc = emitSparseGradientUltraHighFinalReduceKernelSource(
            prepared.rowDataType,
            prepared.gradientDataType,
            prepared.embeddingDim,
            prepared.ultraHighFinalReduceKernelName,
            prepared.sparseRowUpdate.has_value() ? &prepared.sparseRowUpdate.value() : nullptr);
        if constexpr (PRINT_GENERATED_EMBEDDING_SPARSE_GRADIENT_KERNELS) {
            std::fprintf(stderr,
                         "\n===== Generated Embedding sparse-gradient ultra-high final reducer CUDA source begin: %s =====\n%s\n===== "
                         "Generated Embedding sparse-gradient ultra-high final reducer CUDA source end: %s =====\n",
                         prepared.ultraHighFinalReduceKernelName.c_str(),
                         ultraFinalSrc.c_str(),
                         prepared.ultraHighFinalReduceKernelName.c_str());
            std::fflush(stderr);
        }
        CompiledEmbeddingSparseGradientCudaKernel compiledUltraFinal =
            compileEmbeddingSparseGradientCudaKernel(ultraFinalSrc, prepared.ultraHighFinalReduceKernelName, prepared.deviceNum);
        prepared.ultraHighFinalReduceModule = compiledUltraFinal.module;
        prepared.ultraHighFinalReduceKernel = compiledUltraFinal.function;
    }
}
template <typename RowT>
void allocateTypedPreparedBuffers(PreparedEmbeddingSparseGradient& prepared,
                                  const TensorPlacement& placement,
                                  SparseRowGradient& outputGradient,
                                  uint64_t numTokens,
                                  int cubItems) {
    if (!rowTypeCanRepresentVocabularySentinel<RowT>(prepared.vocabularySize)) {
        throw std::invalid_argument("Embedding sparse-gradient row dtype " + dtypeName(rowDTypeForCppType<RowT>()) +
                                    " cannot represent vocabulary_size as the invalid-row sentinel.");
    }

    prepared.rowKeys = Tensor(placement, TensorDescriptor(rowDTypeForCppType<RowT>(), {numTokens}));
    prepared.tokenIds = Tensor(placement, TensorDescriptor(DataType::UINT32, {numTokens}));
    prepared.sortedRowKeys = Tensor(placement, TensorDescriptor(rowDTypeForCppType<RowT>(), {numTokens}));
    prepared.sortedTokenIds = Tensor(placement, TensorDescriptor(DataType::UINT32, {numTokens}));
    prepared.runCounts = Tensor(placement, TensorDescriptor(DataType::UINT32, {numTokens}));
    prepared.runOffsets = Tensor(placement, TensorDescriptor(DataType::UINT32, {numTokens}));
    prepared.numRuns = Tensor(placement, TensorDescriptor(DataType::UINT32, {1}));
    prepared.lowRunRows = Tensor(placement, TensorDescriptor(DataType::UINT32, {numTokens}));
    prepared.highRunRows = Tensor(placement, TensorDescriptor(DataType::UINT32, {numTokens}));
    prepared.ultraHighRunRows = Tensor(placement, TensorDescriptor(DataType::UINT32, {numTokens}));
    prepared.ultraHighRunPartialCounts = Tensor(placement, TensorDescriptor(DataType::UINT32, {numTokens}));
    prepared.ultraHighRunPartialOffsets = Tensor(placement, TensorDescriptor(DataType::UINT32, {numTokens}));
    prepared.maxUltraHighPartials = maxUltraHighPartialsForEmbeddingSparseGradient(numTokens, prepared.runBucketConfig);
    prepared.ultraHighPartialRunRows = Tensor(placement, TensorDescriptor(DataType::UINT32, {prepared.maxUltraHighPartials}));
    prepared.ultraHighPartialTokenOffsets = Tensor(placement, TensorDescriptor(DataType::UINT32, {prepared.maxUltraHighPartials}));
    const uint64_t ultraHighScratchDim =
        hasUltraHighTwoStageEmbeddingSparseGradientReducer(prepared.embeddingDim) ? prepared.embeddingDim : 1ULL;
    prepared.ultraHighPartialSums =
        Tensor(placement, TensorDescriptor(DataType::FP32, {prepared.maxUltraHighPartials, ultraHighScratchDim}));
    prepared.numLowRunRows = Tensor(placement, TensorDescriptor(DataType::UINT32, {1}));
    prepared.numHighRunRows = Tensor(placement, TensorDescriptor(DataType::UINT32, {1}));
    prepared.numUltraHighRunRows = Tensor(placement, TensorDescriptor(DataType::UINT32, {1}));
    prepared.numUltraHighPartials = Tensor(placement, TensorDescriptor(DataType::UINT32, {1}));

    if (useTwoStageEmbeddingSparseGradientFinalize(numTokens)) {
        const uint64_t stageBlocks = twoStageEmbeddingSparseGradientFinalizeBlockCount(numTokens);
        prepared.twoStageLowRunRowsScratch = Tensor(placement, TensorDescriptor(DataType::UINT32, {numTokens}));
        prepared.twoStageHighRunRowsScratch = Tensor(placement, TensorDescriptor(DataType::UINT32, {numTokens}));
        prepared.twoStageUltraHighRunRowsScratch = Tensor(placement, TensorDescriptor(DataType::UINT32, {numTokens}));
        prepared.twoStageUltraHighRunPartialCountsScratch = Tensor(placement, TensorDescriptor(DataType::UINT32, {numTokens}));
        prepared.twoStageUltraHighRunPartialOffsetsScratch = Tensor(placement, TensorDescriptor(DataType::UINT32, {numTokens}));
        prepared.twoStageLowRunRowCounts = Tensor(placement, TensorDescriptor(DataType::UINT32, {stageBlocks}));
        prepared.twoStageHighRunRowCounts = Tensor(placement, TensorDescriptor(DataType::UINT32, {stageBlocks}));
        prepared.twoStageUltraHighRunRowCounts = Tensor(placement, TensorDescriptor(DataType::UINT32, {stageBlocks}));
        prepared.twoStageUltraHighPartialCounts = Tensor(placement, TensorDescriptor(DataType::UINT32, {stageBlocks}));
    }

    prepared.sortTempBytes = querySortTempBytes(prepared.rowKeys.getMemPtr<RowT>(),
                                                prepared.sortedRowKeys.getMemPtr<RowT>(),
                                                prepared.tokenIds.getMemPtr<uint32_t>(),
                                                prepared.sortedTokenIds.getMemPtr<uint32_t>(),
                                                cubItems);
    prepared.rleTempBytes = queryRleTempBytes(prepared.sortedRowKeys.getMemPtr<RowT>(),
                                              outputGradient.rows.getMemPtr<RowT>(),
                                              prepared.runCounts.getMemPtr<uint32_t>(),
                                              prepared.numRuns.getMemPtr<uint32_t>(),
                                              cubItems);
    prepared.scanTempBytes =
        queryScanTempBytes(prepared.runCounts.getMemPtr<uint32_t>(), prepared.runOffsets.getMemPtr<uint32_t>(), cubItems);
    prepared.sortTempStorage = allocateByteScratch(placement, prepared.sortTempBytes);
    prepared.rleTempStorage = allocateByteScratch(placement, prepared.rleTempBytes);
    prepared.scanTempStorage = allocateByteScratch(placement, prepared.scanTempBytes);
}

template <typename IndexT, typename RowT>
void launchMaterializeSortPairsTyped(const Tensor& indices, PreparedEmbeddingSparseGradient& prepared, Stream stream) {
    const uint32_t block = THREADS_PER_BLOCK;
    const uint32_t grid = static_cast<uint32_t>((prepared.numTokens + block - 1) / block);
    materializeEmbeddingSparseGradientSortPairsKernel<IndexT, RowT>
        <<<grid, block, 0, stream.getStream()>>>(indices.getMemPtr<IndexT>(),
                                                 prepared.rowKeys.getMemPtr<RowT>(),
                                                 prepared.tokenIds.getMemPtr<uint32_t>(),
                                                 prepared.numTokens,
                                                 prepared.vocabularySize,
                                                 prepared.paddingIndex,
                                                 prepared.hasPaddingIndex);
    CUDA_CHECK(cudaPeekAtLastError());
}

template <typename RowT>
void launchMaterializeSortPairsForRowType(const Tensor& indices, PreparedEmbeddingSparseGradient& prepared, Stream stream) {
    switch (prepared.indexDataType) {
        case DataType::UINT32:
            launchMaterializeSortPairsTyped<uint32_t, RowT>(indices, prepared, stream);
            break;
        case DataType::UINT64:
            launchMaterializeSortPairsTyped<uint64_t, RowT>(indices, prepared, stream);
            break;
        default:
            throw std::runtime_error("Prepared Embedding sparse-gradient producer has unsupported index dtype.");
    }
}

void launchMaterializeSortPairs(const Tensor& indices, PreparedEmbeddingSparseGradient& prepared, Stream stream) {
    switch (prepared.rowDataType) {
        case DataType::UINT16:
            launchMaterializeSortPairsForRowType<uint16_t>(indices, prepared, stream);
            break;
        case DataType::UINT32:
            launchMaterializeSortPairsForRowType<uint32_t>(indices, prepared, stream);
            break;
        case DataType::UINT64:
            launchMaterializeSortPairsForRowType<uint64_t>(indices, prepared, stream);
            break;
        default:
            throw std::runtime_error("Prepared Embedding sparse-gradient producer has unsupported row dtype.");
    }
}

template <typename RowT, typename GradT, uint32_t EmbeddingDim>
void launchReduceValuesFixedDim(PreparedEmbeddingSparseGradient& prepared,
                                const Tensor& upstreamGradient,
                                SparseRowGradient& outputGradient,
                                const Tensor& runRows,
                                const Tensor& numRunRows,
                                Stream stream) {
    using Geometry = FixedEmbeddingReducerGeometry<EmbeddingDim>;
    constexpr uint32_t BLOCK_THREADS = Geometry::BLOCK_THREADS;
    const dim3 block(BLOCK_THREADS);
    const dim3 grid(static_cast<uint32_t>((outputGradient.capacity + Geometry::ROWS_PER_LOW_BLOCK - 1ULL) / Geometry::ROWS_PER_LOW_BLOCK),
                    static_cast<uint32_t>((Geometry::NUM_VECTORS + Geometry::VECTORS_PER_TILE - 1ULL) / Geometry::VECTORS_PER_TILE));
    reduceEmbeddingSparseGradientValuesFixedDimKernel<RowT, GradT, EmbeddingDim>
        <<<grid, block, 0, stream.getStream()>>>(prepared.sortedTokenIds.getMemPtr<uint32_t>(),
                                                 prepared.runOffsets.getMemPtr<uint32_t>(),
                                                 prepared.runCounts.getMemPtr<uint32_t>(),
                                                 numRunRows.getMemPtr<uint32_t>(),
                                                 runRows.getMemPtr<uint32_t>(),
                                                 upstreamGradient.getMemPtr<GradT>(),
                                                 outputGradient.values.getMemPtr<float>());
}

template <typename RowT, typename GradT>
void launchReduceValuesTyped(PreparedEmbeddingSparseGradient& prepared,
                             const Tensor& upstreamGradient,
                             SparseRowGradient& outputGradient,
                             const Tensor& runRows,
                             const Tensor& numRunRows,
                             Stream stream);

template <typename RowT, typename GradT, uint32_t EmbeddingDim>
void launchHighRunReduceValuesFixedDim(PreparedEmbeddingSparseGradient& prepared,
                                       const Tensor& upstreamGradient,
                                       SparseRowGradient& outputGradient,
                                       const Tensor& runRows,
                                       const Tensor& numRunRows,
                                       Stream stream) {
    using Geometry = FixedEmbeddingReducerGeometry<EmbeddingDim>;
    constexpr uint32_t BLOCK_THREADS = Geometry::BLOCK_THREADS;
    const dim3 block(BLOCK_THREADS);
    const dim3 grid(static_cast<uint32_t>(outputGradient.capacity),
                    static_cast<uint32_t>((Geometry::NUM_VECTORS + Geometry::VECTORS_PER_TILE - 1ULL) / Geometry::VECTORS_PER_TILE));
    reduceEmbeddingSparseGradientValuesHighRunFixedDimKernel<RowT, GradT, EmbeddingDim>
        <<<grid, block, 0, stream.getStream()>>>(prepared.sortedTokenIds.getMemPtr<uint32_t>(),
                                                 prepared.runOffsets.getMemPtr<uint32_t>(),
                                                 prepared.runCounts.getMemPtr<uint32_t>(),
                                                 numRunRows.getMemPtr<uint32_t>(),
                                                 runRows.getMemPtr<uint32_t>(),
                                                 upstreamGradient.getMemPtr<GradT>(),
                                                 outputGradient.values.getMemPtr<float>());
}

template <typename RowT, typename GradT>
void launchHighRunReduceValuesTyped(PreparedEmbeddingSparseGradient& prepared,
                                    const Tensor& upstreamGradient,
                                    SparseRowGradient& outputGradient,
                                    const Tensor& runRows,
                                    const Tensor& numRunRows,
                                    Stream stream) {
    const uint64_t maxRows = outputGradient.capacity;
    if (maxRows == 0) {
        return;
    }

    switch (prepared.embeddingDim) {
        case 4:
            launchHighRunReduceValuesFixedDim<RowT, GradT, 4U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 8:
            launchHighRunReduceValuesFixedDim<RowT, GradT, 8U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 16:
            launchHighRunReduceValuesFixedDim<RowT, GradT, 16U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 32:
            launchHighRunReduceValuesFixedDim<RowT, GradT, 32U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 64:
            launchHighRunReduceValuesFixedDim<RowT, GradT, 64U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 128:
            launchHighRunReduceValuesFixedDim<RowT, GradT, 128U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 256:
            launchHighRunReduceValuesFixedDim<RowT, GradT, 256U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 512:
            launchHighRunReduceValuesFixedDim<RowT, GradT, 512U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 768:
            launchHighRunReduceValuesFixedDim<RowT, GradT, 768U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 1024:
            launchHighRunReduceValuesFixedDim<RowT, GradT, 1024U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 1280:
            launchHighRunReduceValuesFixedDim<RowT, GradT, 1280U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 1536:
            launchHighRunReduceValuesFixedDim<RowT, GradT, 1536U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 1792:
            launchHighRunReduceValuesFixedDim<RowT, GradT, 1792U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 2048:
            launchHighRunReduceValuesFixedDim<RowT, GradT, 2048U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 2304:
            launchHighRunReduceValuesFixedDim<RowT, GradT, 2304U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 2560:
            launchHighRunReduceValuesFixedDim<RowT, GradT, 2560U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 2816:
            launchHighRunReduceValuesFixedDim<RowT, GradT, 2816U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 3072:
            launchHighRunReduceValuesFixedDim<RowT, GradT, 3072U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 3328:
            launchHighRunReduceValuesFixedDim<RowT, GradT, 3328U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 3584:
            launchHighRunReduceValuesFixedDim<RowT, GradT, 3584U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 3840:
            launchHighRunReduceValuesFixedDim<RowT, GradT, 3840U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 4096:
            launchHighRunReduceValuesFixedDim<RowT, GradT, 4096U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        default:
            launchReduceValuesTyped<RowT, GradT>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            return;
    }
    CUDA_CHECK(cudaPeekAtLastError());
}

template <typename RowT, typename GradT>
void launchReduceValuesTyped(PreparedEmbeddingSparseGradient& prepared,
                             const Tensor& upstreamGradient,
                             SparseRowGradient& outputGradient,
                             const Tensor& runRows,
                             const Tensor& numRunRows,
                             Stream stream) {
    const uint64_t maxRows = outputGradient.capacity;
    if (maxRows == 0) {
        return;
    }

    switch (prepared.embeddingDim) {
        case 4:
            launchReduceValuesFixedDim<RowT, GradT, 4U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 8:
            launchReduceValuesFixedDim<RowT, GradT, 8U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 16:
            launchReduceValuesFixedDim<RowT, GradT, 16U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 32:
            launchReduceValuesFixedDim<RowT, GradT, 32U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 64:
            launchReduceValuesFixedDim<RowT, GradT, 64U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 128:
            launchReduceValuesFixedDim<RowT, GradT, 128U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 256:
            launchReduceValuesFixedDim<RowT, GradT, 256U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 512:
            launchReduceValuesFixedDim<RowT, GradT, 512U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 768:
            launchReduceValuesFixedDim<RowT, GradT, 768U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 1024:
            launchReduceValuesFixedDim<RowT, GradT, 1024U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 1280:
            launchReduceValuesFixedDim<RowT, GradT, 1280U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 1536:
            launchReduceValuesFixedDim<RowT, GradT, 1536U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 1792:
            launchReduceValuesFixedDim<RowT, GradT, 1792U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 2048:
            launchReduceValuesFixedDim<RowT, GradT, 2048U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 2304:
            launchReduceValuesFixedDim<RowT, GradT, 2304U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 2560:
            launchReduceValuesFixedDim<RowT, GradT, 2560U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 2816:
            launchReduceValuesFixedDim<RowT, GradT, 2816U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 3072:
            launchReduceValuesFixedDim<RowT, GradT, 3072U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 3328:
            launchReduceValuesFixedDim<RowT, GradT, 3328U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 3584:
            launchReduceValuesFixedDim<RowT, GradT, 3584U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 3840:
            launchReduceValuesFixedDim<RowT, GradT, 3840U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case 4096:
            launchReduceValuesFixedDim<RowT, GradT, 4096U>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        default: {
            const dim3 block(THREADS_PER_BLOCK);
            const dim3 grid(static_cast<uint32_t>(maxRows),
                            static_cast<uint32_t>((prepared.embeddingDim + THREADS_PER_BLOCK - 1ULL) / THREADS_PER_BLOCK));
            reduceEmbeddingSparseGradientValuesTiledKernel<RowT, GradT>
                <<<grid, block, 0, stream.getStream()>>>(prepared.sortedTokenIds.getMemPtr<uint32_t>(),
                                                         prepared.runOffsets.getMemPtr<uint32_t>(),
                                                         prepared.runCounts.getMemPtr<uint32_t>(),
                                                         numRunRows.getMemPtr<uint32_t>(),
                                                         runRows.getMemPtr<uint32_t>(),
                                                         upstreamGradient.getMemPtr<GradT>(),
                                                         outputGradient.values.getMemPtr<float>(),
                                                         prepared.embeddingDim);
            break;
        }
    }
    CUDA_CHECK(cudaPeekAtLastError());
}

template <typename RowT>
void launchReduceValuesForRowType(PreparedEmbeddingSparseGradient& prepared,
                                  const Tensor& upstreamGradient,
                                  SparseRowGradient& outputGradient,
                                  const Tensor& runRows,
                                  const Tensor& numRunRows,
                                  Stream stream) {
    switch (prepared.gradientDataType) {
        case DataType::FP16:
            launchReduceValuesTyped<RowT, __half>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case DataType::BF16:
            launchReduceValuesTyped<RowT, __nv_bfloat16>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case DataType::FP32:
            launchReduceValuesTyped<RowT, float>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        default:
            throw std::runtime_error("Prepared Embedding sparse-gradient producer has unsupported upstream gradient dtype.");
    }
}

template <typename RowT>
void launchHighRunReduceValuesForRowType(PreparedEmbeddingSparseGradient& prepared,
                                         const Tensor& upstreamGradient,
                                         SparseRowGradient& outputGradient,
                                         const Tensor& runRows,
                                         const Tensor& numRunRows,
                                         Stream stream) {
    switch (prepared.gradientDataType) {
        case DataType::FP16:
            launchHighRunReduceValuesTyped<RowT, __half>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case DataType::BF16:
            launchHighRunReduceValuesTyped<RowT, __nv_bfloat16>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case DataType::FP32:
            launchHighRunReduceValuesTyped<RowT, float>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        default:
            throw std::runtime_error("Prepared Embedding sparse-gradient producer has unsupported upstream gradient dtype.");
    }
}

void launchReduceValuesForRunRows(PreparedEmbeddingSparseGradient& prepared,
                                  const Tensor& upstreamGradient,
                                  SparseRowGradient& outputGradient,
                                  const Tensor& runRows,
                                  const Tensor& numRunRows,
                                  Stream stream) {
    if (prepared.sparseRowUpdate.has_value()) {
        throw std::invalid_argument(
            "Prepared Embedding sparse-gradient producer with fused sparse-row update must use the fused update launcher.");
    }
    switch (prepared.rowDataType) {
        case DataType::UINT16:
            launchReduceValuesForRowType<uint16_t>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case DataType::UINT32:
            launchReduceValuesForRowType<uint32_t>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case DataType::UINT64:
            launchReduceValuesForRowType<uint64_t>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        default:
            throw std::runtime_error("Prepared Embedding sparse-gradient producer has unsupported row dtype.");
    }
}

void launchHighRunReduceValuesForRunRows(PreparedEmbeddingSparseGradient& prepared,
                                         const Tensor& upstreamGradient,
                                         SparseRowGradient& outputGradient,
                                         const Tensor& runRows,
                                         const Tensor& numRunRows,
                                         Stream stream) {
    if (prepared.sparseRowUpdate.has_value()) {
        throw std::invalid_argument(
            "Prepared Embedding sparse-gradient producer with fused sparse-row update must use the fused update launcher.");
    }
    switch (prepared.rowDataType) {
        case DataType::UINT16:
            launchHighRunReduceValuesForRowType<uint16_t>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case DataType::UINT32:
            launchHighRunReduceValuesForRowType<uint32_t>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        case DataType::UINT64:
            launchHighRunReduceValuesForRowType<uint64_t>(prepared, upstreamGradient, outputGradient, runRows, numRunRows, stream);
            break;
        default:
            throw std::runtime_error("Prepared Embedding sparse-gradient producer has unsupported row dtype.");
    }
}

struct SparseRowUpdateFusionKernelArgs {
    std::vector<const void*> tensorInputPtrs;
    std::vector<void*> tensorOutputPtrs;
    std::vector<float> scalarValues;
    std::vector<void*> args;
};

SparseRowUpdateFusionKernelArgs buildSparseRowUpdateFusionKernelArgs(const SparseRowUpdateFusionSource& source,
                                                                     const std::unordered_map<std::string, float>& runtimeScalars) {
    SparseRowUpdateFusionKernelArgs out;
    out.tensorInputPtrs.reserve(source.kernelInputSlots.size());
    out.tensorOutputPtrs.reserve(source.outputSlots.size());
    out.scalarValues.reserve(source.kernelInputSlots.size());

    for (const SparseRowUpdatePlan::RuntimeInputSlot& slot : source.kernelInputSlots) {
        if (slot.inputKind == NamedInput::Kind::Tensor) {
            out.tensorInputPtrs.push_back(slot.tensor.getMemPtr());
        } else if (slot.inputKind == NamedInput::Kind::RuntimeScalarFp32) {
            auto it = runtimeScalars.find(slot.name);
            if (it == runtimeScalars.end()) {
                throw std::invalid_argument("Fused embedding sparse-row update missing runtime scalar '" + slot.name + "'.");
            }
            out.scalarValues.push_back(it->second);
        } else {
            throw std::runtime_error("Fused embedding sparse-row update encountered unsupported runtime input kind.");
        }
    }
    for (const SparseRowUpdatePlan::RuntimeOutputSlot& slot : source.outputSlots) {
        out.tensorOutputPtrs.push_back(const_cast<void*>(static_cast<const void*>(slot.tensor.getMemPtr())));
    }

    out.args.reserve(source.kernelInputSlots.size() + source.outputSlots.size());
    size_t tensorInputIndex = 0;
    size_t scalarIndex = 0;
    for (const SparseRowUpdatePlan::RuntimeInputSlot& slot : source.kernelInputSlots) {
        if (slot.inputKind == NamedInput::Kind::Tensor) {
            out.args.push_back((void*)&out.tensorInputPtrs[tensorInputIndex++]);
        } else {
            out.args.push_back((void*)&out.scalarValues[scalarIndex++]);
        }
    }
    for (void*& ptr : out.tensorOutputPtrs) {
        out.args.push_back((void*)&ptr);
    }
    return out;
}

void launchReduceValuesWithSparseRowUpdateForRunRows(PreparedEmbeddingSparseGradient& prepared,
                                                     const Tensor& upstreamGradient,
                                                     SparseRowGradient& outputGradient,
                                                     const std::unordered_map<std::string, float>& runtimeScalars,
                                                     const Tensor& runRows,
                                                     const Tensor& numRunRows,
                                                     Stream stream,
                                                     bool highRunBucket);

void prepareUltraHighPartialMetadata(PreparedEmbeddingSparseGradient& prepared, Stream stream) {
    const uint32_t block = THREADS_PER_BLOCK;
    const uint64_t maxUltraRows =
        maxUltraHighRunRowsForEmbeddingSparseGradient(prepared.numTokens, prepared.numTokens, prepared.runBucketConfig);
    const uint32_t grid =
        static_cast<uint32_t>(std::max<uint64_t>(1ULL, std::min<uint64_t>((maxUltraRows + block - 1ULL) / block, 4096ULL)));
    expandEmbeddingSparseGradientUltraHighPartialMetadataKernel<<<grid, block, 0, stream.getStream()>>>(
        prepared.numUltraHighRunRows.getMemPtr<uint32_t>(),
        prepared.ultraHighRunRows.getMemPtr<uint32_t>(),
        prepared.ultraHighRunPartialCounts.getMemPtr<uint32_t>(),
        prepared.ultraHighRunPartialOffsets.getMemPtr<uint32_t>(),
        prepared.ultraHighPartialRunRows.getMemPtr<uint32_t>(),
        prepared.ultraHighPartialTokenOffsets.getMemPtr<uint32_t>(),
        prepared.runBucketConfig.ultraHighTokensPerPartial);
    CUDA_CHECK(cudaPeekAtLastError());
}

void launchUltraHighPartialReduceKernel(PreparedEmbeddingSparseGradient& prepared, const Tensor& upstreamGradient, Stream stream) {
    if (prepared.ultraHighPartialReduceKernel == nullptr) {
        throw std::runtime_error("Prepared Embedding sparse-gradient ultra-high partial reducer kernel was not compiled.");
    }

    const void* sortedTokenIdsPtr = prepared.sortedTokenIds.getMemPtr();
    const void* runOffsetsPtr = prepared.runOffsets.getMemPtr();
    const void* runCountsPtr = prepared.runCounts.getMemPtr();
    const void* numUltraHighPartialsPtr = prepared.numUltraHighPartials.getMemPtr();
    const void* partialRunRowsPtr = prepared.ultraHighPartialRunRows.getMemPtr();
    const void* partialTokenOffsetsPtr = prepared.ultraHighPartialTokenOffsets.getMemPtr();
    const void* upstreamPtr = upstreamGradient.getMemPtr();
    void* partialSumsPtr = prepared.ultraHighPartialSums.getMemPtr();

    std::vector<void*> args;
    args.reserve(8);
    args.push_back((void*)&sortedTokenIdsPtr);
    args.push_back((void*)&runOffsetsPtr);
    args.push_back((void*)&runCountsPtr);
    args.push_back((void*)&numUltraHighPartialsPtr);
    args.push_back((void*)&partialRunRowsPtr);
    args.push_back((void*)&partialTokenOffsetsPtr);
    args.push_back((void*)&upstreamPtr);
    args.push_back((void*)&partialSumsPtr);

    const uint32_t blockThreads = ultraHighReduceBlockThreadsForEmbeddingSparseGradient(prepared.embeddingDim);
    const uint32_t gridX = ultraHighPartialReduceGridDimXForEmbeddingSparseGradient(prepared.maxUltraHighPartials);
    const uint32_t gridY = ultraHighPartialReduceGridDimYForEmbeddingSparseGradient(prepared.embeddingDim);
    CU_CHECK(cuLaunchKernel(prepared.ultraHighPartialReduceKernel, gridX, gridY, 1, blockThreads, 1, 1, 0, stream, args.data(), nullptr));
}

void launchUltraHighFinalReduceKernel(PreparedEmbeddingSparseGradient& prepared,
                                      SparseRowGradient& outputGradient,
                                      const std::unordered_map<std::string, float>* runtimeScalars,
                                      Stream stream) {
    if (prepared.ultraHighFinalReduceKernel == nullptr) {
        throw std::runtime_error("Prepared Embedding sparse-gradient ultra-high final reducer kernel was not compiled.");
    }
    const bool fusedSparseRowUpdate = prepared.sparseRowUpdate.has_value();
    if (fusedSparseRowUpdate && runtimeScalars == nullptr) {
        throw std::invalid_argument("Prepared Embedding sparse-gradient ultra-high final fused reducer missing runtime scalars.");
    }

    const void* numRunRowsPtr = prepared.numUltraHighRunRows.getMemPtr();
    const void* runRowsPtr = prepared.ultraHighRunRows.getMemPtr();
    const void* partialCountsPtr = prepared.ultraHighRunPartialCounts.getMemPtr();
    const void* partialOffsetsPtr = prepared.ultraHighRunPartialOffsets.getMemPtr();
    const void* partialSumsPtr = prepared.ultraHighPartialSums.getMemPtr();

    std::optional<SparseRowUpdateFusionKernelArgs> updateArgs;
    std::vector<void*> args;
    args.reserve(fusedSparseRowUpdate ? 6 + prepared.sparseRowUpdate->kernelInputSlots.size() + prepared.sparseRowUpdate->outputSlots.size()
                                      : 6);
    args.push_back((void*)&numRunRowsPtr);
    args.push_back((void*)&runRowsPtr);
    args.push_back((void*)&partialCountsPtr);
    args.push_back((void*)&partialOffsetsPtr);
    args.push_back((void*)&partialSumsPtr);

    const void* outputRowsPtr = nullptr;
    void* outputValuesPtr = nullptr;
    if (fusedSparseRowUpdate) {
        outputRowsPtr = outputGradient.rows.getMemPtr();
        args.push_back((void*)&outputRowsPtr);
        updateArgs.emplace(buildSparseRowUpdateFusionKernelArgs(prepared.sparseRowUpdate.value(), *runtimeScalars));
        args.insert(args.end(), updateArgs->args.begin(), updateArgs->args.end());
    } else {
        outputValuesPtr = outputGradient.values.getMemPtr();
        args.push_back((void*)&outputValuesPtr);
    }

    const uint32_t blockThreads = ultraHighReduceBlockThreadsForEmbeddingSparseGradient(prepared.embeddingDim);
    const uint64_t maxUltraRows =
        maxUltraHighRunRowsForEmbeddingSparseGradient(outputGradient.capacity, prepared.numTokens, prepared.runBucketConfig);
    const uint32_t gridX = ultraHighFinalReduceGridDimXForEmbeddingSparseGradient(maxUltraRows, prepared.embeddingDim);
    const uint32_t gridY = ultraHighFinalReduceGridDimYForEmbeddingSparseGradient(prepared.embeddingDim);
    CU_CHECK(cuLaunchKernel(prepared.ultraHighFinalReduceKernel, gridX, gridY, 1, blockThreads, 1, 1, 0, stream, args.data(), nullptr));
}

void launchUltraHighRunReduceValues(PreparedEmbeddingSparseGradient& prepared,
                                    const Tensor& upstreamGradient,
                                    SparseRowGradient& outputGradient,
                                    const std::unordered_map<std::string, float>* runtimeScalars,
                                    Stream stream) {
    if (!hasUltraHighTwoStageEmbeddingSparseGradientReducer(prepared.embeddingDim)) {
        if (runtimeScalars != nullptr) {
            launchReduceValuesWithSparseRowUpdateForRunRows(prepared,
                                                            upstreamGradient,
                                                            outputGradient,
                                                            *runtimeScalars,
                                                            prepared.ultraHighRunRows,
                                                            prepared.numUltraHighRunRows,
                                                            stream,
                                                            /*highRunBucket=*/false);
        } else {
            launchReduceValuesForRunRows(
                prepared, upstreamGradient, outputGradient, prepared.ultraHighRunRows, prepared.numUltraHighRunRows, stream);
        }
        return;
    }

    prepareUltraHighPartialMetadata(prepared, stream);
    launchUltraHighPartialReduceKernel(prepared, upstreamGradient, stream);
    launchUltraHighFinalReduceKernel(prepared, outputGradient, runtimeScalars, stream);
}

void launchReduceValues(PreparedEmbeddingSparseGradient& prepared,
                        const Tensor& upstreamGradient,
                        SparseRowGradient& outputGradient,
                        Stream stream) {
    Stream highRunStream = Stream::getNextGradientUpdateStream(prepared.deviceNum);
    Stream ultraHighRunStream = Stream::getNextGradientUpdateStream(prepared.deviceNum);
    Event metadataReady = stream.putEvent(/*enableTiming=*/false);
    highRunStream.waitEvent(metadataReady);
    ultraHighRunStream.waitEvent(metadataReady);

    launchReduceValuesForRunRows(prepared, upstreamGradient, outputGradient, prepared.lowRunRows, prepared.numLowRunRows, stream);
    launchHighRunReduceValuesForRunRows(
        prepared, upstreamGradient, outputGradient, prepared.highRunRows, prepared.numHighRunRows, highRunStream);
    launchUltraHighRunReduceValues(prepared, upstreamGradient, outputGradient, /*runtimeScalars=*/nullptr, ultraHighRunStream);

    Event highRunDone = highRunStream.putEvent(/*enableTiming=*/false);
    Event ultraHighRunDone = ultraHighRunStream.putEvent(/*enableTiming=*/false);
    stream.waitEvent(highRunDone);
    stream.waitEvent(ultraHighRunDone);
}

void launchReduceValuesWithSparseRowUpdateForRunRows(PreparedEmbeddingSparseGradient& prepared,
                                                     const Tensor& upstreamGradient,
                                                     SparseRowGradient& outputGradient,
                                                     const std::unordered_map<std::string, float>& runtimeScalars,
                                                     const Tensor& runRows,
                                                     const Tensor& numRunRows,
                                                     Stream stream,
                                                     bool highRunBucket = false) {
    if (!prepared.sparseRowUpdate.has_value()) {
        throw std::invalid_argument("Prepared Embedding sparse-gradient producer does not have a fused sparse-row update.");
    }
    CUfunction reduceKernel =
        highRunBucket && prepared.highRunReduceKernel != nullptr ? prepared.highRunReduceKernel : prepared.reduceKernel;
    if (reduceKernel == nullptr) {
        throw std::runtime_error("Prepared Embedding sparse-gradient fused reducer kernel was not compiled.");
    }

    const uint32_t reduceGridDimY = highRunBucket ? highRunReduceGridDimYForEmbeddingSparseGradient(prepared.embeddingDim)
                                                  : reduceGridDimYForEmbeddingSparseGradient(prepared.embeddingDim);
    const uint32_t reduceBlockThreads = highRunBucket ? highRunReduceBlockThreadsForEmbeddingSparseGradient(prepared.embeddingDim)
                                                      : reduceBlockThreadsForEmbeddingSparseGradient(prepared.embeddingDim);
    const uint64_t maxBucketRows =
        highRunBucket ? maxHighRunRowsForEmbeddingSparseGradient(outputGradient.capacity, prepared.numTokens, prepared.runBucketConfig)
                      : outputGradient.capacity;
    const uint32_t gridX = highRunBucket ? highRunReduceGridDimXForEmbeddingSparseGradient(maxBucketRows, prepared.embeddingDim)
                                         : reduceGridDimXForEmbeddingSparseGradient(maxBucketRows, prepared.embeddingDim);
    const dim3 grid(gridX, reduceGridDimY, 1U);
    const dim3 block(reduceBlockThreads, 1U, 1U);

    const void* sortedTokenIdsPtr = prepared.sortedTokenIds.getMemPtr();
    const void* runOffsetsPtr = prepared.runOffsets.getMemPtr();
    const void* runCountsPtr = prepared.runCounts.getMemPtr();
    const void* numRunRowsPtr = numRunRows.getMemPtr();
    const void* runRowsPtr = runRows.getMemPtr();
    const void* upstreamPtr = upstreamGradient.getMemPtr();
    const void* outputRowsPtr = outputGradient.rows.getMemPtr();

    SparseRowUpdateFusionKernelArgs updateArgs = buildSparseRowUpdateFusionKernelArgs(prepared.sparseRowUpdate.value(), runtimeScalars);
    std::vector<void*> args;
    args.reserve(7 + updateArgs.args.size());
    args.push_back((void*)&sortedTokenIdsPtr);
    args.push_back((void*)&runOffsetsPtr);
    args.push_back((void*)&runCountsPtr);
    args.push_back((void*)&numRunRowsPtr);
    args.push_back((void*)&runRowsPtr);
    args.push_back((void*)&upstreamPtr);
    args.push_back((void*)&outputRowsPtr);
    args.insert(args.end(), updateArgs.args.begin(), updateArgs.args.end());

    CU_CHECK(cuLaunchKernel(reduceKernel, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, stream, args.data(), nullptr));
}

void launchReduceValuesWithSparseRowUpdate(PreparedEmbeddingSparseGradient& prepared,
                                           const Tensor& upstreamGradient,
                                           SparseRowGradient& outputGradient,
                                           const std::unordered_map<std::string, float>& runtimeScalars,
                                           Stream stream) {
    Stream highRunStream = Stream::getNextGradientUpdateStream(prepared.deviceNum);
    Stream ultraHighRunStream = Stream::getNextGradientUpdateStream(prepared.deviceNum);
    Event metadataReady = stream.putEvent(/*enableTiming=*/false);
    highRunStream.waitEvent(metadataReady);
    ultraHighRunStream.waitEvent(metadataReady);

    launchReduceValuesWithSparseRowUpdateForRunRows(
        prepared, upstreamGradient, outputGradient, runtimeScalars, prepared.lowRunRows, prepared.numLowRunRows, stream);
    launchReduceValuesWithSparseRowUpdateForRunRows(prepared,
                                                    upstreamGradient,
                                                    outputGradient,
                                                    runtimeScalars,
                                                    prepared.highRunRows,
                                                    prepared.numHighRunRows,
                                                    highRunStream,
                                                    /*highRunBucket=*/true);
    launchUltraHighRunReduceValues(prepared, upstreamGradient, outputGradient, &runtimeScalars, ultraHighRunStream);

    Event highRunDone = highRunStream.putEvent(/*enableTiming=*/false);
    Event ultraHighRunDone = ultraHighRunStream.putEvent(/*enableTiming=*/false);
    stream.waitEvent(highRunDone);
    stream.waitEvent(ultraHighRunDone);
}

struct OptionalReduceGridUpdate {
    const DeviceUpdatableKernelNodeDeviceHandle* lowReduceNodeHandle = nullptr;
    const DeviceUpdatableKernelNodeDeviceHandle* highReduceNodeHandle = nullptr;
    const DeviceUpdatableKernelNodeDeviceHandle* ultraHighPartialReduceNodeHandle = nullptr;
    const DeviceUpdatableKernelNodeDeviceHandle* ultraHighReduceNodeHandle = nullptr;
    const DeviceUpdatableKernelNodeDeviceHandle* twoStageClassifyNodeHandle = nullptr;
    const DeviceUpdatableKernelNodeDeviceHandle* twoStageAccumulateNodeHandle = nullptr;
    EmbeddingSparseGradientReduceGridUpdateConfig lowReduceGridConfig;
    EmbeddingSparseGradientReduceGridUpdateConfig highReduceGridConfig;
    EmbeddingSparseGradientReduceGridUpdateConfig ultraHighPartialReduceGridConfig;
    EmbeddingSparseGradientReduceGridUpdateConfig ultraHighReduceGridConfig;
    bool runtimeTwoStageFinalizeDelegate = false;
};

void launchFinalizeRows(PreparedEmbeddingSparseGradient& prepared,
                        SparseRowGradient& outputGradient,
                        Stream stream,
                        OptionalReduceGridUpdate reduceGridUpdate = {}) {
    const bool runtimeTwoStageFinalizeDelegate =
        reduceGridUpdate.runtimeTwoStageFinalizeDelegate && useTwoStageEmbeddingSparseGradientFinalize(prepared.numTokens);

    launchFinalizeAndBucketizeEmbeddingSparseGradientRows(outputGradient.rows.getMemPtr<void>(),
                                                          prepared.numRuns.getMemPtr<uint32_t>(),
                                                          outputGradient.numRows.getMemPtr<void>(),
                                                          prepared.runCounts.getMemPtr<uint32_t>(),
                                                          prepared.lowRunRows.getMemPtr<uint32_t>(),
                                                          prepared.highRunRows.getMemPtr<uint32_t>(),
                                                          prepared.ultraHighRunRows.getMemPtr<uint32_t>(),
                                                          prepared.ultraHighRunPartialCounts.getMemPtr<uint32_t>(),
                                                          prepared.ultraHighRunPartialOffsets.getMemPtr<uint32_t>(),
                                                          prepared.numUltraHighPartials.getMemPtr<uint32_t>(),
                                                          prepared.numLowRunRows.getMemPtr<uint32_t>(),
                                                          prepared.numHighRunRows.getMemPtr<uint32_t>(),
                                                          prepared.numUltraHighRunRows.getMemPtr<uint32_t>(),
                                                          prepared.vocabularySize,
                                                          static_cast<uint32_t>(prepared.numTokens),
                                                          prepared.rowDataType,
                                                          prepared.runBucketConfig.lowRunMax,
                                                          prepared.runBucketConfig.ultraHighRunMin,
                                                          prepared.runBucketConfig.ultraHighTokensPerPartial,
                                                          reduceGridUpdate.lowReduceNodeHandle,
                                                          reduceGridUpdate.highReduceNodeHandle,
                                                          reduceGridUpdate.ultraHighPartialReduceNodeHandle,
                                                          reduceGridUpdate.ultraHighReduceNodeHandle,
                                                          reduceGridUpdate.twoStageClassifyNodeHandle,
                                                          reduceGridUpdate.twoStageAccumulateNodeHandle,
                                                          reduceGridUpdate.lowReduceGridConfig,
                                                          reduceGridUpdate.highReduceGridConfig,
                                                          reduceGridUpdate.ultraHighPartialReduceGridConfig,
                                                          reduceGridUpdate.ultraHighReduceGridConfig,
                                                          runtimeTwoStageFinalizeDelegate,
                                                          runtimeTwoStageEmbeddingSparseGradientFinalizeRunThreshold(),
                                                          stream);
}

template <typename RowT>
void sortPairsTyped(PreparedEmbeddingSparseGradient& prepared, int cubItems, Stream stream) {
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(prepared.sortTempStorage.getMemPtr<void>(),
                                               prepared.sortTempBytes,
                                               prepared.rowKeys.getMemPtr<RowT>(),
                                               prepared.sortedRowKeys.getMemPtr<RowT>(),
                                               prepared.tokenIds.getMemPtr<uint32_t>(),
                                               prepared.sortedTokenIds.getMemPtr<uint32_t>(),
                                               cubItems,
                                               0,
                                               8 * sizeof(RowT),
                                               stream.getStream()));
}

void sortPairs(PreparedEmbeddingSparseGradient& prepared, int cubItems, Stream stream) {
    switch (prepared.rowDataType) {
        case DataType::UINT16:
            sortPairsTyped<uint16_t>(prepared, cubItems, stream);
            break;
        case DataType::UINT32:
            sortPairsTyped<uint32_t>(prepared, cubItems, stream);
            break;
        case DataType::UINT64:
            sortPairsTyped<uint64_t>(prepared, cubItems, stream);
            break;
        default:
            throw std::runtime_error("Prepared Embedding sparse-gradient producer has unsupported row dtype.");
    }
}

template <typename RowT>
void rleRowsTyped(PreparedEmbeddingSparseGradient& prepared, SparseRowGradient& outputGradient, int cubItems, Stream stream) {
    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(prepared.rleTempStorage.getMemPtr<void>(),
                                                  prepared.rleTempBytes,
                                                  prepared.sortedRowKeys.getMemPtr<RowT>(),
                                                  outputGradient.rows.getMemPtr<RowT>(),
                                                  prepared.runCounts.getMemPtr<uint32_t>(),
                                                  prepared.numRuns.getMemPtr<uint32_t>(),
                                                  cubItems,
                                                  stream.getStream()));
}

void rleRows(PreparedEmbeddingSparseGradient& prepared, SparseRowGradient& outputGradient, int cubItems, Stream stream) {
    switch (prepared.rowDataType) {
        case DataType::UINT16:
            rleRowsTyped<uint16_t>(prepared, outputGradient, cubItems, stream);
            break;
        case DataType::UINT32:
            rleRowsTyped<uint32_t>(prepared, outputGradient, cubItems, stream);
            break;
        case DataType::UINT64:
            rleRowsTyped<uint64_t>(prepared, outputGradient, cubItems, stream);
            break;
        default:
            throw std::runtime_error("Prepared Embedding sparse-gradient producer has unsupported row dtype.");
    }
}

std::shared_ptr<PreparedEmbeddingSparseGradient> prepareEmbeddingSparseGradientImpl(
    const Tensor& indices,
    const Tensor& upstreamGradient,
    SparseRowGradient& outputGradient,
    std::optional<uint64_t> paddingIndex,
    std::optional<SparseRowUpdateFusionSource> sparseRowUpdate) {
    if (!indices.isInitialized() || !upstreamGradient.isInitialized()) {
        throw std::invalid_argument("Embedding sparse-gradient indices and upstream gradient tensors must be initialized.");
    }
    outputGradient.validate();
    validateVectorizedJitEmbeddingSparseGradientDim(outputGradient.embeddingDim);
    if (indices.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::invalid_argument("Embedding sparse-gradient indices tensor must live on GPU.");
    }
    const TensorPlacement placement = indices.getPlacement();
    validateSamePlacement(upstreamGradient, placement, "upstream gradient");
    validateSamePlacement(outputGradient.rows, placement, "output rows");
    validateSamePlacement(outputGradient.values, placement, "output values");
    validateSamePlacement(outputGradient.numRows, placement, "output numRows");
    validateDenseContiguous(indices, "indices");
    validateDenseContiguous(upstreamGradient, "upstream gradient");

    if (!isSupportedIndexType(indices.getDataType())) {
        throw std::invalid_argument("Embedding sparse-gradient indices dtype must be uint32 or uint64. Got " +
                                    dtypeName(indices.getDataType()) + ".");
    }
    if (!isSupportedGradientType(upstreamGradient.getDataType())) {
        throw std::invalid_argument("Embedding sparse-gradient upstream dtype must be fp16, bf16, or fp32. Got " +
                                    dtypeName(upstreamGradient.getDataType()) + ".");
    }
    if (!isSupportedRowType(outputGradient.rowDataType) || outputGradient.rows.getDataType() != outputGradient.rowDataType ||
        outputGradient.numRows.getDataType() != outputGradient.rowDataType) {
        throw std::invalid_argument("Embedding sparse-gradient producer requires uint16, uint32, or uint64 SparseRowGradient row storage.");
    }
    if (outputGradient.accumulationDataType != DataType::FP32 || outputGradient.values.getDataType() != DataType::FP32) {
        throw std::invalid_argument("Embedding sparse-gradient producer currently requires fp32 SparseRowGradient value storage.");
    }

    const uint64_t numTokens = indices.getTotalNumElements();
    if (numTokens == 0) {
        throw std::invalid_argument("Embedding sparse-gradient indices tensor must contain at least one token.");
    }
    const int cubItems = checkedCubItems(numTokens, "token count");

    const std::vector<uint64_t> upstreamDims = upstreamGradient.getDimensions();
    if (upstreamDims.empty() || upstreamDims.back() != outputGradient.embeddingDim) {
        throw std::invalid_argument("Embedding sparse-gradient upstream gradient must have trailing dimension embedding_dim.");
    }
    uint64_t upstreamTokens = 1;
    for (size_t i = 0; i + 1 < upstreamDims.size(); ++i) {
        upstreamTokens = checkedProduct(upstreamTokens, upstreamDims[i], "upstream token count");
    }
    if (upstreamTokens != numTokens) {
        throw std::invalid_argument("Embedding sparse-gradient upstream gradient leading dimensions must match the indices tensor shape.");
    }
    if (outputGradient.capacity != std::min<uint64_t>(numTokens, outputGradient.vocabularySize)) {
        throw std::invalid_argument("Embedding sparse-gradient output capacity must equal min(num_tokens, vocabulary_size).");
    }
    if (paddingIndex.has_value() && paddingIndex.value() >= outputGradient.vocabularySize) {
        throw std::invalid_argument("Embedding sparse-gradient padding_index must be less than vocabulary_size.");
    }

    auto prepared = std::make_shared<PreparedEmbeddingSparseGradient>();
    prepared->numTokens = numTokens;
    prepared->vocabularySize = outputGradient.vocabularySize;
    prepared->embeddingDim = outputGradient.embeddingDim;
    prepared->paddingIndex = paddingIndex.value_or(0);
    prepared->hasPaddingIndex = paddingIndex.has_value();
    prepared->deviceNum = placement.getDeviceNum();
    prepared->indexDataType = indices.getDataType();
    prepared->gradientDataType = upstreamGradient.getDataType();
    prepared->rowDataType = outputGradient.rowDataType;
    prepared->runBucketConfig = currentEmbeddingSparseGradientRunBucketConfig();
    validateEmbeddingSparseGradientRunBucketConfig(prepared->runBucketConfig);
    prepared->sparseRowUpdate = std::move(sparseRowUpdate);

    ScopedGpu scopedGpu(placement.getDeviceNum());
    switch (prepared->rowDataType) {
        case DataType::UINT16:
            allocateTypedPreparedBuffers<uint16_t>(*prepared, placement, outputGradient, numTokens, cubItems);
            break;
        case DataType::UINT32:
            allocateTypedPreparedBuffers<uint32_t>(*prepared, placement, outputGradient, numTokens, cubItems);
            break;
        case DataType::UINT64:
            allocateTypedPreparedBuffers<uint64_t>(*prepared, placement, outputGradient, numTokens, cubItems);
            break;
        default:
            throw std::runtime_error("Prepared Embedding sparse-gradient producer has unsupported row dtype.");
    }
    compileGraphReduceKernel(*prepared);

    return prepared;
}

std::shared_ptr<PreparedEmbeddingSparseGradient> prepareEmbeddingSparseGradient(const Tensor& indices,
                                                                                const Tensor& upstreamGradient,
                                                                                SparseRowGradient& outputGradient,
                                                                                std::optional<uint64_t> paddingIndex) {
    return prepareEmbeddingSparseGradientImpl(indices, upstreamGradient, outputGradient, paddingIndex, std::nullopt);
}

std::shared_ptr<PreparedEmbeddingSparseGradient> prepareEmbeddingSparseGradientWithSparseRowUpdate(
    const Tensor& indices,
    const Tensor& upstreamGradient,
    SparseRowGradient& outputGradient,
    PhysicalOutputs updateOutputs,
    const std::unordered_map<std::string, SparseRowUpdateTensorBinding>& updateInputs,
    const std::unordered_map<std::string, Tensor>& indexedUpdateOutputs,
    std::optional<uint64_t> paddingIndex) {
    SparseRowUpdateFusionSource source = SparseRowUpdatePlan::emitFusionSource(
        std::move(updateOutputs), outputGradient.rows, outputGradient.numRows, updateInputs, indexedUpdateOutputs, {{"gradient", "sum"}});
    return prepareEmbeddingSparseGradientImpl(indices, upstreamGradient, outputGradient, paddingIndex, std::move(source));
}

bool preparedEmbeddingSparseGradientHasSparseRowUpdate(const PreparedEmbeddingSparseGradient& prepared) {
    return prepared.sparseRowUpdate.has_value();
}

bool preparedEmbeddingSparseGradientUsesTwoStageFinalize(const PreparedEmbeddingSparseGradient& prepared) {
    return useTwoStageEmbeddingSparseGradientFinalize(prepared.numTokens);
}

void validatePreparedEmbeddingSparseGradientInvocation(PreparedEmbeddingSparseGradient& prepared,
                                                       const Tensor& indices,
                                                       const Tensor& upstreamGradient,
                                                       SparseRowGradient& outputGradient) {
    outputGradient.validate();
    if (indices.getDataType() != prepared.indexDataType || upstreamGradient.getDataType() != prepared.gradientDataType) {
        throw std::invalid_argument(
            "Prepared Embedding sparse-gradient producer received tensors with dtypes different from the prepared plan.");
    }
    if (indices.getTotalNumElements() != prepared.numTokens) {
        throw std::invalid_argument("Prepared Embedding sparse-gradient producer received an indices tensor with a different token count.");
    }
    if (upstreamGradient.getTotalNumElements() != checkedProduct(prepared.numTokens, prepared.embeddingDim, "upstream element count")) {
        throw std::invalid_argument("Prepared Embedding sparse-gradient producer received an upstream gradient with a different shape.");
    }
    if (outputGradient.capacity != std::min<uint64_t>(prepared.numTokens, prepared.vocabularySize) ||
        outputGradient.vocabularySize != prepared.vocabularySize || outputGradient.embeddingDim != prepared.embeddingDim ||
        outputGradient.rowDataType != prepared.rowDataType) {
        throw std::invalid_argument("Prepared Embedding sparse-gradient producer received an output gradient with incompatible metadata.");
    }
}

void launchPreparedEmbeddingSparseGradient(PreparedEmbeddingSparseGradient& prepared,
                                           const Tensor& indices,
                                           const Tensor& upstreamGradient,
                                           SparseRowGradient& outputGradient,
                                           Stream stream) {
    validatePreparedEmbeddingSparseGradientInvocation(prepared, indices, upstreamGradient, outputGradient);
    if (prepared.sparseRowUpdate.has_value()) {
        throw std::invalid_argument(
            "Prepared Embedding sparse-gradient producer has a fused sparse-row update; use "
            "launchPreparedEmbeddingSparseGradientWithSparseRowUpdate.");
    }

    ScopedGpu scopedGpu(prepared.deviceNum);

    launchMaterializeSortPairs(indices, prepared, stream);

    const int cubItems = checkedCubItems(prepared.numTokens, "token count");
    sortPairs(prepared, cubItems, stream);

    // RLE writes only the first numRuns counts.
    rleRows(prepared, outputGradient, cubItems, stream);

    launchFinalizeRows(prepared, outputGradient, stream);

    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(prepared.scanTempStorage.getMemPtr<void>(),
                                             prepared.scanTempBytes,
                                             prepared.runCounts.getMemPtr<uint32_t>(),
                                             prepared.runOffsets.getMemPtr<uint32_t>(),
                                             cubItems,
                                             stream.getStream()));

    launchReduceValues(prepared, upstreamGradient, outputGradient, stream);
}

void launchPreparedEmbeddingSparseGradientWithSparseRowUpdate(PreparedEmbeddingSparseGradient& prepared,
                                                              const Tensor& indices,
                                                              const Tensor& upstreamGradient,
                                                              SparseRowGradient& outputGradient,
                                                              const std::unordered_map<std::string, float>& runtimeScalars,
                                                              Stream stream) {
    validatePreparedEmbeddingSparseGradientInvocation(prepared, indices, upstreamGradient, outputGradient);
    if (!prepared.sparseRowUpdate.has_value()) {
        throw std::invalid_argument(
            "Prepared Embedding sparse-gradient producer does not have a fused sparse-row update; use "
            "launchPreparedEmbeddingSparseGradient.");
    }

    ScopedGpu scopedGpu(prepared.deviceNum);

    launchMaterializeSortPairs(indices, prepared, stream);

    const int cubItems = checkedCubItems(prepared.numTokens, "token count");
    sortPairs(prepared, cubItems, stream);

    // RLE writes only the first numRuns counts.
    rleRows(prepared, outputGradient, cubItems, stream);

    launchFinalizeRows(prepared, outputGradient, stream);

    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(prepared.scanTempStorage.getMemPtr<void>(),
                                             prepared.scanTempBytes,
                                             prepared.runCounts.getMemPtr<uint32_t>(),
                                             prepared.runOffsets.getMemPtr<uint32_t>(),
                                             cubItems,
                                             stream.getStream()));

    launchReduceValuesWithSparseRowUpdate(prepared, upstreamGradient, outputGradient, runtimeScalars, stream);
}

EmbeddingSparseGradientProfileResult profilePreparedEmbeddingSparseGradient(PreparedEmbeddingSparseGradient& prepared,
                                                                            const Tensor& indices,
                                                                            const Tensor& upstreamGradient,
                                                                            SparseRowGradient& outputGradient,
                                                                            Stream stream) {
    validatePreparedEmbeddingSparseGradientInvocation(prepared, indices, upstreamGradient, outputGradient);
    if (prepared.sparseRowUpdate.has_value()) {
        throw std::invalid_argument("Embedding sparse-gradient materialization profiler does not support fused sparse-row updates yet.");
    }

    ScopedGpu scopedGpu(prepared.deviceNum);

    EmbeddingSparseGradientProfileResult result;
    result.numTokens = prepared.numTokens;
    result.vocabularySize = prepared.vocabularySize;
    result.embeddingDim = prepared.embeddingDim;
    result.capacity = outputGradient.capacity;
    result.indexDataType = prepared.indexDataType;
    result.gradientDataType = prepared.gradientDataType;
    result.rowDataType = prepared.rowDataType;
    result.fusedSparseRowUpdate = false;
    result.sortTempBytes = prepared.sortTempBytes;
    result.rleTempBytes = prepared.rleTempBytes;
    result.scanTempBytes = prepared.scanTempBytes;

    Event totalStart(prepared.deviceNum, /*enableTiming=*/true);
    Event materializeEnd(prepared.deviceNum, /*enableTiming=*/true);
    Event sortEnd(prepared.deviceNum, /*enableTiming=*/true);
    Event rleEnd(prepared.deviceNum, /*enableTiming=*/true);
    Event finalizeEnd(prepared.deviceNum, /*enableTiming=*/true);
    Event scanEnd(prepared.deviceNum, /*enableTiming=*/true);
    Event reduceEnd(prepared.deviceNum, /*enableTiming=*/true);

    totalStart.record(stream);

    launchMaterializeSortPairs(indices, prepared, stream);
    materializeEnd.record(stream);

    const int cubItems = checkedCubItems(prepared.numTokens, "token count");
    sortPairs(prepared, cubItems, stream);
    sortEnd.record(stream);

    rleRows(prepared, outputGradient, cubItems, stream);
    rleEnd.record(stream);

    launchFinalizeRows(prepared, outputGradient, stream);
    finalizeEnd.record(stream);

    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(prepared.scanTempStorage.getMemPtr<void>(),
                                             prepared.scanTempBytes,
                                             prepared.runCounts.getMemPtr<uint32_t>(),
                                             prepared.runOffsets.getMemPtr<uint32_t>(),
                                             cubItems,
                                             stream.getStream()));
    scanEnd.record(stream);

    launchReduceValues(prepared, upstreamGradient, outputGradient, stream);
    reduceEnd.record(stream);

    populateRunCountProfileStats(result, prepared, outputGradient, stream);

    result.materializeSortPairsMs = materializeEnd.synchronizeAndReportElapsedTimeInMilliseconds(totalStart);
    result.cubSortMs = sortEnd.synchronizeAndReportElapsedTimeInMilliseconds(materializeEnd);
    result.clearRunCountsMs = 0.0f;
    result.cubRleMs = rleEnd.synchronizeAndReportElapsedTimeInMilliseconds(sortEnd);
    result.finalizeRowsMs = finalizeEnd.synchronizeAndReportElapsedTimeInMilliseconds(rleEnd);
    result.cubScanOffsetsMs = scanEnd.synchronizeAndReportElapsedTimeInMilliseconds(finalizeEnd);
    result.reduceValuesMs = reduceEnd.synchronizeAndReportElapsedTimeInMilliseconds(scanEnd);
    result.totalMs = reduceEnd.synchronizeAndReportElapsedTimeInMilliseconds(totalStart);
    return result;
}

EmbeddingSparseGradientProfileResult profilePreparedEmbeddingSparseGradientWithSparseRowUpdate(
    PreparedEmbeddingSparseGradient& prepared,
    const Tensor& indices,
    const Tensor& upstreamGradient,
    SparseRowGradient& outputGradient,
    const std::unordered_map<std::string, float>& runtimeScalars,
    Stream stream) {
    validatePreparedEmbeddingSparseGradientInvocation(prepared, indices, upstreamGradient, outputGradient);
    if (!prepared.sparseRowUpdate.has_value()) {
        throw std::invalid_argument(
            "Embedding sparse-gradient fused-update profiler requires a prepared producer with a fused sparse-row update.");
    }

    ScopedGpu scopedGpu(prepared.deviceNum);

    EmbeddingSparseGradientProfileResult result;
    result.numTokens = prepared.numTokens;
    result.vocabularySize = prepared.vocabularySize;
    result.embeddingDim = prepared.embeddingDim;
    result.capacity = outputGradient.capacity;
    result.indexDataType = prepared.indexDataType;
    result.gradientDataType = prepared.gradientDataType;
    result.rowDataType = prepared.rowDataType;
    result.fusedSparseRowUpdate = true;
    result.sortTempBytes = prepared.sortTempBytes;
    result.rleTempBytes = prepared.rleTempBytes;
    result.scanTempBytes = prepared.scanTempBytes;

    Event totalStart(prepared.deviceNum, /*enableTiming=*/true);
    Event materializeEnd(prepared.deviceNum, /*enableTiming=*/true);
    Event sortEnd(prepared.deviceNum, /*enableTiming=*/true);
    Event rleEnd(prepared.deviceNum, /*enableTiming=*/true);
    Event finalizeEnd(prepared.deviceNum, /*enableTiming=*/true);
    Event scanEnd(prepared.deviceNum, /*enableTiming=*/true);
    Event reduceUpdateEnd(prepared.deviceNum, /*enableTiming=*/true);

    totalStart.record(stream);

    launchMaterializeSortPairs(indices, prepared, stream);
    materializeEnd.record(stream);

    const int cubItems = checkedCubItems(prepared.numTokens, "token count");
    sortPairs(prepared, cubItems, stream);
    sortEnd.record(stream);

    rleRows(prepared, outputGradient, cubItems, stream);
    rleEnd.record(stream);

    launchFinalizeRows(prepared, outputGradient, stream);
    finalizeEnd.record(stream);

    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(prepared.scanTempStorage.getMemPtr<void>(),
                                             prepared.scanTempBytes,
                                             prepared.runCounts.getMemPtr<uint32_t>(),
                                             prepared.runOffsets.getMemPtr<uint32_t>(),
                                             cubItems,
                                             stream.getStream()));
    scanEnd.record(stream);

    launchReduceValuesWithSparseRowUpdate(prepared, upstreamGradient, outputGradient, runtimeScalars, stream);
    reduceUpdateEnd.record(stream);

    populateRunCountProfileStats(result, prepared, outputGradient, stream);

    result.materializeSortPairsMs = materializeEnd.synchronizeAndReportElapsedTimeInMilliseconds(totalStart);
    result.cubSortMs = sortEnd.synchronizeAndReportElapsedTimeInMilliseconds(materializeEnd);
    result.clearRunCountsMs = 0.0f;
    result.cubRleMs = rleEnd.synchronizeAndReportElapsedTimeInMilliseconds(sortEnd);
    result.finalizeRowsMs = finalizeEnd.synchronizeAndReportElapsedTimeInMilliseconds(rleEnd);
    result.cubScanOffsetsMs = scanEnd.synchronizeAndReportElapsedTimeInMilliseconds(finalizeEnd);
    // In this profiler the "reduce" stage is the production fused reducer+optimizer-update kernel.
    result.reduceValuesMs = reduceUpdateEnd.synchronizeAndReportElapsedTimeInMilliseconds(scanEnd);
    result.totalMs = reduceUpdateEnd.synchronizeAndReportElapsedTimeInMilliseconds(totalStart);
    return result;
}

namespace {

void capturePreparedEmbeddingSparseGradientImpl(CudaGraphCaptureBuilder& builder,
                                                PreparedEmbeddingSparseGradient& prepared,
                                                const Tensor& indices,
                                                const Tensor& upstreamGradient,
                                                SparseRowGradient& outputGradient,
                                                const std::unordered_map<std::string, float>* runtimeScalars,
                                                CapturedEmbeddingSparseGradient& captured) {
    validatePreparedEmbeddingSparseGradientInvocation(prepared, indices, upstreamGradient, outputGradient);
    const bool fusedSparseRowUpdate = prepared.sparseRowUpdate.has_value();
    if (fusedSparseRowUpdate && runtimeScalars == nullptr) {
        throw std::invalid_argument(
            "Prepared Embedding sparse-gradient graph capture with fused sparse-row update requires runtime scalar bindings.");
    }
    if (!fusedSparseRowUpdate && runtimeScalars != nullptr) {
        throw std::invalid_argument(
            "Prepared Embedding sparse-gradient graph capture received runtime scalar bindings for a non-fused sparse-gradient producer.");
    }
    auto validateReduceNodeHandle = [&](const DeviceUpdatableKernelNodeDeviceHandle& handle, const char* label) {
        if (!handle.isInitialized()) {
            throw std::invalid_argument(std::string("Prepared Embedding sparse-gradient graph capture requires a preallocated ") + label +
                                        " reduce-node handle. Allocate CapturedEmbeddingSparseGradient before stream capture begins.");
        }
        if (handle.getGpuNum() != prepared.deviceNum) {
            throw std::invalid_argument(std::string("Prepared Embedding sparse-gradient graph capture ") + label +
                                        " reduce-node handle must live on the prepared GPU.");
        }
    };
    validateReduceNodeHandle(captured.lowReduceNodeHandle, "low-run");
    validateReduceNodeHandle(captured.highReduceNodeHandle, "high-run");
    if (hasUltraHighTwoStageEmbeddingSparseGradientReducer(prepared.embeddingDim)) {
        validateReduceNodeHandle(captured.ultraHighPartialReduceNodeHandle, "ultra-high-partial-run");
    }
    validateReduceNodeHandle(captured.ultraHighReduceNodeHandle, "ultra-high-run");
    const bool useTwoStageFinalize = useTwoStageEmbeddingSparseGradientFinalize(prepared.numTokens);
    if (useTwoStageFinalize) {
        validateReduceNodeHandle(captured.twoStageFinalizeClassifyNodeHandle, "two-stage-finalize-classify");
        validateReduceNodeHandle(captured.twoStageFinalizeAccumulateNodeHandle, "two-stage-finalize-accumulate");
    }
    if (builder.stream().getGpuNum() != prepared.deviceNum) {
        throw std::invalid_argument("Prepared Embedding sparse-gradient graph capture stream must be on the prepared GPU.");
    }
    if (prepared.reduceKernel == nullptr) {
        throw std::runtime_error("Prepared Embedding sparse-gradient graph reducer kernel was not compiled.");
    }

    Stream stream = builder.stream();
    ScopedGpu scopedGpu(prepared.deviceNum);

    launchMaterializeSortPairs(indices, prepared, stream);

    const int cubItems = checkedCubItems(prepared.numTokens, "token count");
    sortPairs(prepared, cubItems, stream);

    // RLE writes only the first numRuns counts.
    rleRows(prepared, outputGradient, cubItems, stream);

    const EmbeddingSparseGradientReduceGridUpdateConfig normalReduceGridConfig =
        reduceGridUpdateConfigForEmbeddingSparseGradient(outputGradient.capacity, prepared.embeddingDim);
    const uint64_t maxHighRows =
        maxHighRunRowsForEmbeddingSparseGradient(outputGradient.capacity, prepared.numTokens, prepared.runBucketConfig);
    const EmbeddingSparseGradientReduceGridUpdateConfig highRunReduceGridConfig =
        highRunReduceGridUpdateConfigForEmbeddingSparseGradient(maxHighRows, prepared.embeddingDim);
    const EmbeddingSparseGradientReduceGridUpdateConfig ultraHighPartialReduceGridConfig =
        ultraHighPartialReduceGridUpdateConfigForEmbeddingSparseGradient(prepared.maxUltraHighPartials, prepared.embeddingDim);
    const uint64_t maxUltraRows =
        maxUltraHighRunRowsForEmbeddingSparseGradient(outputGradient.capacity, prepared.numTokens, prepared.runBucketConfig);
    const EmbeddingSparseGradientReduceGridUpdateConfig ultraHighFinalReduceGridConfig =
        ultraHighFinalReduceGridUpdateConfigForEmbeddingSparseGradient(maxUltraRows, prepared.embeddingDim);
    if (normalReduceGridConfig.reduceRowsPerGridX == 0 || normalReduceGridConfig.reduceGridDimY == 0 ||
        highRunReduceGridConfig.reduceRowsPerGridX == 0 || highRunReduceGridConfig.reduceGridDimY == 0 ||
        ultraHighPartialReduceGridConfig.reduceRowsPerGridX == 0 || ultraHighPartialReduceGridConfig.reduceGridDimY == 0 ||
        ultraHighFinalReduceGridConfig.reduceRowsPerGridX == 0 || ultraHighFinalReduceGridConfig.reduceGridDimY == 0) {
        throw std::runtime_error("Prepared Embedding sparse-gradient graph reducer requires a non-empty embedding dimension.");
    }

    launchFinalizeRows(prepared,
                       outputGradient,
                       stream,
                       OptionalReduceGridUpdate{&captured.lowReduceNodeHandle,
                                                &captured.highReduceNodeHandle,
                                                hasUltraHighTwoStageEmbeddingSparseGradientReducer(prepared.embeddingDim)
                                                    ? &captured.ultraHighPartialReduceNodeHandle
                                                    : nullptr,
                                                &captured.ultraHighReduceNodeHandle,
                                                useTwoStageFinalize ? &captured.twoStageFinalizeClassifyNodeHandle : nullptr,
                                                useTwoStageFinalize ? &captured.twoStageFinalizeAccumulateNodeHandle : nullptr,
                                                normalReduceGridConfig,
                                                highRunReduceGridConfig,
                                                ultraHighPartialReduceGridConfig,
                                                ultraHighFinalReduceGridConfig,
                                                useTwoStageFinalize});

    if (useTwoStageFinalize) {
        EmbeddingSparseGradientTwoStageFinalizeCapturedNodes twoStageNodes = captureTwoStageFinalizeAndBucketizeEmbeddingSparseGradientRows(
            outputGradient.rows.getMemPtr<void>(),
            prepared.numRuns.getMemPtr<uint32_t>(),
            outputGradient.numRows.getMemPtr<void>(),
            prepared.runCounts.getMemPtr<uint32_t>(),
            prepared.lowRunRows.getMemPtr<uint32_t>(),
            prepared.highRunRows.getMemPtr<uint32_t>(),
            prepared.ultraHighRunRows.getMemPtr<uint32_t>(),
            prepared.ultraHighRunPartialCounts.getMemPtr<uint32_t>(),
            prepared.ultraHighRunPartialOffsets.getMemPtr<uint32_t>(),
            prepared.numUltraHighPartials.getMemPtr<uint32_t>(),
            prepared.numLowRunRows.getMemPtr<uint32_t>(),
            prepared.numHighRunRows.getMemPtr<uint32_t>(),
            prepared.numUltraHighRunRows.getMemPtr<uint32_t>(),
            prepared.twoStageLowRunRowsScratch.getMemPtr<uint32_t>(),
            prepared.twoStageHighRunRowsScratch.getMemPtr<uint32_t>(),
            prepared.twoStageUltraHighRunRowsScratch.getMemPtr<uint32_t>(),
            prepared.twoStageUltraHighRunPartialCountsScratch.getMemPtr<uint32_t>(),
            prepared.twoStageUltraHighRunPartialOffsetsScratch.getMemPtr<uint32_t>(),
            prepared.twoStageLowRunRowCounts.getMemPtr<uint32_t>(),
            prepared.twoStageHighRunRowCounts.getMemPtr<uint32_t>(),
            prepared.twoStageUltraHighRunRowCounts.getMemPtr<uint32_t>(),
            prepared.twoStageUltraHighPartialCounts.getMemPtr<uint32_t>(),
            prepared.vocabularySize,
            static_cast<uint32_t>(prepared.numTokens),
            prepared.rowDataType,
            prepared.runBucketConfig.lowRunMax,
            prepared.runBucketConfig.ultraHighRunMin,
            prepared.runBucketConfig.ultraHighTokensPerPartial,
            &captured.lowReduceNodeHandle,
            &captured.highReduceNodeHandle,
            hasUltraHighTwoStageEmbeddingSparseGradientReducer(prepared.embeddingDim) ? &captured.ultraHighPartialReduceNodeHandle
                                                                                      : nullptr,
            &captured.ultraHighReduceNodeHandle,
            normalReduceGridConfig,
            highRunReduceGridConfig,
            ultraHighPartialReduceGridConfig,
            ultraHighFinalReduceGridConfig,
            stream);
        captured.twoStageFinalizeClassifyNode = twoStageNodes.classifyNode;
        captured.twoStageFinalizeAccumulateNode = twoStageNodes.accumulateNode;
    }

    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(prepared.scanTempStorage.getMemPtr<void>(),
                                             prepared.scanTempBytes,
                                             prepared.runCounts.getMemPtr<uint32_t>(),
                                             prepared.runOffsets.getMemPtr<uint32_t>(),
                                             cubItems,
                                             stream.getStream()));

    auto captureReduceBucket = [&](const Tensor& runRows,
                                   const Tensor& numRunRows,
                                   CUfunction reduceKernel,
                                   uint32_t reduceBlockThreads,
                                   uint32_t reduceGridDimY,
                                   Stream captureStream) {
        const void* sortedTokenIdsPtr = prepared.sortedTokenIds.getMemPtr();
        const void* runOffsetsPtr = prepared.runOffsets.getMemPtr();
        const void* runCountsPtr = prepared.runCounts.getMemPtr();
        const void* numRunRowsPtr = numRunRows.getMemPtr();
        const void* runRowsPtr = runRows.getMemPtr();
        const void* upstreamPtr = upstreamGradient.getMemPtr();

        std::vector<void*> args;
        args.reserve(fusedSparseRowUpdate
                         ? 7 + prepared.sparseRowUpdate->kernelInputSlots.size() + prepared.sparseRowUpdate->outputSlots.size()
                         : 7);
        args.push_back((void*)&sortedTokenIdsPtr);
        args.push_back((void*)&runOffsetsPtr);
        args.push_back((void*)&runCountsPtr);
        args.push_back((void*)&numRunRowsPtr);
        args.push_back((void*)&runRowsPtr);
        args.push_back((void*)&upstreamPtr);

        std::optional<SparseRowUpdateFusionKernelArgs> updateArgs;
        const void* outputRowsPtr = nullptr;
        void* outputValuesPtr = nullptr;
        if (fusedSparseRowUpdate) {
            outputRowsPtr = outputGradient.rows.getMemPtr();
            args.push_back((void*)&outputRowsPtr);
            updateArgs.emplace(buildSparseRowUpdateFusionKernelArgs(prepared.sparseRowUpdate.value(), *runtimeScalars));
            args.insert(args.end(), updateArgs->args.begin(), updateArgs->args.end());
        } else {
            outputValuesPtr = outputGradient.values.getMemPtr();
            args.push_back((void*)&outputValuesPtr);
        }

        return builder.captureDeviceUpdatableKernelOnStream(
            CudaGraphKernelLaunch{reduceKernel, dim3(1, reduceGridDimY, 1), dim3(reduceBlockThreads, 1, 1), 0, args.data(), nullptr},
            captureStream);
    };

    auto captureUltraHighPartialReduce = [&](Stream captureStream) {
        const void* sortedTokenIdsPtr = prepared.sortedTokenIds.getMemPtr();
        const void* runOffsetsPtr = prepared.runOffsets.getMemPtr();
        const void* runCountsPtr = prepared.runCounts.getMemPtr();
        const void* numUltraHighPartialsPtr = prepared.numUltraHighPartials.getMemPtr();
        const void* partialRunRowsPtr = prepared.ultraHighPartialRunRows.getMemPtr();
        const void* partialTokenOffsetsPtr = prepared.ultraHighPartialTokenOffsets.getMemPtr();
        const void* upstreamPtr = upstreamGradient.getMemPtr();
        void* partialSumsPtr = prepared.ultraHighPartialSums.getMemPtr();

        std::vector<void*> args;
        args.reserve(8);
        args.push_back((void*)&sortedTokenIdsPtr);
        args.push_back((void*)&runOffsetsPtr);
        args.push_back((void*)&runCountsPtr);
        args.push_back((void*)&numUltraHighPartialsPtr);
        args.push_back((void*)&partialRunRowsPtr);
        args.push_back((void*)&partialTokenOffsetsPtr);
        args.push_back((void*)&upstreamPtr);
        args.push_back((void*)&partialSumsPtr);

        return builder.captureDeviceUpdatableKernelOnStream(
            CudaGraphKernelLaunch{prepared.ultraHighPartialReduceKernel,
                                  dim3(1, ultraHighPartialReduceGridDimYForEmbeddingSparseGradient(prepared.embeddingDim), 1),
                                  dim3(ultraHighReduceBlockThreadsForEmbeddingSparseGradient(prepared.embeddingDim), 1, 1),
                                  0,
                                  args.data(),
                                  nullptr},
            captureStream);
    };

    auto captureUltraHighFinalReduce = [&](Stream captureStream) {
        const void* numRunRowsPtr = prepared.numUltraHighRunRows.getMemPtr();
        const void* runRowsPtr = prepared.ultraHighRunRows.getMemPtr();
        const void* partialCountsPtr = prepared.ultraHighRunPartialCounts.getMemPtr();
        const void* partialOffsetsPtr = prepared.ultraHighRunPartialOffsets.getMemPtr();
        const void* partialSumsPtr = prepared.ultraHighPartialSums.getMemPtr();

        std::vector<void*> args;
        args.reserve(fusedSparseRowUpdate
                         ? 6 + prepared.sparseRowUpdate->kernelInputSlots.size() + prepared.sparseRowUpdate->outputSlots.size()
                         : 6);
        args.push_back((void*)&numRunRowsPtr);
        args.push_back((void*)&runRowsPtr);
        args.push_back((void*)&partialCountsPtr);
        args.push_back((void*)&partialOffsetsPtr);
        args.push_back((void*)&partialSumsPtr);

        std::optional<SparseRowUpdateFusionKernelArgs> updateArgs;
        const void* outputRowsPtr = nullptr;
        void* outputValuesPtr = nullptr;
        if (fusedSparseRowUpdate) {
            outputRowsPtr = outputGradient.rows.getMemPtr();
            args.push_back((void*)&outputRowsPtr);
            updateArgs.emplace(buildSparseRowUpdateFusionKernelArgs(prepared.sparseRowUpdate.value(), *runtimeScalars));
            args.insert(args.end(), updateArgs->args.begin(), updateArgs->args.end());
        } else {
            outputValuesPtr = outputGradient.values.getMemPtr();
            args.push_back((void*)&outputValuesPtr);
        }

        return builder.captureDeviceUpdatableKernelOnStream(
            CudaGraphKernelLaunch{prepared.ultraHighFinalReduceKernel,
                                  dim3(1, ultraHighFinalReduceGridDimYForEmbeddingSparseGradient(prepared.embeddingDim), 1),
                                  dim3(ultraHighReduceBlockThreadsForEmbeddingSparseGradient(prepared.embeddingDim), 1, 1),
                                  0,
                                  args.data(),
                                  nullptr},
            captureStream);
    };

    // The scan produces the final runOffsets consumed by all bucket reducers. From here the low, high,
    // and ultra-high buckets are independent, so capture them as sibling graph branches instead of
    // serializing them on the main capture stream. The captured object owns dedicated helper streams so
    // capture never depends on unrelated work queued on the reusable gradient-update stream pool.
    Event reducersReady = stream.putEvent(false);
    captured.highRunCaptureStream.waitEvent(reducersReady);
    captured.ultraHighRunCaptureStream.waitEvent(reducersReady);

    CUfunction highRunReduceKernel = prepared.highRunReduceKernel != nullptr ? prepared.highRunReduceKernel : prepared.reduceKernel;
    captured.lowReduceNode = captureReduceBucket(prepared.lowRunRows,
                                                 prepared.numLowRunRows,
                                                 prepared.reduceKernel,
                                                 reduceBlockThreadsForEmbeddingSparseGradient(prepared.embeddingDim),
                                                 reduceGridDimYForEmbeddingSparseGradient(prepared.embeddingDim),
                                                 stream);
    captured.highReduceNode = captureReduceBucket(prepared.highRunRows,
                                                  prepared.numHighRunRows,
                                                  highRunReduceKernel,
                                                  highRunReduceBlockThreadsForEmbeddingSparseGradient(prepared.embeddingDim),
                                                  highRunReduceGridDimYForEmbeddingSparseGradient(prepared.embeddingDim),
                                                  captured.highRunCaptureStream);
    if (hasUltraHighTwoStageEmbeddingSparseGradientReducer(prepared.embeddingDim)) {
        if (prepared.ultraHighPartialReduceKernel == nullptr || prepared.ultraHighFinalReduceKernel == nullptr) {
            throw std::runtime_error("Prepared Embedding sparse-gradient graph ultra-high reducer kernels were not compiled.");
        }
        prepareUltraHighPartialMetadata(prepared, captured.ultraHighRunCaptureStream);
        captured.ultraHighPartialReduceNode = captureUltraHighPartialReduce(captured.ultraHighRunCaptureStream);
        captured.ultraHighReduceNode = captureUltraHighFinalReduce(captured.ultraHighRunCaptureStream);
    } else {
        captured.ultraHighReduceNode = captureReduceBucket(prepared.ultraHighRunRows,
                                                           prepared.numUltraHighRunRows,
                                                           prepared.reduceKernel,
                                                           reduceBlockThreadsForEmbeddingSparseGradient(prepared.embeddingDim),
                                                           reduceGridDimYForEmbeddingSparseGradient(prepared.embeddingDim),
                                                           captured.ultraHighRunCaptureStream);
    }

    stream.waitEvent(captured.highRunCaptureStream.putEvent(false));
    stream.waitEvent(captured.ultraHighRunCaptureStream.putEvent(false));
}

}  // namespace

EmbeddingSparseGradientRunBucketConfig defaultEmbeddingSparseGradientRunBucketConfig() {
    return EmbeddingSparseGradientRunBucketConfig{EMBEDDING_SPARSE_GRADIENT_LOW_RUN_MAX,
                                                  EMBEDDING_SPARSE_GRADIENT_ULTRA_HIGH_RUN_MIN,
                                                  DEFAULT_EMBEDDING_SPARSE_GRADIENT_ULTRA_HIGH_TOKENS_PER_PARTIAL};
}

bool supportsEmbeddingSparseGradientFusedSparseRowUpdate(uint64_t embeddingDim) {
    return hasVectorizedJitEmbeddingSparseGradientReducer(embeddingDim);
}

void setEmbeddingSparseGradientRunBucketConfigOverrideForTesting(std::optional<EmbeddingSparseGradientRunBucketConfig> config) {
    if (config.has_value()) {
        validateEmbeddingSparseGradientRunBucketConfig(config.value());
    }
    std::lock_guard<std::mutex> lock(gEmbeddingSparseGradientRunBucketConfigMutex);
    gEmbeddingSparseGradientRunBucketConfigOverride = std::move(config);
}

void capturePreparedEmbeddingSparseGradient(CudaGraphCaptureBuilder& builder,
                                            PreparedEmbeddingSparseGradient& prepared,
                                            const Tensor& indices,
                                            const Tensor& upstreamGradient,
                                            SparseRowGradient& outputGradient,
                                            CapturedEmbeddingSparseGradient& captured) {
    capturePreparedEmbeddingSparseGradientImpl(builder, prepared, indices, upstreamGradient, outputGradient, nullptr, captured);
}

void capturePreparedEmbeddingSparseGradientWithSparseRowUpdate(CudaGraphCaptureBuilder& builder,
                                                               PreparedEmbeddingSparseGradient& prepared,
                                                               const Tensor& indices,
                                                               const Tensor& upstreamGradient,
                                                               SparseRowGradient& outputGradient,
                                                               const std::unordered_map<std::string, float>& runtimeScalars,
                                                               CapturedEmbeddingSparseGradient& captured) {
    capturePreparedEmbeddingSparseGradientImpl(builder, prepared, indices, upstreamGradient, outputGradient, &runtimeScalars, captured);
}

}  // namespace ThorImplementation
