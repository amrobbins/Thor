#include "Utilities/TensorOperations/Embedding/EmbeddingSparseGradient.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/Event.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/Embedding/EmbeddingSparseGradientCudaCompile.h"

#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <memory>
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

using DataType = TensorDescriptor::DataType;

constexpr uint32_t THREADS_PER_BLOCK = 256;
constexpr bool PRINT_GENERATED_EMBEDDING_SPARSE_GRADIENT_KERNELS = true;
constexpr uint32_t WARP_SIZE_EMBEDDING = 32;
constexpr uint32_t WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE_EMBEDDING;
static_assert(THREADS_PER_BLOCK == 256);
static_assert(WARPS_PER_BLOCK == 8);

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


template <typename GradT, uint32_t EmbeddingDim>
struct EmbeddingSparseGradientVectorOps {
    static_assert(EmbeddingDim % 4U == 0U, "Fixed embedding sparse-gradient vector reducer requires D divisible by 4.");
    static constexpr uint32_t DIMS_PER_THREAD = 4U;
    static constexpr uint32_t THREADS_PER_ROW = EmbeddingDim / DIMS_PER_THREAD;
    static constexpr uint32_t ROWS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_ROW;

    __device__ __forceinline__ static float4 load(uint64_t token,
                                                  uint32_t vectorIndex,
                                                  const GradT* __restrict__ upstreamGradient) {
        const uint64_t base = token * static_cast<uint64_t>(EmbeddingDim) +
                              static_cast<uint64_t>(vectorIndex) * DIMS_PER_THREAD;
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

template <uint32_t EmbeddingDim>
struct EmbeddingSparseGradientVectorOps<float, EmbeddingDim> {
    static_assert(EmbeddingDim % 4U == 0U, "FP32 vector reducer requires D divisible by 4.");
    static constexpr uint32_t DIMS_PER_THREAD = 4U;
    static constexpr uint32_t THREADS_PER_ROW = EmbeddingDim / DIMS_PER_THREAD;
    static constexpr uint32_t ROWS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_ROW;

    __device__ __forceinline__ static float4 load(uint64_t token,
                                                  uint32_t vectorIndex,
                                                  const float* __restrict__ upstreamGradient) {
        const float4* __restrict__ input = reinterpret_cast<const float4*>(upstreamGradient + token * static_cast<uint64_t>(EmbeddingDim));
        return input[vectorIndex];
    }

    __device__ __forceinline__ static void store(uint64_t row, uint32_t vectorIndex, float4 value, float* __restrict__ outputValues) {
        float4* __restrict__ output = reinterpret_cast<float4*>(outputValues + row * static_cast<uint64_t>(EmbeddingDim));
        output[vectorIndex] = value;
    }
};

template <uint32_t EmbeddingDim>
struct EmbeddingSparseGradientVectorOps<__half, EmbeddingDim> {
    static_assert(EmbeddingDim % 4U == 0U, "FP16 vector reducer requires D divisible by 4.");
    static constexpr uint32_t DIMS_PER_THREAD = 4U;
    static constexpr uint32_t THREADS_PER_ROW = EmbeddingDim / DIMS_PER_THREAD;
    static constexpr uint32_t ROWS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_ROW;

    __device__ __forceinline__ static float4 load(uint64_t token,
                                                  uint32_t vectorIndex,
                                                  const __half* __restrict__ upstreamGradient) {
        const unsigned long long* __restrict__ input = reinterpret_cast<const unsigned long long*>(upstreamGradient + token * static_cast<uint64_t>(EmbeddingDim));
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

template <uint32_t EmbeddingDim>
struct EmbeddingSparseGradientVectorOps<__nv_bfloat16, EmbeddingDim> {
    static_assert(EmbeddingDim % 4U == 0U, "BF16 vector reducer requires D divisible by 4.");
    static constexpr uint32_t DIMS_PER_THREAD = 4U;
    static constexpr uint32_t THREADS_PER_ROW = EmbeddingDim / DIMS_PER_THREAD;
    static constexpr uint32_t ROWS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_ROW;

    __device__ __forceinline__ static float4 load(uint64_t token,
                                                  uint32_t vectorIndex,
                                                  const __nv_bfloat16* __restrict__ upstreamGradient) {
        const unsigned long long* __restrict__ input = reinterpret_cast<const unsigned long long*>(upstreamGradient + token * static_cast<uint64_t>(EmbeddingDim));
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

__device__ __forceinline__ cudaGraphDeviceNode_t loadEmbeddingSparseGradientTargetNode(const cudaGraphDeviceNode_t* targetNode) {
    if (targetNode == nullptr) {
        asm("trap;");
    }
    cudaGraphDeviceNode_t node = *targetNode;
    if (node == nullptr) {
        asm("trap;");
    }
    return node;
}

__device__ __forceinline__ uint32_t checkedEmbeddingSparseGradientGridDim(uint64_t value, uint32_t minGrid, uint32_t maxGrid) {
    if (minGrid == 0U || maxGrid == 0U || minGrid > maxGrid) {
        asm("trap;");
    }
    uint64_t clamped = value;
    if (clamped < static_cast<uint64_t>(minGrid)) {
        clamped = static_cast<uint64_t>(minGrid);
    }
    if (clamped > static_cast<uint64_t>(maxGrid) || clamped > 0xffffffffULL) {
        asm("trap;");
    }
    return static_cast<uint32_t>(clamped);
}

template <typename RowT>
__global__ void finalizeEmbeddingSparseGradientRowsKernel(const RowT* __restrict__ outputRows,
                                                          const uint32_t* __restrict__ numRuns,
                                                          RowT* __restrict__ outputNumRows,
                                                          uint64_t vocabularySize,
                                                          const cudaGraphDeviceNode_t* reduceNode,
                                                          uint32_t reduceRowsPerGridX,
                                                          uint32_t reduceGridDimY,
                                                          uint32_t minReduceGridDimX,
                                                          uint32_t maxReduceGridDimX,
                                                          uint32_t maxReduceGridDimY) {
    uint32_t validRuns = numRuns[0];
    if (validRuns != 0 && static_cast<uint64_t>(outputRows[validRuns - 1]) == vocabularySize) {
        validRuns -= 1;
    }

    outputNumRows[0] = static_cast<RowT>(validRuns);

    if (reduceNode != nullptr) {
        if (reduceRowsPerGridX == 0U || reduceGridDimY == 0U || reduceGridDimY > maxReduceGridDimY) {
            asm("trap;");
        }
        const uint64_t packedGridX = (static_cast<uint64_t>(validRuns) + reduceRowsPerGridX - 1ULL) / reduceRowsPerGridX;
        const uint32_t gridX = checkedEmbeddingSparseGradientGridDim(packedGridX,
                                                                     minReduceGridDimX,
                                                                     maxReduceGridDimX);
        cudaError_t status = cudaGraphKernelNodeSetGridDim(loadEmbeddingSparseGradientTargetNode(reduceNode),
                                                           dim3(gridX, reduceGridDimY, 1U));
        if (status != cudaSuccess) {
            asm("trap;");
        }
    }
}

template <uint32_t EmbeddingDim>
struct IsFixedEmbeddingReducerDim {
    static constexpr bool value = EmbeddingDim == 16U || EmbeddingDim == 32U || EmbeddingDim == 64U || EmbeddingDim == 128U ||
                                  EmbeddingDim == 256U;
};

template <typename RowT, typename GradT, uint32_t EmbeddingDim>
__global__ void reduceEmbeddingSparseGradientValuesFixedDimKernel(const uint32_t* __restrict__ sortedTokenIds,
                                                                  const uint32_t* __restrict__ runOffsets,
                                                                  const uint32_t* __restrict__ runCounts,
                                                                  const RowT* __restrict__ numValidRows,
                                                                  const GradT* __restrict__ upstreamGradient,
                                                                  float* __restrict__ outputValues) {
    static_assert(IsFixedEmbeddingReducerDim<EmbeddingDim>::value, "Unsupported fixed embedding sparse-gradient reducer dimension.");
    static_assert(EmbeddingDim % 4U == 0U, "Fixed embedding sparse-gradient vector reducer requires D divisible by 4.");

    using VecOps = EmbeddingSparseGradientVectorOps<GradT, EmbeddingDim>;
    constexpr uint32_t THREADS_PER_ROW = VecOps::THREADS_PER_ROW;
    constexpr uint32_t ROWS_PER_BLOCK = VecOps::ROWS_PER_BLOCK;
    static_assert(THREADS_PER_ROW >= 1U, "Fixed embedding sparse-gradient vector reducer requires at least one thread per row.");
    static_assert(ROWS_PER_BLOCK >= 1U, "Fixed embedding sparse-gradient vector reducer requires at least one row per block.");
    static_assert(THREADS_PER_ROW * ROWS_PER_BLOCK == THREADS_PER_BLOCK,
                  "Fixed embedding sparse-gradient vector reducer must use the whole block.");

    const uint32_t rowInBlock = threadIdx.x / THREADS_PER_ROW;
    const uint32_t vectorIndex = threadIdx.x - rowInBlock * THREADS_PER_ROW;
    const uint64_t row = static_cast<uint64_t>(blockIdx.x) * ROWS_PER_BLOCK + rowInBlock;
    if (row >= static_cast<uint64_t>(numValidRows[0])) {
        return;
    }

    const uint64_t begin = static_cast<uint64_t>(runOffsets[row]);
    const uint64_t count = static_cast<uint64_t>(runCounts[row]);
    const uint64_t firstToken = static_cast<uint64_t>(sortedTokenIds[begin]);

    // The larger side of the singleton/reduce copy is the FP32 output, so each active thread writes exactly one
    // coalesced 16-byte float4. FP32 upstream also reads 16 bytes; FP16/BF16 upstream read the corresponding 8 bytes.
    // Packing ROWS_PER_BLOCK rows into one CTA keeps all 256 threads useful for the fixed D={16,32,64,128,256} cases.
    float4 sum = VecOps::load(firstToken, vectorIndex, upstreamGradient);
    for (uint64_t i = 1ULL; i < count; ++i) {
        const uint64_t token = static_cast<uint64_t>(sortedTokenIds[begin + i]);
        sum = thor_embedding_add_float4(sum, VecOps::load(token, vectorIndex, upstreamGradient));
    }
    VecOps::store(row, vectorIndex, sum, outputValues);
}

template <typename RowT, typename GradT>
__global__ void reduceEmbeddingSparseGradientValuesTiledKernel(const uint32_t* __restrict__ sortedTokenIds,
                                                               const uint32_t* __restrict__ runOffsets,
                                                               const uint32_t* __restrict__ runCounts,
                                                               const RowT* __restrict__ numValidRows,
                                                               const GradT* __restrict__ upstreamGradient,
                                                               float* __restrict__ outputValues,
                                                               uint64_t embeddingDim) {
    const uint64_t row = static_cast<uint64_t>(blockIdx.x);
    if (row >= static_cast<uint64_t>(numValidRows[0])) {
        return;
    }

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

bool hasFixedDimEmbeddingSparseGradientReducer(uint64_t embeddingDim) {
    return embeddingDim == 16ULL || embeddingDim == 32ULL || embeddingDim == 64ULL || embeddingDim == 128ULL || embeddingDim == 256ULL;
}

uint32_t reduceRowsPerGridXForEmbeddingSparseGradient(uint64_t embeddingDim) {
    if (!hasFixedDimEmbeddingSparseGradientReducer(embeddingDim)) {
        return 1U;
    }
    // Fixed-D reducers use one 16-byte FP32 output vector per thread, so each row consumes D / 4 threads.
    // Pack as many rows as possible into one 256-thread CTA so every lane has useful row work.
    return static_cast<uint32_t>(THREADS_PER_BLOCK / (embeddingDim / 4ULL));
}

uint32_t reduceGridDimXForEmbeddingSparseGradient(uint64_t numRows, uint64_t embeddingDim) {
    const uint32_t rowsPerGridX = reduceRowsPerGridXForEmbeddingSparseGradient(embeddingDim);
    return static_cast<uint32_t>((numRows + rowsPerGridX - 1ULL) / rowsPerGridX);
}

uint32_t reduceGridDimYForEmbeddingSparseGradient(uint64_t embeddingDim) {
    if (hasFixedDimEmbeddingSparseGradientReducer(embeddingDim)) {
        return 1U;
    }
    return static_cast<uint32_t>((embeddingDim + THREADS_PER_BLOCK - 1ULL) / THREADS_PER_BLOCK);
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

std::string emitSparseGradientReduceKernelSource(DataType rowDType,
                                                DataType gradDType,
                                                uint64_t embeddingDim,
                                                const std::string& kernelName,
                                                const SparseRowUpdateFusionSource* sparseRowUpdate) {
    if (sparseRowUpdate != nullptr && !hasFixedDimEmbeddingSparseGradientReducer(embeddingDim)) {
        throw std::invalid_argument("Embedding sparse-gradient fused sparse-row update currently requires a fixed D={16,32,64,128,256} reducer.");
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
    ss << "extern \"C\" __global__\n";
    ss << "void " << kernelName << "(const unsigned int* sortedTokenIds, const unsigned int* runOffsets, const unsigned int* runCounts, ";
    ss << "const " << rowType << "* numValidRows, const " << gradType << "* upstreamGradient";
    if (sparseRowUpdate != nullptr) {
        ss << ", const " << rowType << "* outputRows" << sparseRowUpdate->parameterSource;
    } else {
        ss << ", float* outputValues";
    }
    ss << ") {\n";

    if (hasFixedDimEmbeddingSparseGradientReducer(embeddingDim)) {
        const uint64_t threadsPerRow = embeddingDim / 4ULL;
        const uint64_t rowsPerBlock = THREADS_PER_BLOCK / threadsPerRow;
        ss << "  constexpr unsigned int THREADS_PER_BLOCK_LOCAL = " << THREADS_PER_BLOCK << "U;\n";
        ss << "  constexpr unsigned int embeddingDim = " << embeddingDim << "U;\n";
        ss << "  constexpr unsigned int dimsPerThread = 4U;\n";
        ss << "  constexpr unsigned int threadsPerRow = " << threadsPerRow << "U;\n";
        ss << "  constexpr unsigned int rowsPerBlock = " << rowsPerBlock << "U;\n";
        ss << "  const unsigned int rowInBlock = threadIdx.x / threadsPerRow;\n";
        ss << "  const unsigned int vectorIndex = threadIdx.x - rowInBlock * threadsPerRow;\n";
        ss << "  const unsigned long long row = static_cast<unsigned long long>(blockIdx.x) * rowsPerBlock + rowInBlock;\n";
        ss << "  if (row >= static_cast<unsigned long long>(numValidRows[0])) return;\n";
        ss << "  const unsigned long long begin = static_cast<unsigned long long>(runOffsets[row]);\n";
        ss << "  const unsigned long long count = static_cast<unsigned long long>(runCounts[row]);\n";
        ss << "  const unsigned long long firstToken = static_cast<unsigned long long>(sortedTokenIds[begin]);\n";

        auto emitLoad = [&](const std::string& tokenExpr, const std::string& valueName) {
            if (gradDType == DataType::FP32) {
                ss << "  const float4* __restrict__ input_" << valueName << " = reinterpret_cast<const float4*>(upstreamGradient + (" << tokenExpr << ") * embeddingDim);\n";
                ss << "  " << (valueName == "sum" ? "float4 " : "const float4 ") << valueName << " = input_" << valueName << "[vectorIndex];\n";
            } else if (gradDType == DataType::FP16) {
                ss << "  const unsigned long long* __restrict__ input_" << valueName << " = reinterpret_cast<const unsigned long long*>(upstreamGradient + (" << tokenExpr << ") * embeddingDim);\n";
                ss << "  const unsigned long long packed_" << valueName << " = input_" << valueName << "[vectorIndex];\n";
                ss << "  const float2 lo_" << valueName << " = thor_embedding_half2_bits_to_float2(static_cast<unsigned int>(packed_" << valueName << "));\n";
                ss << "  const float2 hi_" << valueName << " = thor_embedding_half2_bits_to_float2(static_cast<unsigned int>(packed_" << valueName << " >> 32U));\n";
                ss << "  " << (valueName == "sum" ? "float4 " : "const float4 ") << valueName << " = make_float4(lo_" << valueName << ".x, lo_" << valueName << ".y, hi_" << valueName << ".x, hi_" << valueName << ".y);\n";
            } else if (gradDType == DataType::BF16) {
                ss << "  const unsigned long long* __restrict__ input_" << valueName << " = reinterpret_cast<const unsigned long long*>(upstreamGradient + (" << tokenExpr << ") * embeddingDim);\n";
                ss << "  const unsigned long long packed_" << valueName << " = input_" << valueName << "[vectorIndex];\n";
                ss << "  const float2 lo_" << valueName << " = thor_embedding_bfloat162_bits_to_float2(static_cast<unsigned int>(packed_" << valueName << "));\n";
                ss << "  const float2 hi_" << valueName << " = thor_embedding_bfloat162_bits_to_float2(static_cast<unsigned int>(packed_" << valueName << " >> 32U));\n";
                ss << "  " << (valueName == "sum" ? "float4 " : "const float4 ") << valueName << " = make_float4(lo_" << valueName << ".x, lo_" << valueName << ".y, hi_" << valueName << ".x, hi_" << valueName << ".y);\n";
            } else {
                ss << "  const unsigned long long base_" << valueName << " = (" << tokenExpr << ") * static_cast<unsigned long long>(embeddingDim) + static_cast<unsigned long long>(vectorIndex) * dimsPerThread;\n";
                ss << "  " << (valueName == "sum" ? "float4 " : "const float4 ") << valueName << " = make_float4(thor_embedding_grad_to_float(upstreamGradient[base_" << valueName << " + 0ULL]), thor_embedding_grad_to_float(upstreamGradient[base_" << valueName << " + 1ULL]), thor_embedding_grad_to_float(upstreamGradient[base_" << valueName << " + 2ULL]), thor_embedding_grad_to_float(upstreamGradient[base_" << valueName << " + 3ULL]));\n";
            }
        };

        emitLoad("firstToken", "sum");
        ss << "  for (unsigned long long i = 1ULL; i < count; ++i) {\n";
        ss << "    const unsigned long long token = static_cast<unsigned long long>(sortedTokenIds[begin + i]);\n";
        if (gradDType == DataType::FP32) {
            ss << "    const float4* __restrict__ input_v = reinterpret_cast<const float4*>(upstreamGradient + token * embeddingDim);\n";
            ss << "    const float4 v = input_v[vectorIndex];\n";
        } else if (gradDType == DataType::FP16) {
            ss << "    const unsigned long long* __restrict__ input_v = reinterpret_cast<const unsigned long long*>(upstreamGradient + token * embeddingDim);\n";
            ss << "    const unsigned long long packed_v = input_v[vectorIndex];\n";
            ss << "    const float2 lo_v = thor_embedding_half2_bits_to_float2(static_cast<unsigned int>(packed_v));\n";
            ss << "    const float2 hi_v = thor_embedding_half2_bits_to_float2(static_cast<unsigned int>(packed_v >> 32U));\n";
            ss << "    const float4 v = make_float4(lo_v.x, lo_v.y, hi_v.x, hi_v.y);\n";
        } else if (gradDType == DataType::BF16) {
            ss << "    const unsigned long long* __restrict__ input_v = reinterpret_cast<const unsigned long long*>(upstreamGradient + token * embeddingDim);\n";
            ss << "    const unsigned long long packed_v = input_v[vectorIndex];\n";
            ss << "    const float2 lo_v = thor_embedding_bfloat162_bits_to_float2(static_cast<unsigned int>(packed_v));\n";
            ss << "    const float2 hi_v = thor_embedding_bfloat162_bits_to_float2(static_cast<unsigned int>(packed_v >> 32U));\n";
            ss << "    const float4 v = make_float4(lo_v.x, lo_v.y, hi_v.x, hi_v.y);\n";
        } else {
            ss << "    const unsigned long long base_v = token * static_cast<unsigned long long>(embeddingDim) + static_cast<unsigned long long>(vectorIndex) * dimsPerThread;\n";
            ss << "    const float4 v = make_float4(thor_embedding_grad_to_float(upstreamGradient[base_v + 0ULL]), thor_embedding_grad_to_float(upstreamGradient[base_v + 1ULL]), thor_embedding_grad_to_float(upstreamGradient[base_v + 2ULL]), thor_embedding_grad_to_float(upstreamGradient[base_v + 3ULL]));\n";
        }
        ss << "    sum = thor_embedding_add_float4(sum, v);\n";
        ss << "  }\n";
        if (sparseRowUpdate != nullptr) {
            ss << "  const unsigned long long sru_logical_row = row;\n";
            ss << "  const unsigned long long sru_indexed_row = static_cast<unsigned long long>(outputRows[row]);\n";
            ss << "  const unsigned long long sru_vector_index = static_cast<unsigned long long>(vectorIndex);\n";
            ss << sparseRowUpdate->bodySource;
        } else {
            ss << "  float4* __restrict__ output = reinterpret_cast<float4*>(outputValues + row * embeddingDim);\n";
            ss << "  output[vectorIndex] = sum;\n";
        }
    } else {
        ss << "  constexpr unsigned int THREADS_PER_BLOCK_LOCAL = " << THREADS_PER_BLOCK << "U;\n";
        ss << "  const unsigned long long row = static_cast<unsigned long long>(blockIdx.x);\n";
        ss << "  if (row >= static_cast<unsigned long long>(numValidRows[0])) return;\n";
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

    Tensor rowKeys;
    Tensor tokenIds;
    Tensor sortedRowKeys;
    Tensor sortedTokenIds;
    Tensor runCounts;
    Tensor runOffsets;
    Tensor numRuns;
    Tensor sortTempStorage;
    Tensor rleTempStorage;
    Tensor scanTempStorage;
    size_t sortTempBytes = 0;
    size_t rleTempBytes = 0;
    size_t scanTempBytes = 0;

    CUmodule reduceModule = nullptr;
    CUfunction reduceKernel = nullptr;
    std::string reduceKernelName;
    std::optional<SparseRowUpdateFusionSource> sparseRowUpdate;
};

PreparedEmbeddingSparseGradient::~PreparedEmbeddingSparseGradient() {
    if (reduceModule != nullptr) {
        try {
            CU_CHECK(cuModuleUnload(reduceModule));
        } catch (...) {
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
    uint32_t maxRunCount = 0;
    for (uint32_t count : runCounts) {
        if (count == 1U) {
            ++singletonRows;
        } else if (count > 1U) {
            ++duplicateRows;
        }
        maxRunCount = std::max(maxRunCount, count);
    }
    result.singletonRows = singletonRows;
    result.duplicateRows = duplicateRows;
    result.maxRunCount = maxRunCount;
}

void compileGraphReduceKernel(PreparedEmbeddingSparseGradient& prepared) {
    if (prepared.reduceKernel != nullptr) {
        return;
    }

    prepared.reduceKernelName = sparseGradientReduceKernelName(prepared.rowDataType,
                                                              prepared.gradientDataType,
                                                              prepared.embeddingDim,
                                                              prepared.sparseRowUpdate.has_value());
    const std::string src = emitSparseGradientReduceKernelSource(prepared.rowDataType,
                                                                 prepared.gradientDataType,
                                                                 prepared.embeddingDim,
                                                                 prepared.reduceKernelName,
                                                                 prepared.sparseRowUpdate.has_value() ? &prepared.sparseRowUpdate.value() : nullptr);
    if constexpr (PRINT_GENERATED_EMBEDDING_SPARSE_GRADIENT_KERNELS) {
        std::fprintf(stderr,
                     "\n===== Generated Embedding sparse-gradient reducer CUDA source begin: %s =====\n%s\n===== Generated Embedding sparse-gradient reducer CUDA source end: %s =====\n",
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
    prepared.scanTempBytes = queryScanTempBytes(prepared.runCounts.getMemPtr<uint32_t>(), prepared.runOffsets.getMemPtr<uint32_t>(), cubItems);

    prepared.sortTempStorage = allocateByteScratch(placement, prepared.sortTempBytes);
    prepared.rleTempStorage = allocateByteScratch(placement, prepared.rleTempBytes);
    prepared.scanTempStorage = allocateByteScratch(placement, prepared.scanTempBytes);
}

template <typename IndexT, typename RowT>
void launchMaterializeSortPairsTyped(const Tensor& indices, PreparedEmbeddingSparseGradient& prepared, Stream stream) {
    const uint32_t block = THREADS_PER_BLOCK;
    const uint32_t grid = static_cast<uint32_t>((prepared.numTokens + block - 1) / block);
    materializeEmbeddingSparseGradientSortPairsKernel<IndexT, RowT><<<grid, block, 0, stream.getStream()>>>(
        indices.getMemPtr<IndexT>(),
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
                                Stream stream) {
    const dim3 block(THREADS_PER_BLOCK);
    const dim3 grid(reduceGridDimXForEmbeddingSparseGradient(outputGradient.capacity, EmbeddingDim), 1U);
    reduceEmbeddingSparseGradientValuesFixedDimKernel<RowT, GradT, EmbeddingDim><<<grid, block, 0, stream.getStream()>>>(
        prepared.sortedTokenIds.getMemPtr<uint32_t>(),
        prepared.runOffsets.getMemPtr<uint32_t>(),
        prepared.runCounts.getMemPtr<uint32_t>(),
        outputGradient.numRows.getMemPtr<RowT>(),
        upstreamGradient.getMemPtr<GradT>(),
        outputGradient.values.getMemPtr<float>());
}

template <typename RowT, typename GradT>
void launchReduceValuesTyped(PreparedEmbeddingSparseGradient& prepared,
                             const Tensor& upstreamGradient,
                             SparseRowGradient& outputGradient,
                             Stream stream) {
    const uint64_t maxRows = outputGradient.capacity;
    if (maxRows == 0) {
        return;
    }

    switch (prepared.embeddingDim) {
        case 16:
            launchReduceValuesFixedDim<RowT, GradT, 16U>(prepared, upstreamGradient, outputGradient, stream);
            break;
        case 32:
            launchReduceValuesFixedDim<RowT, GradT, 32U>(prepared, upstreamGradient, outputGradient, stream);
            break;
        case 64:
            launchReduceValuesFixedDim<RowT, GradT, 64U>(prepared, upstreamGradient, outputGradient, stream);
            break;
        case 128:
            launchReduceValuesFixedDim<RowT, GradT, 128U>(prepared, upstreamGradient, outputGradient, stream);
            break;
        case 256:
            launchReduceValuesFixedDim<RowT, GradT, 256U>(prepared, upstreamGradient, outputGradient, stream);
            break;
        default: {
            const dim3 block(THREADS_PER_BLOCK);
            const dim3 grid(reduceGridDimXForEmbeddingSparseGradient(maxRows, prepared.embeddingDim),
                            reduceGridDimYForEmbeddingSparseGradient(prepared.embeddingDim));
            reduceEmbeddingSparseGradientValuesTiledKernel<RowT, GradT><<<grid, block, 0, stream.getStream()>>>(
                prepared.sortedTokenIds.getMemPtr<uint32_t>(),
                prepared.runOffsets.getMemPtr<uint32_t>(),
                prepared.runCounts.getMemPtr<uint32_t>(),
                outputGradient.numRows.getMemPtr<RowT>(),
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
                                  Stream stream) {
    switch (prepared.gradientDataType) {
        case DataType::FP16:
            launchReduceValuesTyped<RowT, __half>(prepared, upstreamGradient, outputGradient, stream);
            break;
        case DataType::BF16:
            launchReduceValuesTyped<RowT, __nv_bfloat16>(prepared, upstreamGradient, outputGradient, stream);
            break;
        case DataType::FP32:
            launchReduceValuesTyped<RowT, float>(prepared, upstreamGradient, outputGradient, stream);
            break;
        default:
            throw std::runtime_error("Prepared Embedding sparse-gradient producer has unsupported upstream gradient dtype.");
    }
}

void launchReduceValues(PreparedEmbeddingSparseGradient& prepared,
                        const Tensor& upstreamGradient,
                        SparseRowGradient& outputGradient,
                        Stream stream) {
    if (prepared.sparseRowUpdate.has_value()) {
        throw std::invalid_argument("Prepared Embedding sparse-gradient producer with fused sparse-row update must use the fused update launcher.");
    }
    switch (prepared.rowDataType) {
        case DataType::UINT16:
            launchReduceValuesForRowType<uint16_t>(prepared, upstreamGradient, outputGradient, stream);
            break;
        case DataType::UINT32:
            launchReduceValuesForRowType<uint32_t>(prepared, upstreamGradient, outputGradient, stream);
            break;
        case DataType::UINT64:
            launchReduceValuesForRowType<uint64_t>(prepared, upstreamGradient, outputGradient, stream);
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

void launchReduceValuesWithSparseRowUpdate(PreparedEmbeddingSparseGradient& prepared,
                                           const Tensor& upstreamGradient,
                                           SparseRowGradient& outputGradient,
                                           const std::unordered_map<std::string, float>& runtimeScalars,
                                           Stream stream) {
    if (!prepared.sparseRowUpdate.has_value()) {
        throw std::invalid_argument("Prepared Embedding sparse-gradient producer does not have a fused sparse-row update.");
    }
    if (prepared.reduceKernel == nullptr) {
        throw std::runtime_error("Prepared Embedding sparse-gradient fused reducer kernel was not compiled.");
    }

    const uint32_t reduceRowsPerGridX = reduceRowsPerGridXForEmbeddingSparseGradient(prepared.embeddingDim);
    const uint32_t reduceGridDimY = reduceGridDimYForEmbeddingSparseGradient(prepared.embeddingDim);
    const dim3 grid(reduceGridDimXForEmbeddingSparseGradient(outputGradient.capacity, prepared.embeddingDim), reduceGridDimY, 1U);
    const dim3 block(THREADS_PER_BLOCK, 1U, 1U);

    const void* sortedTokenIdsPtr = prepared.sortedTokenIds.getMemPtr();
    const void* runOffsetsPtr = prepared.runOffsets.getMemPtr();
    const void* runCountsPtr = prepared.runCounts.getMemPtr();
    const void* numRowsPtr = outputGradient.numRows.getMemPtr();
    const void* upstreamPtr = upstreamGradient.getMemPtr();
    const void* outputRowsPtr = outputGradient.rows.getMemPtr();

    SparseRowUpdateFusionKernelArgs updateArgs = buildSparseRowUpdateFusionKernelArgs(prepared.sparseRowUpdate.value(), runtimeScalars);
    std::vector<void*> args;
    args.reserve(6 + updateArgs.args.size());
    args.push_back((void*)&sortedTokenIdsPtr);
    args.push_back((void*)&runOffsetsPtr);
    args.push_back((void*)&runCountsPtr);
    args.push_back((void*)&numRowsPtr);
    args.push_back((void*)&upstreamPtr);
    args.push_back((void*)&outputRowsPtr);
    args.insert(args.end(), updateArgs.args.begin(), updateArgs.args.end());

    (void)reduceRowsPerGridX;
    CU_CHECK(cuLaunchKernel(prepared.reduceKernel, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, stream, args.data(), nullptr));
}

struct OptionalReduceGridUpdate {
    const DeviceUpdatableKernelNodeDeviceHandle* reduceNodeHandle = nullptr;
    uint32_t reduceRowsPerGridX = 1;
    uint32_t reduceGridDimY = 0;
    uint32_t minReduceGridDimX = 1;
    uint32_t maxReduceGridDimX = 1;
    uint32_t maxReduceGridDimY = 1;
};

template <typename RowT>
void launchFinalizeRowsTyped(PreparedEmbeddingSparseGradient& prepared,
                             SparseRowGradient& outputGradient,
                             Stream stream,
                             OptionalReduceGridUpdate reduceGridUpdate) {
    const cudaGraphDeviceNode_t* reduceNodePtr =
        reduceGridUpdate.reduceNodeHandle != nullptr ? reduceGridUpdate.reduceNodeHandle->devicePtr() : nullptr;
    finalizeEmbeddingSparseGradientRowsKernel<RowT><<<1, 1, 0, stream.getStream()>>>(outputGradient.rows.getMemPtr<RowT>(),
                                                                                    prepared.numRuns.getMemPtr<uint32_t>(),
                                                                                    outputGradient.numRows.getMemPtr<RowT>(),
                                                                                    prepared.vocabularySize,
                                                                                    reduceNodePtr,
                                                                                    reduceGridUpdate.reduceRowsPerGridX,
                                                                                    reduceGridUpdate.reduceGridDimY,
                                                                                    reduceGridUpdate.minReduceGridDimX,
                                                                                    reduceGridUpdate.maxReduceGridDimX,
                                                                                    reduceGridUpdate.maxReduceGridDimY);
    CUDA_CHECK(cudaPeekAtLastError());
}

void launchFinalizeRows(PreparedEmbeddingSparseGradient& prepared,
                        SparseRowGradient& outputGradient,
                        Stream stream,
                        OptionalReduceGridUpdate reduceGridUpdate = {}) {
    switch (prepared.rowDataType) {
        case DataType::UINT16:
            launchFinalizeRowsTyped<uint16_t>(prepared, outputGradient, stream, reduceGridUpdate);
            break;
        case DataType::UINT32:
            launchFinalizeRowsTyped<uint32_t>(prepared, outputGradient, stream, reduceGridUpdate);
            break;
        case DataType::UINT64:
            launchFinalizeRowsTyped<uint64_t>(prepared, outputGradient, stream, reduceGridUpdate);
            break;
        default:
            throw std::runtime_error("Prepared Embedding sparse-gradient producer has unsupported row dtype.");
    }
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
        throw std::invalid_argument("Embedding sparse-gradient indices dtype must be uint32 or uint64. Got " + dtypeName(indices.getDataType()) + ".");
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
    SparseRowUpdateFusionSource source = SparseRowUpdatePlan::emitFusionSource(std::move(updateOutputs),
                                                                               outputGradient.rows,
                                                                               outputGradient.numRows,
                                                                               updateInputs,
                                                                               indexedUpdateOutputs,
                                                                               {{"gradient", "sum"}});
    return prepareEmbeddingSparseGradientImpl(indices, upstreamGradient, outputGradient, paddingIndex, std::move(source));
}

bool preparedEmbeddingSparseGradientHasSparseRowUpdate(const PreparedEmbeddingSparseGradient& prepared) {
    return prepared.sparseRowUpdate.has_value();
}

void validatePreparedEmbeddingSparseGradientInvocation(PreparedEmbeddingSparseGradient& prepared,
                                                       const Tensor& indices,
                                                       const Tensor& upstreamGradient,
                                                       SparseRowGradient& outputGradient) {
    outputGradient.validate();
    if (indices.getDataType() != prepared.indexDataType || upstreamGradient.getDataType() != prepared.gradientDataType) {
        throw std::invalid_argument("Prepared Embedding sparse-gradient producer received tensors with dtypes different from the prepared plan.");
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
            "Prepared Embedding sparse-gradient producer has a fused sparse-row update; use launchPreparedEmbeddingSparseGradientWithSparseRowUpdate.");
    }

    ScopedGpu scopedGpu(prepared.deviceNum);

    launchMaterializeSortPairs(indices, prepared, stream);

    const int cubItems = checkedCubItems(prepared.numTokens, "token count");
    sortPairs(prepared, cubItems, stream);

    // RLE writes only the first numRuns counts. Clear the full counts buffer so the full-capacity prefix scan below is deterministic.
    prepared.runCounts.memsetAsync(stream, 0);
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
            "Prepared Embedding sparse-gradient producer does not have a fused sparse-row update; use launchPreparedEmbeddingSparseGradient.");
    }

    ScopedGpu scopedGpu(prepared.deviceNum);

    launchMaterializeSortPairs(indices, prepared, stream);

    const int cubItems = checkedCubItems(prepared.numTokens, "token count");
    sortPairs(prepared, cubItems, stream);

    // RLE writes only the first numRuns counts. Clear the full counts buffer so the full-capacity prefix scan below is deterministic.
    prepared.runCounts.memsetAsync(stream, 0);
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
    result.sortTempBytes = prepared.sortTempBytes;
    result.rleTempBytes = prepared.rleTempBytes;
    result.scanTempBytes = prepared.scanTempBytes;

    Event totalStart(prepared.deviceNum, /*enableTiming=*/true);
    Event materializeEnd(prepared.deviceNum, /*enableTiming=*/true);
    Event sortEnd(prepared.deviceNum, /*enableTiming=*/true);
    Event clearCountsEnd(prepared.deviceNum, /*enableTiming=*/true);
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

    // RLE writes only the first numRuns counts. Clear the full counts buffer so the full-capacity prefix scan below is deterministic.
    prepared.runCounts.memsetAsync(stream, 0);
    clearCountsEnd.record(stream);

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
    result.clearRunCountsMs = clearCountsEnd.synchronizeAndReportElapsedTimeInMilliseconds(sortEnd);
    result.cubRleMs = rleEnd.synchronizeAndReportElapsedTimeInMilliseconds(clearCountsEnd);
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
    result.sortTempBytes = prepared.sortTempBytes;
    result.rleTempBytes = prepared.rleTempBytes;
    result.scanTempBytes = prepared.scanTempBytes;

    Event totalStart(prepared.deviceNum, /*enableTiming=*/true);
    Event materializeEnd(prepared.deviceNum, /*enableTiming=*/true);
    Event sortEnd(prepared.deviceNum, /*enableTiming=*/true);
    Event clearCountsEnd(prepared.deviceNum, /*enableTiming=*/true);
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

    // RLE writes only the first numRuns counts. Clear the full counts buffer so the full-capacity prefix scan below is deterministic.
    prepared.runCounts.memsetAsync(stream, 0);
    clearCountsEnd.record(stream);

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
    result.clearRunCountsMs = clearCountsEnd.synchronizeAndReportElapsedTimeInMilliseconds(sortEnd);
    result.cubRleMs = rleEnd.synchronizeAndReportElapsedTimeInMilliseconds(clearCountsEnd);
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
        throw std::invalid_argument("Prepared Embedding sparse-gradient graph capture with fused sparse-row update requires runtime scalar bindings.");
    }
    if (!fusedSparseRowUpdate && runtimeScalars != nullptr) {
        throw std::invalid_argument("Prepared Embedding sparse-gradient graph capture received runtime scalar bindings for a non-fused sparse-gradient producer.");
    }
    if (!captured.reduceNodeHandle.isInitialized()) {
        throw std::invalid_argument("Prepared Embedding sparse-gradient graph capture requires a preallocated reduce-node handle. Allocate CapturedEmbeddingSparseGradient before stream capture begins.");
    }
    if (captured.reduceNodeHandle.getGpuNum() != prepared.deviceNum) {
        throw std::invalid_argument("Prepared Embedding sparse-gradient graph capture reduce-node handle must live on the prepared GPU.");
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

    // RLE writes only the first numRuns counts. Clear the full counts buffer so the full-capacity prefix scan below is deterministic.
    prepared.runCounts.memsetAsync(stream, 0);
    rleRows(prepared, outputGradient, cubItems, stream);

    const uint32_t reduceRowsPerGridX = reduceRowsPerGridXForEmbeddingSparseGradient(prepared.embeddingDim);
    const uint32_t reduceGridDimY = reduceGridDimYForEmbeddingSparseGradient(prepared.embeddingDim);
    if (reduceRowsPerGridX == 0 || reduceGridDimY == 0) {
        throw std::runtime_error("Prepared Embedding sparse-gradient graph reducer requires a non-empty embedding dimension.");
    }

    launchFinalizeRows(prepared,
                       outputGradient,
                       stream,
                       OptionalReduceGridUpdate{&captured.reduceNodeHandle,
                                                reduceRowsPerGridX,
                                                reduceGridDimY,
                                                /*minReduceGridDimX=*/1,
                                                reduceGridDimXForEmbeddingSparseGradient(outputGradient.capacity, prepared.embeddingDim),
                                                reduceGridDimY});

    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(prepared.scanTempStorage.getMemPtr<void>(),
                                             prepared.scanTempBytes,
                                             prepared.runCounts.getMemPtr<uint32_t>(),
                                             prepared.runOffsets.getMemPtr<uint32_t>(),
                                             cubItems,
                                             stream.getStream()));

    const void* sortedTokenIdsPtr = prepared.sortedTokenIds.getMemPtr();
    const void* runOffsetsPtr = prepared.runOffsets.getMemPtr();
    const void* runCountsPtr = prepared.runCounts.getMemPtr();
    const void* numRowsPtr = outputGradient.numRows.getMemPtr();
    const void* upstreamPtr = upstreamGradient.getMemPtr();

    std::vector<void*> args;
    args.reserve(6);
    args.push_back((void*)&sortedTokenIdsPtr);
    args.push_back((void*)&runOffsetsPtr);
    args.push_back((void*)&runCountsPtr);
    args.push_back((void*)&numRowsPtr);
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

    captured.reduceNode = builder.captureDeviceUpdatableKernel(
        CudaGraphKernelLaunch{prepared.reduceKernel, dim3(1, reduceGridDimY, 1), dim3(THREADS_PER_BLOCK, 1, 1), 0, args.data(), nullptr});
}

}  // namespace

void capturePreparedEmbeddingSparseGradient(CudaGraphCaptureBuilder& builder,
                                            PreparedEmbeddingSparseGradient& prepared,
                                            const Tensor& indices,
                                            const Tensor& upstreamGradient,
                                            SparseRowGradient& outputGradient,
                                            CapturedEmbeddingSparseGradient& captured) {
    capturePreparedEmbeddingSparseGradientImpl(builder, prepared, indices, upstreamGradient, outputGradient, nullptr, captured);
}

void capturePreparedEmbeddingSparseGradientWithSparseRowUpdate(
    CudaGraphCaptureBuilder& builder,
    PreparedEmbeddingSparseGradient& prepared,
    const Tensor& indices,
    const Tensor& upstreamGradient,
    SparseRowGradient& outputGradient,
    const std::unordered_map<std::string, float>& runtimeScalars,
    CapturedEmbeddingSparseGradient& captured) {
    capturePreparedEmbeddingSparseGradientImpl(builder, prepared, indices, upstreamGradient, outputGradient, &runtimeScalars, captured);
}

void launchEmbeddingSparseGradient(const Tensor& indices,
                                   const Tensor& upstreamGradient,
                                   SparseRowGradient& outputGradient,
                                   std::optional<uint64_t> paddingIndex,
                                   Stream stream) {
    auto prepared = prepareEmbeddingSparseGradient(indices, upstreamGradient, outputGradient, paddingIndex);
    launchPreparedEmbeddingSparseGradient(*prepared, indices, upstreamGradient, outputGradient, stream);
}

}  // namespace ThorImplementation
