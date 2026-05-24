#include "Utilities/TensorOperations/Embedding/EmbeddingSparseGradient.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/CudaDriver/CudaGraphDynamicGrid.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/Embedding/EmbeddingSparseGradientCudaCompile.h"

#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace ThorImplementation {
namespace {

using DataType = TensorDescriptor::DataType;

constexpr uint32_t THREADS_PER_BLOCK = 256;
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

template <typename RowT>
__global__ void finalizeEmbeddingSparseGradientRowsKernel(const RowT* __restrict__ uniqueRows,
                                                          const uint32_t* __restrict__ numRuns,
                                                          RowT* __restrict__ outputRows,
                                                          RowT* __restrict__ outputNumRows,
                                                          uint64_t vocabularySize) {
    uint32_t validRuns = numRuns[0];
    if (validRuns != 0 && static_cast<uint64_t>(uniqueRows[validRuns - 1]) == vocabularySize) {
        validRuns -= 1;
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        outputNumRows[0] = static_cast<RowT>(validRuns);
    }

    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < static_cast<uint64_t>(validRuns);
         i += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
        outputRows[i] = uniqueRows[i];
    }
}

template <typename RowT, typename GradT>
__global__ void reduceEmbeddingSparseGradientValuesKernel(const uint32_t* __restrict__ sortedTokenIds,
                                                          const uint32_t* __restrict__ runOffsets,
                                                          const uint32_t* __restrict__ runCounts,
                                                          const RowT* __restrict__ numValidRows,
                                                          const GradT* __restrict__ upstreamGradient,
                                                          float* __restrict__ outputValues,
                                                          uint64_t embeddingDim) {
    const uint64_t row = blockIdx.x;
    if (row >= static_cast<uint64_t>(numValidRows[0])) {
        return;
    }

    const uint32_t lane = threadIdx.x & (WARP_SIZE_EMBEDDING - 1);
    const uint32_t warpInBlock = threadIdx.x >> 5;
    const uint64_t dim = static_cast<uint64_t>(blockIdx.y) * WARPS_PER_BLOCK + warpInBlock;
    if (dim >= embeddingDim) {
        return;
    }

    const uint64_t begin = static_cast<uint64_t>(runOffsets[row]);
    const uint64_t count = static_cast<uint64_t>(runCounts[row]);
    float sum = 0.0f;
    for (uint64_t i = lane; i < count; i += WARP_SIZE_EMBEDDING) {
        const uint64_t token = static_cast<uint64_t>(sortedTokenIds[begin + i]);
        sum += thor_embedding_grad_to_float(upstreamGradient[token * embeddingDim + dim]);
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }

    if (lane == 0) {
        outputValues[row * embeddingDim + dim] = sum;
    }
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

std::string sparseGradientReduceKernelName(DataType rowDType, DataType gradDType, uint64_t embeddingDim) {
    std::ostringstream ss;
    ss << "thor_embedding_sparse_reduce_r" << static_cast<uint64_t>(rowDType) << "_g" << static_cast<uint64_t>(gradDType) << "_d"
       << embeddingDim;
    return ss.str();
}

std::string emitSparseGradientReduceKernelSource(DataType rowDType, DataType gradDType, uint64_t embeddingDim, const std::string& kernelName) {
    const std::string rowType = sparseGradientScalarType(rowDType);
    const std::string gradType = sparseGradientScalarType(gradDType);

    std::ostringstream ss;
    ss << "#include <cuda_fp16.h>\n";
    ss << "#include <cuda_bf16.h>\n\n";
    ss << "__device__ __forceinline__ float thor_embedding_grad_to_float(float v) { return v; }\n";
    ss << "__device__ __forceinline__ float thor_embedding_grad_to_float(__half v) { return __half2float(v); }\n";
    ss << "__device__ __forceinline__ float thor_embedding_grad_to_float(__nv_bfloat16 v) { return __bfloat162float(v); }\n\n";
    ss << "extern \"C\" __global__\n";
    ss << "void " << kernelName << "(const unsigned int* sortedTokenIds, const unsigned int* runOffsets, const unsigned int* runCounts, ";
    ss << "const " << rowType << "* numValidRows, const " << gradType << "* upstreamGradient, float* outputValues) {\n";
    ss << "  const unsigned long long row = static_cast<unsigned long long>(blockIdx.x);\n";
    ss << "  if (row >= static_cast<unsigned long long>(numValidRows[0])) return;\n";
    ss << "  const unsigned int lane = threadIdx.x & 31U;\n";
    ss << "  const unsigned int warpInBlock = threadIdx.x >> 5;\n";
    ss << "  const unsigned long long embeddingDim = " << embeddingDim << "ULL;\n";
    ss << "  const unsigned long long dim = static_cast<unsigned long long>(blockIdx.y) * " << WARPS_PER_BLOCK << "ULL + warpInBlock;\n";
    ss << "  if (dim >= embeddingDim) return;\n";
    ss << "  const unsigned long long begin = static_cast<unsigned long long>(runOffsets[row]);\n";
    ss << "  const unsigned long long count = static_cast<unsigned long long>(runCounts[row]);\n";
    ss << "  float sum = 0.0f;\n";
    ss << "  for (unsigned long long i = lane; i < count; i += 32ULL) {\n";
    ss << "    const unsigned long long token = static_cast<unsigned long long>(sortedTokenIds[begin + i]);\n";
    ss << "    sum += thor_embedding_grad_to_float(upstreamGradient[token * embeddingDim + dim]);\n";
    ss << "  }\n";
    ss << "#pragma unroll\n";
    ss << "  for (int offset = 16; offset > 0; offset >>= 1) {\n";
    ss << "    sum += __shfl_down_sync(0xffffffffu, sum, offset);\n";
    ss << "  }\n";
    ss << "  if (lane == 0U) outputValues[row * embeddingDim + dim] = sum;\n";
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
    Tensor uniqueRows;
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
};

PreparedEmbeddingSparseGradient::~PreparedEmbeddingSparseGradient() {
    if (reduceModule != nullptr) {
        try {
            CU_CHECK(cuModuleUnload(reduceModule));
        } catch (...) {
        }
    }
}

void compileGraphReduceKernel(PreparedEmbeddingSparseGradient& prepared) {
    if (prepared.reduceKernel != nullptr) {
        return;
    }

    prepared.reduceKernelName = sparseGradientReduceKernelName(prepared.rowDataType, prepared.gradientDataType, prepared.embeddingDim);
    const std::string src = emitSparseGradientReduceKernelSource(prepared.rowDataType,
                                                                 prepared.gradientDataType,
                                                                 prepared.embeddingDim,
                                                                 prepared.reduceKernelName);
    CUDA_CHECK(cudaFree(nullptr));
    CompiledEmbeddingSparseGradientCudaKernel compiled =
        compileEmbeddingSparseGradientCudaKernel(src, prepared.reduceKernelName, prepared.deviceNum);
    prepared.reduceModule = compiled.module;
    prepared.reduceKernel = compiled.function;
}

template <typename RowT>
void allocateTypedPreparedBuffers(PreparedEmbeddingSparseGradient& prepared, const TensorPlacement& placement, uint64_t numTokens, int cubItems) {
    if (!rowTypeCanRepresentVocabularySentinel<RowT>(prepared.vocabularySize)) {
        throw std::invalid_argument("Embedding sparse-gradient row dtype " + dtypeName(rowDTypeForCppType<RowT>()) +
                                    " cannot represent vocabulary_size as the invalid-row sentinel.");
    }

    prepared.rowKeys = Tensor(placement, TensorDescriptor(rowDTypeForCppType<RowT>(), {numTokens}));
    prepared.tokenIds = Tensor(placement, TensorDescriptor(DataType::UINT32, {numTokens}));
    prepared.sortedRowKeys = Tensor(placement, TensorDescriptor(rowDTypeForCppType<RowT>(), {numTokens}));
    prepared.sortedTokenIds = Tensor(placement, TensorDescriptor(DataType::UINT32, {numTokens}));
    prepared.uniqueRows = Tensor(placement, TensorDescriptor(rowDTypeForCppType<RowT>(), {numTokens}));
    prepared.runCounts = Tensor(placement, TensorDescriptor(DataType::UINT32, {numTokens}));
    prepared.runOffsets = Tensor(placement, TensorDescriptor(DataType::UINT32, {numTokens}));
    prepared.numRuns = Tensor(placement, TensorDescriptor(DataType::UINT32, {1}));

    prepared.sortTempBytes = querySortTempBytes(prepared.rowKeys.getMemPtr<RowT>(),
                                                prepared.sortedRowKeys.getMemPtr<RowT>(),
                                                prepared.tokenIds.getMemPtr<uint32_t>(),
                                                prepared.sortedTokenIds.getMemPtr<uint32_t>(),
                                                cubItems);
    prepared.rleTempBytes = queryRleTempBytes(prepared.sortedRowKeys.getMemPtr<RowT>(),
                                              prepared.uniqueRows.getMemPtr<RowT>(),
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

template <typename RowT, typename GradT>
void launchReduceValuesTyped(PreparedEmbeddingSparseGradient& prepared,
                             const Tensor& upstreamGradient,
                             SparseRowGradient& outputGradient,
                             Stream stream) {
    const uint64_t maxRows = outputGradient.capacity;
    if (maxRows == 0) {
        return;
    }
    const dim3 block(THREADS_PER_BLOCK);
    const dim3 grid(static_cast<uint32_t>(maxRows), static_cast<uint32_t>((prepared.embeddingDim + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK));
    reduceEmbeddingSparseGradientValuesKernel<RowT, GradT><<<grid, block, 0, stream.getStream()>>>(
        prepared.sortedTokenIds.getMemPtr<uint32_t>(),
        prepared.runOffsets.getMemPtr<uint32_t>(),
        prepared.runCounts.getMemPtr<uint32_t>(),
        outputGradient.numRows.getMemPtr<RowT>(),
        upstreamGradient.getMemPtr<GradT>(),
        outputGradient.values.getMemPtr<float>(),
        prepared.embeddingDim);
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

template <typename RowT>
void launchFinalizeRowsTyped(PreparedEmbeddingSparseGradient& prepared, SparseRowGradient& outputGradient, Stream stream) {
    const uint32_t finalizeBlock = THREADS_PER_BLOCK;
    const uint32_t finalizeGrid = static_cast<uint32_t>((outputGradient.capacity + finalizeBlock - 1) / finalizeBlock);
    finalizeEmbeddingSparseGradientRowsKernel<RowT><<<std::max<uint32_t>(finalizeGrid, 1u), finalizeBlock, 0, stream.getStream()>>>(
        prepared.uniqueRows.getMemPtr<RowT>(),
        prepared.numRuns.getMemPtr<uint32_t>(),
        outputGradient.rows.getMemPtr<RowT>(),
        outputGradient.numRows.getMemPtr<RowT>(),
        prepared.vocabularySize);
    CUDA_CHECK(cudaPeekAtLastError());
}

void launchFinalizeRows(PreparedEmbeddingSparseGradient& prepared, SparseRowGradient& outputGradient, Stream stream) {
    switch (prepared.rowDataType) {
        case DataType::UINT16:
            launchFinalizeRowsTyped<uint16_t>(prepared, outputGradient, stream);
            break;
        case DataType::UINT32:
            launchFinalizeRowsTyped<uint32_t>(prepared, outputGradient, stream);
            break;
        case DataType::UINT64:
            launchFinalizeRowsTyped<uint64_t>(prepared, outputGradient, stream);
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
void rleRowsTyped(PreparedEmbeddingSparseGradient& prepared, int cubItems, Stream stream) {
    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(prepared.rleTempStorage.getMemPtr<void>(),
                                                  prepared.rleTempBytes,
                                                  prepared.sortedRowKeys.getMemPtr<RowT>(),
                                                  prepared.uniqueRows.getMemPtr<RowT>(),
                                                  prepared.runCounts.getMemPtr<uint32_t>(),
                                                  prepared.numRuns.getMemPtr<uint32_t>(),
                                                  cubItems,
                                                  stream.getStream()));
}

void rleRows(PreparedEmbeddingSparseGradient& prepared, int cubItems, Stream stream) {
    switch (prepared.rowDataType) {
        case DataType::UINT16:
            rleRowsTyped<uint16_t>(prepared, cubItems, stream);
            break;
        case DataType::UINT32:
            rleRowsTyped<uint32_t>(prepared, cubItems, stream);
            break;
        case DataType::UINT64:
            rleRowsTyped<uint64_t>(prepared, cubItems, stream);
            break;
        default:
            throw std::runtime_error("Prepared Embedding sparse-gradient producer has unsupported row dtype.");
    }
}

std::shared_ptr<PreparedEmbeddingSparseGradient> prepareEmbeddingSparseGradient(const Tensor& indices,
                                                                               const Tensor& upstreamGradient,
                                                                               SparseRowGradient& outputGradient,
                                                                               std::optional<uint64_t> paddingIndex) {
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

    ScopedGpu scopedGpu(placement.getDeviceNum());
    switch (prepared->rowDataType) {
        case DataType::UINT16:
            allocateTypedPreparedBuffers<uint16_t>(*prepared, placement, numTokens, cubItems);
            break;
        case DataType::UINT32:
            allocateTypedPreparedBuffers<uint32_t>(*prepared, placement, numTokens, cubItems);
            break;
        case DataType::UINT64:
            allocateTypedPreparedBuffers<uint64_t>(*prepared, placement, numTokens, cubItems);
            break;
        default:
            throw std::runtime_error("Prepared Embedding sparse-gradient producer has unsupported row dtype.");
    }
    compileGraphReduceKernel(*prepared);

    return prepared;
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

    ScopedGpu scopedGpu(prepared.deviceNum);

    launchMaterializeSortPairs(indices, prepared, stream);

    const int cubItems = checkedCubItems(prepared.numTokens, "token count");
    sortPairs(prepared, cubItems, stream);

    // RLE writes only the first numRuns counts. Clear the full counts buffer so the full-capacity prefix scan below is deterministic.
    prepared.runCounts.memsetAsync(stream, 0);
    rleRows(prepared, cubItems, stream);

    launchFinalizeRows(prepared, outputGradient, stream);

    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(prepared.scanTempStorage.getMemPtr<void>(),
                                             prepared.scanTempBytes,
                                             prepared.runCounts.getMemPtr<uint32_t>(),
                                             prepared.runOffsets.getMemPtr<uint32_t>(),
                                             cubItems,
                                             stream.getStream()));

    launchReduceValues(prepared, upstreamGradient, outputGradient, stream);
}

void capturePreparedEmbeddingSparseGradient(CudaGraphCaptureBuilder& builder,
                                            PreparedEmbeddingSparseGradient& prepared,
                                            const Tensor& indices,
                                            const Tensor& upstreamGradient,
                                            SparseRowGradient& outputGradient,
                                            CapturedEmbeddingSparseGradient& captured) {
    validatePreparedEmbeddingSparseGradientInvocation(prepared, indices, upstreamGradient, outputGradient);
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
    rleRows(prepared, cubItems, stream);

    launchFinalizeRows(prepared, outputGradient, stream);

    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(prepared.scanTempStorage.getMemPtr<void>(),
                                             prepared.scanTempBytes,
                                             prepared.runCounts.getMemPtr<uint32_t>(),
                                             prepared.runOffsets.getMemPtr<uint32_t>(),
                                             cubItems,
                                             stream.getStream()));

    const uint32_t dimTiles = static_cast<uint32_t>((prepared.embeddingDim + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    if (dimTiles == 0) {
        throw std::runtime_error("Prepared Embedding sparse-gradient graph reducer requires a non-empty embedding dimension.");
    }

    launchUpdateDeviceGrid2DFromScalar(DynamicGrid2DFromScalarDescriptor{&captured.reduceNodeHandle,
                                                                         outputGradient.numRows,
                                                                         dimTiles,
                                                                         /*gridDimXPerRow=*/1,
                                                                         /*minGridDimX=*/1,
                                                                         static_cast<uint32_t>(outputGradient.capacity),
                                                                         dimTiles},
                                       stream);

    const void* sortedTokenIdsPtr = prepared.sortedTokenIds.getMemPtr();
    const void* runOffsetsPtr = prepared.runOffsets.getMemPtr();
    const void* runCountsPtr = prepared.runCounts.getMemPtr();
    const void* numRowsPtr = outputGradient.numRows.getMemPtr();
    const void* upstreamPtr = upstreamGradient.getMemPtr();
    void* outputValuesPtr = outputGradient.values.getMemPtr();
    void* args[] = {&sortedTokenIdsPtr, &runOffsetsPtr, &runCountsPtr, &numRowsPtr, &upstreamPtr, &outputValuesPtr};

    captured.reduceNode = builder.captureDeviceUpdatableKernel(
        CudaGraphKernelLaunch{prepared.reduceKernel, dim3(1, dimTiles, 1), dim3(THREADS_PER_BLOCK, 1, 1), 0, args, nullptr});
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
