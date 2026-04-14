#include "Utilities/TensorOperations/GpuMatrixTranspose/gpuMatrixTranspose.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "Utilities/Expression/CudaHelpers.h"

namespace ThorImplementation {
namespace {

constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;

template <typename T, int N>
struct alignas(sizeof(T) * N) PackedValues {
    T v[N];
};

template <typename T>
__device__ inline float transposeToFloat(T v) {
    return static_cast<float>(v);
}

// template <>
// __device__ inline float transposeToFloat<float>(float v) {
//     return v;
// }

template <>
__device__ inline float transposeToFloat<__half>(__half v) {
    return __half2float(v);
}

template <>
__device__ inline float transposeToFloat<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <>
__device__ inline float transposeToFloat<__nv_fp8_e4m3>(__nv_fp8_e4m3 v) {
    return __half2float(__half(v));
}

template <>
__device__ inline float transposeToFloat<__nv_fp8_e5m2>(__nv_fp8_e5m2 v) {
    return __half2float(__half(v));
}

template <typename T>
__device__ inline T transposeFromFloat(float v) {
    return static_cast<T>(v);
}

// template <>
// __device__ inline float transposeFromFloat<float>(float v) {
//     return v;
// }

template <>
__device__ inline __half transposeFromFloat<__half>(float v) {
    return __float2half_rn(v);
}

template <>
__device__ inline __nv_bfloat16 transposeFromFloat<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

template <>
__device__ inline __nv_fp8_e4m3 transposeFromFloat<__nv_fp8_e4m3>(float v) {
    return __nv_fp8_e4m3(__float2half_rn(v));
}

template <>
__device__ inline __nv_fp8_e5m2 transposeFromFloat<__nv_fp8_e5m2>(float v) {
    return __nv_fp8_e5m2(__float2half_rn(v));
}

template <typename OutT, typename InT>
__device__ inline OutT transposeConvert(InT v) {
    if constexpr (std::is_same_v<OutT, InT>) {
        return v;
    } else if constexpr (std::is_same_v<OutT, float>) {
        return transposeToFloat(v);
    } else if constexpr (std::is_same_v<InT, float>) {
        return transposeFromFloat<OutT>(v);
    } else if constexpr (std::is_same_v<OutT, __half> || std::is_same_v<OutT, __nv_bfloat16> || std::is_same_v<OutT, __nv_fp8_e4m3> ||
                         std::is_same_v<OutT, __nv_fp8_e5m2> || std::is_same_v<InT, __half> || std::is_same_v<InT, __nv_bfloat16> ||
                         std::is_same_v<InT, __nv_fp8_e4m3> || std::is_same_v<InT, __nv_fp8_e5m2>) {
        return transposeFromFloat<OutT>(transposeToFloat(v));
    } else {
        return static_cast<OutT>(v);
    }
}

template <typename T, int N>
__device__ inline uint32_t packToU32(const PackedValues<T, N>& values) {
    union U {
        uint32_t raw;
        PackedValues<T, N> packed;
    } u{};
    u.raw = 0;
    u.packed = values;
    return u.raw;
}

template <typename T, int N>
__device__ inline PackedValues<T, N> unpackFromU32(uint32_t raw) {
    union U {
        uint32_t raw;
        PackedValues<T, N> packed;
    } u{};
    u.raw = raw;
    return u.packed;
}

template <typename T, int N>
__device__ inline bool packedPointerIsAligned(const T* ptr) {
    constexpr uintptr_t kAlignment = alignof(PackedValues<T, N>);
    return (reinterpret_cast<uintptr_t>(ptr) & (kAlignment - 1)) == 0;
}

template <typename T, int N>
__device__ inline PackedValues<T, N> loadPackedValues(const T* ptr) {
    PackedValues<T, N> values{};
    if constexpr (sizeof(T) * N == 4 || sizeof(T) * N == 2 || sizeof(T) * N == 1) {
        if (packedPointerIsAligned<T, N>(ptr)) {
            values = *reinterpret_cast<const PackedValues<T, N>*>(ptr);
        } else {
#pragma unroll
            for (int i = 0; i < N; ++i) {
                values.v[i] = ptr[i];
            }
        }
    } else {
#pragma unroll
        for (int i = 0; i < N; ++i) {
            values.v[i] = ptr[i];
        }
    }
    return values;
}

template <typename T, int N>
__device__ inline void storePackedValues(T* ptr, const PackedValues<T, N>& values) {
    if constexpr (sizeof(T) * N == 4 || sizeof(T) * N == 2 || sizeof(T) * N == 1) {
        if (packedPointerIsAligned<T, N>(ptr)) {
            *reinterpret_cast<PackedValues<T, N>*>(ptr) = values;
        } else {
#pragma unroll
            for (int i = 0; i < N; ++i) {
                ptr[i] = values.v[i];
            }
        }
    } else {
#pragma unroll
        for (int i = 0; i < N; ++i) {
            ptr[i] = values.v[i];
        }
    }
}

template <typename InT, typename OutT, int PACK>
__global__ void matrixTransposeCastKernel(OutT* transposedMatrix, const InT* matrix, int numRows, int numCols) {
    __shared__ uint32_t tile[PACK * TILE_DIM][TILE_DIM + 1];

    constexpr int TILE_SCALARS = PACK * TILE_DIM;
    constexpr int LOAD_ITERS = TILE_SCALARS / BLOCK_ROWS;
    constexpr int STORE_ITERS = TILE_DIM / BLOCK_ROWS;

    const int warpRow = static_cast<int>(threadIdx.y);
    const int lane = static_cast<int>(threadIdx.x);

    const int packedInputCols = (numCols + PACK - 1) / PACK;
    const int fullPackedInputCols = numCols / PACK;
    const int packedOutputCols = (numRows + PACK - 1) / PACK;
    const int fullPackedOutputCols = numRows / PACK;

    const int packedCol = lane + static_cast<int>(blockIdx.x) * TILE_DIM;
    const int matrixRowStart = static_cast<int>(blockIdx.y) * TILE_SCALARS;

    const int outputPackedCol = lane + static_cast<int>(blockIdx.y) * TILE_DIM;
    const int outputRowStart = static_cast<int>(blockIdx.x) * TILE_SCALARS;

    const int packedColBase = static_cast<int>(blockIdx.x) * TILE_DIM;
    const int outputPackedColBase = static_cast<int>(blockIdx.y) * TILE_DIM;
    const bool warpHasFullInputCols = packedColBase + TILE_DIM <= fullPackedInputCols;
    const bool warpHasFullOutputCols = outputPackedColBase + TILE_DIM <= fullPackedOutputCols;

    const int lastLoadRow = matrixRowStart + warpRow + (LOAD_ITERS - 1) * BLOCK_ROWS;
    const bool warpHasFullInputRows = lastLoadRow < numRows;

    const int firstStoreRow = outputRowStart + PACK * warpRow;
    const int lastStoreRow = firstStoreRow + (STORE_ITERS - 1) * PACK * BLOCK_ROWS + (PACK - 1);
    const bool warpHasFullOutputRows = lastStoreRow < numCols;

    const uintptr_t firstWarpLoadAddr =
        reinterpret_cast<uintptr_t>(matrix) +
        (static_cast<uintptr_t>(matrixRowStart + warpRow) * static_cast<uintptr_t>(numCols) + static_cast<uintptr_t>(packedCol) * PACK) *
            sizeof(InT);
    const bool warpHasPackedLoadAlignment = (firstWarpLoadAddr & (alignof(PackedValues<InT, PACK>) - 1)) == 0;

    const size_t outputRowStrideBytes = static_cast<size_t>(numRows) * sizeof(OutT);
    const uintptr_t firstWarpStoreAddr =
        reinterpret_cast<uintptr_t>(transposedMatrix) +
        (static_cast<uintptr_t>(firstStoreRow) * static_cast<uintptr_t>(numRows) + static_cast<uintptr_t>(outputPackedCol) * PACK) *
            sizeof(OutT);
    const bool warpHasPackedStoreAlignment = (outputRowStrideBytes % alignof(PackedValues<OutT, PACK>) == 0) &&
                                             ((firstWarpStoreAddr & (alignof(PackedValues<OutT, PACK>) - 1)) == 0);

    const bool warpOnEdge = !(warpHasFullInputCols && warpHasFullOutputCols && warpHasFullInputRows && warpHasFullOutputRows);
    const bool warpCanUseFastPath = !warpOnEdge && warpHasPackedLoadAlignment && warpHasPackedStoreAlignment;

    int tileRow = warpRow;
    if (warpCanUseFastPath) {
#pragma unroll
        for (int iter = 0; iter < LOAD_ITERS; ++iter) {
            const int matrixRow = matrixRowStart + warpRow + iter * BLOCK_ROWS;
            const InT* src = matrix + static_cast<size_t>(matrixRow) * static_cast<size_t>(numCols) + static_cast<size_t>(packedCol) * PACK;
            tile[tileRow][lane] = packToU32(loadPackedValues<InT, PACK>(src));
            tileRow += BLOCK_ROWS;
        }
    } else {
#pragma unroll
        for (int iter = 0; iter < LOAD_ITERS; ++iter) {
            const int matrixRow = matrixRowStart + warpRow + iter * BLOCK_ROWS;
            uint32_t raw = 0;
            if (matrixRow < numRows && packedCol < packedInputCols) {
                PackedValues<InT, PACK> values{};
                const int remaining = numCols - packedCol * PACK;
                const int validLanes = remaining > 0 ? ((remaining < PACK) ? remaining : PACK) : 0;
                const InT* src =
                    matrix + static_cast<size_t>(matrixRow) * static_cast<size_t>(numCols) + static_cast<size_t>(packedCol) * PACK;
#pragma unroll
                for (int i = 0; i < PACK; ++i) {
                    if (i < validLanes) {
                        values.v[i] = src[i];
                    }
                }
                raw = packToU32(values);
            }
            tile[tileRow][lane] = raw;
            tileRow += BLOCK_ROWS;
        }
    }

    __syncthreads();

    int tileCol = warpRow;
    if (warpCanUseFastPath) {
#pragma unroll
        for (int iter = 0; iter < STORE_ITERS; ++iter) {
            const int matrixRow = outputRowStart + PACK * warpRow + iter * PACK * BLOCK_ROWS;
#pragma unroll
            for (int rowLane = 0; rowLane < PACK; ++rowLane) {
                PackedValues<OutT, PACK> outputValues{};
#pragma unroll
                for (int colLane = 0; colLane < PACK; ++colLane) {
                    const PackedValues<InT, PACK> inputValues = unpackFromU32<InT, PACK>(tile[PACK * lane + colLane][tileCol]);
                    outputValues.v[colLane] = transposeConvert<OutT>(inputValues.v[rowLane]);
                }
                storePackedValues<OutT, PACK>(transposedMatrix + static_cast<size_t>(matrixRow + rowLane) * static_cast<size_t>(numRows) +
                                                  static_cast<size_t>(outputPackedCol) * PACK,
                                              outputValues);
            }
            tileCol += BLOCK_ROWS;
        }
    } else {
#pragma unroll
        for (int iter = 0; iter < STORE_ITERS; ++iter) {
            const int matrixRow = outputRowStart + PACK * warpRow + iter * PACK * BLOCK_ROWS;
            const int remainingCols = numRows - outputPackedCol * PACK;
            const int validOutputLanes =
                (outputPackedCol < packedOutputCols && remainingCols > 0) ? ((remainingCols < PACK) ? remainingCols : PACK) : 0;
#pragma unroll
            for (int rowLane = 0; rowLane < PACK; ++rowLane) {
                const int outputRow = matrixRow + rowLane;
                if (outputRow < numCols && validOutputLanes > 0) {
#pragma unroll
                    for (int colLane = 0; colLane < PACK; ++colLane) {
                        const PackedValues<InT, PACK> inputValues = unpackFromU32<InT, PACK>(tile[PACK * lane + colLane][tileCol]);
                        if (colLane < validOutputLanes) {
                            transposedMatrix[static_cast<size_t>(outputRow) * static_cast<size_t>(numRows) +
                                             static_cast<size_t>(outputPackedCol) * PACK + colLane] =
                                transposeConvert<OutT>(inputValues.v[rowLane]);
                        }
                    }
                }
            }
            tileCol += BLOCK_ROWS;
        }
    }
}

template <typename InT, typename OutT, int PACK>
void launchTypedMatrixTranspose(void* output, const void* input, uint32_t numRows, uint32_t numCols, cudaStream_t stream) {
    constexpr dim3 blockSize(TILE_DIM, BLOCK_ROWS);
    constexpr uint32_t tileScalars = PACK * TILE_DIM;
    const dim3 gridSize((numCols + tileScalars - 1u) / tileScalars, (numRows + tileScalars - 1u) / tileScalars);
    matrixTransposeCastKernel<InT, OutT, PACK><<<gridSize, blockSize, 0, stream>>>(
        static_cast<OutT*>(output), static_cast<const InT*>(input), static_cast<int>(numRows), static_cast<int>(numCols));
    CUDA_CHECK(cudaGetLastError());
}

static inline size_t scalarSizeBytes(TensorDescriptor::DataType dtype) {
    switch (dtype) {
        case TensorDescriptor::DataType::FP32:
            return 4;
        case TensorDescriptor::DataType::FP16:
            return 2;
        case TensorDescriptor::DataType::BF16:
            return 2;
        case TensorDescriptor::DataType::FP8_E4M3:
            return 1;
        case TensorDescriptor::DataType::FP8_E5M2:
            return 1;
        case TensorDescriptor::DataType::UINT8:
            return 1;
        case TensorDescriptor::DataType::UINT16:
            return 2;
        case TensorDescriptor::DataType::UINT32:
            return 4;
        case TensorDescriptor::DataType::INT32:
            return 4;
        default:
            throw std::runtime_error("Unsupported dtype in transpose dispatch.");
    }
}

static inline int chooseTransposePack(uint32_t /*numRows*/,
                                      uint32_t /*numCols*/,
                                      TensorDescriptor::DataType input_dtype,
                                      TensorDescriptor::DataType output_dtype) {
    const size_t larger_bytes = std::max(scalarSizeBytes(input_dtype), scalarSizeBytes(output_dtype));
    return static_cast<int>(4 / larger_bytes);
}

template <typename InT, int PACK>
void dispatchOutputType(
    void* output, const void* input, uint32_t numRows, uint32_t numCols, TensorDescriptor::DataType output_dtype, cudaStream_t stream) {
    switch (output_dtype) {
        case TensorDescriptor::DataType::FP32:
            launchTypedMatrixTranspose<InT, float, PACK>(output, input, numRows, numCols, stream);
            return;
        case TensorDescriptor::DataType::FP16:
            launchTypedMatrixTranspose<InT, __half, PACK>(output, input, numRows, numCols, stream);
            return;
        case TensorDescriptor::DataType::BF16:
            launchTypedMatrixTranspose<InT, __nv_bfloat16, PACK>(output, input, numRows, numCols, stream);
            return;
        case TensorDescriptor::DataType::FP8_E4M3:
            launchTypedMatrixTranspose<InT, __nv_fp8_e4m3, PACK>(output, input, numRows, numCols, stream);
            return;
        case TensorDescriptor::DataType::FP8_E5M2:
            launchTypedMatrixTranspose<InT, __nv_fp8_e5m2, PACK>(output, input, numRows, numCols, stream);
            return;
        case TensorDescriptor::DataType::UINT8:
            launchTypedMatrixTranspose<InT, uint8_t, PACK>(output, input, numRows, numCols, stream);
            return;
        case TensorDescriptor::DataType::UINT16:
            launchTypedMatrixTranspose<InT, uint16_t, PACK>(output, input, numRows, numCols, stream);
            return;
        case TensorDescriptor::DataType::UINT32:
            launchTypedMatrixTranspose<InT, uint32_t, PACK>(output, input, numRows, numCols, stream);
            return;
        case TensorDescriptor::DataType::INT32:
            launchTypedMatrixTranspose<InT, int32_t, PACK>(output, input, numRows, numCols, stream);
            return;
        default:
            throw std::runtime_error("Unsupported transpose output dtype.");
    }
}

template <typename InT>
void dispatchPackAndOutputType(void* output,
                               const void* input,
                               uint32_t numRows,
                               uint32_t numCols,
                               TensorDescriptor::DataType input_dtype,
                               TensorDescriptor::DataType output_dtype,
                               cudaStream_t stream) {
    switch (chooseTransposePack(numRows, numCols, input_dtype, output_dtype)) {
        case 4:
            dispatchOutputType<InT, 4>(output, input, numRows, numCols, output_dtype, stream);
            return;
        case 2:
            dispatchOutputType<InT, 2>(output, input, numRows, numCols, output_dtype, stream);
            return;
        default:
            dispatchOutputType<InT, 1>(output, input, numRows, numCols, output_dtype, stream);
            return;
    }
}

}  // namespace

void launchMatrixTransposeByType(void* output,
                                 const void* input,
                                 uint32_t numRows,
                                 uint32_t numCols,
                                 TensorDescriptor::DataType input_dtype,
                                 TensorDescriptor::DataType output_dtype,
                                 cudaStream_t stream) {
    if (output == nullptr || input == nullptr) {
        throw std::runtime_error("launchMatrixTransposeByType received a null pointer.");
    }
    if (numRows == 0 || numCols == 0) {
        return;
    }

    switch (input_dtype) {
        case TensorDescriptor::DataType::FP32:
            dispatchPackAndOutputType<float>(output, input, numRows, numCols, input_dtype, output_dtype, stream);
            return;
        case TensorDescriptor::DataType::FP16:
            dispatchPackAndOutputType<__half>(output, input, numRows, numCols, input_dtype, output_dtype, stream);
            return;
        case TensorDescriptor::DataType::BF16:
            dispatchPackAndOutputType<__nv_bfloat16>(output, input, numRows, numCols, input_dtype, output_dtype, stream);
            return;
        case TensorDescriptor::DataType::FP8_E4M3:
            dispatchPackAndOutputType<__nv_fp8_e4m3>(output, input, numRows, numCols, input_dtype, output_dtype, stream);
            return;
        case TensorDescriptor::DataType::FP8_E5M2:
            dispatchPackAndOutputType<__nv_fp8_e5m2>(output, input, numRows, numCols, input_dtype, output_dtype, stream);
            return;
        case TensorDescriptor::DataType::UINT8:
            dispatchPackAndOutputType<uint8_t>(output, input, numRows, numCols, input_dtype, output_dtype, stream);
            return;
        case TensorDescriptor::DataType::UINT16:
            dispatchPackAndOutputType<uint16_t>(output, input, numRows, numCols, input_dtype, output_dtype, stream);
            return;
        case TensorDescriptor::DataType::UINT32:
            dispatchPackAndOutputType<uint32_t>(output, input, numRows, numCols, input_dtype, output_dtype, stream);
            return;
        case TensorDescriptor::DataType::INT32:
            dispatchPackAndOutputType<int32_t>(output, input, numRows, numCols, input_dtype, output_dtype, stream);
            return;
        default:
            throw std::runtime_error("Unsupported transpose input dtype.");
    }
}

void matrixTranspose(float* transposedMatrix_d, const float* matrix_d, int numRows, int numCols, cudaStream_t stream) {
    launchMatrixTransposeByType(transposedMatrix_d,
                                matrix_d,
                                static_cast<uint32_t>(numRows),
                                static_cast<uint32_t>(numCols),
                                TensorDescriptor::DataType::FP32,
                                TensorDescriptor::DataType::FP32,
                                stream);
}

void matrixTranspose(__half* transposedMatrix_d, const __half* matrix_d, int numRows, int numCols, cudaStream_t stream) {
    launchMatrixTransposeByType(transposedMatrix_d,
                                matrix_d,
                                static_cast<uint32_t>(numRows),
                                static_cast<uint32_t>(numCols),
                                TensorDescriptor::DataType::FP16,
                                TensorDescriptor::DataType::FP16,
                                stream);
}

void matrixTranspose(__nv_bfloat16* transposedMatrix_d, const __nv_bfloat16* matrix_d, int numRows, int numCols, cudaStream_t stream) {
    launchMatrixTransposeByType(transposedMatrix_d,
                                matrix_d,
                                static_cast<uint32_t>(numRows),
                                static_cast<uint32_t>(numCols),
                                TensorDescriptor::DataType::BF16,
                                TensorDescriptor::DataType::BF16,
                                stream);
}

void matrixTranspose(__nv_fp8_e4m3* transposedMatrix_d, const __nv_fp8_e4m3* matrix_d, int numRows, int numCols, cudaStream_t stream) {
    launchMatrixTransposeByType(transposedMatrix_d,
                                matrix_d,
                                static_cast<uint32_t>(numRows),
                                static_cast<uint32_t>(numCols),
                                TensorDescriptor::DataType::FP8_E4M3,
                                TensorDescriptor::DataType::FP8_E4M3,
                                stream);
}

void matrixTranspose(__nv_fp8_e5m2* transposedMatrix_d, const __nv_fp8_e5m2* matrix_d, int numRows, int numCols, cudaStream_t stream) {
    launchMatrixTransposeByType(transposedMatrix_d,
                                matrix_d,
                                static_cast<uint32_t>(numRows),
                                static_cast<uint32_t>(numCols),
                                TensorDescriptor::DataType::FP8_E5M2,
                                TensorDescriptor::DataType::FP8_E5M2,
                                stream);
}

}  // namespace ThorImplementation
