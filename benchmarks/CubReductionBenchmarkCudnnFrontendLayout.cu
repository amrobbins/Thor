#include "benchmarks/CubReductionBenchmarkCudnnFrontendLayout.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ThorImplementation::CubReductionBenchmarking {
namespace {

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t BLOCK_ROWS = 8;

template <typename StorageT>
__global__ void packedNchwToNhwcKernel(const StorageT* __restrict__ source,
                                      StorageT* __restrict__ destination,
                                      uint64_t channels,
                                      uint64_t spatial) {
    // For each batch item, transpose the packed [C, H*W] matrix into
    // [H*W, C]. Reads and writes are both coalesced; +1 removes the classic
    // shared-memory bank conflict in the transposed access.
    __shared__ StorageT tile[TILE_DIM][TILE_DIM + 1];

    const uint64_t batch = blockIdx.z;
    const uint64_t spatial_in = static_cast<uint64_t>(blockIdx.x) * TILE_DIM + threadIdx.x;
    const uint64_t channel_in = static_cast<uint64_t>(blockIdx.y) * TILE_DIM + threadIdx.y;
    const uint64_t source_batch_base = batch * channels * spatial;

#pragma unroll
    for (uint32_t row = 0; row < TILE_DIM; row += BLOCK_ROWS) {
        const uint64_t channel = channel_in + row;
        if (channel < channels && spatial_in < spatial) {
            tile[threadIdx.y + row][threadIdx.x] =
                source[source_batch_base + channel * spatial + spatial_in];
        }
    }

    __syncthreads();

    const uint64_t channel_out = static_cast<uint64_t>(blockIdx.y) * TILE_DIM + threadIdx.x;
    const uint64_t spatial_out = static_cast<uint64_t>(blockIdx.x) * TILE_DIM + threadIdx.y;
    const uint64_t destination_batch_base = batch * spatial * channels;

#pragma unroll
    for (uint32_t row = 0; row < TILE_DIM; row += BLOCK_ROWS) {
        const uint64_t spatial_index = spatial_out + row;
        if (channel_out < channels && spatial_index < spatial) {
            destination[destination_batch_base + spatial_index * channels + channel_out] =
                tile[threadIdx.x][threadIdx.y + row];
        }
    }
}

void checkLaunch(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename StorageT>
void launchTyped(const void* source,
                 void* destination,
                 uint64_t n,
                 uint64_t c,
                 uint64_t spatial,
                 cudaStream_t stream) {
    if (n > 65535) {
        throw std::invalid_argument("cuDNN Frontend reduction benchmark NCHW->NHWC transpose requires N <= 65535.");
    }
    if (c > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) * TILE_DIM
        || spatial > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) * TILE_DIM) {
        throw std::invalid_argument("cuDNN Frontend reduction benchmark transpose grid exceeds CUDA grid range.");
    }

    const dim3 block(TILE_DIM, BLOCK_ROWS, 1);
    const dim3 grid(static_cast<uint32_t>((spatial + TILE_DIM - 1) / TILE_DIM),
                    static_cast<uint32_t>((c + TILE_DIM - 1) / TILE_DIM),
                    static_cast<uint32_t>(n));
    packedNchwToNhwcKernel<StorageT><<<grid, block, 0, stream>>>(
        static_cast<const StorageT*>(source), static_cast<StorageT*>(destination), c, spatial);
    checkLaunch(cudaGetLastError(), "Failed to launch benchmark NCHW->NHWC transpose");
}

}  // namespace

void launchPackedNchwToNhwc(const void* source,
                            void* destination,
                            uint32_t element_bytes,
                            uint64_t n,
                            uint64_t c,
                            uint64_t h,
                            uint64_t w,
                            cudaStream_t stream) {
    if (source == nullptr || destination == nullptr) {
        throw std::invalid_argument("cuDNN Frontend reduction benchmark transpose requires non-null buffers.");
    }
    if (n == 0 || c == 0 || h == 0 || w == 0) {
        throw std::invalid_argument("cuDNN Frontend reduction benchmark transpose requires non-zero dimensions.");
    }
    if (h > std::numeric_limits<uint64_t>::max() / w) {
        throw std::overflow_error("cuDNN Frontend reduction benchmark spatial extent overflow.");
    }
    const uint64_t spatial = h * w;

    switch (element_bytes) {
        case 2:
            launchTyped<uint16_t>(source, destination, n, c, spatial, stream);
            return;
        case 4:
            launchTyped<uint32_t>(source, destination, n, c, spatial, stream);
            return;
        default:
            throw std::invalid_argument("cuDNN Frontend reduction benchmark transpose supports 2- or 4-byte elements only.");
    }
}

}  // namespace ThorImplementation::CubReductionBenchmarking
