// 1388: Thor's dense copy of a strided span (StridedCopy.h; the plan in DenseCopyPlan.h; 1389: the transpose's loads in
// flight, rectangular tiles and any number of batch axes, after thor_dense_copy_audit; 1389b: PackedGather). Three
// kernels over the word width (1, 2, 4, 8, 16 bytes): RowCopy, in a 32-bit and a 64-bit index form, PackedGather
// (words of 1, 2 or 4 bytes) and Transpose. They copy words, so they are independent of the dtype. Through 1387 any
// span that was neither dense nor one pitch of dense rows ran a gather of one element per thread with a 64-bit
// division and modulo per axis per element. 1389e: a MatrixTranspose plan (a plain rotation of a dense matrix)
// launches the matrix transpose kernel of gpuMatrixTransposeKernels.cu on unsigned words (1394: moved here). 1389g:
// the transpose takes its tiles in either order (DenseCopyPlan::tile_v_first), for the audit's probe; the planner's
// plans are as before.
// 1389h: a launch is one CTA per unit of work (DenseCopyInternal::gridOf). Through 1389g it was the CTAs resident at
// once, each looping over its share of the units; the kernels' loops over the units are kept, so they still run on
// any smaller grid (the probe's and the tests'). 1389j: transposes of single bytes take denseTransposeBytesKernel,
// four bytes at a time. 1389k: the matrix transpose kernel for words of 2 or 4 bytes only. 1389n: the row copy took
// its words per thread from the plan (row_items), so the audit's probe could vary them; 1395 compiles the planner's
// counts only.
// 1389o: the planner's row copy is one 4-byte word or two 2-byte words a thread and step; the transpose's loads in
// flight are as before (transposeLoadItems). 1389p: that short step by size and stride only (rowCopyStepItems).

#include "Utilities/TensorOperations/Copy/StridedCopy.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/Cub/FastDivU32.h"

namespace ThorImplementation {
namespace {

using CubReductionInternal::FastDivU32;
using DenseCopyInternal::MAX_RANK;
using DenseCopyInternal::THREADS;

template <uint32_t Bytes>
struct CopyWord;
template <>
struct CopyWord<1> {
    using Type = uint8_t;
};
template <>
struct CopyWord<2> {
    using Type = uint16_t;
};
template <>
struct CopyWord<4> {
    using Type = uint32_t;
};
template <>
struct CopyWord<8> {
    using Type = uint2;
};
template <>
struct CopyWord<16> {
    using Type = uint4;
};

// RowCopy: destination word j of the row-major [outer..., p] is source word e plus sum c_k stride_k, the outer axes
// held innermost first. The kernel takes them as a __grid_constant__ parameter, so the loop over the axes reads them
// in place (no local copy of the arrays) without unrolling 32 axes into every item.
struct RowCopyArgs {
    uint64_t words = 0, p = 1;
    uint32_t outer_rank = 0;
    uint64_t dims[MAX_RANK]{};     // innermost outer axis first
    uint64_t strides[MAX_RANK]{};  // in words, likewise
    FastDivU32 p_div;
    FastDivU32 dim_divs[MAX_RANK];
};

template <bool Fast>
__device__ __forceinline__ uint64_t rowCopySourceWord(const RowCopyArgs& a, uint64_t j) {
    if constexpr (Fast) {
        uint32_t rest = a.p_div.divide(static_cast<uint32_t>(j));
        uint64_t offset = j - static_cast<uint64_t>(rest) * a.p;
#pragma unroll 1
        for (uint32_t i = 0; i + 1 < a.outer_rank; ++i) {
            const uint32_t q = a.dim_divs[i].divide(rest);
            offset += static_cast<uint64_t>(rest - q * static_cast<uint32_t>(a.dims[i])) * a.strides[i];
            rest = q;
        }
        return offset + static_cast<uint64_t>(rest) * a.strides[a.outer_rank - 1];
    } else {
        uint64_t rest = j / a.p;
        uint64_t offset = j - rest * a.p;
#pragma unroll 1
        for (uint32_t i = 0; i + 1 < a.outer_rank; ++i) {
            const uint64_t q = rest / a.dims[i];
            offset += (rest - q * a.dims[i]) * a.strides[i];
            rest = q;
        }
        return offset + rest * a.strides[a.outer_rank - 1];
    }
}

// transposeLoadItems is a host constexpr function (DenseCopyPlan.h is CUDA-free), so the transpose kernel reads it
// through a namespace-scope constexpr variable, evaluated on the host side.
template <typename Word>
constexpr uint32_t TRANSPOSE_LOAD_ITEMS = DenseCopyInternal::transposeLoadItems(sizeof(Word));

// The destination in order, so writes are coalesced; each thread loads all of its words before storing any.
// 1389n: ITEMS words a thread per step (the plan's rowCopyStepItems; the planner's is rowCopyItems of the word).
template <typename Word, bool Fast, uint32_t ITEMS>
__global__ __launch_bounds__(THREADS) void denseRowCopyKernel(const Word* __restrict__ source,
                                                              Word* __restrict__ destination,
                                                              const __grid_constant__ RowCopyArgs a) {
    constexpr uint64_t STEP_WORDS = uint64_t{THREADS} * ITEMS;
    const uint64_t step = static_cast<uint64_t>(gridDim.x) * STEP_WORDS;
    for (uint64_t base = static_cast<uint64_t>(blockIdx.x) * STEP_WORDS; base < a.words; base += step) {
        Word words[ITEMS];
#pragma unroll
        for (uint32_t k = 0; k < ITEMS; ++k) {
            const uint64_t j = base + k * THREADS + threadIdx.x;
            if (j < a.words) words[k] = source[rowCopySourceWord<Fast>(a, j)];
        }
#pragma unroll
        for (uint32_t k = 0; k < ITEMS; ++k) {
            const uint64_t j = base + k * THREADS + threadIdx.x;
            if (j < a.words) destination[j] = words[k];
        }
    }
}

// PackedGather (1389b): the source word of destination word j of single words (p = 1) over two or more outer axes,
// with its coordinates on the two innermost outer axes. Indices below 2^31 (the planner checks).
__device__ __forceinline__ uint64_t packedGatherSourceWord(const RowCopyArgs& a, uint32_t j, uint32_t& c0,
                                                           uint32_t& c1) {
    uint32_t rest = a.dim_divs[0].divide(j);
    c0 = j - rest * static_cast<uint32_t>(a.dims[0]);
    uint64_t offset = static_cast<uint64_t>(c0) * a.strides[0];
    if (a.outer_rank == 2) {
        c1 = rest;
        return offset + static_cast<uint64_t>(rest) * a.strides[1];
    }
    const uint32_t above = a.dim_divs[1].divide(rest);
    c1 = rest - above * static_cast<uint32_t>(a.dims[1]);
    offset += static_cast<uint64_t>(c1) * a.strides[1];
    rest = above;
#pragma unroll 1
    for (uint32_t i = 2; i + 1 < a.outer_rank; ++i) {
        const uint32_t q = a.dim_divs[i].divide(rest);
        offset += static_cast<uint64_t>(rest - q * static_cast<uint32_t>(a.dims[i])) * a.strides[i];
        rest = q;
    }
    return offset + static_cast<uint64_t>(rest) * a.strides[a.outer_rank - 1];
}

// The 16 / W words of one 16-byte destination word, the first at the lowest address.
template <typename Word>
__device__ __forceinline__ uint4 packWords(const Word* w) {
    if constexpr (sizeof(Word) == 4) {
        return make_uint4(w[0], w[1], w[2], w[3]);
    } else if constexpr (sizeof(Word) == 2) {
        return make_uint4(static_cast<uint32_t>(w[0]) | (static_cast<uint32_t>(w[1]) << 16),
                          static_cast<uint32_t>(w[2]) | (static_cast<uint32_t>(w[3]) << 16),
                          static_cast<uint32_t>(w[4]) | (static_cast<uint32_t>(w[5]) << 16),
                          static_cast<uint32_t>(w[6]) | (static_cast<uint32_t>(w[7]) << 16));
    } else {
        uint32_t quads[4];
#pragma unroll
        for (uint32_t q = 0; q < 4; ++q)
            quads[q] = static_cast<uint32_t>(w[4 * q]) | (static_cast<uint32_t>(w[4 * q + 1]) << 8) |
                       (static_cast<uint32_t>(w[4 * q + 2]) << 16) | (static_cast<uint32_t>(w[4 * q + 3]) << 24);
        return make_uint4(quads[0], quads[1], quads[2], quads[3]);
    }
}

// PackedGather (1389b): each thread gathers the 16 / W consecutive destination words of one 16-byte destination word
// (a pack), all loaded before the one store. Through 1389a these were row-copy words: an address decode over every
// outer axis and a 1-, 2- or 4-byte store for each. Here the source address is decoded once a pack and then steps
// along the innermost outer axis, carrying into the next one; a carry past that decodes again. The packs are in
// destination order across the threads, so a warp's store is 512 consecutive bytes. The words past the last whole
// pack (fewer than 16 / W) are stored singly by the first block.
template <typename Word>
__global__ __launch_bounds__(THREADS) void densePackedGatherKernel(const Word* __restrict__ source,
                                                                   Word* __restrict__ destination,
                                                                   const __grid_constant__ RowCopyArgs a) {
    constexpr uint32_t PACK = static_cast<uint32_t>(DenseCopyInternal::PACK_BYTES / sizeof(Word));
    uint4* packed = reinterpret_cast<uint4*>(destination);
    const uint32_t words = static_cast<uint32_t>(a.words);
    const uint32_t packs = words / PACK;
    const uint32_t d0 = static_cast<uint32_t>(a.dims[0]), d1 = static_cast<uint32_t>(a.dims[1]);
    const uint64_t s0 = a.strides[0];
    const uint64_t carry = a.strides[1] - static_cast<uint64_t>(d0 - 1) * s0;  // modulo 2^64
    const uint64_t step = static_cast<uint64_t>(gridDim.x) * THREADS;
    for (uint64_t pack = static_cast<uint64_t>(blockIdx.x) * THREADS + threadIdx.x; pack < packs; pack += step) {
        const uint32_t j = static_cast<uint32_t>(pack) * PACK;
        uint32_t c0 = 0, c1 = 0;
        uint64_t offset = packedGatherSourceWord(a, j, c0, c1);
        Word w[PACK];
#pragma unroll
        for (uint32_t k = 0; k < PACK; ++k) {
            w[k] = source[offset];
            if (k + 1 < PACK) {
                // Branches. As selects (1389c) the step measured up to 1.5x faster on packed gathers under 5 MiB
                // and 1.14x to 2.27x slower on two of the audit's general permutations of 24 and 31 MiB at 1- and
                // 2-byte words (58.8 to 133 us at worst). The cause was not found; 1389d put the branches back.
                if (++c0 < d0) {
                    offset += s0;
                } else {
                    c0 = 0;
                    if (++c1 < d1) offset += carry;
                    else offset = packedGatherSourceWord(a, j + k + 1, c0, c1);
                }
            }
        }
        packed[pack] = packWords<Word>(w);
    }
    if (blockIdx.x == 0 && threadIdx.x < words - packs * PACK) {
        const uint32_t j = packs * PACK + threadIdx.x;
        uint32_t c0 = 0, c1 = 0;
        destination[j] = source[packedGatherSourceWord(a, j, c0, c1)];
    }
}

// Transpose: the batch axes held innermost first, read in place as RowCopyArgs' axes are.
struct TransposeArgs {
    uint32_t x = 1, v = 1, p = 1, tile_x = 32, tile_v = 32;
    uint32_t width = 32;      // words of a tile row: tile_x vectors
    uint32_t out_width = 32;  // words of a destination segment: tile_v vectors
    uint32_t pitch = 33;      // shared words between tile rows
    uint32_t tiles_x = 1, tiles_v = 1, tiles_xv = 1, tiles = 1, batch_rank = 0;
    uint32_t v_first = 0;  // 1389g: the plan's tile_v_first
    uint32_t quads = 1;    // 1389j, packed bytes: blocks of four v rows in a tile
    uint64_t x_dest_stride = 0, v_source_stride = 0;
    uint64_t batch_dims[MAX_RANK]{};
    uint64_t batch_source[MAX_RANK]{};
    uint64_t batch_dest[MAX_RANK]{};
    FastDivU32 batch_divs[MAX_RANK];
    FastDivU32 tiles_x_div, tiles_v_div, tiles_xv_div, width_div, out_width_div, p_div;
};

// A tile is tile_v v rows (its rows) of tile_x x vectors of p words (width words, pitch apart). Source v rows are read
// in order into the tile, each thread loading ITEMS words before it stores any (1389: through 1388 one load at a time,
// so a tile cost one memory latency per word per thread); destination x rows are written in order from it. Every
// index is below 2^31 (the planner checks); offsets are 64-bit. 1389g: the tiles of a plane are taken along x within a
// band of tile_v v rows, or (v_first, which only the audit's probe sets) along v within tile_x x rows.
template <typename Word>
__global__ __launch_bounds__(THREADS) void denseTransposeKernel(const Word* __restrict__ source,
                                                                Word* __restrict__ destination,
                                                                const __grid_constant__ TransposeArgs a) {
    constexpr uint32_t ITEMS = TRANSPOSE_LOAD_ITEMS<Word>;
    constexpr uint32_t NONE = 0xFFFFFFFFu;
    extern __shared__ __align__(16) unsigned char dense_copy_tile_bytes[];
    Word* tile = reinterpret_cast<Word*>(dense_copy_tile_bytes);
    const uint32_t tile_words = a.tile_v * a.width;  // also tile_x * out_width
    for (uint32_t t = blockIdx.x; t < a.tiles; t += gridDim.x) {
        uint32_t b = a.tiles_xv_div.divide(t);
        const uint32_t txv = t - b * a.tiles_xv;
        uint32_t tx = 0, tv = 0;
        if (a.v_first != 0) {
            tx = a.tiles_v_div.divide(txv);
            tv = txv - tx * a.tiles_v;
        } else {
            tv = a.tiles_x_div.divide(txv);
            tx = txv - tv * a.tiles_x;
        }
        uint64_t source_base = 0, dest_base = 0;
#pragma unroll 1
        for (uint32_t i = 0; i < a.batch_rank; ++i) {
            const uint32_t q = a.batch_divs[i].divide(b);
            const uint64_t c = b - q * static_cast<uint32_t>(a.batch_dims[i]);
            source_base += c * a.batch_source[i];
            dest_base += c * a.batch_dest[i];
            b = q;
        }
        const uint32_t x0 = tx * a.tile_x, v0 = tv * a.tile_v;
        const uint32_t rows_here = min(a.tile_v, a.v - v0);  // v rows in this tile
        const uint32_t x_here = min(a.tile_x, a.x - x0);     // x vectors in this tile
        const uint32_t words_here = x_here * a.p;            // words of each tile row
        const Word* source_tile = source + source_base + static_cast<uint64_t>(x0) * a.p;
        for (uint32_t base = 0; base < tile_words; base += THREADS * ITEMS) {
            Word words[ITEMS];
            uint32_t at[ITEMS];
#pragma unroll
            for (uint32_t k = 0; k < ITEMS; ++k) {
                const uint32_t index = base + k * THREADS + threadIdx.x;
                at[k] = NONE;
                if (index < tile_words) {
                    const uint32_t r = a.width_div.divide(index);
                    const uint32_t c = index - r * a.width;
                    if (r < rows_here && c < words_here) {
                        words[k] = source_tile[static_cast<uint64_t>(v0 + r) * a.v_source_stride + c];
                        at[k] = r * a.pitch + c;
                    }
                }
            }
#pragma unroll
            for (uint32_t k = 0; k < ITEMS; ++k)
                if (at[k] != NONE) tile[at[k]] = words[k];
        }
        __syncthreads();
        const uint32_t out_here = rows_here * a.p;  // words of each destination x row segment in this tile
        Word* dest_tile = destination + dest_base + static_cast<uint64_t>(v0) * a.p;
        for (uint32_t index = threadIdx.x; index < tile_words; index += THREADS) {
            const uint32_t xr = a.out_width_div.divide(index);
            const uint32_t rem = index - xr * a.out_width;
            if (xr < x_here && rem < out_here) {
                const uint32_t vv = a.p_div.divide(rem);
                dest_tile[static_cast<uint64_t>(x0 + xr) * a.x_dest_stride + rem] =
                    tile[vv * a.pitch + xr * a.p + (rem - vv * a.p)];
            }
        }
        __syncthreads();
    }
}

// 1389j: the four bytes row[at], ..., row[at + 3] in one word, the first in the low byte, from the aligned words that
// hold them. Bytes past row[valid - 1] are undefined: a word with none of row[0], ..., row[valid - 1] is not read.
__device__ __forceinline__ uint32_t loadFourBytes(const uint8_t* row, uint32_t at, uint32_t valid) {
    const uintptr_t first = reinterpret_cast<uintptr_t>(row + at);
    const uint32_t* word = reinterpret_cast<const uint32_t*>(first & ~uintptr_t{3});
    const uint32_t shift = static_cast<uint32_t>(first & 3) * 8;
    const uint32_t low = __ldg(word);
    if (shift == 0) return low;
    const bool next = reinterpret_cast<uintptr_t>(word + 1) <= reinterpret_cast<uintptr_t>(row + valid - 1);
    return __funnelshift_r(low, next ? __ldg(word + 1) : 0u, shift);
}

// 1389j: w[k] holds byte j of row k (columns in the bytes); afterwards w[j] holds row k in its byte k, column j.
__device__ __forceinline__ void transposeFourByFour(uint32_t (&w)[4]) {
    const uint32_t a = __byte_perm(w[0], w[1], 0x5140), b = __byte_perm(w[0], w[1], 0x7362);
    const uint32_t c = __byte_perm(w[2], w[3], 0x5140), d = __byte_perm(w[2], w[3], 0x7362);
    w[0] = __byte_perm(a, c, 0x5410);
    w[1] = __byte_perm(a, c, 0x7632);
    w[2] = __byte_perm(b, d, 0x5410);
    w[3] = __byte_perm(b, d, 0x7632);
}

// Transpose of single bytes (1389j: plans of 1-byte words with p = 1). As denseTransposeKernel, with the bytes four at
// a time: a thread loads a block of 4 x 4 bytes (four v rows, four x columns, each row's four bytes as one 4-byte
// word from the aligned words that hold them, so any base and stride), transposes it in registers and writes the four
// words, each four v rows of one x row, into the tile; the destination x rows are then written in aligned 4-byte
// words where the whole word lies in the tile's v rows, and byte by byte at the two ends of a row otherwise. Through
// 1389i every byte had its own load, shared store, shared load and store, with an index decode per byte: the 1-byte
// transposes ran at 0.45 to 0.88 of the copy curve from 16 MiB, the same rotations at 2-byte words at 0.97 to 1.04.
// The tile is tile_x rows (rounded up to 4) of `pitch` words, a power of two; word r of x row c sits at
// r ^ ((c / 4) % pitch), so the four words a block writes and the words a destination row reads fall in distinct
// banks.
__global__ __launch_bounds__(THREADS) void denseTransposeBytesKernel(const uint8_t* __restrict__ source,
                                                                     uint8_t* __restrict__ destination,
                                                                     const __grid_constant__ TransposeArgs a) {
    extern __shared__ __align__(16) unsigned char dense_copy_tile_bytes[];
    uint32_t* tile = reinterpret_cast<uint32_t*>(dense_copy_tile_bytes);
    const uint32_t blocks = a.quads * a.width;   // a.width: blocks of four x columns
    const uint32_t slots = a.tile_x * a.out_width;  // a.out_width: destination words a tile row can touch
    const uint32_t swizzle = a.pitch - 1;
    for (uint32_t t = blockIdx.x; t < a.tiles; t += gridDim.x) {
        uint32_t b = a.tiles_xv_div.divide(t);
        const uint32_t txv = t - b * a.tiles_xv;
        uint32_t tx = 0, tv = 0;
        if (a.v_first != 0) {
            tx = a.tiles_v_div.divide(txv);
            tv = txv - tx * a.tiles_v;
        } else {
            tv = a.tiles_x_div.divide(txv);
            tx = txv - tv * a.tiles_x;
        }
        uint64_t source_base = 0, dest_base = 0;
#pragma unroll 1
        for (uint32_t i = 0; i < a.batch_rank; ++i) {
            const uint32_t q = a.batch_divs[i].divide(b);
            const uint64_t c = b - q * static_cast<uint32_t>(a.batch_dims[i]);
            source_base += c * a.batch_source[i];
            dest_base += c * a.batch_dest[i];
            b = q;
        }
        const uint32_t x0 = tx * a.tile_x, v0 = tv * a.tile_v;
        const uint32_t rows_here = min(a.tile_v, a.v - v0);  // v rows in this tile
        const uint32_t x_here = min(a.tile_x, a.x - x0);     // x bytes in this tile
        const uint8_t* source_tile = source + source_base + x0;
        for (uint32_t i = threadIdx.x; i < blocks; i += THREADS) {
            const uint32_t quad = a.width_div.divide(i);
            const uint32_t column = i - quad * a.width;
            const uint32_t at = 4 * column;
            uint32_t w[4];
#pragma unroll
            for (uint32_t k = 0; k < 4; ++k) {
                const uint32_t r = 4 * quad + k;
                w[k] = r < rows_here && at < x_here
                           ? loadFourBytes(source_tile + static_cast<uint64_t>(v0 + r) * a.v_source_stride, at, x_here)
                           : 0u;
            }
            transposeFourByFour(w);
            const uint32_t slot = quad ^ (column & swizzle);
#pragma unroll
            for (uint32_t k = 0; k < 4; ++k) tile[(at + k) * a.pitch + slot] = w[k];
        }
        __syncthreads();
        uint8_t* dest_tile = destination + dest_base + v0;
        for (uint32_t i = threadIdx.x; i < slots; i += THREADS) {
            const uint32_t xr = a.out_width_div.divide(i);
            const uint32_t k = i - xr * a.out_width;
            uint8_t* row = dest_tile + static_cast<uint64_t>(x0 + xr) * a.x_dest_stride;
            const uint32_t misaligned = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(row) & 3);
            const int32_t first = static_cast<int32_t>(4 * k) - static_cast<int32_t>(misaligned);  // its v row
            if (xr >= x_here || first >= static_cast<int32_t>(rows_here)) continue;
            const uint32_t* tile_row = tile + xr * a.pitch;
            const uint32_t s = (xr / 4) & swizzle;
            uint32_t* out = reinterpret_cast<uint32_t*>(row - misaligned) + k;
            if (first >= 0 && first + 4 <= static_cast<int32_t>(rows_here)) {
                const uint32_t u = static_cast<uint32_t>(first);
                const uint32_t low = tile_row[(u / 4) ^ s];
                *out = (u & 3) == 0 ? low : __funnelshift_r(low, tile_row[(u / 4 + 1) ^ s], (u & 3) * 8);
            } else {
                uint8_t* bytes = reinterpret_cast<uint8_t*>(out);
#pragma unroll
                for (int32_t j = 0; j < 4; ++j) {
                    const int32_t u = first + j;
                    if (u >= 0 && u < static_cast<int32_t>(rows_here))
                        bytes[j] = reinterpret_cast<const uint8_t*>(tile_row + ((u / 4) ^ s))[u & 3];
                }
            }
        }
        __syncthreads();
    }
}

[[nodiscard]] RowCopyArgs rowCopyArgs(const DenseCopyPlan& plan) {
    RowCopyArgs a;
    a.words = plan.words;
    a.p = plan.p;
    a.outer_rank = static_cast<uint32_t>(plan.outer_dims.size());
    for (uint32_t i = 0; i < a.outer_rank; ++i) {  // innermost first
        a.dims[i] = plan.outer_dims[a.outer_rank - 1 - i];
        a.strides[i] = plan.outer_strides[a.outer_rank - 1 - i];
    }
    if (plan.words < DenseCopyInternal::MAX_32BIT_INDEX) {
        a.p_div = FastDivU32::make(static_cast<uint32_t>(plan.p));
        for (uint32_t k = 0; k < a.outer_rank; ++k) a.dim_divs[k] = FastDivU32::make(static_cast<uint32_t>(a.dims[k]));
    }
    return a;
}

[[nodiscard]] TransposeArgs transposeArgs(const DenseCopyPlan& plan) {
    TransposeArgs a;
    a.x = static_cast<uint32_t>(plan.x);
    a.v = static_cast<uint32_t>(plan.v);
    a.p = static_cast<uint32_t>(plan.p);
    a.tile_x = plan.tile_x;
    a.tile_v = plan.tile_v;
    a.width = static_cast<uint32_t>(plan.tile_x * plan.p);
    a.out_width = static_cast<uint32_t>(plan.tile_v * plan.p);
    a.pitch = static_cast<uint32_t>(DenseCopyInternal::transposePitchWords(plan.tile_x, plan.p, plan.word_bytes));
    a.tiles_x = static_cast<uint32_t>((plan.x + plan.tile_x - 1) / plan.tile_x);
    a.tiles_v = static_cast<uint32_t>((plan.v + plan.tile_v - 1) / plan.tile_v);
    a.tiles_xv = a.tiles_x * a.tiles_v;
    a.v_first = plan.tile_v_first ? 1 : 0;
    a.tiles = static_cast<uint32_t>(DenseCopyInternal::workUnits(plan));
    a.batch_rank = static_cast<uint32_t>(plan.batch_dims.size());
    for (uint32_t i = 0; i < a.batch_rank; ++i) {  // innermost first
        const size_t k = a.batch_rank - 1 - i;
        a.batch_dims[i] = plan.batch_dims[k];
        a.batch_source[i] = plan.batch_source[k];
        a.batch_dest[i] = plan.batch_dest[k];
        a.batch_divs[i] = FastDivU32::make(static_cast<uint32_t>(plan.batch_dims[k]));
    }
    a.x_dest_stride = plan.x_dest_stride;
    a.v_source_stride = plan.v_source_stride;
    a.tiles_x_div = FastDivU32::make(a.tiles_x);
    a.tiles_v_div = FastDivU32::make(a.tiles_v);
    a.tiles_xv_div = FastDivU32::make(a.tiles_xv);
    a.width_div = FastDivU32::make(a.width);
    a.out_width_div = FastDivU32::make(a.out_width);
    a.p_div = FastDivU32::make(a.p);
    if (DenseCopyInternal::transposePacksBytes(plan.p, plan.word_bytes)) {  // 1389j: denseTransposeBytesKernel
        a.width = (plan.tile_x + 3) / 4;
        a.quads = (plan.tile_v + 3) / 4;
        a.out_width = (plan.tile_v + 6) / 4;
        a.pitch = static_cast<uint32_t>(DenseCopyInternal::transposeByteRowWords(plan.tile_v));
        a.width_div = FastDivU32::make(a.width);
        a.out_width_div = FastDivU32::make(a.out_width);
    }
    return a;
}

// The row copy on the planner's words per thread (rowCopyStepItems). 1395: its two counts per word are the only ones
// compiled (one for 1-, 8- and 16-byte words): 14 kernels. 1389n-1394 compiled 1, 2, 4, 8 and 16 for every word (50),
// for the audit's probe.
template <typename Word, bool Fast>
void launchRowCopy(const DenseCopyPlan& plan, const Word* in, Word* out, cudaStream_t stream) {
    constexpr uint32_t LONG_ITEMS = DenseCopyInternal::rowCopyItems(sizeof(Word));
    constexpr uint32_t SHORT_ITEMS = DenseCopyInternal::rowCopyShortItems(sizeof(Word));
    const RowCopyArgs a = rowCopyArgs(plan);
    if (SHORT_ITEMS != LONG_ITEMS && DenseCopyInternal::rowCopyTakesShortSteps(plan)) {
        denseRowCopyKernel<Word, Fast, SHORT_ITEMS><<<plan.grid, THREADS, 0, stream>>>(in, out, a);
    } else {
        denseRowCopyKernel<Word, Fast, LONG_ITEMS><<<plan.grid, THREADS, 0, stream>>>(in, out, a);
    }
}

template <uint32_t Bytes>
void launchTyped(const DenseCopyPlan& plan, const void* source, void* destination, cudaStream_t stream) {
    using Word = typename CopyWord<Bytes>::Type;
    const Word* in = static_cast<const Word*>(source);
    Word* out = static_cast<Word*>(destination);
    if (plan.family == DenseCopyFamily::Transpose && DenseCopyInternal::transposePacksBytes(plan.p, Bytes)) {
        denseTransposeBytesKernel<<<plan.grid, THREADS, DenseCopyInternal::sharedBytes(plan), stream>>>(
            reinterpret_cast<const uint8_t*>(in), reinterpret_cast<uint8_t*>(out), transposeArgs(plan));
    } else if (plan.family == DenseCopyFamily::Transpose) {
        denseTransposeKernel<Word><<<plan.grid, THREADS, DenseCopyInternal::sharedBytes(plan), stream>>>(
            in, out, transposeArgs(plan));
    } else if (plan.family == DenseCopyFamily::PackedGather) {
        if constexpr (Bytes <= 4) {
            densePackedGatherKernel<Word><<<plan.grid, THREADS, 0, stream>>>(in, out, rowCopyArgs(plan));
        } else {
            throw std::invalid_argument("Dense copy: a packed gather has words of 1, 2 or 4 bytes.");
        }
    } else if (plan.words < DenseCopyInternal::MAX_32BIT_INDEX) {
        launchRowCopy<Word, true>(plan, in, out, stream);
    } else {
        launchRowCopy<Word, false>(plan, in, out, stream);
    }
    CUDA_CHECK(cudaPeekAtLastError());
}

// 1394: the matrix transpose kernel of a MatrixTranspose plan, moved here from gpuMatrixTransposeKernels.cu (deleted
// with its dtype-cast path and typed entry points; its one production caller, the cuBLAS FP8 operand transpose, is a
// dense copy now). The body is that kernel's with the input and output word one type, so nothing converts. 32 x 32
// tiles of packed 4-byte words (PACK words of 4 / PACK bytes each), 8 rows of threads; int indices and at most 65,535
// row tiles, which matrixTransposeFits (DenseCopyPlan.cpp) checks. Words of 2 or 4 bytes (1389k).
constexpr int MATRIX_TILE = 32;
constexpr int MATRIX_BLOCK_ROWS = 8;

template <typename Word, int PACK>
struct alignas(sizeof(Word) * PACK) MatrixWords {
    Word v[PACK];
};

template <typename Word, int PACK>
__device__ inline uint32_t matrixWordsToU32(const MatrixWords<Word, PACK>& values) {
    union U {
        uint32_t raw;
        MatrixWords<Word, PACK> packed;
    } u{};
    u.raw = 0;
    u.packed = values;
    return u.raw;
}

template <typename Word, int PACK>
__device__ inline MatrixWords<Word, PACK> matrixWordsFromU32(uint32_t raw) {
    union U {
        uint32_t raw;
        MatrixWords<Word, PACK> packed;
    } u{};
    u.raw = raw;
    return u.packed;
}

template <typename Word, int PACK>
__device__ inline bool matrixWordsAligned(const Word* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) & (alignof(MatrixWords<Word, PACK>) - 1)) == 0;
}

template <typename Word, int PACK>
__device__ inline MatrixWords<Word, PACK> loadMatrixWords(const Word* ptr) {
    MatrixWords<Word, PACK> values{};
    if (matrixWordsAligned<Word, PACK>(ptr)) {
        values = *reinterpret_cast<const MatrixWords<Word, PACK>*>(ptr);
    } else {
#pragma unroll
        for (int i = 0; i < PACK; ++i) values.v[i] = ptr[i];
    }
    return values;
}

template <typename Word, int PACK>
__device__ inline void storeMatrixWords(Word* ptr, const MatrixWords<Word, PACK>& values) {
    if (matrixWordsAligned<Word, PACK>(ptr)) {
        *reinterpret_cast<MatrixWords<Word, PACK>*>(ptr) = values;
    } else {
#pragma unroll
        for (int i = 0; i < PACK; ++i) ptr[i] = values.v[i];
    }
}

// The [numCols, numRows] transpose of the dense [numRows, numCols] matrix of words.
template <typename Word, int PACK>
__global__ void denseMatrixTransposeKernel(Word* transposedMatrix, const Word* matrix, int numRows, int numCols) {
    static_assert(sizeof(Word) * PACK == 4, "packed 4-byte words");
    __shared__ uint32_t tile[PACK * MATRIX_TILE][MATRIX_TILE + 1];

    constexpr int TILE_SCALARS = PACK * MATRIX_TILE;
    constexpr int LOAD_ITERS = TILE_SCALARS / MATRIX_BLOCK_ROWS;
    constexpr int STORE_ITERS = MATRIX_TILE / MATRIX_BLOCK_ROWS;

    const int warpRow = static_cast<int>(threadIdx.y);
    const int lane = static_cast<int>(threadIdx.x);

    const int packedInputCols = (numCols + PACK - 1) / PACK;
    const int fullPackedInputCols = numCols / PACK;
    const int packedOutputCols = (numRows + PACK - 1) / PACK;
    const int fullPackedOutputCols = numRows / PACK;

    const int packedCol = lane + static_cast<int>(blockIdx.x) * MATRIX_TILE;
    const int matrixRowStart = static_cast<int>(blockIdx.y) * TILE_SCALARS;

    const int outputPackedCol = lane + static_cast<int>(blockIdx.y) * MATRIX_TILE;
    const int outputRowStart = static_cast<int>(blockIdx.x) * TILE_SCALARS;

    const int packedColBase = static_cast<int>(blockIdx.x) * MATRIX_TILE;
    const int outputPackedColBase = static_cast<int>(blockIdx.y) * MATRIX_TILE;
    const bool warpHasFullInputCols = packedColBase + MATRIX_TILE <= fullPackedInputCols;
    const bool warpHasFullOutputCols = outputPackedColBase + MATRIX_TILE <= fullPackedOutputCols;

    const int lastLoadRow = matrixRowStart + warpRow + (LOAD_ITERS - 1) * MATRIX_BLOCK_ROWS;
    const bool warpHasFullInputRows = lastLoadRow < numRows;

    const int firstStoreRow = outputRowStart + PACK * warpRow;
    const int lastStoreRow = firstStoreRow + (STORE_ITERS - 1) * PACK * MATRIX_BLOCK_ROWS + (PACK - 1);
    const bool warpHasFullOutputRows = lastStoreRow < numCols;

    const uintptr_t firstWarpLoadAddr =
        reinterpret_cast<uintptr_t>(matrix) +
        (static_cast<uintptr_t>(matrixRowStart + warpRow) * static_cast<uintptr_t>(numCols) +
         static_cast<uintptr_t>(packedCol) * PACK) *
            sizeof(Word);
    const bool warpHasPackedLoadAlignment = (firstWarpLoadAddr & (alignof(MatrixWords<Word, PACK>) - 1)) == 0;

    const size_t outputRowStrideBytes = static_cast<size_t>(numRows) * sizeof(Word);
    const uintptr_t firstWarpStoreAddr =
        reinterpret_cast<uintptr_t>(transposedMatrix) +
        (static_cast<uintptr_t>(firstStoreRow) * static_cast<uintptr_t>(numRows) +
         static_cast<uintptr_t>(outputPackedCol) * PACK) *
            sizeof(Word);
    const bool warpHasPackedStoreAlignment = (outputRowStrideBytes % alignof(MatrixWords<Word, PACK>) == 0) &&
                                             ((firstWarpStoreAddr & (alignof(MatrixWords<Word, PACK>) - 1)) == 0);

    const bool warpOnEdge =
        !(warpHasFullInputCols && warpHasFullOutputCols && warpHasFullInputRows && warpHasFullOutputRows);
    const bool warpCanUseFastPath = !warpOnEdge && warpHasPackedLoadAlignment && warpHasPackedStoreAlignment;

    int tileRow = warpRow;
    if (warpCanUseFastPath) {
#pragma unroll
        for (int iter = 0; iter < LOAD_ITERS; ++iter) {
            const int matrixRow = matrixRowStart + warpRow + iter * MATRIX_BLOCK_ROWS;
            const Word* src = matrix + static_cast<size_t>(matrixRow) * static_cast<size_t>(numCols) +
                              static_cast<size_t>(packedCol) * PACK;
            tile[tileRow][lane] = matrixWordsToU32(loadMatrixWords<Word, PACK>(src));
            tileRow += MATRIX_BLOCK_ROWS;
        }
    } else {
#pragma unroll
        for (int iter = 0; iter < LOAD_ITERS; ++iter) {
            const int matrixRow = matrixRowStart + warpRow + iter * MATRIX_BLOCK_ROWS;
            uint32_t raw = 0;
            if (matrixRow < numRows && packedCol < packedInputCols) {
                MatrixWords<Word, PACK> values{};
                const int remaining = numCols - packedCol * PACK;
                const int validLanes = remaining > 0 ? ((remaining < PACK) ? remaining : PACK) : 0;
                const Word* src = matrix + static_cast<size_t>(matrixRow) * static_cast<size_t>(numCols) +
                                  static_cast<size_t>(packedCol) * PACK;
#pragma unroll
                for (int i = 0; i < PACK; ++i) {
                    if (i < validLanes) values.v[i] = src[i];
                }
                raw = matrixWordsToU32(values);
            }
            tile[tileRow][lane] = raw;
            tileRow += MATRIX_BLOCK_ROWS;
        }
    }

    __syncthreads();

    int tileCol = warpRow;
    if (warpCanUseFastPath) {
#pragma unroll
        for (int iter = 0; iter < STORE_ITERS; ++iter) {
            const int matrixRow = outputRowStart + PACK * warpRow + iter * PACK * MATRIX_BLOCK_ROWS;
#pragma unroll
            for (int rowLane = 0; rowLane < PACK; ++rowLane) {
                MatrixWords<Word, PACK> outputValues{};
#pragma unroll
                for (int colLane = 0; colLane < PACK; ++colLane) {
                    outputValues.v[colLane] =
                        matrixWordsFromU32<Word, PACK>(tile[PACK * lane + colLane][tileCol]).v[rowLane];
                }
                Word* dst = transposedMatrix + static_cast<size_t>(matrixRow + rowLane) * static_cast<size_t>(numRows) +
                            static_cast<size_t>(outputPackedCol) * PACK;
                storeMatrixWords<Word, PACK>(dst, outputValues);
            }
            tileCol += MATRIX_BLOCK_ROWS;
        }
    } else {
#pragma unroll
        for (int iter = 0; iter < STORE_ITERS; ++iter) {
            const int matrixRow = outputRowStart + PACK * warpRow + iter * PACK * MATRIX_BLOCK_ROWS;
            const int remainingCols = numRows - outputPackedCol * PACK;
            const int validOutputLanes = (outputPackedCol < packedOutputCols && remainingCols > 0)
                                             ? ((remainingCols < PACK) ? remainingCols : PACK)
                                             : 0;
#pragma unroll
            for (int rowLane = 0; rowLane < PACK; ++rowLane) {
                const int outputRow = matrixRow + rowLane;
                if (outputRow < numCols && validOutputLanes > 0) {
#pragma unroll
                    for (int colLane = 0; colLane < PACK; ++colLane) {
                        const MatrixWords<Word, PACK> inputValues =
                            matrixWordsFromU32<Word, PACK>(tile[PACK * lane + colLane][tileCol]);
                        if (colLane < validOutputLanes) {
                            transposedMatrix[static_cast<size_t>(outputRow) * static_cast<size_t>(numRows) +
                                             static_cast<size_t>(outputPackedCol) * PACK + colLane] =
                                inputValues.v[rowLane];
                        }
                    }
                }
            }
            tileCol += MATRIX_BLOCK_ROWS;
        }
    }
}

template <typename Word>
void launchMatrixTranspose(const DenseCopyPlan& plan, const void* source, void* destination, cudaStream_t stream) {
    constexpr int PACK = static_cast<int>(4 / sizeof(Word));
    const uint32_t rows = static_cast<uint32_t>(plan.v), cols = static_cast<uint32_t>(plan.x);
    constexpr uint32_t tile_words = PACK * MATRIX_TILE;
    const dim3 grid((cols + tile_words - 1u) / tile_words, (rows + tile_words - 1u) / tile_words);
    denseMatrixTransposeKernel<Word, PACK><<<grid, dim3(MATRIX_TILE, MATRIX_BLOCK_ROWS), 0, stream>>>(
        static_cast<Word*>(destination), static_cast<const Word*>(source), static_cast<int>(rows),
        static_cast<int>(cols));
    CUDA_CHECK(cudaPeekAtLastError());
}

void requireSameGpu(const StridedSpan& source, const StridedSpan& destination) {
    if (source.getPlacement() != destination.getPlacement() ||
        source.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::invalid_argument("Dense copy requires source and destination on the same GPU placement.");
    }
}

}  // namespace

DenseCopyPlan planCopyToDense(const StridedSpan& source, const StridedSpan& destination) {
    if (source.getDescriptor() != destination.getDescriptor()) {
        throw std::invalid_argument("Dense copy requires identical source and destination descriptors. Source=" +
                                    source.getDescriptor().toString() +
                                    ", destination=" + destination.getDescriptor().toString() + ".");
    }
    requireSameGpu(source, destination);
    if (!destination.isDenseContiguous()) throw std::invalid_argument("Dense copy requires a dense destination.");
    const uint32_t element_bytes =
        static_cast<uint32_t>(TensorDescriptor::getElementSizeInBytes(source.getDataType()));
    DenseCopyPlan plan = DenseCopyInternal::planLayout(source.getDimensions(), source.getStridesElements(),
                                                       element_bytes,
                                                       reinterpret_cast<uintptr_t>(source.getMemPtr<void>()),
                                                       reinterpret_cast<uintptr_t>(destination.getMemPtr<void>()));
    plan.grid = DenseCopyInternal::gridOf(plan);
    return plan;
}

void launchCopyToDense(const DenseCopyPlan& plan, const StridedSpan& source, const StridedSpan& destination,
                       Stream stream) {
    requireSameGpu(source, destination);
    if (source.getDimensions() != plan.source_dims || source.getStridesElements() != plan.source_strides ||
        destination.getDimensions() != plan.source_dims || !destination.isDenseContiguous() ||
        source.getDataType() != destination.getDataType() ||
        TensorDescriptor::getElementSizeInBytes(source.getDataType()) != plan.element_bytes) {
        throw std::invalid_argument("Dense copy: the spans do not have the plan's layout.");
    }
    if (plan.family == DenseCopyFamily::Empty) return;
    const void* in = source.getMemPtr<void>();
    void* out = destination.getMemPtr<void>();
    const uint64_t out_alignment =
        plan.family == DenseCopyFamily::PackedGather ? DenseCopyInternal::PACK_BYTES : plan.word_bytes;
    if (reinterpret_cast<uintptr_t>(in) % plan.word_bytes != 0 || reinterpret_cast<uintptr_t>(out) % out_alignment != 0)
        throw std::invalid_argument("Dense copy: an address is not aligned to the plan's " +
                                    std::to_string(plan.word_bytes) + "-byte word (the destination of a packed " +
                                    "gather: 16 bytes).");
    ScopedGpu scoped_gpu(source.getPlacement().getDeviceNum());
    if (plan.family == DenseCopyFamily::Memcpy) {
        CUDA_CHECK(
            cudaMemcpyAsync(out, in, plan.words * plan.word_bytes, cudaMemcpyDeviceToDevice, stream.getStream()));
        return;
    }
    if (plan.family == DenseCopyFamily::MatrixTranspose) {
        // 1389e: the source is the dense [v, x] matrix of words (the planner checked); the kernel writes its
        // [x, v] transpose. Unsigned words, copied as they are whatever the dtype (1394: denseMatrixTransposeKernel).
        // 1389k: words of 2 or 4 bytes; single bytes are Transposes (the packed byte tile).
        if (plan.word_bytes == 2) {
            launchMatrixTranspose<uint16_t>(plan, in, out, stream.getStream());
        } else if (plan.word_bytes == 4) {
            launchMatrixTranspose<uint32_t>(plan, in, out, stream.getStream());
        } else {
            throw std::invalid_argument("Dense copy: a matrix transpose has words of 2 or 4 bytes.");
        }
        return;
    }
    switch (plan.word_bytes) {
        case 1: launchTyped<1>(plan, in, out, stream.getStream()); break;
        case 2: launchTyped<2>(plan, in, out, stream.getStream()); break;
        case 4: launchTyped<4>(plan, in, out, stream.getStream()); break;
        case 8: launchTyped<8>(plan, in, out, stream.getStream()); break;
        case 16: launchTyped<16>(plan, in, out, stream.getStream()); break;
        default: throw std::invalid_argument("Dense copy: words are 1, 2, 4, 8 or 16 bytes.");
    }
}

void copyToDenseAsync(const StridedSpan& source, Tensor& destination, Stream stream) {
    const StridedSpan dense(destination);
    launchCopyToDense(planCopyToDense(source, dense), source, dense, stream);
}

}  // namespace ThorImplementation
