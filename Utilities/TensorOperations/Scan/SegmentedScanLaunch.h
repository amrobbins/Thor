#pragma once

// What the host and the segmented scan kernel share: the ops, the tile shape, the row-length remainder, the
// workspace layout and one launch's arguments. Includes no Thor header, so the kernel builds on its own
// (SegmentedScanKernel.cuh); the API is SegmentedScan.h.

#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace ThorImplementation {

// ArgMax and ArgMin write, for each element, the flat UINT32 index of the largest (smallest) element so far in its row.
enum class SegmentedScanOp : uint8_t { Sum = 0, Max = 1, Min = 2, Product = 3, ArgMax = 4, ArgMin = 5 };

__host__ __device__ constexpr bool isArg(SegmentedScanOp op) {
    return op == SegmentedScanOp::ArgMax || op == SegmentedScanOp::ArgMin;
}

// Inclusive: each element's output takes in the element. Exclusive: only the elements before it in its row; a row's
// first element takes the op's identity (SegmentedScan.h).
enum class SegmentedScanMode : uint8_t { Inclusive = 0, Exclusive = 1 };

// Forward: each row from its first element to its last. Reverse: from its last to its first (an arg scan's index is
// then the last of equal elements, the first met).
enum class SegmentedScanDirection : uint8_t { Forward = 0, Reverse = 1 };

namespace SegmentedScanKernels {

inline constexpr uint32_t kThreads = 256;
inline constexpr uint32_t kGroup = 32;       // tiles per group
inline constexpr uint32_t kSuper = 32;       // groups per supergroup
inline constexpr uint32_t kChunks = 4;       // chunks per thread
inline constexpr uint32_t kChunkBytes = 32;  // bytes of the wider of input and output per chunk

// Elements per chunk and per tile for element sizes of input and output: a chunk is 32 bytes of the wider side, so
// every scan loads or stores 32 bytes per chunk (plans/Thor_Scan_Kernel_Log9_2026-11-18.md, which kernels to keep).
inline constexpr uint32_t chunkElements(uint32_t input_bytes, uint32_t output_bytes) {
    return kChunkBytes / (input_bytes > output_bytes ? input_bytes : output_bytes);
}

inline constexpr uint64_t tileElements(uint32_t input_bytes, uint32_t output_bytes) {
    return static_cast<uint64_t>(kThreads) * kChunks * chunkElements(input_bytes, output_bytes);
}

// The row length and its remainder by multiplication: a 64-bit % is a call to a 58-instruction routine, so the
// host works out a magic multiplier once per launch (Granlund and Montgomery, "Division by invariant integers using
// multiplication", 1994, figure 4.1: exact for every 64-bit dividend and every divisor from 1 up).
struct Columns {
    uint64_t count = 0;  // 0 for ragged rows, whose remainders are never taken
    uint64_t magic = 0;
    uint32_t shift1 = 0;
    uint32_t shift2 = 0;

    __host__ __device__ uint64_t mod(uint64_t x) const {
#if defined(__CUDA_ARCH__)
        const uint64_t t = __umul64hi(magic, x);
#else
        const uint64_t t = static_cast<uint64_t>((static_cast<unsigned __int128>(magic) * x) >> 64);
#endif
        const uint64_t q = (t + ((x - t) >> shift1)) >> shift2;
        return x - q * count;
    }
};

inline Columns columnsOf(uint64_t count) {
    Columns columns;
    columns.count = count;
    if (count == 0) return columns;
    uint32_t l = 0;  // ceil(log2 count)
    while (l < 64 && (uint64_t{1} << l) < count) ++l;
    const unsigned __int128 power = static_cast<unsigned __int128>(1) << l;
    columns.magic = static_cast<uint64_t>(((static_cast<unsigned __int128>(1) << 64) * (power - count)) / count + 1);
    columns.shift1 = l < 1 ? l : 1;
    columns.shift2 = l > 0 ? l - 1 : 0;
    return columns;
}

// The runs the tiles publish, all in one UINT64 workspace: the tiles', then the groups', then the supergroups' (each
// supergroup's inclusive run). A run is one 64-bit word, or two for an arg scan, whose runs carry a 64-bit key: run r
// is words runWords x r and on. The workspace is cleared once, when it is made.
struct Workspace {
    unsigned long long* tile_runs = nullptr;
    unsigned long long* group_runs = nullptr;
    unsigned long long* super_runs = nullptr;
    uint64_t num_tiles = 0;
};

__host__ __device__ constexpr uint32_t runWords(SegmentedScanOp op) { return isArg(op) ? 2U : 1U; }

// Workspace words for n elements in tiles of tile_elements.
inline uint64_t workspaceWords(SegmentedScanOp op, uint64_t n, uint64_t tile_elements) {
    const uint64_t tiles = (n + tile_elements - 1) / tile_elements;
    const uint64_t groups = (tiles + kGroup - 1) / kGroup;
    const uint64_t supers = (groups + kSuper - 1) / kSuper;
    return runWords(op) * (tiles + groups + supers);
}

// A published word holds its launch's epoch in 31 bits, 1 to kMaxEpoch; 0 is a cleared word. Every launch writes
// every run word, so at a launch's start every word holds the last launch's epoch (or 0), never this one's; a tile
// waits for this one's, so only an epoch different from the last one's is needed, and the wrap is safe.
inline constexpr uint32_t kMaxEpoch = 0x7FFFFFFFU;

__host__ __device__ inline uint32_t nextEpoch(uint32_t last) { return last >= kMaxEpoch ? 1U : last + 1U; }

// The output of a float sum or product of 2 or 1 bytes, whose format the launch chooses: FP16 or BF16, E4M3 or E5M2.
// One kernel serves both formats of a size (Launch::alternate_output: BF16, E5M2).
template <uint32_t BYTES>
struct SharedFloat {
    std::conditional_t<BYTES == 2, uint16_t, uint8_t> bits;
};
using SharedFloat16 = SharedFloat<2>;
using SharedFloat8 = SharedFloat<1>;

template <typename O>
inline constexpr bool kSharedFloat = std::is_same_v<O, SharedFloat16> || std::is_same_v<O, SharedFloat8>;

struct Launch {
    const void* input = nullptr;
    void* output = nullptr;
    uint64_t n = 0;
    Columns columns;                // dense rows
    const void* offsets = nullptr;  // ragged rows: rows + 1 offsets of OFF
    uint32_t rows = 0;
    Workspace workspace;  // num_tiles is the grid; the pointers are null when no tile publishes
    void* arg_values = nullptr;  // an arg scan's winning values, in the input's dtype; null: not written
    bool alternate_output = false;  // a SharedFloat output: BF16 (not FP16), E5M2 (not E4M3)
    bool exclusive = false;         // the mode: exclusive (not inclusive)
    bool arg_min = false;           // an arg scan (the ArgMax kernel) is argmin
    bool reverse = false;           // the direction: reverse (not forward)
};

// One launch of the kernel reading T and writing O: dense rows when OFF is void, ragged rows (offsets of OFF)
// otherwise; the mode and the direction are the launch's (Launch::exclusive, Launch::reverse). Instantiated for the
// supported pairs in SegmentedScanFp32.cu, SegmentedScanF16.cu, SegmentedScanFp8.cu and SegmentedScanArg.cu (OP ArgMax
// only: argmin is the same kernel with Launch::arg_min).
template <typename T, typename O, SegmentedScanOp OP, typename OFF>
void launch(const Launch& args, cudaStream_t stream);

}  // namespace SegmentedScanKernels
}  // namespace ThorImplementation
