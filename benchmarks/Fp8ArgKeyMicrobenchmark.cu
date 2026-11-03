// ARG-A1 standalone microbenchmark for the FP8 ARG key scheme (Utilities/TensorOperations/Cub/Fp8ArgKey.h).
//
// Not a production kernel and not part of the Thor library: it answers "does the packed key + lane-candidate update
// fit under DRAM line rate?" before any ARG kernel family exists, and gives a small kernel to inspect in SASS.
// One thread owns one PacketBytes-wide column packet of an [outer, R, K] FP8 tensor and walks its R rows; there is no
// R split, no load-ahead and no shared memory, so shapes with few threads (small outer*K) under-fill the GPU in both
// kernels alike. Two kernels with identical loads and identical UINT32-per-component output traffic:
//   load: XOR-folds the loaded words (the bandwidth roofline of this loop shape)
//   arg:  argmin/argmax via fp8ArgKeys4 + one lane candidate min per component, writes the winning row
// The interesting number is arg/load: ~1.0 means the per-element ARG work hides under memory traffic.
// Variants: p16, p16_b5 (16 components per lane capped at 48 registers), p8, p4 (components per lane = bytes).
// ARG-A3f (1114): p8_lazy / p16_lazy keep one byte-SIMD running key minimum per word (__vminu4 over a 4-row batch) and
// resolve rows only when a batch strictly improves a byte (__vcmpltu4 / __vcmpeq4, rare after the first batches on
// random data); the per-element candidate build of the eager kernels disappears. `occ` rows run every kernel with
// dynamic shared memory that caps residency at 2 blocks (16 warps) per SM, where per-element ALU work stops hiding
// under memory traffic (production ARG kernels run at 4-8 warps per scheduler, not the microbenchmark's 16).
//
//   ./thor_fp8_arg_key_microbenchmark            # all shapes, both packets
//   cuobjdump -res-usage thor_fp8_arg_key_microbenchmark
//   cuobjdump -sass -fun '<mangled arg kernel>' thor_fp8_arg_key_microbenchmark

#include "Utilities/TensorOperations/Cub/Fp8ArgKey.h"
#include "test/Utilities/TensorOperations/Fp8ArgReference.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace C = ThorImplementation::CubReductionInternal;
namespace Reference = ThorImplementation::Fp8ArgReference;

namespace {

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

// One non-blocking stream for everything (Thor's convention; no default-stream memory APIs).
cudaStream_t benchmarkStream() {
    static cudaStream_t stream = [] {
        cudaStream_t s = nullptr;
        check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
        return s;
    }();
    return stream;
}

constexpr uint32_t Threads = 256;
constexpr uint64_t BufferBytes = 192ULL << 20;  // per working set (2x the 5090's L2)
constexpr int BufferCount = 4;                  // rotated per launch so every launch reads cold
constexpr int Warmups = 3;
constexpr int Samples = 15;

template <int PacketBytes>
struct Packet;
template <>
struct Packet<4> {
    using Type = uint32_t;
    __device__ static void words(const Type& p, uint32_t (&w)[1]) { w[0] = p; }
};
template <>
struct Packet<8> {
    using Type = uint2;
    __device__ static void words(const Type& p, uint32_t (&w)[2]) {
        w[0] = p.x;
        w[1] = p.y;
    }
};
template <>
struct Packet<16> {
    using Type = uint4;
    __device__ static void words(const Type& p, uint32_t (&w)[4]) {
        w[0] = p.x;
        w[1] = p.y;
        w[2] = p.z;
        w[3] = p.w;
    }
};

// MinBlocks > 1 caps registers (5 x 256 threads: <= 48, under the 49-register cliff).
template <int PacketBytes, bool Arg, int MinBlocks>
__global__ __launch_bounds__(Threads, MinBlocks) void fp8ArgMicroKernel(const uint8_t* __restrict__ input,
                                                             uint32_t* __restrict__ output,
                                                             uint32_t outer,
                                                             uint32_t rows,
                                                             uint32_t k,
                                                             C::Fp8ArgFold fold) {
    constexpr int Words = PacketBytes / 4;
    using P = Packet<PacketBytes>;
    (void)fold;
    const uint32_t packets = k / PacketBytes;
    const uint64_t thread = static_cast<uint64_t>(blockIdx.x) * Threads + threadIdx.x;
    if (thread >= static_cast<uint64_t>(outer) * packets) return;
    const uint32_t o = static_cast<uint32_t>(thread / packets);
    const uint32_t p = static_cast<uint32_t>(thread % packets);
    const typename P::Type* source = reinterpret_cast<const typename P::Type*>(
        input + static_cast<uint64_t>(o) * rows * k + static_cast<uint64_t>(p) * PacketBytes);
    const uint32_t stride = k / PacketBytes;

    uint32_t state[PacketBytes];
#pragma unroll
    for (int c = 0; c < PacketBytes; ++c) state[c] = Arg ? C::FP8_ARG_LANE_IDENTITY : 0u;

#pragma unroll 4
    for (uint32_t r = 0; r < rows; ++r) {
        uint32_t w[Words];
        P::words(source[static_cast<uint64_t>(r) * stride], w);
#pragma unroll
        for (int i = 0; i < Words; ++i) {
            if constexpr (Arg) {
                const C::Fp8ArgKeys4 keys = C::fp8ArgKeys4(w[i], fold);
#pragma unroll
                for (int b = 0; b < 4; ++b)
                    state[4 * i + b] = C::fp8ArgMin(state[4 * i + b], C::fp8ArgLaneCandidate(keys, b, r));
            } else {
                state[4 * i] ^= w[i];
            }
        }
    }

    uint4* target = reinterpret_cast<uint4*>(output + static_cast<uint64_t>(o) * k + static_cast<uint64_t>(p) * PacketBytes);
#pragma unroll
    for (int c = 0; c < PacketBytes; c += 4) {
        uint32_t v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
            v[j] = Arg ? (state[c + j] >> 1) & (C::FP8_ARG_MAX_LANE_ROWS - 1u) : state[c + j];
        target[c / 4] = make_uint4(v[0], v[1], v[2], v[3]);
    }
}

// Lazy-index variant: per word, the running byte-wise key minimum `best`, the winning row per byte and the winner's
// negative-zero bits. A 4-row batch reduces to one byte-wise minimum; only bytes it strictly improves look for their
// first equal row (descending, so the lowest row writes last), which keeps the lowest index among equal keys.
template <int PacketBytes>
__device__ __forceinline__ void lazyArgBatch(uint32_t (&best)[PacketBytes / 4],
                                             uint32_t (&negative_zero)[PacketBytes / 4],
                                             uint32_t (&rows)[PacketBytes],
                                             const C::Fp8ArgKeys4 (&keys)[4][PacketBytes / 4],
                                             int batch_rows,
                                             uint32_t first_row) {
#pragma unroll
    for (int i = 0; i < PacketBytes / 4; ++i) {
        uint32_t m = keys[0][i].keys;
#pragma unroll
        for (int r = 1; r < 4; ++r)
            if (r < batch_rows) m = __vminu4(m, keys[r][i].keys);
        const uint32_t improved = __vcmpltu4(m, best[i]);
        if (improved != 0) {
#pragma unroll
            for (int r = 3; r >= 0; --r) {
                if (r >= batch_rows) continue;
                const uint32_t equal = __vcmpeq4(keys[r][i].keys, m) & improved;
#pragma unroll
                for (int b = 0; b < 4; ++b)
                    if ((equal >> (8 * b)) & 1u) rows[4 * i + b] = first_row + static_cast<uint32_t>(r);
                negative_zero[i] = (negative_zero[i] & ~equal) | (keys[r][i].negative_zero & equal);
            }
            best[i] = __vminu4(best[i], m);
        }
    }
}

template <int PacketBytes, int MinBlocks>
__global__ __launch_bounds__(Threads, MinBlocks) void fp8ArgLazyMicroKernel(const uint8_t* __restrict__ input,
                                                                 uint32_t* __restrict__ output,
                                                                 uint32_t outer,
                                                                 uint32_t rows,
                                                                 uint32_t k,
                                                                 C::Fp8ArgFold fold) {
    constexpr int Words = PacketBytes / 4;
    using P = Packet<PacketBytes>;
    const uint32_t packets = k / PacketBytes;
    const uint64_t thread = static_cast<uint64_t>(blockIdx.x) * Threads + threadIdx.x;
    if (thread >= static_cast<uint64_t>(outer) * packets) return;
    const uint32_t o = static_cast<uint32_t>(thread / packets);
    const uint32_t p = static_cast<uint32_t>(thread % packets);
    const typename P::Type* source = reinterpret_cast<const typename P::Type*>(
        input + static_cast<uint64_t>(o) * rows * k + static_cast<uint64_t>(p) * PacketBytes);
    const uint32_t stride = k / PacketBytes;

    uint32_t best[Words];
    uint32_t negative_zero[Words];
    uint32_t winner[PacketBytes];
#pragma unroll
    for (int i = 0; i < Words; ++i) {
        best[i] = 0xFFFFFFFFu;  // key 0xFF never occurs
        negative_zero[i] = 0;
    }
#pragma unroll
    for (int c = 0; c < PacketBytes; ++c) winner[c] = 0;

    uint32_t r = 0;
    for (; r + 4 <= rows; r += 4) {
        C::Fp8ArgKeys4 keys[4][Words];
#pragma unroll
        for (int rr = 0; rr < 4; ++rr) {
            uint32_t w[Words];
            P::words(source[static_cast<uint64_t>(r + rr) * stride], w);
#pragma unroll
            for (int i = 0; i < Words; ++i) keys[rr][i] = C::fp8ArgKeys4(w[i], fold);
        }
        lazyArgBatch<PacketBytes>(best, negative_zero, winner, keys, 4, r);
    }
    for (; r < rows; ++r) {
        C::Fp8ArgKeys4 keys[4][Words]{};
        uint32_t w[Words];
        P::words(source[static_cast<uint64_t>(r) * stride], w);
#pragma unroll
        for (int i = 0; i < Words; ++i) keys[0][i] = C::fp8ArgKeys4(w[i], fold);
        lazyArgBatch<PacketBytes>(best, negative_zero, winner, keys, 1, r);
    }
    (void)negative_zero;  // production rebuilds the value byte from best + negative_zero; the benchmark writes rows

    uint4* target = reinterpret_cast<uint4*>(output + static_cast<uint64_t>(o) * k + static_cast<uint64_t>(p) * PacketBytes);
#pragma unroll
    for (int c = 0; c < PacketBytes; c += 4) target[c / 4] = make_uint4(winner[c], winner[c + 1], winner[c + 2], winner[c + 3]);
}

__global__ void fillKernel(uint8_t* data, uint64_t bytes, uint64_t seed) {
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < bytes;
         i += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
        uint64_t x = (i + seed) * 0x9E3779B97F4A7C15ULL;
        x ^= x >> 29;
        data[i] = static_cast<uint8_t>(x * 0xBF58476D1CE4E5B9ULL >> 56);
    }
}

// Kind: 0 load, 1 eager ARG, 2 lazy ARG. shared_bytes > 0 caps residency (see `occ`).
template <int PacketBytes, int Kind, int MinBlocks = 1>
void launch(const uint8_t* input, uint32_t* output, uint32_t outer, uint32_t rows, uint32_t k, C::Fp8ArgFold fold,
            size_t shared_bytes = 0) {
    const uint64_t threads = static_cast<uint64_t>(outer) * (k / PacketBytes);
    const uint32_t blocks = static_cast<uint32_t>((threads + Threads - 1) / Threads);
    if constexpr (Kind == 2) {
        auto* kernel = fp8ArgLazyMicroKernel<PacketBytes, MinBlocks>;
        check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_bytes)),
              "smem attribute");
        kernel<<<blocks, Threads, shared_bytes, benchmarkStream()>>>(input, output, outer, rows, k, fold);
    } else {
        auto* kernel = fp8ArgMicroKernel<PacketBytes, Kind == 1, MinBlocks>;
        check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_bytes)),
              "smem attribute");
        kernel<<<blocks, Threads, shared_bytes, benchmarkStream()>>>(input, output, outer, rows, k, fold);
    }
    check(cudaGetLastError(), "microkernel launch");
}

// Dynamic shared memory per block that leaves room for `blocks` resident blocks per SM.
size_t residencyCapSharedBytes(int blocks) {
    int device = 0;
    check(cudaGetDevice(&device), "device");
    int per_sm = 0;
    check(cudaDeviceGetAttribute(&per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device), "smem per SM");
    int per_block = 0;
    check(cudaDeviceGetAttribute(&per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device), "smem per block");
    return static_cast<size_t>(std::min(per_block, per_sm / blocks - 1024));
}

// Device ARG result vs the FP32 reference on a small specials-heavy tensor.
template <int PacketBytes, int MinBlocks = 1, int Kind = 1>
void validate() {
    constexpr uint32_t outer = 13, rows = 37, k = 64;
    std::mt19937 rng(PacketBytes);
    static constexpr std::array<uint8_t, 8> SPECIALS{{0x00, 0x80, 0x7F, 0xFF, 0x7E, 0xFE, 0x7C, 0xFC}};
    std::vector<uint8_t> host(static_cast<size_t>(outer) * rows * k);
    for (uint8_t& byte : host) byte = (rng() & 3u) == 0 ? SPECIALS[rng() % SPECIALS.size()] : static_cast<uint8_t>(rng());
    uint8_t* input = nullptr;
    uint32_t* output = nullptr;
    check(cudaMalloc(&input, host.size()), "malloc validate input");
    check(cudaMalloc(&output, static_cast<size_t>(outer) * k * sizeof(uint32_t)), "malloc validate output");
    check(cudaMemcpyAsync(input, host.data(), host.size(), cudaMemcpyHostToDevice, benchmarkStream()),
          "copy validate input");
    for (const auto& [maximum, e5m2] : {std::pair<bool, bool>{false, false}, {true, true}, {true, false}, {false, true}}) {
        launch<PacketBytes, Kind, MinBlocks>(input, output, outer, rows, k, C::fp8ArgFold(maximum, e5m2));
        std::vector<uint32_t> index(static_cast<size_t>(outer) * k);
        check(cudaMemcpyAsync(index.data(), output, index.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost,
                              benchmarkStream()),
              "copy index");
        check(cudaStreamSynchronize(benchmarkStream()), "sync validate");
        for (uint32_t o = 0; o < outer; ++o) {
            for (uint32_t c = 0; c < k; ++c) {
                const auto at = [&](uint32_t r) { return host[(static_cast<size_t>(o) * rows + r) * k + c]; };
                Reference::Candidate best{0, Reference::decode(at(0), e5m2)};
                for (uint32_t r = 1; r < rows; ++r) best = Reference::combine(best, {r, Reference::decode(at(r), e5m2)}, maximum);
                if (index[static_cast<size_t>(o) * k + c] != best.index)
                    throw std::runtime_error(std::string(Kind == 2 ? "lazy " : "") + "ARG microkernel mismatch at P" +
                                             std::to_string(PacketBytes) + " o=" +
                                             std::to_string(o) + " c=" + std::to_string(c));
            }
        }
    }
    check(cudaFree(input), "free");
    check(cudaFree(output), "free");
}

template <int PacketBytes, int Kind, int MinBlocks = 1>
double timeKernel(const std::array<uint8_t*, BufferCount>& inputs, uint32_t* output, uint32_t outer, uint32_t rows, uint32_t k,
                  size_t shared_bytes = 0) {
    const C::Fp8ArgFold fold = C::fp8ArgFold(false, false);
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event");
    check(cudaEventCreate(&stop), "event");
    std::vector<float> samples;
    for (int i = 0; i < Warmups + Samples; ++i) {
        check(cudaEventRecord(start, benchmarkStream()), "record");
        launch<PacketBytes, Kind, MinBlocks>(inputs[i % BufferCount], output, outer, rows, k, fold, shared_bytes);
        check(cudaEventRecord(stop, benchmarkStream()), "record");
        check(cudaEventSynchronize(stop), "sync");
        float ms = 0;
        check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
        if (i >= Warmups) samples.push_back(ms);
    }
    check(cudaEventDestroy(start), "event");
    check(cudaEventDestroy(stop), "event");
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

// Load roofline uses the uncapped load kernel of the same packet; MinBlocks applies to the ARG kernel only.
template <int PacketBytes, int MinBlocks = 1, int Kind = 1>
void sweep(const std::array<uint8_t*, BufferCount>& inputs, uint32_t* output, size_t shared_bytes = 0) {
    constexpr std::array<std::pair<uint32_t, uint32_t>, 9> shapes{
        {{4, 512}, {16, 512}, {64, 512}, {256, 512}, {1024, 512}, {64, 64}, {64, 16}, {1024, 16}, {16, 4096}}};
    for (const auto& [rows, k] : shapes) {
        if (k % PacketBytes != 0) continue;
        const uint32_t outer = static_cast<uint32_t>(BufferBytes / (static_cast<uint64_t>(rows) * k));
        const double traffic = static_cast<double>(outer) * rows * k + static_cast<double>(outer) * k * 4.0;
        const double load_ms = timeKernel<PacketBytes, 0>(inputs, output, outer, rows, k, shared_bytes);
        const double arg_ms = timeKernel<PacketBytes, Kind, MinBlocks>(inputs, output, outer, rows, k, shared_bytes);
        std::cout << "p" << PacketBytes << (MinBlocks > 1 ? "_b" + std::to_string(MinBlocks) : std::string())
                  << (Kind == 2 ? "_lazy" : "") << (shared_bytes != 0 ? "_occ" : "") << ',' << rows << ',' << k << ',' << outer << ',' << std::fixed
                  << std::setprecision(4) << load_ms << ',' << arg_ms << ',' << std::setprecision(3)
                  << traffic / (load_ms * 1e9) << ',' << traffic / (arg_ms * 1e9) << ',' << arg_ms / load_ms << '\n';
    }
}

}  // namespace

int main() {
    try {
        validate<4>();
        validate<8>();
        validate<16>();
        validate<16, 5>();
        validate<8, 1, 2>();
        validate<16, 1, 2>();
        std::cout << "# validation passed (argmin/argmax x E4M3/E5M2 vs FP32 reference; eager and lazy)\n";
        std::array<uint8_t*, BufferCount> inputs{};
        for (int i = 0; i < BufferCount; ++i) {
            check(cudaMalloc(&inputs[i], BufferBytes), "malloc input");
            fillKernel<<<4096, 256, 0, benchmarkStream()>>>(inputs[i], BufferBytes, 977ULL * (i + 1));
            check(cudaGetLastError(), "fill");
        }
        uint32_t* output = nullptr;
        check(cudaMalloc(&output, BufferBytes * sizeof(uint32_t) / 4), "malloc output");  // R >= 4
        check(cudaStreamSynchronize(benchmarkStream()), "fill sync");
        std::cout << "packet,R,K,outer,load_ms,arg_ms,load_TBps,arg_TBps,arg_over_load\n";
        sweep<16>(inputs, output);
        sweep<16, 5>(inputs, output);
        sweep<8>(inputs, output);
        sweep<4>(inputs, output);
        sweep<8, 1, 2>(inputs, output);
        sweep<16, 1, 2>(inputs, output);
        // Two resident 256-thread blocks per SM: load roofline and ARG kernels under the same cap.
        const size_t occ = residencyCapSharedBytes(2);
        sweep<8>(inputs, output, occ);
        sweep<8, 1, 2>(inputs, output, occ);
        sweep<16, 1, 2>(inputs, output, occ);
        for (uint8_t* input : inputs) check(cudaFree(input), "free");
        check(cudaFree(output), "free");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FP8 ARG key microbenchmark failed: " << e.what() << '\n';
        return 1;
    }
}
