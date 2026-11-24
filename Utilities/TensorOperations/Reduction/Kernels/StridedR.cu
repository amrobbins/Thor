// The StridedR kernels and the stride-0 closed form (rank-1 strided views). The design is in StridedReduction.h. Lean
// on purpose: the scalar helpers only (no CUB, Thrust or cooperative groups), as the ragged objects.
//
// One kernel template over the input type and the combine (0 add, 1 product, 2 min): 5 x 3 = 15 kernels, and the
// closed-form kernel. Split finishes inside its own launch (the last CTA to arrive combines the partials).

#include "Utilities/TensorOperations/Reduction/Kernels/StridedReduction.h"

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/TensorOperations/Reduction/Kernels/KernelLaunchers.h"
#include "Utilities/TensorOperations/Reduction/Kernels/ScalarOps.cuh"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ThorImplementation::ReductionInternal {
namespace {

constexpr uint32_t STRIDED_R_WARP_LANES = 32;
constexpr uint32_t STRIDED_R_WARPS = STRIDED_R_THREADS / STRIDED_R_WARP_LANES;
constexpr uint64_t STRIDED_R_MAX_GRID = 2147483647ULL;
static_assert(STRIDED_R_THREADS % STRIDED_R_WARP_LANES == 0 && STRIDED_R_UNROLL == 8);

void stridedRCheck(cudaError_t status, const char* what) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("StridedR: ") + what + ": " + cudaGetErrorString(status));
}

// The input's storage word and its FP32 value. Every element is its own load at base + i s elements (64-bit).
// The load (__ldg of the word) and the decode are separate steps. A warp waits at the first instruction that needs a
// loaded word, so the kernel issues every load of a group first and decodes after: the loads are in flight together.
template <typename InputT>
struct StridedRElement;

template <>
struct StridedRElement<float> {
    using Bits = float;
    __device__ __forceinline__ static float decode(Bits bits) { return bits; }
};

template <>
struct StridedRElement<__half> {
    using Bits = unsigned short;
    __device__ __forceinline__ static float decode(Bits bits) { return __half2float(__ushort_as_half(bits)); }
};

template <>
struct StridedRElement<__nv_bfloat16> {
    using Bits = unsigned short;
    // A BF16 is the high half of its FP32: the widening is exact.
    __device__ __forceinline__ static float decode(Bits bits) {
        return __uint_as_float(static_cast<uint32_t>(bits) << 16);
    }
};

template <>
struct StridedRElement<__nv_fp8_e4m3> {
    using Bits = unsigned char;
    __device__ __forceinline__ static float decode(Bits bits) {
        return static_cast<float>(ThorLowPrecision::fp8E4M3FromBits(static_cast<__nv_fp8_storage_t>(bits)));
    }
};

template <>
struct StridedRElement<__nv_fp8_e5m2> {
    using Bits = unsigned char;
    __device__ __forceinline__ static float decode(Bits bits) {
        return static_cast<float>(ThorLowPrecision::fp8E5M2FromBits(static_cast<__nv_fp8_storage_t>(bits)));
    }
};

// The input transform (0 identity, 1 |x|, 2 x^2) and MAX's sign fold, uniform for the launch.
__device__ __forceinline__ float stridedRTransform(float value, uint32_t transform, bool negate) {
    if (transform == 1U) {
        value = ::fabsf(value);
    } else if (transform == 2U) {
        value = value * value;
    }
    return negate ? -value : value;
}

// AdditiveFinalizeFp32's arithmetic (ReductionOperation.cuh), out of line: MEAN divides, L2 takes the root. Split's
// last CTA finalizes with it.
__device__ __noinline__ float stridedRFinalizeOutOfLine(float value, float divisor, bool square_root) {
    float finalized = value;
    if (divisor == 0.0f) {
        finalized = 0.0f;
    } else if (divisor != 1.0f) {
        finalized /= divisor;
    }
    return square_root ? ::sqrtf(finalized) : finalized;
}

// The StridedR kernel. CTA c of G owns the logical range [c q + min(c, r), (c + 1) q + min(c + 1, r)), n = q G + r.
// Each CTA iteration covers STRIDED_R_TILE elements: warp w owns 32 U consecutive elements of it and its load u reads
// element w 32 U + 32 u + lane, so a warp load is 32 consecutive logical elements. U accumulators per thread, combined
// in a fixed tree, then the warps in order: the result depends on n, s and G only.
// Every load of a group is issued before any is used, in the full tiles and in the last partial tile (loading, decoding
// and combining each element of the partial tile in turn, behind its own range check, ran a thread's loads one after
// another: about 0.4 us each, 3 us or more on every launch).
// One launch. CTA c stores its unfinalized FP32 partial to partials[c], makes it visible (__threadfence) and then
// counts itself in *ticket (one atomicAdd). The CTA whose count reaches G is the last to arrive, so every partial is
// stored and visible to it: its thread t combines partials t, t + 512, ..., the CTA reduces them in the fixed tree,
// finalizes, scales and stores output[0], and sets *ticket back to 0. Which CTA arrives last varies from run to run;
// the partials and the order they are combined in do not, so the result depends on n, s and G only. *ticket must be 0
// at launch (stridedRInitWorkspace once; every launch leaves it 0).
template <typename InputT, int Combine>
__global__ __launch_bounds__(STRIDED_R_THREADS, 2) void stridedRKernel(
    const typename StridedRElement<InputT>::Bits* __restrict__ input, uint64_t elements, uint64_t stride,
    uint32_t transform, bool negate, float* __restrict__ partials, uint32_t* ticket, void* output,
    DataType output_dtype, float finalize_divisor, bool finalize_square_root, float output_scale) {
    using Element = StridedRElement<InputT>;
    using Bits = typename Element::Bits;
    const Fp8TypedReductionOp<Combine> combine{};
    __shared__ float warp_totals[STRIDED_R_WARPS];
    __shared__ uint32_t last_to_arrive;

    const uint64_t ctas = gridDim.x;
    const uint64_t cta = blockIdx.x;
    const uint64_t q = elements / ctas;
    const uint64_t r = elements % ctas;
    const uint64_t begin = cta * q + (cta < r ? cta : r);
    const uint64_t end = begin + q + (cta < r ? 1U : 0U);

    const uint32_t lane = threadIdx.x % STRIDED_R_WARP_LANES;
    const uint32_t warp = threadIdx.x / STRIDED_R_WARP_LANES;
    const uint64_t thread_first = static_cast<uint64_t>(warp) * STRIDED_R_WARP_LANES * STRIDED_R_UNROLL + lane;
    const uint64_t load_step = uint64_t{STRIDED_R_WARP_LANES} * stride;  // elements between a thread's loads
    const uint64_t tile_step = STRIDED_R_TILE * stride;

    float acc[STRIDED_R_UNROLL];
#pragma unroll
    for (uint32_t u = 0; u < STRIDED_R_UNROLL; ++u) acc[u] = combine.identity();

    uint64_t tile = begin;
    const Bits* p = input + (begin + thread_first) * stride;
    for (; tile + STRIDED_R_TILE <= end; tile += STRIDED_R_TILE, p += tile_step) {
        Bits raw[STRIDED_R_UNROLL];
#pragma unroll
        for (uint32_t u = 0; u < STRIDED_R_UNROLL; ++u) raw[u] = __ldg(p + u * load_step);
#pragma unroll
        for (uint32_t u = 0; u < STRIDED_R_UNROLL; ++u)
            acc[u] = combine(acc[u], stridedRTransform(Element::decode(raw[u]), transform, negate));
    }
    // The last partial tile: the same element mapping. This thread's load u is inside the range while u < count (its
    // elements ascend with u). The first loop only loads, so the count loads are in flight together; the second
    // decodes every slot (a slot not loaded holds zero) and combines the loaded ones.
    const uint64_t first = tile + thread_first;
    const uint64_t loads = first < end ? (end - first + (STRIDED_R_WARP_LANES - 1)) / STRIDED_R_WARP_LANES : 0;
    const uint32_t count = loads < STRIDED_R_UNROLL ? static_cast<uint32_t>(loads) : STRIDED_R_UNROLL;
    Bits raw[STRIDED_R_UNROLL];
#pragma unroll
    for (uint32_t u = 0; u < STRIDED_R_UNROLL; ++u) {
        raw[u] = Bits{};
        if (u < count) raw[u] = __ldg(p + u * load_step);
    }
#pragma unroll
    for (uint32_t u = 0; u < STRIDED_R_UNROLL; ++u) {
        const float value = stridedRTransform(Element::decode(raw[u]), transform, negate);
        acc[u] = u < count ? combine(acc[u], value) : acc[u];
    }

    float value = combine(combine(combine(acc[0], acc[1]), combine(acc[2], acc[3])),
                          combine(combine(acc[4], acc[5]), combine(acc[6], acc[7])));
    for (uint32_t offset = STRIDED_R_WARP_LANES / 2; offset != 0; offset >>= 1) {
        value = combine(value, __shfl_xor_sync(0xffffffffU, value, static_cast<int>(offset)));
    }
    if (lane == 0) warp_totals[warp] = value;
    __syncthreads();

    // Thread 0 stores the CTA's partial, fences it, then counts the CTA; the others wait for its answer.
    if (threadIdx.x == 0) {
        float partial = combine.identity();
#pragma unroll
        for (uint32_t w = 0; w < STRIDED_R_WARPS; ++w) partial = combine(partial, warp_totals[w]);
        partials[cta] = partial;
        __threadfence();
        last_to_arrive = atomicAdd(ticket, 1U) + 1U == static_cast<uint32_t>(ctas) ? 1U : 0U;
    }
    __syncthreads();
    if (last_to_arrive == 0U) return;

    // The last CTA to arrive. The partials were stored by other CTAs during this launch, so they are read with plain
    // (volatile) loads, never __ldg. With G <= 512 each thread reads at most one, all in flight together.
    if (threadIdx.x == 0) *ticket = 0U;
    const volatile float* const stored = partials;
    float part = combine.identity();
    for (uint64_t c = threadIdx.x; c < ctas; c += STRIDED_R_THREADS) part = combine(part, stored[c]);
    for (uint32_t offset = STRIDED_R_WARP_LANES / 2; offset != 0; offset >>= 1) {
        part = combine(part, __shfl_xor_sync(0xffffffffU, part, static_cast<int>(offset)));
    }
    if (lane == 0) warp_totals[warp] = part;
    __syncthreads();
    if (threadIdx.x != 0) return;
    float total = combine.identity();
#pragma unroll
    for (uint32_t w = 0; w < STRIDED_R_WARPS; ++w) total = combine(total, warp_totals[w]);
    if constexpr (Combine == 0) {
        if (finalize_divisor != 1.0f || finalize_square_root) {
            total = stridedRFinalizeOutOfLine(total, finalize_divisor, finalize_square_root);
        }
    }
    storeFp32AsRuntimeDType(output, output_dtype, 0, total * output_scale);
}

__device__ __forceinline__ float stridedRLoadRuntime(const void* input, DataType dtype) {
    switch (dtype) {
        case DataType::FP8_E4M3:
            return StridedRElement<__nv_fp8_e4m3>::decode(__ldg(static_cast<const unsigned char*>(input)));
        case DataType::FP8_E5M2:
            return StridedRElement<__nv_fp8_e5m2>::decode(__ldg(static_cast<const unsigned char*>(input)));
        case DataType::FP16:
            return StridedRElement<__half>::decode(__ldg(static_cast<const unsigned short*>(input)));
        case DataType::BF16:
            return StridedRElement<__nv_bfloat16>::decode(__ldg(static_cast<const unsigned short*>(input)));
        default:
            return StridedRElement<float>::decode(__ldg(static_cast<const float*>(input)));
    }
}

// The stride-0 closed form: one thread, x read once, the operation in FP64, one rounding to FP32, then output_scale.
// FP64 holds x n for every FP32 x (|x| n < 2^128 2^64), exactly while n < 2^29. Additive results start from +0, as the
// summation does, so a -0 input gives +0 (MIN / MAX / PRODUCT keep -0's sign as their combines do). PRODUCT is x^n by
// squaring with the integer n, so its sign follows n's parity at every n.
__global__ void stridedRClosedFormKernel(const void* input, DataType input_dtype, uint64_t elements, uint32_t op,
                                         void* output, DataType output_dtype, float output_scale) {
    const double x = static_cast<double>(stridedRLoadRuntime(input, input_dtype));
    const double n = static_cast<double>(elements);
    double result = x;
    switch (static_cast<ReductionOp>(op)) {
        case ReductionOp::Sum:
            result = 0.0 + x * n;
            break;
        case ReductionOp::Mean:
            result = 0.0 + x;
            break;
        case ReductionOp::Min:
        case ReductionOp::Max:
            result = x;
            break;
        case ReductionOp::Product: {
            double base = x;
            result = 1.0;
            for (uint64_t e = elements; e != 0;) {
                if ((e & 1U) != 0U) result *= base;
                e >>= 1;
                if (e != 0) base *= base;
            }
            break;
        }
        case ReductionOp::L1Norm:
            result = ::fabs(x) * n;
            break;
        case ReductionOp::SumSquares:
            result = (x * x) * n;
            break;
        case ReductionOp::L2Norm:
            result = ::fabs(x) * ::sqrt(n);
            break;
    }
    storeFp32AsRuntimeDType(output, output_dtype, 0, static_cast<float>(result) * output_scale);
}

struct StridedROperation {
    int combine = 0;         // 0 add, 1 product, 2 min
    uint32_t transform = 0;  // 0 identity, 1 |x|, 2 x^2
    bool negate = false;     // MAX: the min of -x, with a negated scale
    float finalize_divisor = 1.0f;
    bool finalize_square_root = false;
};

StridedROperation stridedROperation(ReductionOp op, uint64_t elements) {
    StridedROperation out;
    switch (op) {
        case ReductionOp::Sum: break;
        case ReductionOp::Mean: out.finalize_divisor = static_cast<float>(elements); break;
        case ReductionOp::L1Norm: out.transform = 1; break;
        case ReductionOp::SumSquares: out.transform = 2; break;
        case ReductionOp::L2Norm:
            out.transform = 2;
            out.finalize_square_root = true;
            break;
        case ReductionOp::Product: out.combine = 1; break;
        case ReductionOp::Min: out.combine = 2; break;
        case ReductionOp::Max:
            out.combine = 2;
            out.negate = true;
            break;
        default: throw std::invalid_argument("StridedR: unsupported reduction operation");
    }
    return out;
}

template <typename InputT, int Combine>
const void* stridedRKernelAddress() {
    return reinterpret_cast<const void*>(&stridedRKernel<InputT, Combine>);
}

// Calls fn.template operator()<InputT, Combine>() for the runtime input dtype and combine.
template <typename Fn>
decltype(auto) dispatchStridedR(DataType input_dtype, int combine, Fn&& fn) {
    const auto by_combine = [&]<typename InputT>() -> decltype(auto) {
        switch (combine) {
            case 0: return fn.template operator()<InputT, 0>();
            case 1: return fn.template operator()<InputT, 1>();
            case 2: return fn.template operator()<InputT, 2>();
        }
        throw std::logic_error("StridedR: unknown combine");
    };
    switch (input_dtype) {
        case DataType::FP8_E4M3: return by_combine.template operator()<__nv_fp8_e4m3>();
        case DataType::FP8_E5M2: return by_combine.template operator()<__nv_fp8_e5m2>();
        case DataType::FP16: return by_combine.template operator()<__half>();
        case DataType::BF16: return by_combine.template operator()<__nv_bfloat16>();
        case DataType::FP32: return by_combine.template operator()<float>();
        default: break;
    }
    throw std::invalid_argument("StridedR: unsupported input dtype");
}

void requireStridedRInput(const StridedSpan& input) {
    if (!input.isInitialized() || input.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU)
        throw std::invalid_argument("StridedR: the input must be an initialized GPU span");
    if (input.getDimensions().size() != 1 || input.getDimensions()[0] == 0)
        throw std::invalid_argument("StridedR: the input must be a nonempty rank-1 span");
    if (!stridedRSupportsInput(input.getDataType()))
        throw std::invalid_argument("StridedR: unsupported input dtype");
}

}  // namespace

std::string stridedRLaunchName(const StridedRLaunch& launch) {
    switch (launch.family) {
        case StridedRFamily::ClosedForm: return "strided_r:closed_form";
        case StridedRFamily::Split: return "strided_r:split_g" + std::to_string(launch.grid);
    }
    return "strided_r:unknown";
}

bool stridedRSupportsInput(DataType dtype) {
    switch (dtype) {
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
            return true;
        default:
            return false;
    }
}

bool stridedRSupportsOutput(DataType dtype) { return stridedRSupportsInput(dtype); }

uint32_t stridedRPersistentGrid(int gpu) {
    ScopedGpu scoped_gpu(gpu);
    int sms = 0;
    stridedRCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, gpu), "SM count");
    // Every instantiation has the same CTA and launch bounds; the FP32 add kernel stands for them (the register counts
    // differ by a few at most, and the launch bounds cap them at two CTAs' worth).
    int resident = 0;
    stridedRCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, stridedRKernelAddress<float, 0>(),
                                                                static_cast<int>(STRIDED_R_THREADS), 0),
                  "resident CTAs");
    const uint64_t grid = static_cast<uint64_t>(std::max(sms, 1)) * static_cast<uint64_t>(std::max(resident, 1));
    return static_cast<uint32_t>(std::min<uint64_t>(grid, STRIDED_R_MAX_GRID));
}

uint32_t stridedRSplitGrid(uint64_t elements, int gpu) {
    // A gather is latency-bound: every CTA issues all of its loads at once, so the grid stays the persistent one until
    // a CTA would have fewer elements than threads.
    const uint64_t ctas = std::max<uint64_t>(1, (elements + STRIDED_R_THREADS - 1) / STRIDED_R_THREADS);
    return static_cast<uint32_t>(std::min<uint64_t>(stridedRPersistentGrid(gpu), ctas));
}

StridedRLaunch stridedRSplitLaunch(uint32_t grid) {
    if (grid == 0 || grid > STRIDED_R_MAX_GRID) throw std::invalid_argument("StridedR: Split needs 1 to 2^31 - 1 CTAs");
    // G FP32 partials, then the 4-byte ticket.
    return StridedRLaunch{StridedRFamily::Split, grid, (uint64_t{grid} + 1) * sizeof(float)};
}

StridedRLaunch stridedRLaunchFor(const StridedSpan& input, DataType output_dtype, int gpu) {
    requireStridedRInput(input);
    if (!stridedRSupportsOutput(output_dtype)) throw std::invalid_argument("StridedR: unsupported output dtype");
    if (input.getStridesElements()[0] == 0) return StridedRLaunch{StridedRFamily::ClosedForm, 1, 0};
    return stridedRSplitLaunch(stridedRSplitGrid(input.getDimensions()[0], gpu));
}

void stridedRInitWorkspace(const StridedSpan& workspace, const Stream& stream) {
    if (!workspace.isInitialized() || workspace.getArraySizeInBytes() == 0) return;
    ScopedGpu scoped_gpu(stream.getGpuNum());
    stridedRCheck(cudaMemsetAsync(workspace.getMemPtr<void>(), 0, workspace.getArraySizeInBytes(), stream.getStream()),
                  "cudaMemsetAsync(workspace)");
    stridedRCheck(cudaStreamSynchronize(stream.getStream()), "workspace initialization");
}

void launchStridedR(ReductionOp op,
                    const StridedRLaunch& launch,
                    const StridedSpan& input,
                    const StridedSpan& output,
                    const StridedSpan& workspace,
                    float output_scale,
                    Stream& stream) {
    requireStridedRInput(input);
    if (!output.isInitialized() || output.getTotalNumElements() != 1 || !stridedRSupportsOutput(output.getDataType()))
        throw std::invalid_argument("StridedR: the output must be one element of FP8, FP16, BF16 or FP32");
    const uint64_t elements = input.getDimensions()[0];
    const uint64_t stride = input.getStridesElements()[0];
    const StridedROperation operation = stridedROperation(op, elements);
    void* out = output.getMemPtr<void>();
    const DataType out_dtype = output.getDataType();
    ScopedGpu scoped_gpu(stream.getGpuNum());

    switch (launch.family) {
        case StridedRFamily::ClosedForm:
            if (stride != 0) throw std::invalid_argument("StridedR: the closed form needs stride 0");
            stridedRClosedFormKernel<<<1, 1, 0, stream.getStream()>>>(input.getMemPtr<void>(), input.getDataType(),
                                                                      elements, static_cast<uint32_t>(op), out,
                                                                      out_dtype, output_scale);
            stridedRCheck(cudaGetLastError(), "launch");
            return;
        case StridedRFamily::Split:
            break;
        default:
            throw std::invalid_argument("StridedR: unknown family");
    }

    const uint32_t grid = launch.grid;
    if (grid == 0 || grid > STRIDED_R_MAX_GRID) throw std::invalid_argument("StridedR: Split needs 1 to 2^31 - 1 CTAs");
    if (!workspace.isInitialized() || workspace.getArraySizeInBytes() < (uint64_t{grid} + 1) * sizeof(float) ||
        reinterpret_cast<uintptr_t>(workspace.getMemPtr<void>()) % alignof(float) != 0)
        throw std::invalid_argument("StridedR: Split needs a 4-byte aligned workspace of G FP32 partials and the ticket");
    float* const partials = workspace.getMemPtr<float>();
    uint32_t* const ticket = reinterpret_cast<uint32_t*>(partials + grid);  // 0 between launches (stridedRInitWorkspace)
    // MAX: the min of -x, undone by the scale's sign.
    const float scale = operation.negate ? -output_scale : output_scale;
    dispatchStridedR(input.getDataType(), operation.combine, [&]<typename InputT, int Combine>() {
        using Bits = typename StridedRElement<InputT>::Bits;
        stridedRKernel<InputT, Combine><<<grid, STRIDED_R_THREADS, 0, stream.getStream()>>>(
            input.getMemPtr<Bits>(), elements, stride, operation.transform, operation.negate, partials, ticket, out,
            out_dtype, operation.finalize_divisor, operation.finalize_square_root, scale);
    });
    stridedRCheck(cudaGetLastError(), "launch");
}

}  // namespace ThorImplementation::ReductionInternal
