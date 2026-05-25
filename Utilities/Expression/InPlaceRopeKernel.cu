#include "Utilities/Expression/InPlaceRopeKernel.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "Utilities/Expression/CudaHelpers.h"

namespace ThorImplementation {
namespace {

template <typename T>
__device__ float ropeLoad(const T* ptr, uint32_t idx) {
    return static_cast<float>(ptr[idx]);
}

template <>
__device__ float ropeLoad<half>(const half* ptr, uint32_t idx) {
    return __half2float(ptr[idx]);
}

template <>
__device__ float ropeLoad<__nv_bfloat16>(const __nv_bfloat16* ptr, uint32_t idx) {
    return __bfloat162float(ptr[idx]);
}

template <typename T>
__device__ T ropeStore(float value) {
    return static_cast<T>(value);
}

template <>
__device__ half ropeStore<half>(float value) {
    return __float2half(value);
}

template <>
__device__ __nv_bfloat16 ropeStore<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

template <typename T>
struct RopePairsPerThread {
    // Each thread handles 16 bytes worth of input elements and writes 16 bytes worth of output elements.
    // RoPE works on element pairs, so the count is always even for the supported dtypes.
    static constexpr uint32_t value = 16U / (2U * static_cast<uint32_t>(sizeof(T)));
};

static_assert(RopePairsPerThread<half>::value == 4U);
static_assert(RopePairsPerThread<__nv_bfloat16>::value == 4U);
static_assert(RopePairsPerThread<float>::value == 2U);

template <typename T>
__device__ void rotateOneTensorPair(T* ptr,
                                    uint32_t pair_linear,
                                    uint32_t sequence_length,
                                    uint32_t num_heads,
                                    uint32_t head_dim,
                                    uint32_t rotary_dim,
                                    float rope_base_in,
                                    int32_t position_offset,
                                    bool interleaved,
                                    bool inverse,
                                    uint32_t scaling_kind,
                                    float scaling_factor,
                                    uint32_t original_max_position_embeddings,
                                    float attention_factor,
                                    float yarn_beta_fast,
                                    float yarn_beta_slow,
                                    float llama3_low_freq_factor,
                                    float llama3_high_freq_factor) {
    const uint32_t half_dim = rotary_dim >> 1U;
    const uint32_t pair_index = pair_linear % half_dim;
    uint32_t tmp = pair_linear / half_dim;
    const uint32_t head = tmp % num_heads;
    tmp /= num_heads;
    const uint32_t seq = tmp % sequence_length;
    const uint32_t batch = tmp / sequence_length;

    const uint32_t base = (((batch * sequence_length + seq) * num_heads + head) * head_dim);
    const uint32_t lane0 = interleaved ? (base + 2U * pair_index) : (base + pair_index);
    const uint32_t lane1 = interleaved ? (lane0 + 1U) : (base + pair_index + half_dim);

    float position = static_cast<float>(seq) + static_cast<float>(position_offset);
    if (scaling_kind == 1U) {  // RotaryScalingKind::Linear
        position = position / scaling_factor;
    }

    float rope_base = rope_base_in;
    float freq = 0.0f;
    if (scaling_kind == 2U) {  // RotaryScalingKind::DynamicNTK
        const int32_t positive_offset = position_offset > 0 ? position_offset : 0;
        const float seq_len = fmaxf(static_cast<float>(sequence_length) + static_cast<float>(positive_offset), 1.0f);
        const float original_max = static_cast<float>(original_max_position_embeddings);
        if (seq_len > original_max && rotary_dim > 2U) {
            const float ntk_ratio = (scaling_factor * seq_len / original_max) - (scaling_factor - 1.0f);
            rope_base = rope_base * powf(ntk_ratio, static_cast<float>(rotary_dim) / static_cast<float>(rotary_dim - 2U));
        }
        freq = powf(rope_base, -2.0f * static_cast<float>(pair_index) / static_cast<float>(rotary_dim));
    } else if (scaling_kind == 3U) {  // RotaryScalingKind::Yarn
        const float pos_freq = powf(rope_base, 2.0f * static_cast<float>(pair_index) / static_cast<float>(rotary_dim));
        const float inv_freq_extrap = 1.0f / pos_freq;
        const float inv_freq_interp = 1.0f / (scaling_factor * pos_freq);
        const float original_max = static_cast<float>(original_max_position_embeddings);
        const float log_base = logf(rope_base);
        float low = (static_cast<float>(rotary_dim) * logf(original_max / (yarn_beta_fast * 6.2831853071795864769f))) /
                    (2.0f * log_base);
        float high = (static_cast<float>(rotary_dim) * logf(original_max / (yarn_beta_slow * 6.2831853071795864769f))) /
                     (2.0f * log_base);
        low = fmaxf(floorf(low), 0.0f);
        high = fminf(ceilf(high), static_cast<float>(rotary_dim - 1U));
        if (low == high) {
            high += 0.001f;
        }
        const float ramp = fminf(fmaxf((static_cast<float>(pair_index) - low) / (high - low), 0.0f), 1.0f);
        freq = inv_freq_interp * ramp + inv_freq_extrap * (1.0f - ramp);
    } else if (scaling_kind == 5U) {  // RotaryScalingKind::Llama3
        const float inv_freq = powf(rope_base, -2.0f * static_cast<float>(pair_index) / static_cast<float>(rotary_dim));
        const float wavelen = 6.2831853071795864769f / inv_freq;
        const float original_max = static_cast<float>(original_max_position_embeddings);
        const float low_freq_wavelen = original_max / llama3_low_freq_factor;
        const float high_freq_wavelen = original_max / llama3_high_freq_factor;
        if (wavelen > low_freq_wavelen) {
            freq = inv_freq / scaling_factor;
        } else if (wavelen < high_freq_wavelen) {
            freq = inv_freq;
        } else {
            const float smooth = (original_max / wavelen - llama3_low_freq_factor) / (llama3_high_freq_factor - llama3_low_freq_factor);
            freq = (1.0f - smooth) * (inv_freq / scaling_factor) + smooth * inv_freq;
        }
    } else {
        freq = powf(rope_base, -2.0f * static_cast<float>(pair_index) / static_cast<float>(rotary_dim));
    }

    const float theta = position * freq;
    float s;
    float c;
    sincosf(theta, &s, &c);
    s *= attention_factor;
    c *= attention_factor;
    if (inverse) {
        s = -s;
    }

    const float x0 = ropeLoad(ptr, lane0);
    const float x1 = ropeLoad(ptr, lane1);
    const float y0 = x0 * c - x1 * s;
    const float y1 = x1 * c + x0 * s;
    ptr[lane0] = ropeStore<T>(y0);
    ptr[lane1] = ropeStore<T>(y1);
}

template <typename T>
__device__ void rotateTensorChunk(T* ptr,
                                  uint32_t pair_begin,
                                  uint32_t pair_count,
                                  uint32_t sequence_length,
                                  uint32_t num_heads,
                                  uint32_t head_dim,
                                  uint32_t rotary_dim,
                                  float rope_base,
                                  int32_t position_offset,
                                  bool interleaved,
                                  bool inverse,
                                  uint32_t scaling_kind,
                                  float scaling_factor,
                                  uint32_t original_max_position_embeddings,
                                  float attention_factor,
                                  float yarn_beta_fast,
                                  float yarn_beta_slow,
                                  float llama3_low_freq_factor,
                                  float llama3_high_freq_factor) {
    for (uint32_t i = 0; i < RopePairsPerThread<T>::value; ++i) {
        const uint32_t pair = pair_begin + i;
        if (pair < pair_count) {
            rotateOneTensorPair(ptr,
                                pair,
                                sequence_length,
                                num_heads,
                                head_dim,
                                rotary_dim,
                                rope_base,
                                position_offset,
                                interleaved,
                                inverse,
                                scaling_kind,
                                scaling_factor,
                                original_max_position_embeddings,
                                attention_factor,
                                yarn_beta_fast,
                                yarn_beta_slow,
                                llama3_low_freq_factor,
                                llama3_high_freq_factor);
        }
    }
}

template <typename T>
__global__ void groupedInPlaceRopeKernel(T* q,
                                         uint32_t q_pairs,
                                         uint32_t q_sequence_length,
                                         uint32_t q_num_heads,
                                         uint32_t q_head_dim,
                                         T* k,
                                         uint32_t k_pairs,
                                         uint32_t k_sequence_length,
                                         uint32_t k_num_heads,
                                         uint32_t k_head_dim,
                                         uint32_t rotary_dim,
                                         float rope_base,
                                         int32_t position_offset,
                                         bool interleaved,
                                         bool inverse,
                                         uint32_t scaling_kind,
                                         float scaling_factor,
                                         uint32_t original_max_position_embeddings,
                                         float attention_factor,
                                         float yarn_beta_fast,
                                         float yarn_beta_slow,
                                         float llama3_low_freq_factor,
                                         float llama3_high_freq_factor) {
    const uint32_t chunk = (blockIdx.x * blockDim.x + threadIdx.x) * RopePairsPerThread<T>::value;
    if (chunk < q_pairs) {
        rotateTensorChunk(q,
                          chunk,
                          q_pairs,
                          q_sequence_length,
                          q_num_heads,
                          q_head_dim,
                          rotary_dim,
                          rope_base,
                          position_offset,
                          interleaved,
                          inverse,
                          scaling_kind,
                          scaling_factor,
                          original_max_position_embeddings,
                          attention_factor,
                          yarn_beta_fast,
                          yarn_beta_slow,
                          llama3_low_freq_factor,
                          llama3_high_freq_factor);
    }
    if (k != nullptr && chunk < k_pairs) {
        rotateTensorChunk(k,
                          chunk,
                          k_pairs,
                          k_sequence_length,
                          k_num_heads,
                          k_head_dim,
                          rotary_dim,
                          rope_base,
                          position_offset,
                          interleaved,
                          inverse,
                          scaling_kind,
                          scaling_factor,
                          original_max_position_embeddings,
                          attention_factor,
                          yarn_beta_fast,
                          yarn_beta_slow,
                          llama3_low_freq_factor,
                          llama3_high_freq_factor);
    }
}

struct RopeTensorLaunchDims {
    uint32_t batch = 0;
    uint32_t sequence = 0;
    uint32_t heads = 0;
    uint32_t head_dim = 0;
    uint32_t rotary_dim = 0;
    uint32_t pair_count = 0;
};

uint32_t checkedU32(uint64_t value, const char* name) {
    if (value > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error(std::string("In-place RoPE ") + name + " exceeds uint32_t range.");
    }
    return static_cast<uint32_t>(value);
}

int32_t checkedI32(int64_t value, const char* name) {
    if (value < static_cast<int64_t>(std::numeric_limits<int32_t>::min()) ||
        value > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error(std::string("In-place RoPE ") + name + " exceeds int32_t range.");
    }
    return static_cast<int32_t>(value);
}

uint32_t checkedMulU32(uint32_t a, uint32_t b, const char* name) {
    const uint64_t product = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
    if (product > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error(std::string("In-place RoPE ") + name + " exceeds uint32_t range.");
    }
    return static_cast<uint32_t>(product);
}

RopeTensorLaunchDims validateTensorForInPlaceRope(const Tensor& tensor, const RotaryPositionEmbeddingOptions& options) {
    if (!tensor.isInitialized()) {
        throw std::runtime_error("In-place RoPE tensor is not initialized.");
    }
    if (tensor.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::runtime_error("In-place RoPE tensor must be on GPU.");
    }
    const std::vector<uint64_t>& dims = tensor.getDimensions();
    if (dims.size() != 4 || options.sequence_axis != 1 || options.head_dim_axis != 3) {
        throw std::runtime_error("In-place RoPE currently supports dense BSHD tensors only.");
    }
    RopeTensorLaunchDims out;
    out.batch = checkedU32(dims[0], "batch dimension");
    out.sequence = checkedU32(dims[1], "sequence dimension");
    out.heads = checkedU32(dims[2], "head dimension count");
    out.head_dim = checkedU32(dims[3], "head-dim dimension");
    out.rotary_dim = options.rotary_dim == 0 ? out.head_dim : checkedU32(options.rotary_dim, "rotary_dim");
    if (out.rotary_dim == 0 || (out.rotary_dim & 1U) != 0U || out.rotary_dim > out.head_dim) {
        throw std::runtime_error("In-place RoPE rotary_dim must be even, non-zero, and <= head_dim.");
    }
    const uint32_t half_dim = out.rotary_dim >> 1U;
    const uint32_t batch_sequence = checkedMulU32(out.batch, out.sequence, "batch*sequence");
    const uint32_t tokens_heads = checkedMulU32(batch_sequence, out.heads, "batch*sequence*heads");
    out.pair_count = checkedMulU32(tokens_heads, half_dim, "pair count");
    return out;
}

void validateSharedOptions(const std::vector<RotaryPositionEmbeddingOptions>& options) {
    if (options.empty()) {
        throw std::runtime_error("In-place RoPE requires at least one options entry.");
    }
    const RotaryPositionEmbeddingOptions& first = options.front();
    if (first.base <= 0.0 || first.scaling_factor <= 0.0) {
        throw std::runtime_error("In-place RoPE base/scaling_factor must be positive.");
    }
    if (first.scaling_kind == RotaryScalingKind::LongRope) {
        throw std::runtime_error("In-place RoPE does not support LongRoPE factor lists; use out-of-place RoPE materialization.");
    }
    for (const RotaryPositionEmbeddingOptions& opt : options) {
        if (opt.sequence_axis != first.sequence_axis || opt.head_dim_axis != first.head_dim_axis || opt.rotary_dim != first.rotary_dim ||
            opt.base != first.base || opt.position_offset != first.position_offset || opt.interleaved != first.interleaved ||
            opt.inverse != first.inverse || opt.scaling_kind != first.scaling_kind || opt.scaling_factor != first.scaling_factor ||
            opt.original_max_position_embeddings != first.original_max_position_embeddings || opt.attention_factor != first.attention_factor ||
            opt.yarn_beta_fast != first.yarn_beta_fast || opt.yarn_beta_slow != first.yarn_beta_slow ||
            opt.llama3_low_freq_factor != first.llama3_low_freq_factor || opt.llama3_high_freq_factor != first.llama3_high_freq_factor) {
            throw std::runtime_error("Grouped in-place RoPE requires identical RoPE options for all tensors.");
        }
    }
}

template <typename T>
void launchGroupedInPlaceRope(std::vector<Tensor>& tensors,
                              const std::vector<RotaryPositionEmbeddingOptions>& options,
                              const Stream& stream) {
    if (tensors.empty() || tensors.size() > 2 || tensors.size() != options.size()) {
        throw std::runtime_error("Grouped in-place RoPE currently supports one or two tensors.");
    }

    const RotaryPositionEmbeddingOptions& opt = options.front();
    RopeTensorLaunchDims q = validateTensorForInPlaceRope(tensors[0], opt);
    RopeTensorLaunchDims k{};
    T* k_ptr = nullptr;
    if (tensors.size() == 2) {
        k = validateTensorForInPlaceRope(tensors[1], opt);
        k_ptr = static_cast<T*>(tensors[1].getMemPtr());
    }
    const uint32_t launch_pairs = std::max(q.pair_count, k.pair_count);
    if (launch_pairs == 0) {
        return;
    }

    const uint32_t launch_chunks = (launch_pairs + RopePairsPerThread<T>::value - 1U) / RopePairsPerThread<T>::value;
    constexpr uint32_t block = 256;
    const uint32_t grid = (launch_chunks + block - 1U) / block;
    groupedInPlaceRopeKernel<T><<<grid, block, 0, stream.getStream()>>>(static_cast<T*>(tensors[0].getMemPtr()),
                                                                       q.pair_count,
                                                                       q.sequence,
                                                                       q.heads,
                                                                       q.head_dim,
                                                                       k_ptr,
                                                                       k.pair_count,
                                                                       k.sequence,
                                                                       k.heads,
                                                                       k.head_dim,
                                                                       q.rotary_dim,
                                                                       static_cast<float>(opt.base),
                                                                       checkedI32(opt.position_offset, "position_offset"),
                                                                       opt.interleaved,
                                                                       opt.inverse,
                                                                       static_cast<uint32_t>(opt.scaling_kind),
                                                                       static_cast<float>(opt.scaling_factor),
                                                                       checkedU32(opt.original_max_position_embeddings,
                                                                                  "original_max_position_embeddings"),
                                                                       static_cast<float>(opt.attention_factor.value_or(1.0)),
                                                                       static_cast<float>(opt.yarn_beta_fast),
                                                                       static_cast<float>(opt.yarn_beta_slow),
                                                                       static_cast<float>(opt.llama3_low_freq_factor),
                                                                       static_cast<float>(opt.llama3_high_freq_factor));
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void runGroupedInPlaceRotaryPositionEmbedding(std::vector<Tensor>& tensors,
                                              const std::vector<RotaryPositionEmbeddingOptions>& options,
                                              const Stream& stream) {
    if (tensors.empty() || tensors.size() > 2 || tensors.size() != options.size()) {
        throw std::runtime_error("Grouped in-place RoPE currently supports one or two tensors.");
    }
    validateSharedOptions(options);
    const DataType dtype = tensors[0].getDataType();
    for (const Tensor& tensor : tensors) {
        if (tensor.getDataType() != dtype) {
            throw std::runtime_error("Grouped in-place RoPE tensors must have identical dtypes.");
        }
        if (tensor.getPlacement().getDeviceNum() != tensors[0].getPlacement().getDeviceNum()) {
            throw std::runtime_error("Grouped in-place RoPE tensors must be on the same GPU.");
        }
    }

    switch (dtype) {
        case DataType::FP16:
            launchGroupedInPlaceRope<half>(tensors, options, stream);
            break;
        case DataType::BF16:
            launchGroupedInPlaceRope<__nv_bfloat16>(tensors, options, stream);
            break;
        case DataType::FP32:
            launchGroupedInPlaceRope<float>(tensors, options, stream);
            break;
        default:
            throw std::runtime_error("Grouped in-place RoPE supports FP16, BF16, and FP32 tensors.");
    }
}

}  // namespace ThorImplementation
