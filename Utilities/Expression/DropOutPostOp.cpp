#include "Utilities/Expression/DropOutPostOp.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "Utilities/TensorOperations/Scalar/SetScalar.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace ThorImplementation {
namespace {

std::string storagePrelude(DataType dataType, const std::string& debugName) {
    switch (dataType) {
        case DataType::FP16:
            return R"cuda(
#include <cuda_fp16.h>
typedef half ThorDropoutStorage;
typedef float ThorDropoutAccum;
struct __align__(8) ThorDropoutVector4 { ThorDropoutStorage values[4]; };
__device__ __forceinline__ ThorDropoutAccum thor_dropout_load(ThorDropoutStorage v) { return __half2float(v); }
__device__ __forceinline__ ThorDropoutStorage thor_dropout_store(ThorDropoutAccum v) { return __float2half_rn(v); }
)cuda";
        case DataType::BF16:
            return R"cuda(
#include <cuda_bf16.h>
typedef __nv_bfloat16 ThorDropoutStorage;
typedef float ThorDropoutAccum;
struct __align__(8) ThorDropoutVector4 { ThorDropoutStorage values[4]; };
__device__ __forceinline__ ThorDropoutAccum thor_dropout_load(ThorDropoutStorage v) { return __bfloat162float(v); }
__device__ __forceinline__ ThorDropoutStorage thor_dropout_store(ThorDropoutAccum v) { return __float2bfloat16_rn(v); }
)cuda";
        case DataType::FP32:
            return R"cuda(
typedef float ThorDropoutStorage;
typedef float ThorDropoutAccum;
struct __align__(16) ThorDropoutVector4 { ThorDropoutStorage values[4]; };
__device__ __forceinline__ ThorDropoutAccum thor_dropout_load(ThorDropoutStorage v) { return v; }
__device__ __forceinline__ ThorDropoutStorage thor_dropout_store(ThorDropoutAccum v) { return v; }
)cuda";
        default:
            throw std::invalid_argument(debugName + " dropout post-op supports FP16, BF16, and FP32 storage.");
    }
}

std::string offsetPrelude(DataType offsetsDataType, const std::string& debugName) {
    if (offsetsDataType == DataType::UINT32) return "typedef uint32_t ThorDropoutOffset;\n";
    if (offsetsDataType == DataType::UINT64) return "typedef uint64_t ThorDropoutOffset;\n";
    throw std::invalid_argument(debugName + " ragged dropout post-op offsets must use UINT32 or UINT64.");
}

std::string philoxSource() {
    return R"cuda(
struct ThorDropoutUint4 { uint32_t x; uint32_t y; uint32_t z; uint32_t w; };
__device__ __forceinline__ ThorDropoutUint4 thor_dropout_philox_round(ThorDropoutUint4 c, uint32_t k0, uint32_t k1) {
    const uint32_t M0 = 0xD2511F53U;
    const uint32_t M1 = 0xCD9E8D57U;
    const uint32_t hi0 = __umulhi(M0, c.x);
    const uint32_t lo0 = M0 * c.x;
    const uint32_t hi1 = __umulhi(M1, c.z);
    const uint32_t lo1 = M1 * c.z;
    ThorDropoutUint4 out = {hi1 ^ c.y ^ k0, lo1, hi0 ^ c.w ^ k1, lo0};
    return out;
}
__device__ __forceinline__ ThorDropoutUint4 thor_dropout_philox(uint64_t group, uint64_t seed, uint64_t sequence) {
    const uint32_t W0 = 0x9E3779B9U;
    const uint32_t W1 = 0xBB67AE85U;
    ThorDropoutUint4 c = {static_cast<uint32_t>(group), static_cast<uint32_t>(group >> 32U),
                          static_cast<uint32_t>(sequence), static_cast<uint32_t>(sequence >> 32U)};
    uint32_t k0 = static_cast<uint32_t>(seed);
    uint32_t k1 = static_cast<uint32_t>(seed >> 32U);
#pragma unroll
    for (int round = 0; round < 10; ++round) {
        c = thor_dropout_philox_round(c, k0, k1);
        if (round != 9) { k0 += W0; k1 += W1; }
    }
    return c;
}
__device__ __forceinline__ bool thor_dropout_keep(uint32_t bits, float probability) {
    const float unit = (static_cast<float>(bits) + 1.0f) * 2.3283064365386963e-10f;
    return unit > probability;
}
)cuda";
}

uint32_t launchGridForNumel(uint64_t numel) {
    constexpr uint32_t block = 256;
    constexpr uint64_t elementsPerThread = 4;
    const uint64_t threads = (numel + elementsPerThread - 1) / elementsPerThread;
    return static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>((threads + block - 1) / block, 65535)));
}

std::string kernelNameForDebugName(const std::string& debugName) {
    std::string kernelName = debugName;
    for (char& c : kernelName) {
        const unsigned char uc = static_cast<unsigned char>(c);
        if (!std::isalnum(uc) && c != '_') c = '_';
    }
    if (kernelName.empty()) kernelName = "dropout_postop";
    return kernelName;
}

}  // namespace

DropOutRuntimeState::DropOutRuntimeState(int64_t seed, int64_t initialSequence, std::string ownerName)
    : seed(seed), nextSequence(initialSequence), ownerName(std::move(ownerName)) {
    if (initialSequence < 0) {
        throw std::invalid_argument(this->ownerName + " dropout sequence must be non-negative.");
    }
}

void DropOutRuntimeState::setSequenceAdvance(uint64_t advance) {
    if (advance == 0) advance = 1;
    if (advance > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::overflow_error(ownerName + " dropout sequence advance exceeds int64_t range.");
    }
    sequenceAdvance = advance;
}

TensorScalarBinding DropOutRuntimeState::seedBinding(TensorPlacement placement) {
    ensureBuffer(placement);
    return TensorScalarBinding{seedSequenceBuffer, kSeedByteOffset, DataType::INT64};
}

TensorScalarBinding DropOutRuntimeState::sequenceBinding(TensorPlacement placement) {
    ensureBuffer(placement);
    return TensorScalarBinding{seedSequenceBuffer, kSequenceByteOffset, DataType::INT64};
}

void DropOutRuntimeState::uploadForForward(Stream& stream) {
    TensorPlacement placement(TensorPlacement::MemDevices::GPU, stream.getGpuNum());
    ensureBuffer(placement);
    launchSetInt64Pair(seedSequenceBuffer.getMemPtr<int64_t>(), seed, nextSequence, stream);

    const uint64_t remaining = static_cast<uint64_t>(std::numeric_limits<int64_t>::max() - nextSequence);
    if (sequenceAdvance > remaining) {
        throw std::overflow_error(ownerName + " automatic dropout sequence advance would exceed int64_t range.");
    }
    nextSequence += static_cast<int64_t>(sequenceAdvance);
}

void DropOutRuntimeState::ensureBuffer(TensorPlacement placement) {
    if (seedSequenceBuffer.isInitialized() && seedSequenceBuffer.getPlacement() == placement) return;
    TensorDescriptor descriptor(DataType::INT64, {2});
    seedSequenceBuffer = Tensor(placement, descriptor);
}

CudaKernelExpression makeDropOutPostOpKernel(DataType dataType,
                                             float probability,
                                             bool useResidual,
                                             bool ragged,
                                             DataType offsetsDataType,
                                             uint64_t raggedBatchSize,
                                             uint64_t featuresPerValue,
                                             const std::string& debugName) {
    if (!std::isfinite(probability) || probability <= 0.0f || probability >= 1.0f) {
        throw std::invalid_argument(debugName + " dropout post-op requires probability in (0, 1).");
    }
    if (ragged && (raggedBatchSize == 0 || featuresPerValue == 0 ||
                   raggedBatchSize > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
                   featuresPerValue > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))) {
        throw std::invalid_argument(debugName + " ragged dropout geometry must fit int64 and be non-zero.");
    }

    const std::string kernelName = kernelNameForDebugName(debugName);
    const std::string typePrelude = storagePrelude(dataType, debugName);
    const std::string offsetsPrelude = ragged ? offsetPrelude(offsetsDataType, debugName) : std::string{};
    const std::string philox = philoxSource();
    const std::string residualParam = useResidual ? ", const ThorDropoutStorage* residual" : "";
    const std::string offsetsParam = ragged ? ", const ThorDropoutOffset* offsets" : "";
    const std::string activeExpression = ragged
        ? "static_cast<uint64_t>(offsets[batch]) * static_cast<uint64_t>(features_per_value)"
        : "static_cast<uint64_t>(num_elements)";
    const std::string residualVectorLoad = useResidual
        ? "const ThorDropoutVector4 residual_values = reinterpret_cast<const ThorDropoutVector4*>(residual)[group];"
        : "";
    const std::string residualValueVector = useResidual ? " + thor_dropout_load(residual_values.values[lane])" : "";
    const std::string residualValueScalar = useResidual ? " + thor_dropout_load(residual[index])" : "";

    const std::string forwardSource = typePrelude + offsetsPrelude + philox + R"cuda(
extern "C" __global__
void thor_dropout_postop_forward(const ThorDropoutStorage* projected)cuda" + residualParam + R"cuda(,
                                  const int64_t* seed,
                                  const int64_t* sequence)cuda" + offsetsParam + R"cuda(,
                                  ThorDropoutStorage* output,
                                  float probability,
                                  float scale,
                                  int64_t num_elements,
                                  int64_t batch,
                                  int64_t features_per_value) {
    const uint64_t active_elements = )cuda" + activeExpression + R"cuda(;
    const uint64_t first_group = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    const uint64_t groups = (active_elements + 3ULL) / 4ULL;
    for (uint64_t group = first_group; group < groups; group += stride) {
        const ThorDropoutUint4 random = thor_dropout_philox(group, static_cast<uint64_t>(*seed), static_cast<uint64_t>(*sequence));
        const uint32_t bits[4] = {random.x, random.y, random.z, random.w};
        const uint64_t base_index = group * 4ULL;
        if (base_index + 4ULL <= active_elements) {
            const ThorDropoutVector4 projected_values = reinterpret_cast<const ThorDropoutVector4*>(projected)[group];
            )cuda" + residualVectorLoad + R"cuda(
            ThorDropoutVector4 output_values;
#pragma unroll
            for (uint32_t lane = 0; lane < 4; ++lane) {
                const ThorDropoutAccum dropped = thor_dropout_keep(bits[lane], probability)
                    ? thor_dropout_load(projected_values.values[lane]) * scale
                    : static_cast<ThorDropoutAccum>(0);
                const ThorDropoutAccum combined = dropped)cuda" + residualValueVector + R"cuda(;
                output_values.values[lane] = thor_dropout_store(combined);
            }
            reinterpret_cast<ThorDropoutVector4*>(output)[group] = output_values;
        } else {
#pragma unroll
            for (uint32_t lane = 0; lane < 4; ++lane) {
                const uint64_t index = base_index + lane;
                if (index >= active_elements) continue;
                const ThorDropoutAccum dropped = thor_dropout_keep(bits[lane], probability)
                    ? thor_dropout_load(projected[index]) * scale
                    : static_cast<ThorDropoutAccum>(0);
                const ThorDropoutAccum combined = dropped)cuda" + residualValueScalar + R"cuda(;
                output[index] = thor_dropout_store(combined);
            }
        }
    }
}
)cuda";

    const std::string residualBackwardParam = useResidual ? ", ThorDropoutStorage* d_residual" : "";
    const std::string residualBackwardVectorStore =
        useResidual ? "reinterpret_cast<ThorDropoutVector4*>(d_residual)[group] = dy_values;" : "";
    const std::string residualBackwardScalarStore = useResidual ? "d_residual[index] = dy[index];" : "";
    const std::string backwardSource = typePrelude + offsetsPrelude + philox + R"cuda(
extern "C" __global__
void thor_dropout_postop_backward(const ThorDropoutStorage* dy,
                                   const int64_t* seed,
                                   const int64_t* sequence)cuda" + offsetsParam + R"cuda(,
                                   ThorDropoutStorage* d_projected)cuda" + residualBackwardParam + R"cuda(,
                                   float probability,
                                   float scale,
                                   int64_t num_elements,
                                   int64_t batch,
                                   int64_t features_per_value) {
    const uint64_t active_elements = )cuda" + activeExpression + R"cuda(;
    const uint64_t first_group = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    const uint64_t groups = (active_elements + 3ULL) / 4ULL;
    for (uint64_t group = first_group; group < groups; group += stride) {
        const ThorDropoutUint4 random = thor_dropout_philox(group, static_cast<uint64_t>(*seed), static_cast<uint64_t>(*sequence));
        const uint32_t bits[4] = {random.x, random.y, random.z, random.w};
        const uint64_t base_index = group * 4ULL;
        if (base_index + 4ULL <= active_elements) {
            const ThorDropoutVector4 dy_values = reinterpret_cast<const ThorDropoutVector4*>(dy)[group];
            ThorDropoutVector4 projected_gradient_values;
#pragma unroll
            for (uint32_t lane = 0; lane < 4; ++lane) {
                const ThorDropoutAccum grad = thor_dropout_keep(bits[lane], probability)
                    ? thor_dropout_load(dy_values.values[lane]) * scale
                    : static_cast<ThorDropoutAccum>(0);
                projected_gradient_values.values[lane] = thor_dropout_store(grad);
            }
            reinterpret_cast<ThorDropoutVector4*>(d_projected)[group] = projected_gradient_values;
            )cuda" + residualBackwardVectorStore + R"cuda(
        } else {
#pragma unroll
            for (uint32_t lane = 0; lane < 4; ++lane) {
                const uint64_t index = base_index + lane;
                if (index >= active_elements) continue;
                const ThorDropoutAccum grad = thor_dropout_keep(bits[lane], probability)
                    ? thor_dropout_load(dy[index]) * scale
                    : static_cast<ThorDropoutAccum>(0);
                d_projected[index] = thor_dropout_store(grad);
                )cuda" + residualBackwardScalarStore + R"cuda(
            }
        }
    }
}
)cuda";

    auto backwardBuilder = CudaKernelExpression::builder(kernelName + "_dropout_postop_backward")
                               .source(backwardSource)
                               .entry("thor_dropout_postop_backward")
                               .input("dy", dataType)
                               .tensorRuntimeScalarInput("seed", DataType::INT64)
                               .tensorRuntimeScalarInput("sequence", DataType::INT64);
    if (ragged) backwardBuilder.input("offsets", offsetsDataType);
    backwardBuilder.outputLike("d_projected", dataType, "dy");
    if (useResidual) backwardBuilder.outputLike("d_residual", dataType, "dy");
    backwardBuilder.scalar("probability", DataType::FP32, probability)
        .scalar("scale", DataType::FP32, 1.0f / (1.0f - probability))
        .scalar("num_elements", DataType::INT64, CudaKernelExpression::DimExpr::numel("dy"))
        .scalar("batch", DataType::INT64, static_cast<int64_t>(ragged ? raggedBatchSize : 0))
        .scalar("features_per_value", DataType::INT64, static_cast<int64_t>(ragged ? featuresPerValue : 0))
        .launch([](const CudaKernelExpression::LaunchContext& ctx) {
            constexpr uint32_t block = 256;
            return CudaKernelLaunchConfig{dim3(launchGridForNumel(ctx.numel("dy")), 1, 1), dim3(block, 1, 1), 0};
        });
    CudaKernelExpression backward = backwardBuilder.build();

    auto forwardBuilder = CudaKernelExpression::builder(kernelName + "_dropout_postop")
                              .source(forwardSource)
                              .entry("thor_dropout_postop_forward")
                              .input("projected", dataType);
    if (useResidual) forwardBuilder.input("residual", dataType);
    forwardBuilder.tensorRuntimeScalarInput("seed", DataType::INT64)
        .tensorRuntimeScalarInput("sequence", DataType::INT64);
    if (ragged) forwardBuilder.input("offsets", offsetsDataType);
    forwardBuilder.outputLike("output", dataType, "projected")
        .scalar("probability", DataType::FP32, probability)
        .scalar("scale", DataType::FP32, 1.0f / (1.0f - probability))
        .scalar("num_elements", DataType::INT64, CudaKernelExpression::DimExpr::numel("output"))
        .scalar("batch", DataType::INT64, static_cast<int64_t>(ragged ? raggedBatchSize : 0))
        .scalar("features_per_value", DataType::INT64, static_cast<int64_t>(ragged ? featuresPerValue : 0))
        .launch([](const CudaKernelExpression::LaunchContext& ctx) {
            constexpr uint32_t block = 256;
            return CudaKernelLaunchConfig{dim3(launchGridForNumel(ctx.numel("output")), 1, 1), dim3(block, 1, 1), 0};
        });

    std::unordered_map<std::string, std::string> gradients{{"d_projected", "projected"}};
    if (useResidual) gradients.emplace("d_residual", "residual");
    forwardBuilder.backward("output", std::move(backward), "dy", std::move(gradients));
    return forwardBuilder.build();
}

}  // namespace ThorImplementation
