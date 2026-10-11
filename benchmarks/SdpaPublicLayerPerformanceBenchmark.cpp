// SERIALIZATION-OWNERSHIP-2C-G4-C2: optional steady-state host allocator audit.
//
// G4-B1 reports reproducible whole-submission timings (including host staging),
// NOT isolated SDPA kernel latency. CUDA timing-stream events include host enqueue
// gaps and must not be mislabeled as attention-only device time. G4-B2 adds
// a steady-state CUDA-profiler capture range for external GPU-kernel tracing;
// actual device durations come from the profiler, not this CSV.
// The historical fixed and placement-neutral cases below share the same public
// ScaledDotProductAttention builder and the same semantic active input values.

#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Learning/ScaledDotProductAttention.h"
#include "DeepLearning/Api/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Network/StampedNetwork.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "Utilities/Common/Stream.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

// Optional Linux/GLIBC benchmark-only malloc probe loaded via LD_PRELOAD.
// Weak imports preserve the uninstrumented production/benchmark executable.
// See docs/performance/SdpaPublicLayerG4C2.md for attribution limits.
extern "C" {
struct SdpaCudnnAllocPhaseSnapshot {
    uint64_t calls, malloc_calls, calloc_calls, realloc_calls, free_calls;
    uint64_t posix_memalign_calls, aligned_alloc_calls, requested_bytes;
};
struct SdpaCudnnAllocStageSnapshot {
    uint64_t malloc_calls, calloc_calls, free_calls, requested_bytes;
};
struct SdpaHostAllocSnapshot {
    uint64_t malloc_calls, calloc_calls, realloc_calls, free_calls;
    uint64_t posix_memalign_calls, aligned_alloc_calls, requested_bytes;
    SdpaCudnnAllocPhaseSnapshot cudnn_forward, cudnn_backward;
    SdpaCudnnAllocStageSnapshot cudnn_stage[2][3];
};
void thor_sdpa_host_alloc_start() __attribute__((weak));
void thor_sdpa_host_alloc_stop(SdpaHostAllocSnapshot*) __attribute__((weak));
}

namespace {
namespace Api = Thor;
namespace Impl = ThorImplementation;
using DataType = Impl::DataType;

enum class Geometry { Neutral, Fixed, Dense };
enum class LengthPattern { Full, Uneven };

struct Case {
    const char* name;
    uint32_t batch;
    uint32_t maxQuery;
    uint32_t maxKv;
    uint32_t queryHeads;
    uint32_t kvHeads;
    uint32_t headDim;
    LengthPattern lengths;
    bool causal;
    DataType dtype = DataType::FP16;
};

// The first three are intentionally small enough to smoke-test quickly.  B=7
// includes empty Q rows and independently varying KV row lengths.  B=16
// cases exercise larger full or imbalanced causal rows in FP16 and BF16.
const std::vector<Case> kCases = {
    {"b2_full_fp16",       2,   4,   6,  4, 2, 32, LengthPattern::Full,   false},
    {"b7_uneven_fp16",     7,   4,   6,  4, 2, 32, LengthPattern::Uneven, false},
    {"b7_uneven_causal",   7,   4,   6,  4, 2, 32, LengthPattern::Uneven, true},
    {"b16_full_fp16",     16, 128, 128,  8, 2, 64, LengthPattern::Full,   false},
    {"b16_uneven_causal", 16, 128, 128,  8, 2, 64, LengthPattern::Uneven, true},
    {"b2_full_bf16",        2,   4,   6,  4, 2, 32, LengthPattern::Full,   false, DataType::BF16},
    {"b16_full_bf16",      16, 128, 128,  8, 2, 64, LengthPattern::Full,   false, DataType::BF16},
    {"b16_uneven_causal_bf16", 16, 128, 128, 8, 2, 64, LengthPattern::Uneven, true, DataType::BF16},
};

void require(bool condition, const std::string& error) {
    if (!condition) throw std::runtime_error(error);
}

const char* dtypeName(DataType dtype) {
    require(dtype == DataType::FP16 || dtype == DataType::BF16,
            "benchmark supports only FP16 and BF16 storage");
    return dtype == DataType::FP16 ? "fp16" : "bf16";
}

void fillLowPrecision(Impl::Tensor& tensor, uint64_t count, float value) {
    if (tensor.getDataType() == DataType::FP16) {
        auto* data = tensor.getMemPtr<half>();
        std::fill_n(data, count, __float2half(value));
    } else if (tensor.getDataType() == DataType::BF16) {
        auto* data = tensor.getMemPtr<__nv_bfloat16>();
        std::fill_n(data, count, __float2bfloat16(value));
    } else {
        throw std::runtime_error("unsupported low precision tensor dtype");
    }
}

float readLowPrecision(const Impl::Tensor& tensor, uint64_t index) {
    if (tensor.getDataType() == DataType::FP16)
        return __half2float(tensor.getMemPtr<half>()[index]);
    if (tensor.getDataType() == DataType::BF16)
        return __bfloat162float(tensor.getMemPtr<__nv_bfloat16>()[index]);
    throw std::runtime_error("unsupported low precision tensor dtype");
}

std::vector<uint64_t> offsetsFor(const Case& c, bool query) {
    const uint32_t maximum = query ? c.maxQuery : c.maxKv;
    std::vector<uint64_t> offsets(static_cast<size_t>(c.batch) + 1, 0);
    for (uint32_t row = 0; row < c.batch; ++row) {
        uint32_t length = maximum;
        if (c.lengths == LengthPattern::Uneven) {
            // At least one empty Q row; KV nonempty for every active Q row.
            length = query
                ? (row % 4 == 0 ? 0u : 1u + (row * 13u) % maximum)
                : (1u + (row * 7u + 1u) % maximum);
        }
        require(length <= maximum, "internal row length exceeds semantic maximum");
        offsets[row + 1] = offsets[row] + length;
    }
    return offsets;
}

float sample(int component, uint64_t token, uint64_t head, uint64_t dim) {
    return -0.12f + 0.06f * static_cast<float>(component + 1) +
           0.023f * static_cast<float>((token * 3 + head * 5 + dim * 11 + component) % 19);
}

Impl::Tensor hostValues(const Case& c, uint32_t maximum, uint32_t heads,
                        const std::vector<uint64_t>& offsets, int component, bool dense) {
    const Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    std::vector<uint64_t> shape = dense
        ? std::vector<uint64_t>{c.batch, maximum, heads, c.headDim}
        : std::vector<uint64_t>{static_cast<uint64_t>(c.batch) * maximum, heads, c.headDim};
    Impl::Tensor values(cpu, Impl::TensorDescriptor(c.dtype, shape));
    auto* halfData = c.dtype == DataType::FP16 ? values.getMemPtr<half>() : nullptr;
    auto* bfData = c.dtype == DataType::BF16 ? values.getMemPtr<__nv_bfloat16>() : nullptr;
    require(halfData != nullptr || bfData != nullptr, "unsupported input dtype");
    const uint64_t tokens = static_cast<uint64_t>(c.batch) * maximum;
    for (uint64_t token = 0; token < tokens; ++token) {
        for (uint64_t h = 0; h < heads; ++h) {
            for (uint64_t d = 0; d < c.headDim; ++d) {
                const uint64_t index = (token * heads + h) * c.headDim + d;
                const float value = token < offsets.back()
                    ? sample(component, token, h, d) : 0.0f;
                if (halfData)
                    halfData[index] = __float2half(value);
                else
                    bfData[index] = __float2bfloat16(value);
            }
        }
    }
    return values;
}

Impl::Tensor hostOffsets(const Case& c, const std::vector<uint64_t>& offsets, DataType dtype) {
    const Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor out(cpu, Impl::TensorDescriptor(dtype, {static_cast<uint64_t>(c.batch) + 1}));
    if (dtype == DataType::UINT32) {
        auto* p = out.getMemPtr<uint32_t>();
        for (uint32_t i = 0; i <= c.batch; ++i) p[i] = static_cast<uint32_t>(offsets[i]);
    } else {
        auto* p = out.getMemPtr<uint64_t>();
        for (uint32_t i = 0; i <= c.batch; ++i) p[i] = offsets[i];
    }
    return out;
}

struct Built {
    explicit Built(const Case& c, Geometry g)
        : network(std::make_unique<Api::Network>(std::string("sdpa_g4a_") + c.name)), geometry(g) {}
    std::unique_ptr<Api::Network> network;
    Geometry geometry;
    std::optional<Api::RaggedTensor> q;
    std::optional<Api::RaggedTensor> k;
    std::optional<Api::RaggedTensor> v;
    std::optional<Api::RaggedTensor> labels;
    uint64_t sdpaLayerId = 0;
};

Impl::DynamicExpression mse(bool gradient, DataType dtype) {
    auto p = Impl::Expression::input("predictions", dtype, DataType::FP32);
    auto y = Impl::Expression::input("labels", dtype, DataType::FP32);
    const auto delta = p - y;
    const auto result = gradient
        ? delta * Impl::Expression(2.0 * Impl::Loss::getLossScalingFactor())
        : delta * delta;
    return Impl::DynamicExpression::fromExpressionDefinition(
        Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{
            gradient ? "predictions_grad" : "loss", result.withOutputDType(dtype)}})));
}

Built buildNetwork(const Case& c, Geometry geometry, bool training = false) {
    Built built(c, geometry);
    Api::Network& n = *built.network;
    require(!training || geometry != Geometry::Dense,
            "G4-B1 training is matched fixed/neutral ragged only");
    Api::ScaledDotProductAttention::Builder attention;
    attention.network(n).computeDataType(DataType::FP32);
    if (c.causal) attention.maskKind(Impl::AttentionMaskKind::CausalTopLeft);
    if (geometry == Geometry::Dense) {
        require(c.lengths == LengthPattern::Full, "dense is a contextual control only for full rows");
        const auto q = Api::NetworkInput::Builder().network(n).name("q")
            .dimensions({c.maxQuery,c.queryHeads,c.headDim}).dataType(c.dtype).build();
        const auto k = Api::NetworkInput::Builder().network(n).name("k")
            .dimensions({c.maxKv,c.kvHeads,c.headDim}).dataType(c.dtype).build();
        const auto v = Api::NetworkInput::Builder().network(n).name("v")
            .dimensions({c.maxKv,c.kvHeads,c.headDim}).dataType(c.dtype).build();
        attention.bshdLayout().queryInput(q.getFeatureOutput().value())
            .keyInput(k.getFeatureOutput().value()).valueInput(v.getFeatureOutput().value());
    } else {
        Api::RaggedNetworkInput::Builder qBuilder;
        qBuilder.network(n).name("q")
            .valuesDataType(c.dtype).offsetsDataType(DataType::UINT32)
            .trailingDimensions({c.queryHeads,c.headDim});
        Api::RaggedNetworkInput::Builder kBuilder;
        kBuilder.network(n).name("k")
            .valuesDataType(c.dtype).offsetsDataType(DataType::UINT64)
            .trailingDimensions({c.kvHeads,c.headDim});
        if (geometry == Geometry::Neutral) {
            qBuilder.maxValuesPerRow(c.maxQuery);
            kBuilder.maxValuesPerRow(c.maxKv);
        } else {
            qBuilder.batchSize(c.batch).maxTotalValues(static_cast<uint64_t>(c.batch) * c.maxQuery)
                .maxValuesPerRow(c.maxQuery);
            kBuilder.batchSize(c.batch).maxTotalValues(static_cast<uint64_t>(c.batch) * c.maxKv)
                .maxValuesPerRow(c.maxKv);
        }
        built.q = qBuilder.build();
        built.k = kBuilder.build();
        built.v = Api::RaggedNetworkInput::Builder().network(n).name("v")
            .valuesDataType(c.dtype).trailingDimensions({c.kvHeads,c.headDim})
            .partition(*built.k).build();
        if (training) {
            // Real RAW-loss-connected gradients, exactly as in the accepted F1 tests.
            built.labels = Api::RaggedNetworkInput::Builder().network(n).name("labels")
                .valuesDataType(c.dtype).trailingDimensions({c.queryHeads,c.headDim})
                .partition(*built.q).build();
            const auto qr = Api::GradientRivet::Builder().network(n).tensor(built.q->getValues()).build();
            const auto kr = Api::GradientRivet::Builder().network(n).tensor(built.k->getValues()).build();
            const auto vr = Api::GradientRivet::Builder().network(n).tensor(built.v->getValues()).build();
            attention.queryInput(built.q->withValues(qr.getFeatureOutput().value()))
                .keyInput(built.k->withValues(kr.getFeatureOutput().value()))
                .valueInput(built.v->withValues(vr.getFeatureOutput().value()));
        } else {
            attention.queryInput(*built.q).keyInput(*built.k).valueInput(*built.v);
        }
    }
    const auto sdpa = attention.build();
    built.sdpaLayerId = sdpa.getId();
    const auto expectedVersion = geometry == Geometry::Neutral ? "2.2.0" : "2.1.0";
    // Dense is the historical non-neutral public layer API as well.
    require(sdpa.architectureJson().at("version") == expectedVersion,
            std::string("unexpected SDPA architecture version; expected ") + expectedVersion);
    if (geometry == Geometry::Dense) {
        (void)Api::NetworkOutput::Builder().network(n).name("out")
            .inputTensor(sdpa.getFeatureOutput().value()).build();
    } else {
        require(sdpa.getRaggedFeatureOutput().has_value(), "ragged public SDPA must expose ragged output");
        (void)Api::RaggedNetworkOutput::Builder().network(n).name("out")
            .inputTensor(sdpa.getRaggedFeatureOutput().value()).build();
    }
    if (training) {
        const auto loss = Api::RaggedCustomLoss::Builder().network(n)
            .predictions(*sdpa.getRaggedFeatureOutput()).labels(*built.labels)
            .lossExpression(mse(false, c.dtype)).gradientExpression(mse(true, c.dtype)).build();
        (void)Api::NetworkOutput::Builder().network(n).name("raw_loss")
            .inputTensor(loss.getRaggedRawLoss().getValues()).dataType(c.dtype).build();
    }
    return built;
}

Batch makeBatch(const Case& c, const Built& built) {
    const auto qo = offsetsFor(c, true);
    const auto ko = offsetsFor(c, false);
    Batch batch;
    const bool dense = built.geometry == Geometry::Dense;
    auto qValues = hostValues(c, c.maxQuery, c.queryHeads, qo, 0, dense);
    auto kValues = hostValues(c, c.maxKv, c.kvHeads, ko, 1, dense);
    auto vValues = hostValues(c, c.maxKv, c.kvHeads, ko, 2, dense);
    if (dense) {
        batch.insert("q", std::move(qValues));
        batch.insert("k", std::move(kValues));
        batch.insert("v", std::move(vValues));
    } else {
        require(built.q.has_value() && built.k.has_value() && built.v.has_value(),
                "missing ragged inputs");
        Impl::RowPartitionRuntime qpart(
            hostOffsets(c, qo, DataType::UINT32), built.q->getDescriptorForPlacement(c.batch).getRowPartition());
        Impl::RowPartitionRuntime kvpart(
            hostOffsets(c, ko, DataType::UINT64), built.k->getDescriptorForPlacement(c.batch).getRowPartition());
        qpart.setHostOffsets(qo);
        kvpart.setHostOffsets(ko);
        batch.insert("q", Impl::RaggedTensor(std::move(qValues), qpart));
        batch.insert("k", Impl::RaggedTensor(std::move(kValues), kvpart));
        batch.insert("v", Impl::RaggedTensor(std::move(vValues), kvpart));
        if (built.labels.has_value()) {
            const Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
            Impl::Tensor labels(cpu, built.labels->getDescriptorForPlacement(c.batch).getValuesDescriptor());
            fillLowPrecision(labels, labels.getTotalNumElements(), 0.f);
            batch.insert("labels", Impl::RaggedTensor(std::move(labels), qpart));
        }
    }
    batch.setValidExampleCount(c.batch);
    return batch;
}

struct ExecutionResult {
    std::vector<float> activeOutput;
    uint64_t activeQ = 0;
    uint64_t activeKv = 0;
};

ExecutionResult execute(const Case& c, Geometry geometry, int gpu) {
    Built built = buildNetwork(c, geometry);
    std::vector<Event> initDone;
    // The true value selects inference-only for the G4-A smoke path.
    auto placed = built.network->place(c.batch, initDone, true, {gpu}, 1);
    require(placed != nullptr, "public-layer placement returned null");
    for (auto& event : initDone) event.synchronize();
    require(placed->getStampedNetwork(0).getGpuNum() == static_cast<uint32_t>(gpu),
            "placement selected a different GPU than --gpu");
    auto physical = std::dynamic_pointer_cast<Impl::CustomLayer>(
        placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(built.sdpaLayerId));
    require(physical != nullptr && physical->getFeatureOutput().has_value(),
            "SDPA did not produce a physical CustomLayer output");
    require(physical->getFeatureOutput()->getPlacement().getMemDevice() ==
                Impl::TensorPlacement::MemDevices::GPU,
            "stamped public SDPA output is not device-resident");
    const auto physicalShape = physical->getFeatureOutput()->getDimensions();
    if (geometry != Geometry::Dense) {
        require(physicalShape == (std::vector<uint64_t>{
                    static_cast<uint64_t>(c.batch) * c.maxQuery, c.queryHeads, c.headDim}),
                "ragged physical SDPA output does not match B*M capacity");
    }
    const Batch batch = makeBatch(c, built);
    const auto result = placed->inferLogical(batch);
    require(result.count("out") == 1, "missing public output");
    Impl::Tensor output;
    if (geometry == Geometry::Dense) {
        require(std::holds_alternative<Impl::Tensor>(result.at("out")), "expected dense output");
        output = std::get<Impl::Tensor>(result.at("out"));
    } else {
        require(std::holds_alternative<Impl::RaggedTensor>(result.at("out")), "expected ragged output");
        const auto& ragged = std::get<Impl::RaggedTensor>(result.at("out"));
        require(ragged.getHostOffsetsIfAvailable().has_value(), "lost authoritative Q host offsets");
        require(ragged.getHostOffsetsIfAvailable().value() == offsetsFor(c, true),
                "output partition does not match input Q");
        output = ragged.getValues();
    }
    require(output.getDataType() == c.dtype, "unexpected output dtype");
    const auto qo = offsetsFor(c, true);
    const auto ko = offsetsFor(c, false);
    ExecutionResult checked;
    checked.activeQ = qo.back();
    checked.activeKv = ko.back();
    const uint64_t count = qo.back() * c.queryHeads * c.headDim;
    require(count > 0, "smoke case needs nonzero active output");
    checked.activeOutput.reserve(count);
    double maxAbsActiveOutput = 0;
    for (uint64_t i = 0; i < count; ++i) {
        const float value = readLowPrecision(output, i);
        require(std::isfinite(value), "non-finite active SDPA output at index " + std::to_string(i));
        checked.activeOutput.push_back(value);
        maxAbsActiveOutput = std::max(maxAbsActiveOutput, std::abs(static_cast<double>(value)));
    }
    require(maxAbsActiveOutput > 0.001, "all active SDPA outputs are zero or numerically trivial");
    placed->synchronize();
    return checked;
}

void printCase(const Case& c) {
    const auto q = offsetsFor(c, true);
    const auto kv = offsetsFor(c, false);
    std::cout << "case=" << c.name << " batch=" << c.batch
              << " Q_capacity=" << static_cast<uint64_t>(c.batch) * c.maxQuery
              << " KV_capacity=" << static_cast<uint64_t>(c.batch) * c.maxKv
              << " Q_active=" << q.back() << " KV_active=" << kv.back()
              << " Hq=" << c.queryHeads << " Hkv=" << c.kvHeads
              << " D=" << c.headDim << " mask=" << (c.causal ? "causal_top_left" : "none")
              << " dtype=" << dtypeName(c.dtype) << " (fixed+neutral"
              << (c.lengths == LengthPattern::Full ? "+dense_control" : "") << ")\n";
}

void runCase(const Case& c, int gpu) {
    printCase(c);
    const auto fixed = execute(c, Geometry::Fixed, gpu);
    const auto neutral = execute(c, Geometry::Neutral, gpu);
    require(fixed.activeQ == neutral.activeQ && fixed.activeKv == neutral.activeKv,
            "unmatched active counts");
    require(fixed.activeOutput.size() == neutral.activeOutput.size(), "output size mismatch");
    double maxAbsoluteDiff = 0;
    for (size_t i = 0; i < fixed.activeOutput.size(); ++i)
        maxAbsoluteDiff = std::max(maxAbsoluteDiff,
            std::abs(static_cast<double>(fixed.activeOutput[i] - neutral.activeOutput[i])));
    // This smoke checks equivalent public-path values, not a CPU reference.  The
    // numerical test suite is the CPU oracle.  A tight tolerance also catches
    // a different active-length/partition execution before timing starts.
    require(maxAbsoluteDiff <= 0.03, "fixed/neutral active output mismatch: " +
            std::to_string(maxAbsoluteDiff));
    std::cout << "GPU_SMOKE_PASS fixed+neutral: active_output_elements="
              << fixed.activeOutput.size() << " max_abs_difference="
              << std::setprecision(7) << maxAbsoluteDiff << '\n';
    if (c.lengths == LengthPattern::Full) {
        const auto dense = execute(c, Geometry::Dense, gpu);
        require(dense.activeOutput.size() == neutral.activeOutput.size(),
                "dense contextual control shape mismatch");
        double denseDiff = 0;
        for (size_t i = 0; i < dense.activeOutput.size(); ++i)
            denseDiff = std::max(denseDiff,
                std::abs(static_cast<double>(dense.activeOutput[i] - neutral.activeOutput[i])));
        require(denseDiff <= 0.03, "dense contextual control mismatch: " + std::to_string(denseDiff));
        std::cout << "GPU_SMOKE_PASS dense_context_control: max_abs_difference="
                  << denseDiff << '\n';
    }
}


// These timings cover the public submission boundary; unlike kernel-only cuDNN
// events they include host scheduling/staging. Both geometries receive the same
// fixed logical values and partition offsets for each case.
struct TimingOptions {
    int gpu = 0;
    int warmup = 5;
    int trials = 3;
    int iterations = 20;
    int pool = 4;
    std::optional<Geometry> selectedGeometry;
    // unset means measure both inference and connected training
    std::optional<bool> selectedTraining;
    bool captureGpuKernels = false;
    bool auditHostAllocations = false;
    bool auditCudnnScopes = false;
};

void verifyBackward(const Case& c, Impl::CustomLayer& physical) {
    const auto& gradients = physical.getErrorOutputs();
    require(gradients.size() >= 3, "SDPA must have dQ/dK/dV ports");
    const auto q = offsetsFor(c, true).back();
    const auto kv = offsetsFor(c, false).back();
    const uint64_t expected[3] = {q*c.queryHeads*c.headDim,
                                  kv*c.kvHeads*c.headDim, kv*c.kvHeads*c.headDim};
    for (int port = 0; port < 3; ++port) {
        require(gradients[port].has_value(), "missing SDPA gradient port " + std::to_string(port));
        const auto& gradient = gradients[port].value();
        require(gradient.getTotalNumElements() >= expected[port], "short gradient storage");
        const Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
        Impl::Tensor dst(cpu, gradient.getDescriptor());
        Stream download = Stream::getNextDownloadStream(gradient.getPlacement().getDeviceNum());
        dst.copyFromAsync(gradient, download);
        download.synchronize();
        bool nonzero = false;
        for (uint64_t j = 0; j < expected[port]; ++j) {
            const float v = readLowPrecision(dst, j);
            require(std::isfinite(v), "nonfinite gradient port=" + std::to_string(port));
            nonzero = nonzero || std::abs(v) > 0.00001f;
        }
        // Some masking/row distributions may legitimately cancel dQ/dK,
        // but the value gradient must be nonzero with the MSE RAW loss.
        if (port == 2) require(nonzero, "dV is zero: no genuine backward execution");
    }
}

void measureOne(const Case& c, Geometry geometry, bool training,
                const TimingOptions& opt, std::ostream& csv) {
    require(geometry != Geometry::Dense, "matched timing requires fixed/neutral ragged geometry");
    auto built = buildNetwork(c, geometry, training);
    const auto stampStart = std::chrono::steady_clock::now();
    std::vector<Event> initDone;
    auto placed = built.network->place(c.batch, initDone, !training, {opt.gpu}, 1);
    require(placed != nullptr, "place returned null");
    for (auto& event : initDone) event.synchronize();
    const auto stampEnd = std::chrono::steady_clock::now();
    const double stampMs = std::chrono::duration<double, std::milli>(stampEnd-stampStart).count();
    auto physical = std::dynamic_pointer_cast<Impl::CustomLayer>(
        placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(built.sdpaLayerId));
    require(physical != nullptr, "no physical public SDPA layer");
    std::vector<Batch> pool;
    pool.reserve(opt.pool);
    for (int i = 0; i < opt.pool; ++i) pool.emplace_back(makeBatch(c, built));
    // The physical graph and its device allocations stay resident for all trials.
    // Pool rotations use independent *host* buffers; they are not an independent
    // GPU input pool. G4-C must audit this distinction before device parity.
    Stream timingStream(opt.gpu);
    Event first(opt.gpu, true, false), last(opt.gpu, true, true);
    auto submit = [&](int slot, bool timed, double& hostMs, double& windowMs,
                      SdpaHostAllocSnapshot& allocation) {
        std::map<std::string, Impl::Tensor> outputs;
        std::map<std::string, Event> ready;
        const auto before = std::chrono::steady_clock::now();
        // Start after preparation/warmup and before public submission, including
        // any concurrent host work inside this process. Not SDPA-local counts.
        if (timed && opt.auditHostAllocations) thor_sdpa_host_alloc_start();
        if (timed) timingStream.putEvent(first, true, false);
        Event done = placed->submitBatch(0, pool[slot], outputs, ready, !training);
        if (timed) {
            timingStream.waitEvent(done);
            timingStream.putEvent(last, true, true);
        }
        done.synchronize();
        for (auto& [name, readyEvent] : ready) readyEvent.synchronize();
        if (timed) windowMs = last.synchronizeAndReportElapsedTimeInMilliseconds(first);
        hostMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now()-before).count();
        if (timed && opt.auditHostAllocations) thor_sdpa_host_alloc_stop(&allocation);
        if (timed && opt.auditCudnnScopes) {
            require(allocation.cudnn_forward.calls >= 1 &&
                    (!training || allocation.cudnn_backward.calls >= 1),
                    "G4-C3 cuDNN owner scope missing: configure -DTHOR_SDPA_CUDNN_ALLOC_OWNER_AUDIT=ON, rebuild Thor and preload updated probe");
            for (int phase = 0; phase < (training ? 2 : 1); ++phase) {
                const auto& total = phase == 0 ? allocation.cudnn_forward : allocation.cudnn_backward;
                uint64_t mallocSum=0, callocSum=0, freeSum=0, byteSum=0;
                for (const auto& stage : allocation.cudnn_stage[phase]) {
                    mallocSum += stage.malloc_calls;
                    callocSum += stage.calloc_calls;
                    freeSum += stage.free_calls;
                    byteSum += stage.requested_bytes;
                }
                require(mallocSum == total.malloc_calls && callocSum == total.calloc_calls &&
                        freeSum == total.free_calls && byteSum == total.requested_bytes,
                        "G4-C4 phase attribution does not sum to wrapper totals: rebuild diagnostic Thor and the preload probe from the same patch");
            }
        }
    };
    double hostMs = 0., windowMs = 0.;
    SdpaHostAllocSnapshot allocation{};
    for (int i=0; i<opt.warmup; ++i) submit(i % opt.pool, false, hostMs, windowMs, allocation);
    placed->synchronize();
    if (training) verifyBackward(c, *physical);
    std::cerr << "WARMUP_COMPLETE case=" << c.name
              << " geometry=" << (geometry == Geometry::Fixed ? "fixed" : "neutral")
              << " mode=" << (training ? "fwd_plus_true_bwd" : "forward_inference")
              << " count=" << opt.warmup << '\n';
    // Nsight Systems --capture-range=cudaProfilerApi starts here, AFTER
    // placement, one-time cuDNN selection, pool construction and warmup.
    // Capturing one geometry and one mode per process avoids attributing
    // kernels to an unrelated network. This does not alter production SDPA.
    if (opt.captureGpuKernels) {
        require(cudaDeviceSynchronize() == cudaSuccess,
                "GPU profiler pre-capture synchronization failed");
        require(cudaProfilerStart() == cudaSuccess,
                "cudaProfilerStart failed; use nsys --capture-range=cudaProfilerApi");
    }
    for (int trial = 0; trial < opt.trials; ++trial) {
        for (int iter = 0; iter < opt.iterations; ++iter) {
            submit((trial*opt.iterations+iter) % opt.pool, true, hostMs, windowMs, allocation);
            csv << c.name << ',' << (geometry == Geometry::Fixed ? "fixed" : "neutral")
                << ',' << (training ? "fwd_plus_true_bwd" : "forward_inference")
                << ',' << c.batch << ',' << offsetsFor(c,true).back()
                << ',' << offsetsFor(c,false).back()
                << ',' << static_cast<uint64_t>(c.batch)*c.maxQuery
                << ',' << static_cast<uint64_t>(c.batch)*c.maxKv
                << ',' << (c.causal ? "causal_top_left" : "none")
                << "," << dtypeName(c.dtype) << "," << stampMs << ',' << opt.warmup << ',' << trial
                << ',' << iter << ',' << hostMs << ',' << windowMs
                << ",submit_and_wait,host_enqueue_included,unknown"
                << ',' << (opt.auditHostAllocations ? "process_all_threads" : "not_measured")
                << ',' << allocation.malloc_calls << ',' << allocation.calloc_calls
                << ',' << allocation.realloc_calls << ',' << allocation.free_calls
                << ',' << allocation.posix_memalign_calls << ',' << allocation.aligned_alloc_calls
                << ',' << allocation.requested_bytes
                << ',' << allocation.cudnn_forward.calls
                << ',' << allocation.cudnn_forward.malloc_calls
                << ',' << allocation.cudnn_forward.calloc_calls
                << ',' << allocation.cudnn_forward.realloc_calls
                << ',' << allocation.cudnn_forward.free_calls
                << ',' << allocation.cudnn_forward.posix_memalign_calls
                << ',' << allocation.cudnn_forward.aligned_alloc_calls
                << ',' << allocation.cudnn_forward.requested_bytes
                << ',' << allocation.cudnn_backward.calls
                << ',' << allocation.cudnn_backward.malloc_calls
                << ',' << allocation.cudnn_backward.calloc_calls
                << ',' << allocation.cudnn_backward.realloc_calls
                << ',' << allocation.cudnn_backward.free_calls
                << ',' << allocation.cudnn_backward.posix_memalign_calls
                << ',' << allocation.cudnn_backward.aligned_alloc_calls
                << ',' << allocation.cudnn_backward.requested_bytes;
            for (const auto& phase : allocation.cudnn_stage) {
                for (const auto& stage : phase)
                    csv << ',' << stage.malloc_calls << ',' << stage.calloc_calls
                        << ',' << stage.free_calls << ',' << stage.requested_bytes;
            }
            csv << '\n';
        }
        csv.flush();
    }
    if (opt.captureGpuKernels) {
        placed->synchronize();
        require(cudaDeviceSynchronize() == cudaSuccess,
                "GPU profiler post-capture synchronization failed");
        require(cudaProfilerStop() == cudaSuccess,
                "cudaProfilerStop failed");
        std::cerr << "PUBLIC_SDPA_GPU_KERNEL_TRACE_CAPTURED case=" << c.name
                  << " geometry=" << (geometry == Geometry::Fixed ? "fixed" : "neutral")
                  << " mode=" << (training ? "fwd_plus_true_bwd" : "forward_inference")
                  << " iterations=" << (opt.trials * opt.iterations) << '\n';
    }
}

void measureCase(const Case& c, const TimingOptions& opt, std::ostream& csv) {
    // Interleave modes at case granularity; mode-order effects remain explicit in
    // raw rows, and can be investigated with a second run in reversed order.
    for (const bool training : {false, true}) {
        if (opt.selectedTraining.has_value() && training != opt.selectedTraining.value()) continue;
        for (const auto geometry : {Geometry::Fixed, Geometry::Neutral}) {
            if (opt.selectedGeometry.has_value() && geometry != opt.selectedGeometry.value()) continue;
            measureOne(c, geometry, training, opt, csv);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string selected = "b2_full_fp16";
        bool listOnly = false;
        bool matrixCsv = false;
        bool all = false;
        int gpu = 0;
        bool measure = false;
        std::string csvPath;
        TimingOptions timing;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--measure") measure = true;
            else if (arg == "--csv" && i + 1 < argc) csvPath = argv[++i];
            else if (arg == "--warmup" && i + 1 < argc) timing.warmup = std::stoi(argv[++i]);
            else if (arg == "--trials" && i + 1 < argc) timing.trials = std::stoi(argv[++i]);
            else if (arg == "--iterations" && i + 1 < argc) timing.iterations = std::stoi(argv[++i]);
            else if (arg == "--pool" && i + 1 < argc) timing.pool = std::stoi(argv[++i]);
            else if (arg == "--profile-geometry" && i + 1 < argc) {
                const std::string v = argv[++i];
                if (v == "fixed") timing.selectedGeometry = Geometry::Fixed;
                else if (v == "neutral") timing.selectedGeometry = Geometry::Neutral;
                else throw std::invalid_argument("--profile-geometry must be fixed or neutral");
            }
            else if (arg == "--profile-mode" && i + 1 < argc) {
                const std::string v = argv[++i];
                if (v == "inference") timing.selectedTraining = false;
                else if (v == "training") timing.selectedTraining = true;
                else throw std::invalid_argument("--profile-mode must be inference or training");
            }
            else if (arg == "--capture-gpu-kernels") timing.captureGpuKernels = true;
            else if (arg == "--audit-host-allocations") timing.auditHostAllocations = true;
            else if (arg == "--audit-cudnn-scopes") timing.auditCudnnScopes = true;
            else if (arg == "--list") listOnly = true;
            else if (arg == "--matrix-csv") matrixCsv = true;
            else if (arg == "--all") all = true;
            else if (arg == "--smoke") selected = "b2_full_fp16";
            else if (arg == "--case" && i + 1 < argc) selected = argv[++i];
            else if (arg == "--gpu" && i + 1 < argc) gpu = std::stoi(argv[++i]);
            else if (arg == "--help") {
                std::cout << "Usage: thor_sdpa_public_layer_benchmark [--list | --matrix-csv | --smoke | --case NAME | --all | --measure] [--gpu N]\n"
                          << "--measure [--csv PATH] [--warmup N] [--trials N] [--iterations N] [--pool N]\n"
                          << "          [--profile-geometry fixed|neutral] [--profile-mode inference|training]\n"
                          << "          [--capture-gpu-kernels] (requires nsys --capture-range=cudaProfilerApi)\n"
                          << "          [--audit-host-allocations] (requires LD_PRELOAD malloc probe)\n"
                          << "          [--audit-cudnn-scopes] (requires owner-audit Thor build and updated probe)\n"
                          << "Measured CSV data include host enqueue gaps; extract isolated GPU kernels from nsys traces.\n";
                return 0;
            } else throw std::invalid_argument("unknown/unfinished argument: " + arg);
        }
        if (listOnly || matrixCsv) {
            if (matrixCsv)
                std::cout << "name,batch,max_q,max_kv,active_q,active_kv,q_capacity,kv_capacity,hq,hkv,head_dim,mask,dtype,dense_comparable\n";
            for (const auto& c : kCases) {
                if (!matrixCsv) {
                    printCase(c);
                    continue;
                }
                std::cout << c.name << ',' << c.batch << ',' << c.maxQuery << ',' << c.maxKv
                          << ',' << offsetsFor(c, true).back() << ',' << offsetsFor(c, false).back()
                          << ',' << static_cast<uint64_t>(c.batch) * c.maxQuery
                          << ',' << static_cast<uint64_t>(c.batch) * c.maxKv
                          << ',' << c.queryHeads << ',' << c.kvHeads << ',' << c.headDim
                          << ',' << (c.causal ? "causal_top_left" : "none")
                          << "," << dtypeName(c.dtype) << "," << (c.lengths == LengthPattern::Full ? "yes" : "no") << '\n';
            }
            return 0; // Matrix inspection must work without a GPU.
        }
        int devices = 0;
        const auto countStatus = cudaGetDeviceCount(&devices);
        require(countStatus == cudaSuccess && devices > 0, "CUDA GPU required for public SDPA smoke (no skip)");
        require(gpu >= 0 && gpu < devices, "invalid --gpu index");
        const auto selectStatus = cudaSetDevice(gpu);
        require(selectStatus == cudaSuccess, "cudaSetDevice failed");
        cudaDeviceProp properties{};
        require(cudaGetDeviceProperties(&properties, gpu) == cudaSuccess, "cudaGetDeviceProperties failed");
        timing.gpu = gpu;
        if (timing.captureGpuKernels) {
            require(measure, "--capture-gpu-kernels requires --measure");
            require(!all, "--capture-gpu-kernels requires one --case, not --all");
            require(timing.selectedGeometry.has_value() && timing.selectedTraining.has_value(),
                    "--capture-gpu-kernels requires --profile-geometry and --profile-mode");
        }
        if (!measure) {
            require(!timing.selectedGeometry.has_value() && !timing.selectedTraining.has_value(),
                    "--profile-geometry/--profile-mode require --measure");
        }
        if (measure) {
            require(!timing.auditCudnnScopes || timing.auditHostAllocations,
                    "--audit-cudnn-scopes requires --audit-host-allocations");
            if (timing.auditHostAllocations) {
                require(thor_sdpa_host_alloc_start && thor_sdpa_host_alloc_stop,
                        "--audit-host-allocations requires preloading libsdpa_host_alloc_probe.so");
                require(!timing.captureGpuKernels,
                        "run allocator and Nsight capture separately to avoid profiler interference");
                require(timing.selectedGeometry.has_value() && timing.selectedTraining.has_value(),
                        "allocation audit requires one --profile-geometry and one --profile-mode per process");
            }
            require(timing.warmup >= 1 && timing.trials >= 1 && timing.iterations >= 1 && timing.pool >= 2,
                    "--warmup/--trials/--iterations must be positive and --pool >= 2");
            std::ofstream file;
            if (!csvPath.empty()) {
                file.open(csvPath, std::ios::out | std::ios::trunc);
                require(file.good(), "unable to create CSV output: " + csvPath);
            }
            std::ostream& csv = csvPath.empty() ? std::cout : file;
            csv << "case,geometry,mode,batch,q_active,kv_active,q_capacity,kv_capacity,mask,dtype,stamp_ms,warmup,trial,iteration,host_ms,cuda_completion_window_ms,measurement_scope,stream_window_caveat,engine,host_alloc_scope,malloc_calls,calloc_calls,realloc_calls,free_calls,posix_memalign_calls,aligned_alloc_calls,requested_bytes,cudnn_forward_calls,cudnn_forward_malloc_calls,cudnn_forward_calloc_calls,cudnn_forward_realloc_calls,cudnn_forward_free_calls,cudnn_forward_posix_memalign_calls,cudnn_forward_aligned_alloc_calls,cudnn_forward_requested_bytes,cudnn_backward_calls,cudnn_backward_malloc_calls,cudnn_backward_calloc_calls,cudnn_backward_realloc_calls,cudnn_backward_free_calls,cudnn_backward_posix_memalign_calls,cudnn_backward_aligned_alloc_calls,cudnn_backward_requested_bytes,cudnn_forward_binding_validation_malloc_calls,cudnn_forward_binding_validation_calloc_calls,cudnn_forward_binding_validation_free_calls,cudnn_forward_binding_validation_requested_bytes,cudnn_forward_ragged_metadata_malloc_calls,cudnn_forward_ragged_metadata_calloc_calls,cudnn_forward_ragged_metadata_free_calls,cudnn_forward_ragged_metadata_requested_bytes,cudnn_forward_executable_malloc_calls,cudnn_forward_executable_calloc_calls,cudnn_forward_executable_free_calls,cudnn_forward_executable_requested_bytes,cudnn_backward_binding_validation_malloc_calls,cudnn_backward_binding_validation_calloc_calls,cudnn_backward_binding_validation_free_calls,cudnn_backward_binding_validation_requested_bytes,cudnn_backward_ragged_metadata_malloc_calls,cudnn_backward_ragged_metadata_calloc_calls,cudnn_backward_ragged_metadata_free_calls,cudnn_backward_ragged_metadata_requested_bytes,cudnn_backward_executable_malloc_calls,cudnn_backward_executable_calloc_calls,cudnn_backward_executable_free_calls,cudnn_backward_executable_requested_bytes\n";
            bool matched = false;
            for (const auto& c : kCases) {
                if (!all && c.name != selected) continue;
                measureCase(c, timing, csv);
                matched = true;
            }
            require(matched, "unknown timing case: " + selected);
            require(cudaDeviceSynchronize() == cudaSuccess, "measurement CUDA synchronization failed");
            std::cerr << "PUBLIC_SDPA_CONNECTED_BACKWARD_MEASURED gpu=" << properties.name
                      << " csv=" << (csvPath.empty() ? "stdout" : csvPath)
                      << " NOTE: no isolated attention-kernel timing yet\n";
            return 0;
        }
        std::cout << "G4-A public-layer CUDA execution smoke; GPU=" << gpu
                  << " name=" << properties.name << '\n';
        bool executed = false;
        for (const auto& c : kCases) {
            if (!all && selected != c.name) continue;
            runCase(c, gpu);
            executed = true;
        }
        require(executed, "unknown case: " + selected);
        require(cudaDeviceSynchronize() == cudaSuccess, "CUDA execution/synchronization failed");
        std::cout << "PUBLIC_SDPA_GPU_EXECUTED; no performance measurements asserted\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "G4-A failure: " << e.what() << '\n';
        return 1;
    }
}
