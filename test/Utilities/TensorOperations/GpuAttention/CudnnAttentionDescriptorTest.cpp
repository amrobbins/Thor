#include "Utilities/TensorOperations/GpuAttention/CudnnAttention.h"

#include "gtest/gtest.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

using namespace ThorImplementation;

namespace {

CudnnAttentionDescriptor makeDescriptor() {
    CudnnAttentionDescriptor descriptor;
    descriptor.q = AttentionTensorSpec::bhsd(3, 4, 64, 64, DataType::FP16);
    descriptor.k = AttentionTensorSpec::bhsd(3, 4, 80, 64, DataType::FP16);
    descriptor.v = AttentionTensorSpec::bhsd(3, 4, 80, 64, DataType::FP16);
    descriptor.o = AttentionTensorSpec::bhsd(3, 4, 64, 64, DataType::FP16);
    descriptor.computeDataType = DataType::FP32;
    descriptor.intermediateDataType = DataType::FP32;
    return descriptor;
}

CudnnAttentionDescriptor makePackedDescriptor() {
    CudnnAttentionDescriptor descriptor = makeDescriptor();
    descriptor.q = AttentionTensorSpec::bshd(3, 4, 64, 64, DataType::FP16);
    descriptor.k = AttentionTensorSpec::bshd(3, 4, 80, 64, DataType::FP16);
    descriptor.v = AttentionTensorSpec::bshd(3, 4, 80, 64, DataType::FP16);
    descriptor.o = AttentionTensorSpec::bshd(3, 4, 64, 64, DataType::FP16);
    return descriptor;
}

CudnnAttentionDescriptor makePagedDescriptor() {
    CudnnAttentionDescriptor descriptor = makeDescriptor();
    descriptor.k = AttentionTensorSpec::bhsd(6, 4, 64, 64, DataType::FP16);
    descriptor.v = AttentionTensorSpec::bhsd(6, 4, 64, 64, DataType::FP16);
    descriptor.usePaddingMask = true;
    descriptor.usePagedKvCache = true;
    descriptor.pagedKv.maxSequenceLengthKv = 128;
    return descriptor;
}


bool envFlagEnabled(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && std::string(value) == "1";
}

bool runFp8AttentionProbeEnabled() {
    return envFlagEnabled("THOR_RUN_EXPERIMENTAL_CUDNN_ATTENTION_SUPPORT_SURFACE") ||
           envFlagEnabled("THOR_RUN_EXPERIMENTAL_CUDNN_FP8_ATTENTION_SUPPORT_SURFACE");
}

bool expectFp8AttentionProbeSupport() { return envFlagEnabled("THOR_EXPECT_CUDNN_FP8_ATTENTION_SUPPORT"); }

int fp8AttentionProbeGpuNum() {
    const char* value = std::getenv("THOR_CUDNN_FP8_ATTENTION_PROBE_GPU");
    if (value == nullptr)
        return 0;
    return std::atoi(value);
}

std::string attentionDataTypeName(DataType dataType) { return TensorDescriptor::getElementTypeName(dataType); }

std::string attentionLayoutName(AttentionTensorLayout layout) {
    switch (layout) {
        case AttentionTensorLayout::BHSD:
            return "bhsd";
        case AttentionTensorLayout::BSHD:
            return "bshd";
    }
    return "unknown";
}

std::string attentionMaskName(AttentionMaskKind maskKind) {
    switch (maskKind) {
        case AttentionMaskKind::None:
            return "none";
        case AttentionMaskKind::CausalTopLeft:
            return "causal_top_left";
        case AttentionMaskKind::CausalBottomRight:
            return "causal_bottom_right";
        case AttentionMaskKind::SlidingWindowTopLeft:
            return "sliding_window_top_left";
        case AttentionMaskKind::SlidingWindowBottomRight:
            return "sliding_window_bottom_right";
    }
    return "unknown";
}

AttentionTensorSpec scoreBiasSpec(std::vector<int64_t> dims, DataType dataType = DataType::FP32) {
    AttentionTensorSpec spec;
    spec.dimensions = dims;
    spec.strides.resize(spec.dimensions.size(), 1);
    for (int64_t i = static_cast<int64_t>(spec.dimensions.size()) - 2; i >= 0; --i) {
        spec.strides[static_cast<size_t>(i)] = spec.strides[static_cast<size_t>(i + 1)] * spec.dimensions[static_cast<size_t>(i + 1)];
    }
    spec.dataType = dataType;
    spec.ragged = false;
    return spec;
}

struct Fp8AttentionProbeCase {
    std::string name;
    DataType dataType = DataType::FP8_E4M3;
    AttentionTensorLayout layout = AttentionTensorLayout::BSHD;
    int64_t batch = 2;
    int64_t queryHeads = 4;
    int64_t keyValueHeads = 4;
    int64_t queryLength = 16;
    int64_t keyValueLength = 16;
    int64_t qkHeadDim = 64;
    int64_t vHeadDim = 64;
    AttentionMaskKind maskKind = AttentionMaskKind::None;
    bool generateStats = false;
    bool runBackward = false;
    bool useBias = false;
    std::vector<int64_t> biasDimensions;
    bool usePaddingMask = false;
    bool useDropout = false;
};

CudnnAttentionDescriptor makeFp8ProbeDescriptor(const Fp8AttentionProbeCase& probeCase) {
    CudnnAttentionDescriptor descriptor;
    descriptor.q = AttentionTensorSpec::fromLayout(
        probeCase.layout, probeCase.batch, probeCase.queryHeads, probeCase.queryLength, probeCase.qkHeadDim, probeCase.dataType);
    descriptor.k = AttentionTensorSpec::fromLayout(
        probeCase.layout, probeCase.batch, probeCase.keyValueHeads, probeCase.keyValueLength, probeCase.qkHeadDim, probeCase.dataType);
    descriptor.v = AttentionTensorSpec::fromLayout(
        probeCase.layout, probeCase.batch, probeCase.keyValueHeads, probeCase.keyValueLength, probeCase.vHeadDim, probeCase.dataType);
    descriptor.o = AttentionTensorSpec::fromLayout(
        probeCase.layout, probeCase.batch, probeCase.queryHeads, probeCase.queryLength, probeCase.vHeadDim, probeCase.dataType);
    descriptor.computeDataType = DataType::FP32;
    descriptor.intermediateDataType = DataType::FP32;
    descriptor.maskKind = probeCase.maskKind;
    descriptor.generateStats = probeCase.generateStats || probeCase.runBackward;
    descriptor.useFp8 = true;
    descriptor.useBias = probeCase.useBias;
    if (probeCase.useBias) {
        const std::vector<int64_t> dims = probeCase.biasDimensions.empty()
                                          ? std::vector<int64_t>{probeCase.batch,
                                                                 probeCase.queryHeads,
                                                                 probeCase.queryLength,
                                                                 probeCase.keyValueLength}
                                          : probeCase.biasDimensions;
        descriptor.bias = scoreBiasSpec(dims, probeCase.dataType);
    }
    descriptor.usePaddingMask = probeCase.usePaddingMask;
    if (probeCase.useDropout) {
        descriptor.dropout.probability = 0.125f;
        descriptor.dropout.usePhilox = true;
    }
    descriptor.debugName = std::string("thor_fp8_sdpa_probe_") + probeCase.name;
    return descriptor;
}

std::vector<uint64_t> asUint64Vector(const std::vector<int64_t>& values) {
    std::vector<uint64_t> converted;
    converted.reserve(values.size());
    for (int64_t value : values) {
        converted.push_back(static_cast<uint64_t>(value));
    }
    return converted;
}

uint64_t storageElementsForSpec(const AttentionTensorSpec& spec) {
    uint64_t maxOffset = 0;
    for (size_t i = 0; i < spec.dimensions.size(); ++i) {
        maxOffset += static_cast<uint64_t>(spec.dimensions.at(i) - 1) * static_cast<uint64_t>(spec.strides.at(i));
    }
    return maxOffset + 1;
}

Tensor makeTensorForSpec(TensorPlacement placement, const AttentionTensorSpec& spec, Stream stream) {
    Tensor storage = Tensor::zeros(placement, TensorDescriptor(spec.dataType, {storageElementsForSpec(spec)}), stream);
    return storage.aliasView(asUint64Vector(spec.dimensions), asUint64Vector(spec.strides));
}

Tensor makeFp32Scalar(TensorPlacement placement, Stream stream, double value) {
    return Tensor::values(placement, TensorDescriptor(DataType::FP32, {1, 1, 1, 1}), stream, value);
}

Tensor makeFp32Zeros(TensorPlacement placement, const std::vector<uint64_t>& dims, Stream stream) {
    return Tensor::zeros(placement, TensorDescriptor(DataType::FP32, dims), stream);
}

Tensor makeInt32Values(TensorPlacement placement, const std::vector<uint64_t>& dims, Stream stream, double value) {
    return Tensor::values(placement, TensorDescriptor(DataType::INT32, dims), stream, value);
}

Tensor makeInt64Scalar(TensorPlacement placement, Stream stream, double value) {
    return Tensor::values(placement, TensorDescriptor(DataType::INT64, {1, 1, 1, 1}), stream, value);
}

std::string probeCaseLabel(const Fp8AttentionProbeCase& probeCase) {
    std::ostringstream out;
    out << "case=" << probeCase.name << " dtype=" << attentionDataTypeName(probeCase.dataType)
        << " layout=" << attentionLayoutName(probeCase.layout) << " q_heads=" << probeCase.queryHeads
        << " kv_heads=" << probeCase.keyValueHeads << " sq=" << probeCase.queryLength << " skv=" << probeCase.keyValueLength
        << " dqk=" << probeCase.qkHeadDim << " dv=" << probeCase.vHeadDim << " mask=" << attentionMaskName(probeCase.maskKind)
        << " generate_stats=" << ((probeCase.generateStats || probeCase.runBackward) ? 1 : 0)
        << " backward=" << (probeCase.runBackward ? 1 : 0) << " bias=" << (probeCase.useBias ? 1 : 0)
        << " padding=" << (probeCase.usePaddingMask ? 1 : 0) << " dropout=" << (probeCase.useDropout ? 1 : 0);
    if (probeCase.useBias) {
        const std::vector<int64_t> dims = probeCase.biasDimensions.empty()
                                          ? std::vector<int64_t>{probeCase.batch,
                                                                 probeCase.queryHeads,
                                                                 probeCase.queryLength,
                                                                 probeCase.keyValueLength}
                                          : probeCase.biasDimensions;
        out << " bias_dims=" << testing::PrintToString(dims);
    }
    return out.str();
}

}  // namespace

TEST(CudnnAttentionDescriptor, AllowsPackedRaggedQOAndKVOffsetPairs) {
    CudnnAttentionDescriptor descriptor = makePackedDescriptor();
    descriptor.q.ragged = true;
    descriptor.o.ragged = true;
    descriptor.k.ragged = true;
    descriptor.v.ragged = true;
    descriptor.usePaddingMask = true;

    EXPECT_NO_THROW(descriptor.validateForward());
}


TEST(CudnnAttentionDescriptor, RejectsRaggedOffsetsWithoutBshdPackedLayout) {
    CudnnAttentionDescriptor descriptor = makeDescriptor();
    descriptor.q.ragged = true;
    descriptor.o.ragged = true;
    descriptor.k.ragged = true;
    descriptor.v.ragged = true;

    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnAttentionDescriptor, RejectsRaggedQWithoutOutputOffset) {
    CudnnAttentionDescriptor descriptor = makePackedDescriptor();
    descriptor.q.ragged = true;

    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnAttentionDescriptor, RejectsRaggedOutputWithoutQOffset) {
    CudnnAttentionDescriptor descriptor = makePackedDescriptor();
    descriptor.o.ragged = true;

    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnAttentionDescriptor, RejectsRaggedKWithoutVOffset) {
    CudnnAttentionDescriptor descriptor = makePackedDescriptor();
    descriptor.k.ragged = true;

    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnAttentionDescriptor, RejectsRaggedVWithoutKOffset) {
    CudnnAttentionDescriptor descriptor = makePackedDescriptor();
    descriptor.v.ragged = true;

    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnAttentionDescriptor, RejectsRaggedOffsetsWithoutPaddingMaskSequenceLengthMode) {
    CudnnAttentionDescriptor descriptor = makePackedDescriptor();
    descriptor.q.ragged = true;
    descriptor.o.ragged = true;
    descriptor.k.ragged = true;
    descriptor.v.ragged = true;

    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnAttentionDescriptor, AllowsAdditiveBiasBroadcastSurface) {
    const std::vector<std::vector<int64_t>> allowed{{1, 1, 64, 80}, {1, 4, 64, 80}, {3, 1, 64, 80}, {3, 4, 64, 80}};
    for (const auto& dims : allowed) {
        CudnnAttentionDescriptor descriptor = makeDescriptor();
        descriptor.useBias = true;
        descriptor.bias = scoreBiasSpec(dims);
        EXPECT_NO_THROW(descriptor.validateForward()) << "bias dims " << testing::PrintToString(dims);
    }
}

TEST(CudnnAttentionDescriptor, RejectsAlibiCausalTopLeftPositiveRightBoundOutsideProbeSurface) {
    CudnnAttentionDescriptor descriptor = makeDescriptor();
    descriptor.maskKind = AttentionMaskKind::CausalTopLeft;
    descriptor.diagonalRightBound = 1;
    descriptor.useAlibiMask = true;

    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
    descriptor.generateStats = true;
    EXPECT_THROW(descriptor.validateBackward(), std::invalid_argument);
}

TEST(CudnnAttentionDescriptor, RejectsAlibiSlidingWindowTopLeftPositiveRightBoundOutsideProbeSurface) {
    CudnnAttentionDescriptor descriptor = makeDescriptor();
    descriptor.maskKind = AttentionMaskKind::SlidingWindowTopLeft;
    descriptor.diagonalLeftBound = 3;
    descriptor.diagonalRightBound = 1;
    descriptor.useAlibiMask = true;

    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnAttentionDescriptor, RejectsInvalidAdditiveBiasBroadcastShape) {
    CudnnAttentionDescriptor descriptor = makeDescriptor();
    descriptor.useBias = true;
    descriptor.bias = scoreBiasSpec({3, 2, 64, 80});

    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnAttentionDescriptor, AllowsRaggedAttentionWithFullDenseAdditiveBias) {
    CudnnAttentionDescriptor descriptor = makePackedDescriptor();
    descriptor.q.ragged = true;
    descriptor.o.ragged = true;
    descriptor.k.ragged = true;
    descriptor.v.ragged = true;
    descriptor.usePaddingMask = true;
    descriptor.useBias = true;

    EXPECT_NO_THROW(descriptor.validateForward());
}

TEST(CudnnAttentionDescriptor, RejectsRaggedAttentionBackwardWithFullDenseAdditiveBias) {
    CudnnAttentionDescriptor descriptor = makePackedDescriptor();
    descriptor.q.ragged = true;
    descriptor.o.ragged = true;
    descriptor.k.ragged = true;
    descriptor.v.ragged = true;
    descriptor.usePaddingMask = true;
    descriptor.useBias = true;
    descriptor.generateStats = true;

    EXPECT_THROW(descriptor.validateBackward(), std::invalid_argument);
}

TEST(CudnnAttentionDescriptor, RejectsCombiningRaggedOffsetsWithPagedKvCache) {
    CudnnAttentionDescriptor descriptor = makePackedDescriptor();
    descriptor.q.ragged = true;
    descriptor.o.ragged = true;
    descriptor.k.ragged = true;
    descriptor.v.ragged = true;
    descriptor.usePaddingMask = true;
    descriptor.usePagedKvCache = true;
    descriptor.pagedKv.maxSequenceLengthKv = 128;

    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnAttentionDescriptor, AllowsPagedKvForwardDescriptorWithPositiveMaxSequenceLength) {
    CudnnAttentionDescriptor descriptor = makePagedDescriptor();

    EXPECT_NO_THROW(descriptor.validateForward());
}

TEST(CudnnAttentionDescriptor, RejectsPagedKvForwardDescriptorWithoutPositiveMaxSequenceLength) {
    CudnnAttentionDescriptor descriptor = makePagedDescriptor();
    descriptor.pagedKv.maxSequenceLengthKv = 0;

    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnAttentionDescriptor, RejectsPagedKvForwardDescriptorWithoutPaddingMaskSequenceLengths) {
    CudnnAttentionDescriptor descriptor = makePagedDescriptor();
    descriptor.usePaddingMask = false;

    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnAttentionDescriptor, RejectsPagedKvBackwardUntilTrainingSemanticsAreDefined) {
    CudnnAttentionDescriptor descriptor = makePagedDescriptor();
    descriptor.generateStats = true;

    EXPECT_THROW(descriptor.validateBackward(), std::invalid_argument);
}

TEST(CudnnAttentionFrontendFp8Probe, ExperimentalForwardSupportSurface) {
    if (!runFp8AttentionProbeEnabled()) {
        GTEST_SKIP() << "Set THOR_RUN_EXPERIMENTAL_CUDNN_FP8_ATTENTION_SUPPORT_SURFACE=1 or "
                        "THOR_RUN_EXPERIMENTAL_CUDNN_ATTENTION_SUPPORT_SURFACE=1 to probe cuDNN FP8 SDPA support.";
    }

    // This test intentionally asks cuDNN about combinations that Thor does not expose as production-supported yet.
    // Reuse the unified internal bypass so the probe reaches cuDNN Frontend graph construction/execution.
    setenv("THOR_EXPERIMENTAL_CUDNN_ATTENTION_SUPPORT_SURFACE", "1", 1);

    int deviceCount = 0;
    cudaError_t status = cudaGetDeviceCount(&deviceCount);
    if (status != cudaSuccess || deviceCount <= 0) {
        GTEST_SKIP() << "No CUDA GPU is available for the cuDNN FP8 SDPA probe: " << cudaGetErrorString(status);
    }

    const int gpuNum = fp8AttentionProbeGpuNum();
    ASSERT_GE(gpuNum, 0) << "THOR_CUDNN_FP8_ATTENTION_PROBE_GPU must be non-negative.";
    ASSERT_LT(gpuNum, deviceCount) << "THOR_CUDNN_FP8_ATTENTION_PROBE_GPU selects GPU " << gpuNum << ", but only " << deviceCount
                                   << " CUDA device(s) are visible.";

    cudaDeviceProp prop;
    ASSERT_EQ(cudaGetDeviceProperties(&prop, gpuNum), cudaSuccess);
    std::cout << "FP8_CUDNN_SDPA_PROBE_DEVICE gpu=" << gpuNum << " name=\"" << prop.name << "\" sm=" << prop.major << "."
              << prop.minor << std::endl;

    ScopedGpu scopedGpu(gpuNum);
    Stream stream(gpuNum);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, gpuNum);

    const auto make = [](std::string name,
                         DataType dataType,
                         AttentionTensorLayout layout = AttentionTensorLayout::BSHD,
                         int64_t keyValueHeads = 4,
                         AttentionMaskKind maskKind = AttentionMaskKind::None) {
        Fp8AttentionProbeCase c;
        c.name = name;
        c.dataType = dataType;
        c.layout = layout;
        c.keyValueHeads = keyValueHeads;
        c.maskKind = maskKind;
        return c;
    };

    std::vector<Fp8AttentionProbeCase> cases;
    const auto add = [&cases](Fp8AttentionProbeCase c) { cases.push_back(c); };
    const DataType e4m3 = DataType::FP8_E4M3;
    const DataType e5m2 = DataType::FP8_E5M2;

    add(make("e4m3_bshd_mha_none", e4m3));
    add(make("e4m3_bshd_gqa_none", e4m3, AttentionTensorLayout::BSHD, 2));
    add(make("e4m3_bshd_mqa_none", e4m3, AttentionTensorLayout::BSHD, 1));
    add(make("e4m3_bshd_mha_causal_top_left", e4m3, AttentionTensorLayout::BSHD, 4, AttentionMaskKind::CausalTopLeft));
    add(make("e4m3_bshd_mha_causal_bottom_right_api_is_causal", e4m3, AttentionTensorLayout::BSHD, 4, AttentionMaskKind::CausalBottomRight));
    Fp8AttentionProbeCase c = make("e4m3_bshd_mha_none_stats", e4m3);
    c.generateStats = true;
    add(c);
    add(make("e4m3_bhsd_mha_none", e4m3, AttentionTensorLayout::BHSD));
    c = make("e4m3_bshd_mha_d128", e4m3);
    c.qkHeadDim = 128;
    c.vHeadDim = 128;
    add(c);
    c = make("e4m3_bshd_mha_d256", e4m3);
    c.qkHeadDim = 256;
    c.vHeadDim = 256;
    add(c);
    c = make("e4m3_bshd_mha_qk192_v128", e4m3);
    c.qkHeadDim = 192;
    c.vHeadDim = 128;
    add(c);
    c = make("e4m3_bshd_mha_decode_q1_kv64_causal", e4m3, AttentionTensorLayout::BSHD, 4, AttentionMaskKind::CausalTopLeft);
    c.queryLength = 1;
    c.keyValueLength = 64;
    add(c);

    add(make("e5m2_bshd_mha_none", e5m2));
    add(make("e5m2_bshd_gqa_none", e5m2, AttentionTensorLayout::BSHD, 2));
    add(make("e5m2_bshd_mqa_none", e5m2, AttentionTensorLayout::BSHD, 1));
    add(make("e5m2_bshd_mha_causal_top_left", e5m2, AttentionTensorLayout::BSHD, 4, AttentionMaskKind::CausalTopLeft));
    add(make("e5m2_bshd_mha_causal_bottom_right_api_is_causal", e5m2, AttentionTensorLayout::BSHD, 4, AttentionMaskKind::CausalBottomRight));
    c = make("e5m2_bshd_mha_none_stats", e5m2);
    c.generateStats = true;
    add(c);
    add(make("e5m2_bhsd_mha_none", e5m2, AttentionTensorLayout::BHSD));
    c = make("e5m2_bshd_mha_d128", e5m2);
    c.qkHeadDim = 128;
    c.vHeadDim = 128;
    add(c);
    c = make("e5m2_bshd_mha_qk192_v128", e5m2);
    c.qkHeadDim = 192;
    c.vHeadDim = 128;
    add(c);

    c = make("e4m3_bshd_mha_bias_1_1", e4m3);
    c.useBias = true;
    c.biasDimensions = {1, 1, 16, 16};
    add(c);
    c = make("e4m3_bshd_mha_bias_1_h", e4m3);
    c.useBias = true;
    c.biasDimensions = {1, 4, 16, 16};
    add(c);
    c = make("e4m3_bshd_mha_bias_b_1", e4m3);
    c.useBias = true;
    c.biasDimensions = {2, 1, 16, 16};
    add(c);
    c = make("e4m3_bshd_mha_bias_b_h", e4m3);
    c.useBias = true;
    c.biasDimensions = {2, 4, 16, 16};
    add(c);
    c = make("e4m3_bshd_mha_bias_query_broadcast", e4m3);
    c.useBias = true;
    c.biasDimensions = {2, 4, 1, 16};
    add(c);
    c = make("e4m3_bshd_mha_bias_key_broadcast", e4m3);
    c.useBias = true;
    c.biasDimensions = {2, 4, 16, 1};
    add(c);
    c = make("e5m2_bshd_mha_bias_b_h", e5m2);
    c.useBias = true;
    c.biasDimensions = {2, 4, 16, 16};
    add(c);

    c = make("e4m3_bshd_mha_padding", e4m3);
    c.usePaddingMask = true;
    add(c);
    c = make("e4m3_bshd_mha_padding_causal", e4m3, AttentionTensorLayout::BSHD, 4, AttentionMaskKind::CausalTopLeft);
    c.usePaddingMask = true;
    add(c);
    c = make("e4m3_bshd_gqa_padding", e4m3, AttentionTensorLayout::BSHD, 2);
    c.usePaddingMask = true;
    add(c);
    c = make("e5m2_bshd_mha_padding", e5m2);
    c.usePaddingMask = true;
    add(c);

    c = make("e4m3_bshd_mha_dropout", e4m3);
    c.useDropout = true;
    add(c);
    c = make("e4m3_bshd_mha_dropout_causal", e4m3, AttentionTensorLayout::BSHD, 4, AttentionMaskKind::CausalTopLeft);
    c.useDropout = true;
    add(c);
    c = make("e4m3_bshd_mha_bias_dropout", e4m3);
    c.useBias = true;
    c.biasDimensions = {2, 4, 16, 16};
    c.useDropout = true;
    add(c);
    c = make("e5m2_bshd_mha_dropout", e5m2);
    c.useDropout = true;
    add(c);

    c = make("e4m3_bshd_mha_backward_none", e4m3);
    c.runBackward = true;
    add(c);
    c = make("e4m3_bshd_mha_backward_causal", e4m3, AttentionTensorLayout::BSHD, 4, AttentionMaskKind::CausalTopLeft);
    c.runBackward = true;
    add(c);
    c = make("e4m3_bshd_gqa_backward_none", e4m3, AttentionTensorLayout::BSHD, 2);
    c.runBackward = true;
    add(c);
    c = make("e4m3_bhsd_mha_backward_none", e4m3, AttentionTensorLayout::BHSD);
    c.runBackward = true;
    add(c);
    c = make("e4m3_bshd_mha_backward_d128", e4m3);
    c.qkHeadDim = 128;
    c.vHeadDim = 128;
    c.runBackward = true;
    add(c);
    c = make("e4m3_bshd_mha_backward_d256", e4m3);
    c.qkHeadDim = 256;
    c.vHeadDim = 256;
    c.runBackward = true;
    add(c);
    c = make("e4m3_bshd_mha_backward_decode_q1_kv64_causal", e4m3, AttentionTensorLayout::BSHD, 4, AttentionMaskKind::CausalTopLeft);
    c.queryLength = 1;
    c.keyValueLength = 64;
    c.runBackward = true;
    add(c);
    c = make("e5m2_bshd_mha_backward_none", e5m2);
    c.runBackward = true;
    add(c);
    c = make("e5m2_bshd_mha_backward_causal", e5m2, AttentionTensorLayout::BSHD, 4, AttentionMaskKind::CausalTopLeft);
    c.runBackward = true;
    add(c);

    size_t supportedCount = 0;
    for (const Fp8AttentionProbeCase& probeCase : cases) {
        CudnnScaledDotProductAttention::instance().clearCache();
        const CudnnAttentionDescriptor descriptor = makeFp8ProbeDescriptor(probeCase);
        const std::string label = probeCaseLabel(probeCase);

        try {
            Tensor q = makeTensorForSpec(gpuPlacement, descriptor.q, stream);
            Tensor k = makeTensorForSpec(gpuPlacement, descriptor.k, stream);
            Tensor v = makeTensorForSpec(gpuPlacement, descriptor.v, stream);
            Tensor o = makeTensorForSpec(gpuPlacement, descriptor.o, stream);
            Tensor stats = makeFp32Zeros(gpuPlacement,
                                         {static_cast<uint64_t>(descriptor.batchSize()),
                                          static_cast<uint64_t>(descriptor.queryHeads()),
                                          static_cast<uint64_t>(descriptor.queryLength()),
                                          1},
                                         stream);
            Tensor descaleQ = makeFp32Scalar(gpuPlacement, stream, 1.0);
            Tensor descaleK = makeFp32Scalar(gpuPlacement, stream, 1.0);
            Tensor descaleV = makeFp32Scalar(gpuPlacement, stream, 1.0);
            Tensor descaleS = makeFp32Scalar(gpuPlacement, stream, 1.0);
            Tensor scaleS = makeFp32Scalar(gpuPlacement, stream, 1.0);
            Tensor scaleO = makeFp32Scalar(gpuPlacement, stream, 1.0);
            Tensor amaxS = makeFp32Scalar(gpuPlacement, stream, 0.0);
            Tensor amaxO = makeFp32Scalar(gpuPlacement, stream, 0.0);

            std::optional<Tensor> statsOpt;
            if (descriptor.generateStats) {
                statsOpt = stats;
            }
            std::optional<Tensor> biasOpt;
            if (descriptor.useBias) {
                biasOpt = makeTensorForSpec(gpuPlacement, descriptor.bias.value(), stream);
            }
            std::optional<Tensor> seqLenQOpt;
            std::optional<Tensor> seqLenKvOpt;
            if (descriptor.usePaddingMask) {
                seqLenQOpt = makeInt32Values(gpuPlacement, {static_cast<uint64_t>(descriptor.batchSize())}, stream, descriptor.queryLength());
                seqLenKvOpt = makeInt32Values(gpuPlacement, {static_cast<uint64_t>(descriptor.batchSize())}, stream, descriptor.keyValueLength());
            }
            std::optional<Tensor> dropoutSeedOpt;
            std::optional<Tensor> dropoutOffsetOpt;
            if (descriptor.dropout.probability > 0.0f) {
                dropoutSeedOpt = makeInt64Scalar(gpuPlacement, stream, 1234.0);
                dropoutOffsetOpt = makeInt64Scalar(gpuPlacement, stream, 0.0);
            }

            CudnnAttentionForwardArgs forwardArgs{.q = q,
                                                  .k = k,
                                                  .v = v,
                                                  .o = o,
                                                  .stats = statsOpt,
                                                  .bias = biasOpt,
                                                  .seqLenQ = seqLenQOpt,
                                                  .seqLenKv = seqLenKvOpt,
                                                  .dropoutSeed = dropoutSeedOpt,
                                                  .dropoutOffset = dropoutOffsetOpt,
                                                  .descaleQ = descaleQ,
                                                  .descaleK = descaleK,
                                                  .descaleV = descaleV,
                                                  .descaleS = descaleS,
                                                  .scaleS = scaleS,
                                                  .scaleO = scaleO,
                                                  .amaxS = amaxS,
                                                  .amaxO = amaxO};

            CudnnScaledDotProductAttention::instance().forward(descriptor, forwardArgs, stream);

            if (probeCase.runBackward) {
                Tensor dO = makeTensorForSpec(gpuPlacement, descriptor.o, stream);
                Tensor dQ = makeTensorForSpec(gpuPlacement, descriptor.q, stream);
                Tensor dK = makeTensorForSpec(gpuPlacement, descriptor.k, stream);
                Tensor dV = makeTensorForSpec(gpuPlacement, descriptor.v, stream);
                Tensor descaleO = makeFp32Scalar(gpuPlacement, stream, 1.0);
                Tensor descaleDO = makeFp32Scalar(gpuPlacement, stream, 1.0);
                Tensor descaleDP = makeFp32Scalar(gpuPlacement, stream, 1.0);
                Tensor scaleDQ = makeFp32Scalar(gpuPlacement, stream, 1.0);
                Tensor scaleDK = makeFp32Scalar(gpuPlacement, stream, 1.0);
                Tensor scaleDV = makeFp32Scalar(gpuPlacement, stream, 1.0);
                Tensor scaleDP = makeFp32Scalar(gpuPlacement, stream, 1.0);
                Tensor amaxDQ = makeFp32Scalar(gpuPlacement, stream, 0.0);
                Tensor amaxDK = makeFp32Scalar(gpuPlacement, stream, 0.0);
                Tensor amaxDV = makeFp32Scalar(gpuPlacement, stream, 0.0);
                Tensor amaxDP = makeFp32Scalar(gpuPlacement, stream, 0.0);

                CudnnAttentionBackwardArgs backwardArgs{.q = q,
                                                        .k = k,
                                                        .v = v,
                                                        .o = o,
                                                        .dO = dO,
                                                        .stats = stats,
                                                        .dQ = dQ,
                                                        .dK = dK,
                                                        .dV = dV,
                                                        .descaleQ = descaleQ,
                                                        .descaleK = descaleK,
                                                        .descaleV = descaleV,
                                                        .descaleO = descaleO,
                                                        .descaleDO = descaleDO,
                                                        .descaleS = descaleS,
                                                        .descaleDP = descaleDP,
                                                        .scaleS = scaleS,
                                                        .scaleDQ = scaleDQ,
                                                        .scaleDK = scaleDK,
                                                        .scaleDV = scaleDV,
                                                        .scaleDP = scaleDP,
                                                        .amaxDQ = amaxDQ,
                                                        .amaxDK = amaxDK,
                                                        .amaxDV = amaxDV,
                                                        .amaxDP = amaxDP};
                CudnnScaledDotProductAttention::instance().backward(descriptor, backwardArgs, stream);
            }

            stream.synchronize();
            ++supportedCount;
            std::cout << "FP8_CUDNN_SDPA_SUPPORTED " << label << std::endl;
        } catch (const std::exception& e) {
            (void)cudaGetLastError();
            std::cout << "FP8_CUDNN_SDPA_UNSUPPORTED " << label << " reason=\"" << e.what() << "\"" << std::endl;
        }
    }

    if (expectFp8AttentionProbeSupport()) {
        EXPECT_GT(supportedCount, 0U) << "Expected at least one cuDNN Frontend FP8 SDPA probe case to be supported on this GPU.";
    }
}
