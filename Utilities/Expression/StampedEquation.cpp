#include "Utilities/Expression/StampedEquation.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/Expression/EquationRunner.h"
#include "Utilities/Expression/FusedEquation.h"
#include "Utilities/Expression/MatmulScalarKernel.h"
#include "Utilities/Expression/ReduceMinMaxBackwardKernel.h"
#include "Utilities/CudaDriver/CudaGraphConditional.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.h"
#include "Utilities/ComputeTopology/MachineEvaluator.h"

#include <cudnn_frontend.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "DeepLearning/Implementation/ThorError.h"

using namespace std;

namespace ThorImplementation {

namespace fe = cudnn_frontend;

namespace {
constexpr int64_t CUDNN_FRONTEND_CONV_X_UID = 7'100'001;
constexpr int64_t CUDNN_FRONTEND_CONV_W_UID = 7'100'002;
constexpr int64_t CUDNN_FRONTEND_CONV_Y_UID = 7'100'003;

static uint64_t checkedFinalScanAxis(const std::vector<uint64_t>& dims, uint64_t encoded_axis) {
    if (dims.empty()) {
        throw std::runtime_error("Expression scan requires rank >= 1.");
    }
    const uint64_t final_axis = static_cast<uint64_t>(dims.size() - 1);
    const uint64_t axis = (encoded_axis == UINT64_MAX) ? final_axis : encoded_axis;
    if (axis >= dims.size()) {
        throw std::runtime_error("Expression scan axis is out of range for input rank.");
    }
    if (axis != final_axis) {
        throw std::runtime_error("Expression scan currently supports only the final contiguous axis.");
    }
    return axis;
}

static CubScanOp toCubScanOp(ScanOp op) {
    switch (op) {
        case ScanOp::Sum:
            return CubScanOp::Sum;
        case ScanOp::Min:
            return CubScanOp::Min;
        case ScanOp::Max:
            return CubScanOp::Max;
        case ScanOp::Product:
            return CubScanOp::Product;
        case ScanOp::ArgMin:
        case ScanOp::ArgMax:
            break;
    }
    throw std::runtime_error("Unsupported Expression scan op.");
}

static bool isArgScanOp(ScanOp op) { return op == ScanOp::ArgMin || op == ScanOp::ArgMax; }


static bool thorMatmulDiagnosticsEnabled() {
    const char *value = std::getenv("THOR_MATMUL_DIAGNOSTICS");
    return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}

static bool thorMatmulDiagnosticsVerbose() {
    const char *value = std::getenv("THOR_MATMUL_DIAGNOSTICS");
    if (value == nullptr) {
        return false;
    }
    const std::string mode(value);
    return mode == "2" || mode == "verbose" || mode == "VERBOSE" || mode == "full" || mode == "FULL";
}

static const char *matmulExprOpName(ExprOp op) {
    switch (op) {
        case ExprOp::MATMUL:
            return "MATMUL";
        case ExprOp::GEMM:
            return "GEMM";
        default:
            return "OTHER";
    }
}

static const char *matmulEpilogueName(MatmulEpilogue epilogue) {
    switch (epilogue) {
        case MatmulEpilogue::Default:
            return "Default";
        case MatmulEpilogue::Relu:
            return "Relu";
        case MatmulEpilogue::Gelu:
            return "Gelu";
    }
    return "Unknown";
}

static const char *matmulBackwardEpilogueName(MatmulBackwardEpilogue epilogue) {
    switch (epilogue) {
        case MatmulBackwardEpilogue::Default:
            return "Default";
        case MatmulBackwardEpilogue::DRelu:
            return "DRelu";
        case MatmulBackwardEpilogue::DGelu:
            return "DGelu";
    }
    return "Unknown";
}

static bool shouldPrintStampedMatmulDiagnosticOnce(const std::string &key) {
    static std::mutex mutex;
    static std::unordered_set<std::string> printed;
    std::lock_guard<std::mutex> lock(mutex);
    return printed.insert(key).second;
}

static CubArgScanOp toCubArgScanOp(ScanOp op) {
    switch (op) {
        case ScanOp::ArgMin:
            return CubArgScanOp::ArgMin;
        case ScanOp::ArgMax:
            return CubArgScanOp::ArgMax;
        default:
            break;
    }
    throw std::runtime_error("Unsupported Expression arg scan op.");
}

static CubScanMode toCubScanMode(ScanMode mode) {
    switch (mode) {
        case ScanMode::Exclusive:
            return CubScanMode::Exclusive;
        case ScanMode::Inclusive:
            return CubScanMode::Inclusive;
    }
    throw std::runtime_error("Unsupported Expression scan mode.");
}

static CubScanDirection toCubScanDirection(bool reverse) {
    return reverse ? CubScanDirection::Reverse : CubScanDirection::Forward;
}

static int64_t checkedDim(const std::vector<uint64_t>& dims, size_t idx, const char* tensor_name) {
    if (idx >= dims.size()) {
        throw std::runtime_error(std::string("Attention tensor '") + tensor_name + "' must have rank 4.");
    }
    if (dims[idx] > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::runtime_error(std::string("Attention tensor '") + tensor_name + "' dimension exceeds int64_t range.");
    }
    return static_cast<int64_t>(dims[idx]);
}

struct AttentionTensorLogicalDims {
    int64_t batch = 0;
    int64_t heads = 0;
    int64_t sequence_length = 0;
    int64_t head_dim = 0;
};

static AttentionTensorLogicalDims logicalAttentionDims(const std::vector<uint64_t>& dims,
                                                       AttentionTensorLayout layout,
                                                       const char* tensor_name) {
    if (dims.size() != 4) {
        throw std::runtime_error(std::string("Thor attention expression tensor '") + tensor_name + "' must have rank 4.");
    }

    switch (layout) {
        case AttentionTensorLayout::BHSD:
            return {checkedDim(dims, 0, tensor_name),
                    checkedDim(dims, 1, tensor_name),
                    checkedDim(dims, 2, tensor_name),
                    checkedDim(dims, 3, tensor_name)};
        case AttentionTensorLayout::BSHD:
            return {checkedDim(dims, 0, tensor_name),
                    checkedDim(dims, 2, tensor_name),
                    checkedDim(dims, 1, tensor_name),
                    checkedDim(dims, 3, tensor_name)};
        default:
            throw std::runtime_error(std::string("Unsupported attention layout for tensor '") + tensor_name + "'.");
    }
}

static std::vector<uint64_t> cudnnSemanticDims(const Tensor& tensor, AttentionTensorLayout layout, const char* tensor_name) {
    const AttentionTensorLogicalDims logical = logicalAttentionDims(tensor.getDimensions(), layout, tensor_name);
    return {static_cast<uint64_t>(logical.batch),
            static_cast<uint64_t>(logical.heads),
            static_cast<uint64_t>(logical.sequence_length),
            static_cast<uint64_t>(logical.head_dim)};
}

static std::vector<uint64_t> cudnnSemanticStrides(const Tensor& tensor, AttentionTensorLayout layout, const char* tensor_name) {
    const std::vector<uint64_t> dims = tensor.getDimensions();
    const std::vector<uint64_t> strides = tensor.getStridesElements();
    if (dims.size() != 4 || strides.size() != 4) {
        throw std::runtime_error(std::string("Attention tensor '") + tensor_name + "' must have rank-4 strides.");
    }
    switch (layout) {
        case AttentionTensorLayout::BHSD:
            return {strides[0], strides[1], strides[2], strides[3]};
        case AttentionTensorLayout::BSHD:
            return {strides[0], strides[2], strides[1], strides[3]};
        default:
            throw std::runtime_error(std::string("Unsupported attention layout for tensor '") + tensor_name + "'.");
    }
}

static std::vector<int64_t> toInt64Strides(const std::vector<uint64_t>& strides, const char* tensor_name) {
    std::vector<int64_t> out;
    out.reserve(strides.size());
    for (uint64_t stride : strides) {
        if (stride > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            throw std::runtime_error(std::string("Attention tensor '") + tensor_name + "' stride exceeds int64_t range.");
        }
        out.push_back(static_cast<int64_t>(stride));
    }
    return out;
}

static AttentionTensorSpec attentionSpecForTensor(const Tensor& tensor, AttentionTensorLayout layout, const char* tensor_name) {
    const AttentionTensorLogicalDims dims = logicalAttentionDims(tensor.getDimensions(), layout, tensor_name);
    AttentionTensorSpec spec;
    spec.dimensions = {dims.batch, dims.heads, dims.sequence_length, dims.head_dim};
    spec.strides = toInt64Strides(cudnnSemanticStrides(tensor, layout, tensor_name), tensor_name);
    spec.dataType = tensor.getDataType();
    spec.ragged = false;
    return spec;
}

static Tensor cudnnSemanticTensorView(const Tensor& tensor, AttentionTensorLayout layout, const char* tensor_name) {
    if (layout == AttentionTensorLayout::BHSD) {
        return tensor;
    }
    return tensor.aliasView(cudnnSemanticDims(tensor, layout, tensor_name), cudnnSemanticStrides(tensor, layout, tensor_name), 0);
}

}  // namespace

static void putFrontendTensorPointer(std::unordered_map<int64_t, void*>& pack, int64_t uid, const Tensor& tensor);
static void executeFrontendConvolutionGraph(const BuiltConvolution& built,
                                            const Stream& run_stream,
                                            std::unordered_map<int64_t, void*>& tensor_pack,
                                            const std::optional<Tensor>& workspace,
                                            const char* op_name);

CudnnRmsNormDescriptor CompiledRmsNorm::descriptorFor(const Tensor& inputTensor,
                                                      const Tensor& scaleTensor,
                                                      const Tensor& outputTensor) const {
    const std::vector<uint64_t> inputDims = inputTensor.getDimensions();
    const std::vector<uint64_t> scaleDims = scaleTensor.getDimensions();
    const std::vector<uint64_t> outputDims = outputTensor.getDimensions();
    if (inputDims.size() != 2 || outputDims.size() != 2) {
        throw std::runtime_error("Thor RMSNorm expression stage requires rank-2 logical [outer, hidden] input/output tensors.");
    }
    if (scaleDims.size() != 1) {
        throw std::runtime_error("Thor RMSNorm expression stage requires a rank-1 [hidden] scale tensor.");
    }
    if (inputDims != outputDims) {
        throw std::runtime_error("Thor RMSNorm expression stage input/output dimensions must match.");
    }
    if (inputDims[1] != normalized_feature_count || scaleDims[0] != normalized_feature_count) {
        throw std::runtime_error("Thor RMSNorm expression stage hidden dimension does not match the compiled descriptor.");
    }
    if (inputTensor.getDataType() != input_dtype) {
        throw std::runtime_error("Thor RMSNorm expression stage input dtype does not match compiled dtype.");
    }
    if (scaleTensor.getDataType() != scale_dtype) {
        throw std::runtime_error("Thor RMSNorm expression stage scale dtype does not match compiled dtype.");
    }
    if (outputTensor.getDataType() != output_dtype) {
        throw std::runtime_error("Thor RMSNorm expression stage output dtype does not match compiled dtype.");
    }

    CudnnRmsNormDescriptor descriptor;
    descriptor.outerSize = inputDims[0];
    descriptor.normalizedFeatureCount = normalized_feature_count;
    descriptor.inputDataType = input_dtype;
    descriptor.parameterDataType = scale_dtype;
    descriptor.outputDataType = output_dtype;
    descriptor.computeDataType = compute_dtype;
    descriptor.epsilon = static_cast<float>(epsilon);
    descriptor.training = false;
    descriptor.fusedActivation = fused_activation;
    descriptor.debugName = debug_name;
    descriptor.validateForward();
    return descriptor;
}

CudnnAttentionDescriptor CompiledAttention::descriptorFor(const Tensor& qTensor,
                                                          const Tensor& kTensor,
                                                          const Tensor& vTensor,
                                                          const Tensor& oTensor) const {
    CudnnAttentionDescriptor descriptor;
    descriptor.q = attentionSpecForTensor(qTensor, q_layout, "q");
    descriptor.k = attentionSpecForTensor(kTensor, k_layout, "k");
    descriptor.v = attentionSpecForTensor(vTensor, v_layout, "v");
    descriptor.o = attentionSpecForTensor(oTensor, o_layout, "o");
    if (use_ragged_offsets) {
        descriptor.q.ragged = true;
        descriptor.k.ragged = true;
        descriptor.v.ragged = true;
        descriptor.o.ragged = true;
    }
    descriptor.computeDataType = compute_dtype;
    descriptor.intermediateDataType = DataType::FP32;
    descriptor.attentionScale = attention_scale;
    descriptor.maskKind = mask_kind;
    descriptor.diagonalLeftBound = diagonal_left_bound;
    descriptor.diagonalRightBound = diagonal_right_bound;
    descriptor.useAlibiMask = use_alibi_mask;
    descriptor.useBias = use_bias;
    descriptor.usePaddingMask = use_padding_mask;
    descriptor.usePagedKvCache = use_paged_kv_cache;
    descriptor.pagedKv.maxSequenceLengthKv = paged_kv_max_sequence_length;
    descriptor.dropout.probability = dropout_probability;
    descriptor.dropout.usePhilox = true;
    descriptor.debugName = debug_name;
    descriptor.useFp8 =
        qTensor.getDataType() == DataType::FP8_E4M3 || qTensor.getDataType() == DataType::FP8_E5M2 ||
        kTensor.getDataType() == DataType::FP8_E4M3 || kTensor.getDataType() == DataType::FP8_E5M2 ||
        vTensor.getDataType() == DataType::FP8_E4M3 || vTensor.getDataType() == DataType::FP8_E5M2 ||
        oTensor.getDataType() == DataType::FP8_E4M3 || oTensor.getDataType() == DataType::FP8_E5M2;
    descriptor.validateForward();
    return descriptor;
}

CudnnAttentionDescriptor CompiledAttentionBackward::descriptorFor(const Tensor& qTensor,
                                                                  const Tensor& kTensor,
                                                                  const Tensor& vTensor,
                                                                  const Tensor& oTensor) const {
    CudnnAttentionDescriptor descriptor;
    descriptor.q = attentionSpecForTensor(qTensor, q_layout, "q");
    descriptor.k = attentionSpecForTensor(kTensor, k_layout, "k");
    descriptor.v = attentionSpecForTensor(vTensor, v_layout, "v");
    descriptor.o = attentionSpecForTensor(oTensor, o_layout, "o");
    if (use_ragged_offsets) {
        descriptor.q.ragged = true;
        descriptor.k.ragged = true;
        descriptor.v.ragged = true;
        descriptor.o.ragged = true;
    }
    descriptor.computeDataType = compute_dtype;
    descriptor.intermediateDataType = DataType::FP32;
    descriptor.attentionScale = attention_scale;
    descriptor.maskKind = mask_kind;
    descriptor.diagonalLeftBound = diagonal_left_bound;
    descriptor.diagonalRightBound = diagonal_right_bound;
    descriptor.useAlibiMask = use_alibi_mask;
    descriptor.useBias = use_bias;
    descriptor.usePaddingMask = use_padding_mask;
    descriptor.usePagedKvCache = use_paged_kv_cache;
    descriptor.pagedKv.maxSequenceLengthKv = paged_kv_max_sequence_length;
    descriptor.dropout.probability = dropout_probability;
    descriptor.dropout.usePhilox = true;
    descriptor.generateStats = true;
    descriptor.deterministicBackward = deterministic_backward;
    descriptor.debugName = debug_name;
    descriptor.useFp8 =
        qTensor.getDataType() == DataType::FP8_E4M3 || qTensor.getDataType() == DataType::FP8_E5M2 ||
        kTensor.getDataType() == DataType::FP8_E4M3 || kTensor.getDataType() == DataType::FP8_E5M2 ||
        vTensor.getDataType() == DataType::FP8_E4M3 || vTensor.getDataType() == DataType::FP8_E5M2 ||
        oTensor.getDataType() == DataType::FP8_E4M3 || oTensor.getDataType() == DataType::FP8_E5M2;
    descriptor.validateBackward();
    return descriptor;
}

DataType CompiledAttentionBackward::outputDTypeFor(ExprOp op) const {
    switch (op) {
        case ExprOp::ATTENTION_BACKWARD_Q:
            return dQ_dtype;
        case ExprOp::ATTENTION_BACKWARD_K:
            return dK_dtype;
        case ExprOp::ATTENTION_BACKWARD_V:
            return dV_dtype;
        case ExprOp::ATTENTION_BACKWARD_BIAS:
            return dQ_dtype;
        default:
            throw std::runtime_error("CompiledAttentionBackward::outputDTypeFor expected an attention-backward output op.");
    }
}

namespace {

bool sameOptionalFloat(const std::optional<float>& lhs, const std::optional<float>& rhs) {
    if (lhs.has_value() != rhs.has_value()) {
        return false;
    }
    if (!lhs.has_value()) {
        return true;
    }
    return lhs.value() == rhs.value();
}

bool attentionConfigMatchesBackward(const CompiledAttention& forward,
                                    const CompiledAttentionBackward& backward,
                                    DataType output_dtype) {
    return forward.q_layout == backward.q_layout && forward.k_layout == backward.k_layout && forward.v_layout == backward.v_layout &&
           forward.o_layout == backward.o_layout && forward.mask_kind == backward.mask_kind &&
           forward.diagonal_left_bound == backward.diagonal_left_bound && forward.diagonal_right_bound == backward.diagonal_right_bound &&
           sameOptionalFloat(forward.attention_scale, backward.attention_scale) && forward.use_alibi_mask == backward.use_alibi_mask &&
           forward.use_bias == backward.use_bias && forward.use_padding_mask == backward.use_padding_mask &&
           forward.use_ragged_offsets == backward.use_ragged_offsets && forward.use_paged_kv_cache == backward.use_paged_kv_cache &&
           forward.paged_kv_max_sequence_length == backward.paged_kv_max_sequence_length &&
           forward.dropout_probability == backward.dropout_probability && forward.compute_dtype == backward.compute_dtype &&
           forward.output_dtype == output_dtype;
}

bool tensorMatches(const Tensor& lhs, const Tensor& rhs) {
    return lhs.isInitialized() && rhs.isInitialized() && lhs == rhs && lhs.getDimensions() == rhs.getDimensions() &&
           lhs.getDataType() == rhs.getDataType() && lhs.getPlacement() == rhs.getPlacement();
}

}  // namespace

void StampedAttention::run() { runOn(stream); }

void StampedAttention::runOn(Stream& run_stream) const {
    if (!compiled_attention) {
        throw std::runtime_error("StampedAttention::runOn called with null compiled attention payload.");
    }

    CudnnAttentionDescriptor descriptor = compiled_attention->descriptorFor(q, k, v, output);
    Tensor cudnnQ = cudnnSemanticTensorView(q, compiled_attention->q_layout, "q");
    Tensor cudnnK = cudnnSemanticTensorView(k, compiled_attention->k_layout, "k");
    Tensor cudnnV = cudnnSemanticTensorView(v, compiled_attention->v_layout, "v");
    Tensor cudnnO = cudnnSemanticTensorView(output, compiled_attention->o_layout, "o");
    CudnnAttentionForwardArgs args{.q = cudnnQ, .k = cudnnK, .v = cudnnV, .o = cudnnO};
    if (compiled_attention->use_bias) {
        if (!bias.has_value()) {
            throw std::runtime_error("StampedAttention requires an additive bias tensor but none was provided.");
        }
        args.bias = bias.value();
    }
    if (compiled_attention->use_padding_mask) {
        if (!seq_len_q.has_value() || !seq_len_kv.has_value()) {
            throw std::runtime_error("StampedAttention requires q/kv sequence length tensors for padding-mask attention.");
        }
        args.seqLenQ = seq_len_q.value();
        args.seqLenKv = seq_len_kv.value();
    }
    if (compiled_attention->use_ragged_offsets) {
        if (!q_ragged_offsets.has_value() || !kv_ragged_offsets.has_value()) {
            throw std::runtime_error("StampedAttention requires q/kv ragged offset tensors for ragged attention.");
        }
        args.raggedOffsetQ = q_ragged_offsets.value();
        args.raggedOffsetO = q_ragged_offsets.value();
        args.raggedOffsetK = kv_ragged_offsets.value();
        args.raggedOffsetV = kv_ragged_offsets.value();
    }
    if (compiled_attention->use_paged_kv_cache) {
        if (!page_table_k.has_value() || !page_table_v.has_value()) {
            throw std::runtime_error("StampedAttention requires K/V page-table tensors for paged KV attention.");
        }
        args.pageTableK = page_table_k.value();
        args.pageTableV = page_table_v.value();
    }
    if (compiled_attention->dropout_probability > 0.0f) {
        if (!dropout_seed.has_value() || !dropout_offset.has_value()) {
            throw std::runtime_error("StampedAttention requires dropout seed/offset tensors for attention dropout.");
        }
        args.dropoutSeed = dropout_seed.value();
        args.dropoutOffset = dropout_offset.value();
    }
    if (compiled_attention->use_fp8_forward_scaling) {
        if (!descale_q.has_value() || !descale_k.has_value() || !descale_v.has_value() || !descale_s.has_value() || !scale_s.has_value() ||
            !scale_o.has_value() || !amax_s.has_value() || !amax_o.has_value()) {
            throw std::runtime_error("StampedAttention requires all FP8 scale/descale/amax tensors for FP8 attention forward.");
        }
        args.descaleQ = descale_q.value();
        args.descaleK = descale_k.value();
        args.descaleV = descale_v.value();
        args.descaleS = descale_s.value();
        args.scaleS = scale_s.value();
        args.scaleO = scale_o.value();
        args.amaxS = amax_s.value();
        args.amaxO = amax_o.value();
    }

    if (forward_state && forward_state->retain_for_backward) {
        if (!forward_state->stats.isInitialized()) {
            throw std::runtime_error("StampedAttention retained-forward state was requested without an allocated stats tensor.");
        }
        descriptor.generateStats = true;
        args.stats = forward_state->stats;
        forward_state->has_valid_stats = false;
    }

    CudnnScaledDotProductAttention::instance().forward(descriptor, args, run_stream);

    if (forward_state && forward_state->retain_for_backward) {
        forward_state->output = output;
        forward_state->has_valid_stats = true;
    }
}

bool StampedAttention::canProvideForwardStateFor(const CompiledAttentionBackward& backward,
                                                 const Tensor& q_tensor,
                                                 const Tensor& k_tensor,
                                                 const Tensor& v_tensor,
                                                 const std::optional<Tensor>& bias_tensor,
                                                 const std::optional<Tensor>& seq_len_q_tensor,
                                                 const std::optional<Tensor>& seq_len_kv_tensor,
                                                 const std::optional<Tensor>& q_ragged_offsets_tensor,
                                                 const std::optional<Tensor>& kv_ragged_offsets_tensor,
                                                 const std::optional<Tensor>& dropout_seed_tensor,
                                                 const std::optional<Tensor>& dropout_offset_tensor,
                                                 const Tensor& dO_tensor) const {
    if (!compiled_attention || !forward_state) {
        return false;
    }
    if (!tensorMatches(q, q_tensor) || !tensorMatches(k, k_tensor) || !tensorMatches(v, v_tensor)) {
        return false;
    }
    if (compiled_attention->use_bias) {
        if (!bias.has_value() || !bias_tensor.has_value() || !tensorMatches(bias.value(), bias_tensor.value())) {
            return false;
        }
    }
    if (compiled_attention->use_padding_mask) {
        if (!seq_len_q.has_value() || !seq_len_kv.has_value() || !seq_len_q_tensor.has_value() || !seq_len_kv_tensor.has_value() ||
            !tensorMatches(seq_len_q.value(), seq_len_q_tensor.value()) || !tensorMatches(seq_len_kv.value(), seq_len_kv_tensor.value())) {
            return false;
        }
    }
    if (compiled_attention->use_ragged_offsets) {
        if (!q_ragged_offsets.has_value() || !kv_ragged_offsets.has_value() || !q_ragged_offsets_tensor.has_value() ||
            !kv_ragged_offsets_tensor.has_value() || !tensorMatches(q_ragged_offsets.value(), q_ragged_offsets_tensor.value()) ||
            !tensorMatches(kv_ragged_offsets.value(), kv_ragged_offsets_tensor.value())) {
            return false;
        }
    }
    if (compiled_attention->dropout_probability > 0.0f) {
        if (!dropout_seed.has_value() || !dropout_offset.has_value() || !dropout_seed_tensor.has_value() ||
            !dropout_offset_tensor.has_value() || !tensorMatches(dropout_seed.value(), dropout_seed_tensor.value()) ||
            !tensorMatches(dropout_offset.value(), dropout_offset_tensor.value())) {
            return false;
        }
    }
    if (output.getDimensions() != dO_tensor.getDimensions() || output.getDataType() != dO_tensor.getDataType() ||
        output.getPlacement() != dO_tensor.getPlacement()) {
        return false;
    }
    return attentionConfigMatchesBackward(*compiled_attention, backward, dO_tensor.getDataType());
}

StampedAttention::StampedAttention(std::shared_ptr<CompiledAttention> compiled,
                                   const Tensor& q,
                                   const Tensor& k,
                                   const Tensor& v,
                                   const std::optional<Tensor>& bias,
                                   const std::optional<Tensor>& seq_len_q,
                                   const std::optional<Tensor>& seq_len_kv,
                                   const std::optional<Tensor>& q_ragged_offsets,
                                   const std::optional<Tensor>& kv_ragged_offsets,
                                   const std::optional<Tensor>& page_table_k,
                                   const std::optional<Tensor>& page_table_v,
                                   const std::optional<Tensor>& dropout_seed,
                                   const std::optional<Tensor>& dropout_offset,
                                   const std::optional<Tensor>& descale_q,
                                   const std::optional<Tensor>& descale_k,
                                   const std::optional<Tensor>& descale_v,
                                   const std::optional<Tensor>& descale_s,
                                   const std::optional<Tensor>& scale_s,
                                   const std::optional<Tensor>& scale_o,
                                   const std::optional<Tensor>& amax_s,
                                   const std::optional<Tensor>& amax_o,
                                   const Tensor& output,
                                   const Stream& stream,
                                   std::shared_ptr<AttentionForwardState> forward_state)
    : compiled_attention(std::move(compiled)),
      q(q),
      k(k),
      v(v),
      bias(bias),
      seq_len_q(seq_len_q),
      seq_len_kv(seq_len_kv),
      q_ragged_offsets(q_ragged_offsets),
      kv_ragged_offsets(kv_ragged_offsets),
      page_table_k(page_table_k),
      page_table_v(page_table_v),
      dropout_seed(dropout_seed),
      dropout_offset(dropout_offset),
      descale_q(descale_q),
      descale_k(descale_k),
      descale_v(descale_v),
      descale_s(descale_s),
      scale_s(scale_s),
      scale_o(scale_o),
      amax_s(amax_s),
      amax_o(amax_o),
      output(output),
      stream(stream),
      forward_state(std::move(forward_state)) {}

void StampedAttentionBackward::run() { runOn(stream); }

void StampedAttentionBackward::runOn(Stream& run_stream) const {
    if (!compiled_attention_backward) {
        throw std::runtime_error("StampedAttentionBackward::runOn called with null compiled attention-backward payload.");
    }

    const bool use_saved_forward = saved_forward_state != nullptr;
    const Tensor& forwardOutput = use_saved_forward ? saved_forward_state->output : oScratch;
    const Tensor& forwardStats = use_saved_forward ? saved_forward_state->stats : stats;

    if (use_saved_forward) {
        if (!saved_forward_state->has_valid_stats || !forwardOutput.isInitialized() || !forwardStats.isInitialized()) {
            throw std::runtime_error(
                "Attention-backward expected same-plan retained cuDNN forward stats, but the matching forward stage did not populate "
                "them.");
        }
        if (forwardOutput.getDimensions() != dO.getDimensions() || forwardOutput.getDataType() != dO.getDataType() ||
            forwardOutput.getPlacement() != dO.getPlacement()) {
            throw std::runtime_error("Retained attention forward output is incompatible with attention-backward dO.");
        }
    }

    CudnnAttentionDescriptor descriptor = compiled_attention_backward->descriptorFor(q, k, v, forwardOutput);
    descriptor.generateStats = true;

    Tensor cudnnQ = cudnnSemanticTensorView(q, compiled_attention_backward->q_layout, "q");
    Tensor cudnnK = cudnnSemanticTensorView(k, compiled_attention_backward->k_layout, "k");
    Tensor cudnnV = cudnnSemanticTensorView(v, compiled_attention_backward->v_layout, "v");
    Tensor cudnnO = cudnnSemanticTensorView(forwardOutput, compiled_attention_backward->o_layout, "o");
    Tensor cudnnDO = cudnnSemanticTensorView(dO, compiled_attention_backward->o_layout, "dO");
    Tensor cudnnDQ = cudnnSemanticTensorView(dQ, compiled_attention_backward->q_layout, "dQ");
    Tensor cudnnDK = cudnnSemanticTensorView(dK, compiled_attention_backward->k_layout, "dK");
    Tensor cudnnDV = cudnnSemanticTensorView(dV, compiled_attention_backward->v_layout, "dV");

    if (!use_saved_forward) {
        Tensor cudnnOScratch = cudnnSemanticTensorView(oScratch, compiled_attention_backward->o_layout, "oScratch");
        CudnnAttentionForwardArgs fwdArgs{.q = cudnnQ, .k = cudnnK, .v = cudnnV, .o = cudnnOScratch, .stats = stats};
        if (compiled_attention_backward->use_bias) {
            if (!bias.has_value()) {
                throw std::runtime_error("StampedAttentionBackward requires an additive bias tensor but none was provided.");
            }
            fwdArgs.bias = bias.value();
        }
        if (compiled_attention_backward->use_padding_mask) {
            if (!seq_len_q.has_value() || !seq_len_kv.has_value()) {
                throw std::runtime_error("StampedAttentionBackward requires q/kv sequence length tensors for padding-mask attention.");
            }
            fwdArgs.seqLenQ = seq_len_q.value();
            fwdArgs.seqLenKv = seq_len_kv.value();
        }
        if (compiled_attention_backward->use_ragged_offsets) {
            if (!q_ragged_offsets.has_value() || !kv_ragged_offsets.has_value()) {
                throw std::runtime_error("StampedAttentionBackward requires q/kv ragged offset tensors for ragged attention.");
            }
            fwdArgs.raggedOffsetQ = q_ragged_offsets.value();
            fwdArgs.raggedOffsetO = q_ragged_offsets.value();
            fwdArgs.raggedOffsetK = kv_ragged_offsets.value();
            fwdArgs.raggedOffsetV = kv_ragged_offsets.value();
        }
        if (compiled_attention_backward->dropout_probability > 0.0f) {
            if (!dropout_seed.has_value() || !dropout_offset.has_value()) {
                throw std::runtime_error("StampedAttentionBackward requires dropout seed/offset tensors for attention dropout.");
            }
            fwdArgs.dropoutSeed = dropout_seed.value();
            fwdArgs.dropoutOffset = dropout_offset.value();
        }
        CudnnScaledDotProductAttention::instance().forward(descriptor, fwdArgs, run_stream);
    }

    CudnnAttentionBackwardArgs bwdArgs{.q = cudnnQ,
                                       .k = cudnnK,
                                       .v = cudnnV,
                                       .o = cudnnO,
                                       .dO = cudnnDO,
                                       .stats = forwardStats,
                                       .dQ = cudnnDQ,
                                       .dK = cudnnDK,
                                       .dV = cudnnDV};
    if (compiled_attention_backward->use_bias) {
        if (!bias.has_value()) {
            throw std::runtime_error("StampedAttentionBackward requires an additive bias tensor but none was provided.");
        }
        bwdArgs.bias = bias.value();
        if (!dBiasScratch.has_value()) {
            throw std::runtime_error("StampedAttentionBackward requires an additive-bias gradient scratch tensor but none was allocated.");
        }
        bwdArgs.dBias = dBiasScratch.value();
    }
    if (compiled_attention_backward->use_padding_mask) {
        if (!seq_len_q.has_value() || !seq_len_kv.has_value()) {
            throw std::runtime_error("StampedAttentionBackward requires q/kv sequence length tensors for padding-mask attention.");
        }
        bwdArgs.seqLenQ = seq_len_q.value();
        bwdArgs.seqLenKv = seq_len_kv.value();
    }
    if (compiled_attention_backward->use_ragged_offsets) {
        if (!q_ragged_offsets.has_value() || !kv_ragged_offsets.has_value()) {
            throw std::runtime_error("StampedAttentionBackward requires q/kv ragged offset tensors for ragged attention.");
        }
        bwdArgs.raggedOffsetQ = q_ragged_offsets.value();
        bwdArgs.raggedOffsetO = q_ragged_offsets.value();
        bwdArgs.raggedOffsetDO = q_ragged_offsets.value();
        bwdArgs.raggedOffsetDQ = q_ragged_offsets.value();
        bwdArgs.raggedOffsetK = kv_ragged_offsets.value();
        bwdArgs.raggedOffsetV = kv_ragged_offsets.value();
        bwdArgs.raggedOffsetDK = kv_ragged_offsets.value();
        bwdArgs.raggedOffsetDV = kv_ragged_offsets.value();
    }
    if (compiled_attention_backward->dropout_probability > 0.0f) {
        if (!dropout_seed.has_value() || !dropout_offset.has_value()) {
            throw std::runtime_error("StampedAttentionBackward requires dropout seed/offset tensors for attention dropout.");
        }
        bwdArgs.dropoutSeed = dropout_seed.value();
        bwdArgs.dropoutOffset = dropout_offset.value();
    }
    CudnnScaledDotProductAttention::instance().backward(descriptor, bwdArgs, run_stream);
}

StampedAttentionBackward::StampedAttentionBackward(std::shared_ptr<CompiledAttentionBackward> compiled,
                                                   const Tensor& q,
                                                   const Tensor& k,
                                                   const Tensor& v,
                                                   const std::optional<Tensor>& bias,
                                                   const std::optional<Tensor>& seq_len_q,
                                                   const std::optional<Tensor>& seq_len_kv,
                                                   const std::optional<Tensor>& q_ragged_offsets,
                                                   const std::optional<Tensor>& kv_ragged_offsets,
                                                   const std::optional<Tensor>& dropout_seed,
                                                   const std::optional<Tensor>& dropout_offset,
                                                   const Tensor& dO,
                                                   const Tensor& dQ,
                                                   const Tensor& dK,
                                                   const Tensor& dV,
                                                   const Tensor& oScratch,
                                                   const Tensor& stats,
                                                   const std::optional<Tensor>& dBiasScratch,
                                                   const Stream& stream,
                                                   std::shared_ptr<AttentionForwardState> saved_forward_state)
    : compiled_attention_backward(std::move(compiled)),
      q(q),
      k(k),
      v(v),
      bias(bias),
      seq_len_q(seq_len_q),
      seq_len_kv(seq_len_kv),
      q_ragged_offsets(q_ragged_offsets),
      kv_ragged_offsets(kv_ragged_offsets),
      dropout_seed(dropout_seed),
      dropout_offset(dropout_offset),
      dO(dO),
      dQ(dQ),
      dK(dK),
      dV(dV),
      oScratch(oScratch),
      stats(stats),
      dBiasScratch(dBiasScratch),
      stream(stream),
      saved_forward_state(std::move(saved_forward_state)),
      outputs{this->dQ, this->dK, this->dV} {
    if (this->dBiasScratch.has_value()) {
        outputs.push_back(this->dBiasScratch.value());
    }
}

StampedCudaKernel::StampedCudaKernel(std::shared_ptr<CompiledCudaKernel> compiled,
                                     std::vector<Tensor> inputs,
                                     std::vector<TensorScalarBinding> tensor_runtime_scalars,
                                     std::vector<Tensor> outputs,
                                     std::vector<StampedCudaKernelParam> params,
                                     CudaKernelLaunchConfig launch_config,
                                     const Stream& stream)
    : compiled(std::move(compiled)),
      inputs(std::move(inputs)),
      tensor_runtime_scalars(std::move(tensor_runtime_scalars)),
      outputs(std::move(outputs)),
      params(std::move(params)),
      launch_config(launch_config),
      stream(stream) {
    if (!this->compiled) {
        throw std::runtime_error("StampedCudaKernel requires a compiled CUDA kernel.");
    }
    if (this->compiled->kernel == nullptr) {
        throw std::runtime_error("StampedCudaKernel compiled kernel handle is null.");
    }
    if (this->outputs.empty()) {
        throw std::runtime_error("StampedCudaKernel requires at least one output tensor.");
    }
    for (const Tensor& input : this->inputs) {
        if (!input.isInitialized()) {
            throw std::runtime_error("StampedCudaKernel input tensor is not initialized.");
        }
        if (input.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
            throw std::runtime_error("StampedCudaKernel input tensor is not on GPU.");
        }
        if (input.getPlacement().getDeviceNum() != this->compiled->device_num) {
            throw std::runtime_error("StampedCudaKernel input tensor GPU does not match compiled kernel GPU.");
        }
    }
    for (const TensorScalarBinding& binding : this->tensor_runtime_scalars) {
        if (!binding.buffer.isInitialized()) {
            throw std::runtime_error("StampedCudaKernel tensor runtime scalar buffer is not initialized.");
        }
        if (binding.buffer.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
            throw std::runtime_error("StampedCudaKernel tensor runtime scalar buffer is not on GPU.");
        }
        if (binding.buffer.getPlacement().getDeviceNum() != this->compiled->device_num) {
            throw std::runtime_error("StampedCudaKernel tensor runtime scalar GPU does not match compiled kernel GPU.");
        }
    }
    for (const Tensor& output : this->outputs) {
        if (!output.isInitialized()) {
            throw std::runtime_error("StampedCudaKernel output tensor is not initialized.");
        }
        if (output.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
            throw std::runtime_error("StampedCudaKernel output tensor is not on GPU.");
        }
        if (output.getPlacement().getDeviceNum() != this->compiled->device_num) {
            throw std::runtime_error("StampedCudaKernel output tensor GPU does not match compiled kernel GPU.");
        }
    }
    if (this->stream.getGpuNum() != this->compiled->device_num) {
        throw std::runtime_error("StampedCudaKernel stream GPU does not match compiled kernel GPU.");
    }
}

uint32_t StampedCudaKernel::gpuNum() const {
    if (!compiled) {
        throw std::runtime_error("StampedCudaKernel::gpuNum called with no compiled kernel.");
    }
    return static_cast<uint32_t>(compiled->device_num);
}

Tensor StampedCudaKernel::getOutputTensor() const {
    if (outputs.size() != 1) {
        throw std::runtime_error("StampedCudaKernel::getOutputTensor called for a multi-output kernel.");
    }
    return outputs.front();
}

void StampedCudaKernel::run() { runOn(stream); }

void StampedCudaKernel::run(const std::unordered_map<std::string, float>& runtime_scalars) { runOn(stream, runtime_scalars); }

void StampedCudaKernel::runOn(Stream& run_stream) const { runOn(run_stream, {}); }

void StampedCudaKernel::runOn(Stream& run_stream, const std::unordered_map<std::string, float>& runtime_scalars) const {
    if (!compiled || compiled->kernel == nullptr) {
        throw std::runtime_error("StampedCudaKernel::runOn called with no compiled kernel.");
    }
    if (run_stream.getGpuNum() != compiled->device_num) {
        throw std::runtime_error("StampedCudaKernel::runOn stream GPU does not match compiled kernel GPU.");
    }
    if (launch_config.grid.x == 0 || launch_config.grid.y == 0 || launch_config.grid.z == 0 || launch_config.block.x == 0 ||
        launch_config.block.y == 0 || launch_config.block.z == 0) {
        throw std::runtime_error("StampedCudaKernel launch grid/block dimensions must be non-zero.");
    }

    ScopedGpu scoped_gpu(compiled->device_num);

    if (runtime_scalars.empty() && requiresRuntimeScalars()) {
        throw std::runtime_error("StampedCudaKernel::runOn requires runtime scalar values. Call run(runtime_scalars).");
    }

    std::unordered_set<std::string> consumed_runtime_scalar_names;
    consumed_runtime_scalar_names.reserve(runtime_scalars.size());

    std::vector<void*> pointer_values;
    pointer_values.reserve(params.size());
    std::vector<float> runtime_scalar_values;
    runtime_scalar_values.reserve(params.size());
    std::vector<void*> kernel_args;
    kernel_args.reserve(params.size());

    for (const StampedCudaKernelParam& param : params) {
        switch (param.kind) {
            case StampedCudaKernelParam::Kind::TensorInput: {
                if (param.tensor_index >= inputs.size()) {
                    throw std::runtime_error("StampedCudaKernel tensor input parameter index out of range: " + param.name);
                }
                void* ptr = const_cast<void*>(static_cast<const void*>(inputs[param.tensor_index].getMemPtr<void>()));
                pointer_values.push_back(ptr);
                kernel_args.push_back(&pointer_values.back());
                break;
            }
            case StampedCudaKernelParam::Kind::TensorRuntimeScalar: {
                if (param.tensor_index >= tensor_runtime_scalars.size()) {
                    throw std::runtime_error("StampedCudaKernel tensor runtime scalar parameter index out of range: " + param.name);
                }
                const TensorScalarBinding& binding = tensor_runtime_scalars[param.tensor_index];
                auto* base = static_cast<const uint8_t*>(binding.buffer.getMemPtr());
                void* ptr = (void*)(base + binding.byteOffset);
                pointer_values.push_back(ptr);
                kernel_args.push_back(&pointer_values.back());
                break;
            }
            case StampedCudaKernelParam::Kind::HostRuntimeScalar: {
                auto it = runtime_scalars.find(param.name);
                if (it == runtime_scalars.end()) {
                    throw std::runtime_error("Missing value for runtime scalar: " + param.name +
                                             "  - if it was meant to be constant, use a constant scalar instead.");
                }
                runtime_scalar_values.push_back(it->second);
                kernel_args.push_back(&runtime_scalar_values.back());
                consumed_runtime_scalar_names.insert(param.name);
                break;
            }
            case StampedCudaKernelParam::Kind::TensorOutput: {
                if (param.tensor_index >= outputs.size()) {
                    throw std::runtime_error("StampedCudaKernel tensor output parameter index out of range: " + param.name);
                }
                void* ptr = (void*)outputs[param.tensor_index].getMemPtr<void>();
                pointer_values.push_back(ptr);
                kernel_args.push_back(&pointer_values.back());
                break;
            }
            case StampedCudaKernelParam::Kind::Scalar: {
                std::visit([&](const auto& value) { kernel_args.push_back(const_cast<void*>(static_cast<const void*>(&value))); },
                           param.scalar_value);
                break;
            }
            default:
                throw std::runtime_error("StampedCudaKernel encountered unknown parameter kind.");
        }
    }

    for (const auto& [name, _] : runtime_scalars) {
        if (!consumed_runtime_scalar_names.contains(name)) {
            throw std::runtime_error("Unexpected runtime scalar override for stamped CUDA kernel: " + name);
        }
    }

    CU_CHECK(cuLaunchKernel(compiled->kernel,
                            launch_config.grid.x,
                            launch_config.grid.y,
                            launch_config.grid.z,
                            launch_config.block.x,
                            launch_config.block.y,
                            launch_config.block.z,
                            launch_config.dynamic_shared_bytes,
                            reinterpret_cast<CUstream>(run_stream.getStream()),
                            kernel_args.data(),
                            nullptr));
}

bool StampedCudaKernel::requiresRuntimeScalars() const {
    for (const StampedCudaKernelParam& param : params) {
        if (param.kind == StampedCudaKernelParam::Kind::HostRuntimeScalar) {
            return true;
        }
    }
    return false;
}

std::unordered_set<std::string> StampedCudaKernel::runtimeScalarNames() const {
    std::unordered_set<std::string> names;
    for (const StampedCudaKernelParam& param : params) {
        if (param.kind == StampedCudaKernelParam::Kind::HostRuntimeScalar) {
            names.insert(param.name);
        }
    }
    return names;
}

void StampedEquation::run() { runOn(stream); }

void StampedEquation::runOn(Stream& run_stream) const {
    if (!compiledEquation) {
        throw std::runtime_error("StampedEquation::runOn called with null compiled equation.");
    }

    if (outputs.empty()) {
        throw std::runtime_error("StampedEquation::runOn called with no output tensors.");
    }

    for (size_t i = 0; i < compiledEquation->input_kinds.size(); ++i) {
        if (compiledEquation->input_kinds[i] == NamedInput::Kind::RuntimeScalarFp32) {
            throw std::runtime_error("StampedEquation::runOn requires runtime scalar values. Call run(runtime_scalars).");
        }
    }

    EquationRunner::run(compiledEquation, inputs, outputs, run_stream);
}

void StampedEquation::run(const std::unordered_map<std::string, float>& runtime_scalars) { runOn(stream, runtime_scalars); }

void StampedEquation::runOn(Stream& run_stream, const std::unordered_map<std::string, float>& runtime_scalars) const {
    if (!compiledEquation) {
        throw std::runtime_error("StampedEquation::runOn called with null compiled equation.");
    }

    if (outputs.empty()) {
        throw std::runtime_error("StampedEquation::runOn called with no output tensors.");
    }

    if (runtime_scalars.empty()) {
        runOn(run_stream);
        return;
    }

    std::vector<RuntimeInputValue> overridden_inputs = inputs;
    std::unordered_set<std::string> consumed_names;

    for (size_t i = 0; i < compiledEquation->input_names.size(); ++i) {
        if (compiledEquation->input_kinds[i] != NamedInput::Kind::RuntimeScalarFp32) {
            continue;
        }

        const std::string& name = compiledEquation->input_names[i];
        auto it = runtime_scalars.find(name);
        if (it == runtime_scalars.end()) {
            throw std::runtime_error("Missing value for runtime scalar: " + name +
                                     "  - if it was meant to be constant, use a constant scalar instead.");
        }

        overridden_inputs[i] = it->second;
        consumed_names.insert(name);
    }

    for (const auto& [name, _] : runtime_scalars) {
        if (!consumed_names.contains(name)) {
            throw std::runtime_error("Unexpected runtime scalar override for stamped equation: " + name);
        }
    }

    EquationRunner::run(compiledEquation, overridden_inputs, outputs, run_stream);
}

StampedReduction::StampedReduction(
    std::shared_ptr<BuiltReduction> built, const Tensor& input, const Tensor& output, const Stream& stream, std::optional<Tensor> workspace)
    : built_reduction(built), input(input), output(output), workspace(workspace), stream(stream) {
    if (built_reduction->workspace_bytes != 0) {
        THOR_THROW_IF_FALSE(workspace.has_value());
        THOR_THROW_IF_FALSE(workspace.value().getArraySizeInBytes() >= built_reduction->workspace_bytes);
    }
    THOR_THROW_IF_FALSE(input.getDataType() == built_reduction->key.input_dtype);
    THOR_THROW_IF_FALSE(output.getDataType() == built_reduction->key.output_dtype);
}

void StampedReduction::run() { runOn(stream); }

void StampedReduction::runOn(Stream& run_stream) const {
    if (built_reduction->identity_reduction) {
        Tensor output_view = output;
        output_view.copyFromAsync(input, run_stream);
        return;
    }

    void* workspace_ptr = nullptr;
    if (built_reduction->workspace_bytes > 0) {
        THOR_THROW_IF_FALSE(workspace.has_value());
        workspace_ptr = (void*)workspace.value().getMemPtr();
    }

    CUDNN_CHECK(cudnnReduceTensor(run_stream.getCudnnHandle(),
                                  built_reduction->reduce_desc,
                                  nullptr,
                                  0,
                                  workspace_ptr,
                                  built_reduction->workspace_bytes,
                                  alpha,
                                  built_reduction->a_desc,
                                  input.getMemPtr(),
                                  beta,
                                  built_reduction->c_desc,
                                  (void*)output.getMemPtr()));
}

StampedArgMinMax::StampedArgMinMax(std::shared_ptr<BuiltReduction> built,
                                   const Tensor& input,
                                   const Tensor& output,
                                   const Tensor& reduction_value_output,
                                   const Stream& stream,
                                   std::optional<Tensor> workspace)
    : built_reduction(built),
      input(input),
      output(output),
      reduction_value_output(reduction_value_output),
      workspace(workspace),
      stream(stream) {
    if (!built_reduction->key.output_indices) {
        throw std::runtime_error("StampedArgMinMax requires a BuiltReduction configured for indices.");
    }
    if (built_reduction->workspace_bytes != 0) {
        THOR_THROW_IF_FALSE(workspace.has_value());
        THOR_THROW_IF_FALSE(workspace.value().getArraySizeInBytes() >= built_reduction->workspace_bytes);
    }
}

void StampedArgMinMax::run() { runOn(stream); }

void StampedArgMinMax::runOn(Stream& run_stream) const {
    // std::cerr << "[REDUCE_MINMAX_BW] input dtype=" << TensorDescriptor::getElementTypeName(input.getDataType())
    //           << " built.input_dtype=" << TensorDescriptor::getElementTypeName(built_reduction->key.input_dtype)
    //           << " built.output_dtype=" << TensorDescriptor::getElementTypeName(built_reduction->key.output_dtype)
    //           << " built.compute_dtype=" << TensorDescriptor::getElementTypeName(built_reduction->key.compute_dtype)
    //           << " reduction_value_output dtype=" << TensorDescriptor::getElementTypeName(reduction_value_output.getDataType())
    //           << std::endl;

    void* workspace_ptr = nullptr;
    if (built_reduction->workspace_bytes > 0) {
        THOR_THROW_IF_FALSE(workspace.has_value());
        workspace_ptr = (void*)workspace.value().getMemPtr();
    }

    CUDNN_CHECK(cudnnReduceTensor(run_stream.getCudnnHandle(),
                                  built_reduction->reduce_desc,
                                  (void*)output.getMemPtr(),
                                  built_reduction->indices_bytes,
                                  workspace_ptr,
                                  built_reduction->workspace_bytes,
                                  alpha,
                                  built_reduction->a_desc,
                                  input.getMemPtr(),
                                  beta,
                                  built_reduction->c_desc,
                                  (void*)reduction_value_output.getMemPtr()));
}

StampedSegmentedReduction::StampedSegmentedReduction(std::shared_ptr<CompiledSegmentedReduction> compiled,
                                                     const Tensor& input,
                                                     const Tensor& output,
                                                     const Tensor& segment_offsets,
                                                     const Stream& stream)
    : compiled_segmented_reduction(std::move(compiled)), input(input), output(output), segment_offsets(segment_offsets), stream(stream) {
    if (!compiled_segmented_reduction) {
        throw std::runtime_error("StampedSegmentedReduction requires a compiled segmented reduction descriptor.");
    }
    if (input.getDataType() != compiled_segmented_reduction->input_dtype ||
        output.getDataType() != compiled_segmented_reduction->output_dtype ||
        segment_offsets.getDataType() != compiled_segmented_reduction->offset_dtype) {
        throw std::runtime_error("Segmented-reduction tensor dtypes do not match the compiled descriptor.");
    }
    if (input.getPlacement() != output.getPlacement() || input.getPlacement() != segment_offsets.getPlacement()) {
        throw std::runtime_error("Segmented-reduction input, output, and offsets must share one GPU placement.");
    }
    if (input.getDimensions().size() != 1) {
        throw std::runtime_error("Segmented-reduction expression currently supports rank-1 scalar ragged values only.");
    }
    const std::vector<uint64_t> offset_dims = segment_offsets.getDimensions();
    if (offset_dims.size() != 1 || offset_dims[0] == 0) {
        throw std::runtime_error("Segmented-reduction offsets must be a non-empty rank-1 tensor of shape [num_segments + 1].");
    }
    const uint64_t num_items = input.getTotalNumElements();
    const uint64_t num_segments = offset_dims[0] - 1;
    if (output.getDimensions() != std::vector<uint64_t>{num_segments}) {
        throw std::runtime_error("Segmented-reduction output must have shape [num_segments].");
    }

    size_t temp_storage_bytes = 1;
    switch (compiled_segmented_reduction->op) {
        case ExprOp::SEGMENTED_REDUCE_SUM:
            sum_plan = prepareCubDeviceSegmentedReduceSum(input, output, segment_offsets, num_items, num_segments);
            temp_storage_bytes = sum_plan.temp_storage_bytes;
            break;
        case ExprOp::SEGMENTED_REDUCE_MIN:
            min_plan = prepareCubDeviceSegmentedReduceMin(input, output, segment_offsets, num_items, num_segments);
            temp_storage_bytes = min_plan.temp_storage_bytes;
            break;
        case ExprOp::SEGMENTED_REDUCE_MAX:
            max_plan = prepareCubDeviceSegmentedReduceMax(input, output, segment_offsets, num_items, num_segments);
            temp_storage_bytes = max_plan.temp_storage_bytes;
            break;
        default:
            throw std::runtime_error("Unsupported segmented-reduction op.");
    }
    temp_storage = Tensor(input.getPlacement(), TensorDescriptor(DataType::UINT8, {std::max<size_t>(temp_storage_bytes, 1)}));
}

void StampedSegmentedReduction::run() { runOn(stream); }

void StampedSegmentedReduction::runOn(Stream& run_stream) const {
    switch (compiled_segmented_reduction->op) {
        case ExprOp::SEGMENTED_REDUCE_SUM:
            cubDeviceSegmentedReduceSum(sum_plan, temp_storage, input, output, segment_offsets, run_stream);
            break;
        case ExprOp::SEGMENTED_REDUCE_MIN:
            cubDeviceSegmentedReduceMin(min_plan, temp_storage, input, output, segment_offsets, run_stream);
            break;
        case ExprOp::SEGMENTED_REDUCE_MAX:
            cubDeviceSegmentedReduceMax(max_plan, temp_storage, input, output, segment_offsets, run_stream);
            break;
        default:
            throw std::runtime_error("Unsupported segmented-reduction op.");
    }
}

StampedScan::StampedScan(std::shared_ptr<CompiledScan> compiled,
                         const Tensor& input,
                         const Tensor& output,
                         const Stream& stream,
                         std::optional<Tensor> segment_offsets,
                         std::optional<Tensor> value_output)
    : compiled_scan(std::move(compiled)),
      input(input),
      output(output),
      value_output(value_output.value_or(Tensor())),
      segment_offsets(std::move(segment_offsets)),
      has_value_output(value_output.has_value()),
      stream(stream) {
    if (!compiled_scan) {
        throw std::runtime_error("StampedScan requires a compiled scan descriptor.");
    }
    if (input.getDataType() != compiled_scan->input_dtype || output.getDataType() != compiled_scan->output_dtype) {
        throw std::runtime_error("StampedScan tensor dtypes do not match the compiled scan descriptor.");
    }
    const bool arg_scan = isArgScanOp(compiled_scan->op);
    if (arg_scan) {
        if (output.getDataType() != DataType::UINT32) {
            throw std::runtime_error("Expression arg scan output dtype must be UINT32.");
        }
        if (has_value_output) {
            if (value_output.value().getDataType() != input.getDataType()) {
                throw std::runtime_error("Expression paired arg scan value output dtype must match input dtype.");
            }
            if (value_output.value().getPlacement() != input.getPlacement()) {
                throw std::runtime_error("Expression paired arg scan value output must be on the same GPU placement as input.");
            }
            if (value_output.value().getDimensions() != input.getDimensions()) {
                throw std::runtime_error("Expression paired arg scan value output shape must match input shape.");
            }
        }
    } else if (input.getDataType() != output.getDataType()) {
        throw std::runtime_error("Expression scan currently requires input and output dtypes to match.");
    }
    if (input.getPlacement() != output.getPlacement()) {
        throw std::runtime_error("Expression scan input and output must be on the same GPU placement.");
    }
    if (input.getDimensions() != output.getDimensions()) {
        throw std::runtime_error("Expression scan output shape must match input shape.");
    }
    if (compiled_scan->segmented_by_offsets != this->segment_offsets.has_value()) {
        throw std::runtime_error("StampedScan segmented-offset input does not match the compiled scan descriptor.");
    }

    const std::vector<uint64_t> dims = input.getDimensions();
    checkedFinalScanAxis(dims, compiled_scan->axis);
    const uint64_t num_items = input.getTotalNumElements();

    const CubScanMode cub_mode = toCubScanMode(compiled_scan->mode);
    const CubScanDirection cub_direction = toCubScanDirection(compiled_scan->reverse);

    size_t temp_storage_bytes = 1;
    if (compiled_scan->segmented_by_offsets) {
        const Tensor& offsets = this->segment_offsets.value();
        if (!compiled_scan->offset_dtype.has_value() || offsets.getDataType() != compiled_scan->offset_dtype.value()) {
            throw std::runtime_error("StampedScan segment-offset dtype does not match the compiled scan descriptor.");
        }
        if (offsets.getPlacement() != input.getPlacement()) {
            throw std::runtime_error("Expression segmented_scan input and offsets must be on the same GPU placement.");
        }
        const std::vector<uint64_t> offset_dims = offsets.getDimensions();
        if (offset_dims.size() != 1 || offset_dims[0] == 0) {
            throw std::runtime_error("Expression segmented_scan offsets must be a non-empty rank-1 tensor of shape [num_segments + 1].");
        }
        const uint64_t num_segments = offset_dims[0] - 1;
        ragged_segmented = true;
        if (arg_scan) {
            ragged_segmented_arg_scan_plan = prepareCubDeviceSegmentedArgScan(
                input, output, offsets, num_items, num_segments, toCubArgScanOp(compiled_scan->op), cub_mode, cub_direction);
            temp_storage_bytes = ragged_segmented_arg_scan_plan.temp_storage_bytes;
        } else {
            ragged_segmented_scan_plan = prepareCubDeviceSegmentedScan(
                input, output, offsets, num_items, num_segments, toCubScanOp(compiled_scan->op), cub_mode, cub_direction);
            temp_storage_bytes = ragged_segmented_scan_plan.temp_storage_bytes;
        }
    } else {
        const uint64_t segment_size = dims.empty() ? 0 : dims.back();
        const uint64_t num_segments = (segment_size == 0) ? 0 : num_items / segment_size;
        uniform_segmented = num_segments > 1;
        if (uniform_segmented) {
            if (arg_scan) {
                segmented_arg_scan_plan = prepareCubDeviceSegmentedUniformArgScan(
                    input, output, num_items, num_segments, segment_size, toCubArgScanOp(compiled_scan->op), cub_mode, cub_direction);
                temp_storage_bytes = segmented_arg_scan_plan.temp_storage_bytes;
            } else {
                segmented_scan_plan = prepareCubDeviceSegmentedUniformScan(
                    input, output, num_items, num_segments, segment_size, toCubScanOp(compiled_scan->op), cub_mode, cub_direction);
                temp_storage_bytes = segmented_scan_plan.temp_storage_bytes;
            }
        } else {
            if (arg_scan) {
                arg_scan_plan = prepareCubDeviceArgScan(input, output, num_items, toCubArgScanOp(compiled_scan->op), cub_mode, cub_direction);
                temp_storage_bytes = arg_scan_plan.temp_storage_bytes;
            } else {
                scan_plan = prepareCubDeviceScan(input, output, num_items, toCubScanOp(compiled_scan->op), cub_mode, cub_direction);
                temp_storage_bytes = scan_plan.temp_storage_bytes;
            }
        }
    }

    temp_storage = Tensor(input.getPlacement(), TensorDescriptor(DataType::UINT8, {std::max<size_t>(temp_storage_bytes, 1)}));
}

void StampedScan::run() { runOn(stream); }

void StampedScan::runOn(Stream& run_stream) const {
    const bool arg_scan = isArgScanOp(compiled_scan->op);
    if (ragged_segmented) {
        if (arg_scan) {
            cubDeviceSegmentedArgScan(ragged_segmented_arg_scan_plan, temp_storage, input, output, segment_offsets.value(), run_stream);
        } else {
            cubDeviceSegmentedScan(ragged_segmented_scan_plan, temp_storage, input, output, segment_offsets.value(), run_stream);
        }
    } else if (uniform_segmented) {
        if (arg_scan) {
            cubDeviceSegmentedUniformArgScan(segmented_arg_scan_plan, temp_storage, input, output, run_stream);
        } else {
            cubDeviceSegmentedUniformScan(segmented_scan_plan, temp_storage, input, output, run_stream);
        }
    } else {
        if (arg_scan) {
            cubDeviceArgScan(arg_scan_plan, temp_storage, input, output, run_stream);
        } else {
            cubDeviceScan(scan_plan, temp_storage, input, output, run_stream);
        }
    }
    if (arg_scan && has_value_output) {
        cubDeviceArgScanValuesFromIndices(input,
                                          output,
                                          value_output,
                                          input.getTotalNumElements(),
                                          compiled_scan->op == ScanOp::ArgMin ? CubArgScanOp::ArgMin : CubArgScanOp::ArgMax,
                                          run_stream);
    }
}

StampedSoftmax::StampedSoftmax(std::shared_ptr<CompiledSoftmax> compiled,
                               std::shared_ptr<BuiltSoftmax> built,
                               const Tensor& input,
                               const Tensor& output,
                               const Stream& stream)
    : compiled_softmax(std::move(compiled)), built_softmax(std::move(built)), input(input), output(output), stream(stream) {
    if (!compiled_softmax || !built_softmax) {
        throw std::runtime_error("StampedSoftmax requires compiled and built softmax payloads.");
    }
    THOR_THROW_IF_FALSE(input.getDataType() == built_softmax->key.input_dtype);
    THOR_THROW_IF_FALSE(output.getDataType() == built_softmax->key.output_dtype);
}

void StampedSoftmax::run() { runOn(stream); }

void StampedSoftmax::runOn(Stream& run_stream) const {
    CUDNN_CHECK(cudnnSoftmaxForward(run_stream.getCudnnHandle(),
                                    built_softmax->key.algorithm,
                                    built_softmax->key.mode,
                                    alpha,
                                    built_softmax->x_desc,
                                    input.getMemPtr(),
                                    beta,
                                    built_softmax->y_desc,
                                    (void*)output.getMemPtr()));
}

StampedConvolution::StampedConvolution(std::shared_ptr<CompiledConvolution> compiled,
                                       std::shared_ptr<BuiltConvolution> built,
                                       const Tensor& input,
                                       const Tensor& filter,
                                       const Tensor& output,
                                       const Stream& stream,
                                       std::optional<Tensor> workspace)
    : compiled_convolution(std::move(compiled)),
      built_convolution(std::move(built)),
      input(input),
      filter(filter),
      output(output),
      stream(stream),
      workspace(std::move(workspace)) {}

void StampedConvolution::run() { runOn(stream); }

void StampedConvolution::runOn(Stream& run_stream) const {
    if (!built_convolution) {
        throw std::runtime_error("StampedConvolution missing built convolution payload.");
    }

    if (built_convolution->use_cudnn_frontend) {
        std::unordered_map<int64_t, void*> tensor_pack;
        putFrontendTensorPointer(tensor_pack, CUDNN_FRONTEND_CONV_X_UID, input);
        putFrontendTensorPointer(tensor_pack, CUDNN_FRONTEND_CONV_W_UID, filter);
        putFrontendTensorPointer(tensor_pack, CUDNN_FRONTEND_CONV_Y_UID, output);
        executeFrontendConvolutionGraph(
            *built_convolution, run_stream, tensor_pack, workspace, compiled_convolution->is_3d ? "CONV3D forward" : "CONV2D forward");
        return;
    }

    throw std::runtime_error("StampedConvolution received non-frontend convolution payload unexpectedly.");
}

StampedConvolutionBackward::StampedConvolutionBackward(std::shared_ptr<CompiledConvolutionBackward> compiled,
                                                       std::shared_ptr<BuiltConvolution> built,
                                                       const Tensor& input,
                                                       const Tensor& grad_output,
                                                       const Tensor& output,
                                                       const Stream& stream,
                                                       std::optional<Tensor> workspace)
    : compiled_convolution_backward(std::move(compiled)),
      built_convolution(std::move(built)),
      input(input),
      grad_output(grad_output),
      output(output),
      stream(stream),
      workspace(std::move(workspace)) {}

void StampedConvolutionBackward::run() { runOn(stream); }

void StampedConvolutionBackward::runOn(Stream& run_stream) const {
    if (!built_convolution) {
        throw std::runtime_error("StampedConvolutionBackward missing built convolution payload.");
    }
    if (!compiled_convolution_backward) {
        throw std::runtime_error("StampedConvolutionBackward missing compiled convolution payload.");
    }

    if (built_convolution->use_cudnn_frontend) {
        std::unordered_map<int64_t, void*> tensor_pack;
        if (compiled_convolution_backward->op == ExprOp::CONV2D_BACKWARD_DATA ||
            compiled_convolution_backward->op == ExprOp::CONV3D_BACKWARD_DATA) {
            putFrontendTensorPointer(tensor_pack, CUDNN_FRONTEND_CONV_W_UID, input);
            putFrontendTensorPointer(tensor_pack, CUDNN_FRONTEND_CONV_Y_UID, grad_output);
            putFrontendTensorPointer(tensor_pack, CUDNN_FRONTEND_CONV_X_UID, output);
            executeFrontendConvolutionGraph(
                *built_convolution,
                run_stream,
                tensor_pack,
                workspace,
                compiled_convolution_backward->op == ExprOp::CONV3D_BACKWARD_DATA ? "CONV3D backward-data" : "CONV2D backward-data");
            return;
        }
        if (compiled_convolution_backward->op == ExprOp::CONV2D_BACKWARD_FILTER ||
            compiled_convolution_backward->op == ExprOp::CONV3D_BACKWARD_FILTER) {
            putFrontendTensorPointer(tensor_pack, CUDNN_FRONTEND_CONV_X_UID, input);
            putFrontendTensorPointer(tensor_pack, CUDNN_FRONTEND_CONV_Y_UID, grad_output);
            putFrontendTensorPointer(tensor_pack, CUDNN_FRONTEND_CONV_W_UID, output);
            executeFrontendConvolutionGraph(
                *built_convolution,
                run_stream,
                tensor_pack,
                workspace,
                compiled_convolution_backward->op == ExprOp::CONV3D_BACKWARD_FILTER ? "CONV3D backward-filter" : "CONV2D backward-filter");
            return;
        }
        throw std::runtime_error("StampedConvolutionBackward received unsupported cuDNN Frontend convolution backward op.");
    }

    throw std::runtime_error("StampedConvolutionBackward received non-frontend convolution payload unexpectedly.");
}

StampedRmsNorm::StampedRmsNorm(
    std::shared_ptr<CompiledRmsNorm> compiled, const Tensor& input, const Tensor& scale, const Tensor& output, const Stream& stream)
    : compiled_rms_norm(std::move(compiled)), input(input), scale(scale), output(output), stream(stream) {
    if (!compiled_rms_norm) {
        throw std::runtime_error("StampedRmsNorm requires a compiled RMSNorm payload.");
    }
}

void StampedRmsNorm::run() { runOn(stream); }

void StampedRmsNorm::runOn(Stream& run_stream) const {
    const CudnnRmsNormDescriptor descriptor = compiled_rms_norm->descriptorFor(input, scale, output);
    CudnnRmsNormForwardArgs args;
    args.x = input;
    args.scale = scale;
    args.y = output;
    CudnnRmsNorm::instance().forward(descriptor, args, run_stream);
}


StampedEmbeddingLookup::StampedEmbeddingLookup(std::shared_ptr<CompiledEmbeddingLookup> compiled,
                                                 const Tensor& indices,
                                                 const Tensor& weights,
                                                 const Tensor& output,
                                                 const Stream& stream,
                                                 std::vector<Tensor> epilogue_inputs)
    : compiled_embedding_lookup(std::move(compiled)),
      indices(indices),
      weights(weights),
      output(output),
      stream(stream),
      epilogue_inputs(std::move(epilogue_inputs)) {
    if (!compiled_embedding_lookup) {
        throw std::runtime_error("StampedEmbeddingLookup constructed with null compiled payload.");
    }
    prepared_forward = prepareEmbeddingForward(indices,
                                               weights,
                                               output,
                                               compiled_embedding_lookup->has_padding_index
                                                   ? std::optional<uint64_t>(compiled_embedding_lookup->padding_index)
                                                   : std::nullopt,
                                               compiled_embedding_lookup->epilogue);
}

void StampedEmbeddingLookup::runOn(Stream& run_stream) const {
    if (!prepared_forward) {
        throw std::runtime_error("StampedEmbeddingLookup::runOn called with null prepared forward payload.");
    }
    launchPreparedEmbeddingForward(*prepared_forward, indices, weights, output, run_stream, epilogue_inputs);
}

StampedMatmul::StampedMatmul(std::shared_ptr<CompiledMatmul> compiled,
                             std::shared_ptr<BuiltMatmul> built,
                             const Tensor& lhs,
                             const Tensor& rhs,
                             const std::optional<Tensor>& addend,
                             const Tensor& output,
                             const Stream& stream,
                             std::optional<Tensor> workspace,
                             std::optional<RuntimeInputValue> alpha_input,
                             std::optional<RuntimeInputValue> beta_input,
                             std::optional<std::string> alpha_runtime_name,
                             std::optional<std::string> beta_runtime_name,
                             std::optional<Tensor> alpha_device_scratch,
                             std::optional<Tensor> beta_device_scratch,
                             std::optional<Tensor> alpha_host_scratch,
                             std::optional<Tensor> beta_host_scratch,
                             std::optional<Tensor> epilogue_aux,
                             std::optional<Tensor> bgrad_output)
    : compiled_matmul(std::move(compiled)),
      built_matmul(std::move(built)),
      lhs(lhs),
      rhs(rhs),
      addend(addend),
      output(output),
      epilogue_aux(epilogue_aux),
      bgrad_output(bgrad_output),
      stream(stream),
      workspace(workspace),
      alpha_input(alpha_input),
      beta_input(beta_input),
      alpha_runtime_name(std::move(alpha_runtime_name)),
      beta_runtime_name(std::move(beta_runtime_name)),
      alpha_device_scratch(alpha_device_scratch),
      beta_device_scratch(beta_device_scratch),
      alpha_host_scratch(alpha_host_scratch),
      beta_host_scratch(beta_host_scratch) {
    if (!compiled_matmul) {
        throw std::runtime_error("StampedMatmul requires non-null compiled payload.");
    }
    if (!built_matmul) {
        throw std::runtime_error("StampedMatmul requires non-null built matmul payload.");
    }
    if (compiled_matmul->backward_epilogue != MatmulBackwardEpilogue::Default && !epilogue_aux.has_value()) {
        throw std::runtime_error("StampedMatmul backward cuBLASLt epilogue requires epilogue_aux.");
    }
    if (compiled_matmul->bgrad_output_dtype.has_value() && !bgrad_output.has_value()) {
        throw std::runtime_error("StampedMatmul backward cuBLASLt bgrad epilogue requires bgrad_output.");
    }
    if (bgrad_output.has_value() && !compiled_matmul->bgrad_output_dtype.has_value()) {
        throw std::runtime_error("StampedMatmul received bgrad_output but the compiled matmul does not declare a bgrad output.");
    }
    if (built_matmul->workspace_bytes != 0) {
        if (!workspace.has_value()) {
            throw std::runtime_error("StampedMatmul requires workspace for the chosen optimal kernel.");
        }
        THOR_THROW_IF_FALSE(workspace.value().getArraySizeInBytes() >= built_matmul->workspace_bytes);
    }
}

struct ResolvedMatmulScale {
    float host_value = 1.0f;
    const float* ptr = nullptr;
    bool is_device_pointer = false;
    std::optional<Tensor> device_scratch = std::nullopt;
    std::optional<Tensor> host_scratch = std::nullopt;

    explicit ResolvedMatmulScale(std::optional<Tensor> device_scratch = std::nullopt, std::optional<Tensor> host_scratch = std::nullopt)
        : ptr(&host_value), device_scratch(device_scratch), host_scratch(host_scratch) {}

    void refreshHostPointer() {
        if (!is_device_pointer) {
            ptr = &host_value;
        }
    }

    void setDevicePointer(const float* device_ptr) {
        ptr = device_ptr;
        is_device_pointer = true;
    }

    void copyHostValueToDevice(Stream& run_stream) {
        if (!device_scratch.has_value()) {
            throw std::runtime_error("Missing preallocated GEMM device scalar scratch tensor.");
        }
        if (host_scratch.has_value()) {
            std::memcpy(host_scratch.value().getMemPtr(), &host_value, sizeof(float));
            device_scratch.value().copyFromAsync(host_scratch.value(), run_stream);
        } else {
            CUDA_CHECK(cudaMemcpyAsync(device_scratch.value().getMemPtr(), &host_value, sizeof(float), cudaMemcpyHostToDevice, run_stream));
        }
        ptr = reinterpret_cast<const float*>(device_scratch.value().getMemPtr());
        is_device_pointer = true;
    }

    void scaleTensorDeviceValueIntoScratch(const TensorScalarBinding& binding, Stream& run_stream) {
        if (!device_scratch.has_value()) {
            throw std::runtime_error("Missing preallocated GEMM device scalar scratch tensor.");
        }
        if (binding.sourceDType != DataType::FP32) {
            throw std::runtime_error("Dynamic GEMM tensor-backed alpha/beta currently require FP32 source dtype.");
        }
        const char* device_ptr = static_cast<const char*>(binding.buffer.getMemPtr());
        const float* source_ptr = reinterpret_cast<const float*>(device_ptr + binding.byteOffset);
        launchScaleFp32DeviceScalar(source_ptr, static_cast<float*>(device_scratch.value().getMemPtr()), host_value, run_stream);
        ptr = reinterpret_cast<const float*>(device_scratch.value().getMemPtr());
        is_device_pointer = true;
    }

    void copyTensorValueToScratch(const Tensor& tensor, Stream& run_stream) {
        if (!device_scratch.has_value()) {
            throw std::runtime_error("Missing preallocated GEMM device scalar scratch tensor.");
        }
        device_scratch.value().copyFromAsync(tensor, run_stream);
        ptr = reinterpret_cast<const float*>(device_scratch.value().getMemPtr());
        is_device_pointer = true;
    }

    void scaleTensorValueIntoScratch(const Tensor& tensor, Stream& run_stream) {
        if (!device_scratch.has_value()) {
            throw std::runtime_error("Missing preallocated GEMM device scalar scratch tensor.");
        }
        if (tensor.getDataType() == DataType::FP32) {
            launchScaleFp32DeviceScalar(static_cast<const float*>(tensor.getMemPtr()),
                                        static_cast<float*>(device_scratch.value().getMemPtr()),
                                        host_value,
                                        run_stream);
        } else {
            device_scratch.value().copyFromAsync(tensor, run_stream);
            launchScaleFp32DeviceScalar(static_cast<const float*>(device_scratch.value().getMemPtr()),
                                        static_cast<float*>(device_scratch.value().getMemPtr()),
                                        host_value,
                                        run_stream);
        }
        ptr = reinterpret_cast<const float*>(device_scratch.value().getMemPtr());
        is_device_pointer = true;
    }
};

struct ResolvedMatmulScales {
    ResolvedMatmulScale alpha;
    ResolvedMatmulScale beta;
    CublasScalarPointerMode pointer_mode = CublasScalarPointerMode::Host;
};

static const float* getTensorRuntimeScalarDevicePtr(const TensorScalarBinding& binding) {
    if (binding.sourceDType != DataType::FP32) {
        throw std::runtime_error("Dynamic GEMM tensor-backed alpha/beta currently require FP32 source dtype.");
    }
    const char* device_ptr = static_cast<const char*>(binding.buffer.getMemPtr());
    return reinterpret_cast<const float*>(device_ptr + binding.byteOffset);
}

static bool tensorResolvesToSingleElement(const Tensor& tensor) {
    uint64_t numel = 1;
    for (uint64_t d : tensor.getDimensions()) {
        numel *= d;
    }
    return numel == 1;
}

static ResolvedMatmulScale resolveMatmulRuntimeScale(const std::optional<RuntimeInputValue>& bound_input,
                                                     const std::optional<std::string>& runtime_name,
                                                     double base_scale,
                                                     const std::unordered_map<std::string, float>& runtime_scalars,
                                                     const std::optional<Tensor>& device_scratch,
                                                     const std::optional<Tensor>& host_scratch,
                                                     Stream& run_stream) {
    ResolvedMatmulScale resolved(device_scratch, host_scratch);
    resolved.host_value = static_cast<float>(base_scale);
    resolved.ptr = &resolved.host_value;

    bool used_runtime_override = false;
    if (runtime_name.has_value()) {
        auto it = runtime_scalars.find(*runtime_name);
        if (it != runtime_scalars.end()) {
            resolved.host_value *= it->second;
            used_runtime_override = true;
        }
    }
    if (!bound_input.has_value()) {
        return resolved;
    }

    const RuntimeInputValue& value = bound_input.value();
    if (std::holds_alternative<float>(value)) {
        if (!used_runtime_override) {
            resolved.host_value *= std::get<float>(value);
        }
        return resolved;
    }
    if (std::holds_alternative<Tensor>(value)) {
        const Tensor& tensor = std::get<Tensor>(value);
        if (!tensorResolvesToSingleElement(tensor)) {
            throw std::runtime_error("Dynamic GEMM alpha/beta expression must resolve to a single element.");
        }
        if (tensor.getDataType() == DataType::FP32 && resolved.host_value == 1.0f) {
            resolved.setDevicePointer(static_cast<const float*>(tensor.getMemPtr()));
            return resolved;
        }
        if (resolved.host_value == 1.0f) {
            resolved.copyTensorValueToScratch(tensor, run_stream);
            return resolved;
        }
        resolved.scaleTensorValueIntoScratch(tensor, run_stream);
        return resolved;
    }
    if (std::holds_alternative<TensorScalarBinding>(value)) {
        const TensorScalarBinding& binding = std::get<TensorScalarBinding>(value);
        if (resolved.host_value == 1.0f) {
            resolved.setDevicePointer(getTensorRuntimeScalarDevicePtr(binding));
            return resolved;
        }
        resolved.scaleTensorDeviceValueIntoScratch(binding, run_stream);
        return resolved;
    }
    throw std::runtime_error(
        "Dynamic GEMM scale currently requires fp32 runtime scalar, tensor-backed runtime scalar, or single-element tensor bindings.");
}

static ResolvedMatmulScales resolveMatmulRuntimeScales(const std::optional<RuntimeInputValue>& alpha_input,
                                                       const std::optional<RuntimeInputValue>& beta_input,
                                                       const std::optional<std::string>& alpha_runtime_name,
                                                       const std::optional<std::string>& beta_runtime_name,
                                                       double alpha_base_scale,
                                                       double beta_base_scale,
                                                       const std::unordered_map<std::string, float>& runtime_scalars,
                                                       const std::optional<Tensor>& alpha_device_scratch,
                                                       const std::optional<Tensor>& beta_device_scratch,
                                                       const std::optional<Tensor>& alpha_host_scratch,
                                                       const std::optional<Tensor>& beta_host_scratch,
                                                       Stream& run_stream) {
    ResolvedMatmulScales resolved;
    resolved.alpha = resolveMatmulRuntimeScale(
        alpha_input, alpha_runtime_name, alpha_base_scale, runtime_scalars, alpha_device_scratch, alpha_host_scratch, run_stream);
    resolved.beta = resolveMatmulRuntimeScale(
        beta_input, beta_runtime_name, beta_base_scale, runtime_scalars, beta_device_scratch, beta_host_scratch, run_stream);
    resolved.alpha.refreshHostPointer();
    resolved.beta.refreshHostPointer();

    if (resolved.alpha.is_device_pointer || resolved.beta.is_device_pointer) {
        resolved.pointer_mode = CublasScalarPointerMode::Device;
        if (!resolved.alpha.is_device_pointer) {
            resolved.alpha.copyHostValueToDevice(run_stream);
        }
        if (!resolved.beta.is_device_pointer) {
            resolved.beta.copyHostValueToDevice(run_stream);
        }
    }

    return resolved;
}

static CublasMatrixMultiply::EpilogueFusion toCublasEpilogueFusion(MatmulEpilogue epilogue);
static CublasMatrixMultiply::BackwardEpilogueFusion toCublasBackwardEpilogueFusion(MatmulBackwardEpilogue epilogue);

void StampedMatmul::run() { runOn(stream); }

void StampedMatmul::runOn(Stream& run_stream) const { runOn(run_stream, {}); }

void StampedMatmul::runOn(Stream& run_stream, const std::unordered_map<std::string, float>& runtime_scalars) const {
    if (lhs.getDimensions().size() != 2 || rhs.getDimensions().size() != 2 || output.getDimensions().size() != 2) {
        throw std::runtime_error("StampedMatmul currently only supports rank-2 tensors.");
    }

    const auto lhs_dims = lhs.getDimensions();
    const auto rhs_dims = rhs.getDimensions();
    const int32_t a_rows = static_cast<int32_t>(lhs_dims[0]);
    const int32_t a_cols = static_cast<int32_t>(lhs_dims[1]);
    const int32_t b_rows = static_cast<int32_t>(rhs_dims[0]);
    const int32_t b_cols = static_cast<int32_t>(rhs_dims[1]);

    if (compiled_matmul->op == ExprOp::MATMUL) {
        const CublasMatrixMultiply::MatmulDataTypes dataTypes{lhs.getDescriptor().getDataType(),
                                                              rhs.getDescriptor().getDataType(),
                                                              output.getDescriptor().getDataType(),
                                                              output.getDescriptor().getDataType(),
                                                              compiled_matmul->compute_dtype};
        if (compiled_matmul->backward_epilogue != MatmulBackwardEpilogue::Default) {
            if (compiled_matmul->epilogue != MatmulEpilogue::Default) {
                throw std::runtime_error("Stamped MATMUL cannot combine forward and backward cuBLASLt epilogues in one stage.");
            }
            if (!epilogue_aux.has_value()) {
                throw std::runtime_error("Stamped MATMUL backward epilogue requires epilogue_aux.");
            }
            if (compiled_matmul->transpose_lhs || compiled_matmul->transpose_rhs) {
                throw std::runtime_error(
                    "cuBLASLt MATMUL backward epilogue fusion currently supports only non-transposed row-major stages.");
            }
            const float alphaOne = 1.0f;
            const float betaZero = 0.0f;
            CublasMatrixMultiply::instance().gemmWithBackwardEpilogueUsingHeuristicKernelChoice(
                lhs,
                rhs,
                std::nullopt,
                epilogue_aux.value(),
                output,
                bgrad_output,
                a_rows,
                a_cols,
                b_rows,
                b_cols,
                compiled_matmul->transpose_lhs,
                compiled_matmul->transpose_rhs,
                &alphaOne,
                &betaZero,
                dataTypes,
                toCublasBackwardEpilogueFusion(compiled_matmul->backward_epilogue),
                run_stream,
                CublasScalarPointerMode::Host,
                workspace,
                built_matmul->epilogue_algorithm);
            return;
        }

        if (compiled_matmul->epilogue == MatmulEpilogue::Default) {
            CublasMatrixMultiply::instance().multiply(lhs,
                                                      rhs,
                                                      output,
                                                      workspace,
                                                      a_rows,
                                                      a_cols,
                                                      b_rows,
                                                      b_cols,
                                                      compiled_matmul->transpose_lhs,
                                                      compiled_matmul->transpose_rhs,
                                                      false,
                                                      false,
                                                      dataTypes,
                                                      run_stream);
            return;
        }

        if (compiled_matmul->transpose_lhs || compiled_matmul->transpose_rhs) {
            throw std::runtime_error("cuBLASLt MATMUL activation epilogue fusion currently supports only non-transposed row-major stages.");
        }
        const float alphaOne = 1.0f;
        const float betaZero = 0.0f;
        CublasMatrixMultiply::instance().gemmWithEpilogueUsingHeuristicKernelChoice(lhs,
                                                                                    rhs,
                                                                                    std::nullopt,
                                                                                    output,
                                                                                    a_rows,
                                                                                    a_cols,
                                                                                    b_rows,
                                                                                    b_cols,
                                                                                    compiled_matmul->transpose_lhs,
                                                                                    compiled_matmul->transpose_rhs,
                                                                                    &alphaOne,
                                                                                    &betaZero,
                                                                                    dataTypes,
                                                                                    toCublasEpilogueFusion(compiled_matmul->epilogue),
                                                                                    false,
                                                                                    run_stream,
                                                                                    CublasScalarPointerMode::Host,
                                                                                    workspace,
                                                                                    built_matmul->epilogue_algorithm);
        return;
    }

    if (!addend.has_value()) {
        throw std::runtime_error("Stamped GEMM requires an addend tensor.");
    }
    const bool use_bias_epilogue = addend.value().getDimensions().size() == 1;
    if (!use_bias_epilogue && addend.value().getDimensions().size() != 2) {
        throw std::runtime_error("Stamped GEMM currently supports rank-2 addend tensors or rank-1 bias epilogue vectors.");
    }

    ResolvedMatmulScales resolved_scales = resolveMatmulRuntimeScales(alpha_input,
                                                                      beta_input,
                                                                      alpha_runtime_name,
                                                                      beta_runtime_name,
                                                                      compiled_matmul->alpha,
                                                                      compiled_matmul->beta,
                                                                      runtime_scalars,
                                                                      alpha_device_scratch,
                                                                      beta_device_scratch,
                                                                      alpha_host_scratch,
                                                                      beta_host_scratch,
                                                                      run_stream);

    const CublasMatrixMultiply::MatmulDataTypes dataTypes{
        lhs.getDescriptor().getDataType(),
        rhs.getDescriptor().getDataType(),
        use_bias_epilogue ? output.getDescriptor().getDataType() : addend.value().getDescriptor().getDataType(),
        output.getDescriptor().getDataType(),
        compiled_matmul->compute_dtype};
    const bool use_backward_epilogue = compiled_matmul->backward_epilogue != MatmulBackwardEpilogue::Default;
    if (use_backward_epilogue) {
        if (compiled_matmul->epilogue != MatmulEpilogue::Default) {
            throw std::runtime_error("Stamped GEMM cannot combine forward and backward cuBLASLt epilogues in one stage.");
        }
        if (use_bias_epilogue) {
            throw std::runtime_error(
                "Stamped GEMM backward epilogue requires a rank-2 addend or no addend; rank-1 bias addends are forward epilogues.");
        }
        if (compiled_matmul->transpose_aux) {
            throw std::runtime_error("GEMM cuBLASLt backward epilogue fusion does not support transpose_aux.");
        }
        if (compiled_matmul->transpose_lhs || compiled_matmul->transpose_rhs) {
            throw std::runtime_error("GEMM cuBLASLt backward epilogue fusion currently supports only non-transposed row-major stages.");
        }
        CublasMatrixMultiply::instance().gemmWithBackwardEpilogueUsingHeuristicKernelChoice(
            lhs,
            rhs,
            addend,
            epilogue_aux.value(),
            output,
            bgrad_output,
            a_rows,
            a_cols,
            b_rows,
            b_cols,
            compiled_matmul->transpose_lhs,
            compiled_matmul->transpose_rhs,
            resolved_scales.alpha.ptr,
            resolved_scales.beta.ptr,
            dataTypes,
            toCublasBackwardEpilogueFusion(compiled_matmul->backward_epilogue),
            run_stream,
            resolved_scales.pointer_mode,
            workspace,
            built_matmul->epilogue_algorithm);
        return;
    }

    const bool use_cublaslt_epilogue_wrapper =
        use_bias_epilogue || compiled_matmul->epilogue != MatmulEpilogue::Default || use_backward_epilogue;
    if (use_cublaslt_epilogue_wrapper) {
        if (compiled_matmul->transpose_aux) {
            throw std::runtime_error("GEMM cuBLASLt epilogue fusion does not support transpose_aux.");
        }
        if (compiled_matmul->transpose_lhs || compiled_matmul->transpose_rhs) {
            throw std::runtime_error("GEMM cuBLASLt epilogue fusion currently supports only non-transposed row-major stages.");
        }
        if (use_bias_epilogue) {
            if (addend.value().getDescriptor().getDataType() != output.getDescriptor().getDataType()) {
                throw std::runtime_error("GEMM bias epilogue requires the bias dtype to match the output dtype.");
            }
            if (resolved_scales.beta.is_device_pointer || resolved_scales.beta.host_value != 1.0f) {
                throw std::runtime_error("GEMM bias epilogue currently requires an unscaled +bias addend.");
            }
        }
        CublasMatrixMultiply::instance().gemmWithEpilogueUsingHeuristicKernelChoice(lhs,
                                                                                    rhs,
                                                                                    addend.value(),
                                                                                    output,
                                                                                    a_rows,
                                                                                    a_cols,
                                                                                    b_rows,
                                                                                    b_cols,
                                                                                    compiled_matmul->transpose_lhs,
                                                                                    compiled_matmul->transpose_rhs,
                                                                                    resolved_scales.alpha.ptr,
                                                                                    resolved_scales.beta.ptr,
                                                                                    dataTypes,
                                                                                    toCublasEpilogueFusion(compiled_matmul->epilogue),
                                                                                    use_bias_epilogue,
                                                                                    run_stream,
                                                                                    resolved_scales.pointer_mode,
                                                                                    workspace,
                                                                                    built_matmul->epilogue_algorithm);
        return;
    }

    CublasMatrixMultiply::instance().gemm(lhs,
                                          rhs,
                                          addend.value(),
                                          output,
                                          workspace,
                                          a_rows,
                                          a_cols,
                                          b_rows,
                                          b_cols,
                                          compiled_matmul->transpose_lhs,
                                          compiled_matmul->transpose_rhs,
                                          compiled_matmul->transpose_aux,
                                          resolved_scales.alpha.ptr,
                                          resolved_scales.beta.ptr,
                                          dataTypes,
                                          run_stream,
                                          resolved_scales.pointer_mode);
}


StampedScanMinMaxBackward::StampedScanMinMaxBackward(std::shared_ptr<CompiledScanMinMaxBackward> compiled,
                                                     std::shared_ptr<StampedScan> arg_scan,
                                                     std::shared_ptr<BuiltFlatScatterAdd> scatter_add,
                                                     const Tensor& input,
                                                     const Tensor& grad_output,
                                                     const Tensor& output,
                                                     const Tensor& indices,
                                                     const Stream& stream)
    : compiled_scan_minmax_backward(std::move(compiled)),
      arg_scan(std::move(arg_scan)),
      scatter_add(std::move(scatter_add)),
      input(input),
      grad_output(grad_output),
      output(output),
      indices(indices),
      stream(stream) {
    if (!compiled_scan_minmax_backward || !this->arg_scan || !this->scatter_add) {
        throw std::runtime_error("StampedScanMinMaxBackward requires compiled, arg-scan, and scatter-add plans.");
    }
    if (input.getDataType() != compiled_scan_minmax_backward->input_dtype ||
        grad_output.getDataType() != compiled_scan_minmax_backward->grad_output_dtype ||
        output.getDataType() != compiled_scan_minmax_backward->output_dtype) {
        throw std::runtime_error("StampedScanMinMaxBackward tensor dtypes do not match the compiled descriptor.");
    }
    if (grad_output.getDimensions() != input.getDimensions() || output.getDimensions() != input.getDimensions() ||
        indices.getDimensions() != input.getDimensions()) {
        throw std::runtime_error("StampedScanMinMaxBackward expects input, grad, output, and indices with matching shapes.");
    }
    if (indices.getDataType() != DataType::UINT32) {
        throw std::runtime_error("StampedScanMinMaxBackward arg-scan indices must be UINT32.");
    }
}

void StampedScanMinMaxBackward::run() { runOn(stream); }

void StampedScanMinMaxBackward::runOn(Stream& run_stream) {
    arg_scan->runOn(run_stream);
    runFlatScatterAdd(scatter_add, grad_output, indices, output, run_stream);
}

StampedReduceMinMaxBackward::StampedReduceMinMaxBackward(std::shared_ptr<BuiltReduction> built,
                                                         const Tensor& input,
                                                         const Tensor& grad_output,
                                                         const Tensor& output,
                                                         const Tensor& indices,
                                                         const Tensor& reduction_value_output,
                                                         const Stream& stream,
                                                         std::optional<Tensor> workspace)
    : built_reduction(built),
      input(input),
      grad_output(grad_output),
      output(output),
      indices(indices),
      reduction_value_output(reduction_value_output),
      workspace(workspace),
      stream(stream) {
    if (!built_reduction->key.output_indices) {
        throw std::runtime_error("StampedReduceMinMaxBackward requires a BuiltReduction configured for indices.");
    }
    if (built_reduction->workspace_bytes != 0) {
        THOR_THROW_IF_FALSE(workspace.has_value());
        THOR_THROW_IF_FALSE(workspace.value().getArraySizeInBytes() >= built_reduction->workspace_bytes);
    }
}

void StampedReduceMinMaxBackward::run() { runOn(stream); }

void StampedReduceMinMaxBackward::runOn(Stream& run_stream) {
    // std::cerr << "[REDUCE_MINMAX_BW] input dtype=" << TensorDescriptor::getElementTypeName(input.getDataType())
    //           << " built.input_dtype=" << TensorDescriptor::getElementTypeName(built_reduction->key.input_dtype)
    //           << " built.output_dtype=" << TensorDescriptor::getElementTypeName(built_reduction->key.output_dtype)
    //           << " built.compute_dtype=" << TensorDescriptor::getElementTypeName(built_reduction->key.compute_dtype)
    //           << " indices dtype=" << TensorDescriptor::getElementTypeName(indices.getDataType())
    //           << " reduction_value_output dtype=" << TensorDescriptor::getElementTypeName(reduction_value_output.getDataType())
    //           << std::endl;

    void* workspace_ptr = nullptr;
    if (built_reduction->workspace_bytes > 0) {
        THOR_THROW_IF_FALSE(workspace.has_value());
        workspace_ptr = (void*)workspace.value().getMemPtr();
    }

    CUDNN_CHECK(cudnnReduceTensor(run_stream.getCudnnHandle(),
                                  built_reduction->reduce_desc,
                                  (void*)indices.getMemPtr(),
                                  built_reduction->indices_bytes,
                                  workspace_ptr,
                                  built_reduction->workspace_bytes,
                                  alpha,
                                  built_reduction->a_desc,
                                  input.getMemPtr(),
                                  beta,
                                  built_reduction->c_desc,
                                  (void*)reduction_value_output.getMemPtr()));

    output.memsetAsync(run_stream, 0);

    launchReduceMinMaxBackwardScatter(grad_output.getMemPtr(),
                                      static_cast<const uint32_t*>(indices.getMemPtr()),
                                      (void*)output.getMemPtr(),
                                      input.getDimensions(),
                                      built_reduction->key.reduction_axes,
                                      built_reduction->key.squeeze_axes,
                                      grad_output.getDataType(),
                                      output.getDataType(),
                                      run_stream);
}


static uint64_t conditionalPredicateNumel(const Tensor& predicate) {
    uint64_t numel = 1;
    for (uint64_t d : predicate.getDimensions()) {
        numel *= d;
    }
    return numel;
}

static void validateConditionalPredicateTensor(const Tensor& predicate) {
    if (!predicate.isInitialized()) {
        throw std::runtime_error("Graph-level conditional predicate output tensor is not initialized.");
    }
    if (predicate.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::runtime_error("Graph-level conditional predicate output must live on GPU.");
    }
    if (predicate.getDataType() != DataType::BOOLEAN) {
        throw std::runtime_error("Graph-level conditional predicate output must have BOOLEAN dtype.");
    }
    if (conditionalPredicateNumel(predicate) != 1) {
        throw std::runtime_error("Graph-level conditional predicate output must contain exactly one element.");
    }
}

static std::vector<cudaGraphNode_t> graphLeafNodes(cudaGraph_t graph) {
    size_t node_count = 0;
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &node_count));

    std::vector<cudaGraphNode_t> nodes(node_count);
    if (node_count != 0) {
        CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &node_count));
        nodes.resize(node_count);
    }

    std::vector<cudaGraphNode_t> leaves;
    leaves.reserve(nodes.size());
    for (cudaGraphNode_t node : nodes) {
        size_t dependent_count = 0;
        CUDA_CHECK(cudaGraphNodeGetDependentNodes(node, nullptr, nullptr, &dependent_count));
        if (dependent_count == 0) {
            leaves.push_back(node);
        }
    }
    return leaves;
}

static void capturePlanSequentiallyIntoGraph(const StampedExecutionPlan& plan,
                                             cudaGraph_t graph,
                                             Stream& capture_stream,
                                             const std::vector<cudaGraphNode_t>& dependencies = {}) {
    cudaGraph_t captured_graph = nullptr;
    const cudaGraphNode_t* deps = dependencies.empty() ? nullptr : dependencies.data();
    CUDA_CHECK(cudaStreamBeginCaptureToGraph(capture_stream.getStream(),
                                             graph,
                                             deps,
                                             nullptr,
                                             dependencies.size(),
                                             cudaStreamCaptureModeGlobal));

    try {
        plan.runSequentialOn(capture_stream);
        CUDA_CHECK(cudaStreamEndCapture(capture_stream.getStream(), &captured_graph));
    } catch (...) {
        cudaGraph_t aborted_graph = nullptr;
        cudaError_t end_status = cudaStreamEndCapture(capture_stream.getStream(), &aborted_graph);
        if (end_status == cudaSuccess && aborted_graph != nullptr && aborted_graph != graph) {
            (void)cudaGraphDestroy(aborted_graph);
        } else if (end_status != cudaSuccess) {
            (void)cudaGetLastError();
        }
        throw;
    }

    if (captured_graph != graph) {
        throw std::runtime_error("CUDA graph capture for conditional subplan did not return the target graph.");
    }
}

static void captureConditionalSetterIntoGraph(cudaGraphConditionalHandle conditional_handle,
                                                const Tensor& predicate,
                                                cudaGraph_t graph,
                                                Stream& capture_stream,
                                                const std::vector<cudaGraphNode_t>& dependencies) {
    cudaGraph_t captured_graph = nullptr;
    const cudaGraphNode_t* deps = dependencies.empty() ? nullptr : dependencies.data();
    CUDA_CHECK(cudaStreamBeginCaptureToGraph(capture_stream.getStream(),
                                             graph,
                                             deps,
                                             nullptr,
                                             dependencies.size(),
                                             cudaStreamCaptureModeGlobal));

    try {
        launchSetCudaGraphConditionalFromBool(conditional_handle, predicate, capture_stream);
        CUDA_CHECK(cudaStreamEndCapture(capture_stream.getStream(), &captured_graph));
    } catch (...) {
        cudaGraph_t aborted_graph = nullptr;
        cudaError_t end_status = cudaStreamEndCapture(capture_stream.getStream(), &aborted_graph);
        if (end_status == cudaSuccess && aborted_graph != nullptr && aborted_graph != graph) {
            (void)cudaGraphDestroy(aborted_graph);
        } else if (end_status != cudaSuccess) {
            (void)cudaGetLastError();
        }
        throw;
    }

    if (captured_graph != graph) {
        throw std::runtime_error("CUDA graph capture for conditional setter did not return the target graph.");
    }
}

static CudaGraphExecutable buildConditionalCudaGraph(const StampedExecutionPlan& predicate_plan,
                                                     const StampedExecutionPlan& then_plan,
                                                     const StampedExecutionPlan& else_plan,
                                                     const Stream& stream) {
    Stream capture_stream(stream.getGpuNum());

    // Pre-create common library handles outside capture. Some stage types lazily create
    // cuDNN/cuBLAS handles from the stream; doing that while capture is active would make
    // the conditional graph path fragile.
    (void)capture_stream.getCudnnHandle();
    (void)capture_stream.getCublasHandle();

    Tensor predicate = predicate_plan.output();
    validateConditionalPredicateTensor(predicate);

    cudaGraph_t root_graph = nullptr;
    CUDA_CHECK(cudaGraphCreate(&root_graph, 0));

    try {
        cudaGraphConditionalHandle conditional_handle{};
        CUDA_CHECK(cudaGraphConditionalHandleCreate(&conditional_handle, root_graph, 0, 0));

        capturePlanSequentiallyIntoGraph(predicate_plan, root_graph, capture_stream);

        std::vector<cudaGraphNode_t> predicate_leaves = graphLeafNodes(root_graph);
        if (predicate_leaves.empty()) {
            throw std::runtime_error("Graph-level conditional predicate graph produced no CUDA graph nodes.");
        }
        captureConditionalSetterIntoGraph(conditional_handle, predicate, root_graph, capture_stream, predicate_leaves);

        std::vector<cudaGraphNode_t> conditional_dependencies = graphLeafNodes(root_graph);
        if (conditional_dependencies.empty()) {
            throw std::runtime_error("Graph-level conditional setter graph produced no CUDA graph leaf nodes.");
        }

        cudaGraphNodeParams conditional_params{};
        conditional_params.type = cudaGraphNodeTypeConditional;
        conditional_params.conditional.handle = conditional_handle;
        conditional_params.conditional.type = cudaGraphCondTypeIf;
        conditional_params.conditional.size = 2;

        cudaGraphNode_t conditional_node = nullptr;
        CUDA_CHECK(cudaGraphAddNode(&conditional_node,
                                    root_graph,
                                    conditional_dependencies.data(),
                                    nullptr,
                                    conditional_dependencies.size(),
                                    &conditional_params));

        if (conditional_params.conditional.phGraph_out == nullptr) {
            throw std::runtime_error("CUDA did not return body graphs for graph-level conditional node.");
        }

        capturePlanSequentiallyIntoGraph(then_plan, conditional_params.conditional.phGraph_out[0], capture_stream);
        capturePlanSequentiallyIntoGraph(else_plan, conditional_params.conditional.phGraph_out[1], capture_stream);

        CudaGraph graph(root_graph, stream.getGpuNum(), false);
        root_graph = nullptr;
        CudaGraphExecutable executable = graph.instantiate();
        executable.upload(capture_stream);
        return executable;
    } catch (...) {
        if (root_graph != nullptr) {
            (void)cudaGraphDestroy(root_graph);
        }
        throw;
    }
}

StampedConditional::StampedConditional(std::shared_ptr<StampedExecutionPlan> predicate_plan,
                                       std::shared_ptr<StampedExecutionPlan> then_plan,
                                       std::shared_ptr<StampedExecutionPlan> else_plan,
                                       std::vector<std::string> output_names,
                                       const Stream& stream)
    : predicate_plan(std::move(predicate_plan)),
      then_plan(std::move(then_plan)),
      else_plan(std::move(else_plan)),
      output_names(std::move(output_names)),
      stream(stream) {
    if (!this->predicate_plan || !this->then_plan || !this->else_plan) {
        throw std::runtime_error("StampedConditional requires predicate, then, and else plans.");
    }
    if (this->output_names.empty()) {
        throw std::runtime_error("StampedConditional requires at least one output name.");
    }
    if (requiresRuntimeScalars()) {
        throw std::runtime_error(
            "Graph-level conditional Outputs cannot currently use host runtime scalar overrides because the conditional is "
            "captured as a CUDA graph. Use tensor runtime scalars or constant scalars instead.");
    }

    conditional_graph = buildConditionalCudaGraph(*this->predicate_plan, *this->then_plan, *this->else_plan, this->stream);
}

uint32_t StampedConditional::gpuNum() const {
    Tensor predicate = predicate_plan->output();
    if (predicate.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU) {
        return static_cast<uint32_t>(predicate.getPlacement().getDeviceNum());
    }
    for (const std::string& name : output_names) {
        Tensor out = then_plan->output(name);
        if (out.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU) {
            return static_cast<uint32_t>(out.getPlacement().getDeviceNum());
        }
    }
    return 0;
}

bool StampedConditional::requiresRuntimeScalars() const { return !runtimeScalarNames().empty(); }

std::unordered_set<std::string> StampedConditional::runtimeScalarNames() const {
    std::unordered_set<std::string> names = predicate_plan->runtimeScalarNames();
    std::unordered_set<std::string> then_names = then_plan->runtimeScalarNames();
    std::unordered_set<std::string> else_names = else_plan->runtimeScalarNames();
    names.insert(then_names.begin(), then_names.end());
    names.insert(else_names.begin(), else_names.end());
    return names;
}

void StampedConditional::run() { run({}); }

void StampedConditional::run(const std::unordered_map<std::string, float>& runtime_scalars) { runOn(stream, runtime_scalars); }

void StampedConditional::runOn(Stream& run_stream) const { runOn(run_stream, {}); }

void StampedConditional::runOn(Stream& run_stream, const std::unordered_map<std::string, float>& runtime_scalars) const {
    if (!runtime_scalars.empty()) {
        throw std::runtime_error("Graph-level conditional Outputs do not support host runtime scalar overrides.");
    }
    conditional_graph.launch(run_stream);
}

static std::unordered_set<std::string> runtimeScalarNamesForStage(const StampedExecutionStage& stage) {
    std::unordered_set<std::string> stage_names;
    if (stage.kind == StampedExecutionStage::Kind::FusedKernel && stage.kernel != nullptr && stage.kernel->requiresRuntimeScalars()) {
        stage_names = stage.kernel->runtimeScalarNames();
    } else if (stage.kind == StampedExecutionStage::Kind::CudaKernel && stage.cuda_kernel != nullptr &&
               stage.cuda_kernel->requiresRuntimeScalars()) {
        stage_names = stage.cuda_kernel->runtimeScalarNames();
    } else if (stage.kind == StampedExecutionStage::Kind::Matmul && stage.matmul != nullptr) {
        if (stage.matmul->alphaRuntimeName().has_value()) {
            stage_names.insert(*stage.matmul->alphaRuntimeName());
        }
        if (stage.matmul->betaRuntimeName().has_value()) {
            stage_names.insert(*stage.matmul->betaRuntimeName());
        }
    } else if (stage.kind == StampedExecutionStage::Kind::Conditional && stage.conditional != nullptr &&
               stage.conditional->requiresRuntimeScalars()) {
        stage_names = stage.conditional->runtimeScalarNames();
    }
    return stage_names;
}

bool StampedExecutionPlan::requiresRuntimeScalars() const { return !runtimeScalarNames().empty(); }

std::unordered_set<std::string> StampedExecutionPlan::runtimeScalarNames() const {
    std::unordered_set<std::string> names;
    for (const StampedExecutionStage& stage : steps) {
        std::unordered_set<std::string> stage_names = runtimeScalarNamesForStage(stage);
        names.insert(stage_names.begin(), stage_names.end());
    }
    return names;
}

void StampedExecutionPlan::runSequentialOn(Stream& run_stream) const { runSequentialOn(run_stream, {}); }

void StampedExecutionPlan::runSequentialOn(Stream& run_stream, const std::unordered_map<std::string, float>& runtime_scalars) const {
    std::unordered_set<std::string> consumed_runtime_scalar_names;

    for (const StampedExecutionStage& stage : steps) {
        std::unordered_map<std::string, float> stage_runtime_scalars;
        std::unordered_set<std::string> needed_names = runtimeScalarNamesForStage(stage);

        if (!needed_names.empty()) {
            stage_runtime_scalars.reserve(needed_names.size());
            for (const std::string& name : needed_names) {
                auto it = runtime_scalars.find(name);
                if (it == runtime_scalars.end()) {
                    throw std::runtime_error("Missing value for runtime scalar: " + name +
                                             "  - if it was meant to be constant, use a constant scalar instead.");
                }
                stage_runtime_scalars.emplace(name, it->second);
                consumed_runtime_scalar_names.insert(name);
            }
        }

        if (stage_runtime_scalars.empty()) {
            stage.runOn(run_stream);
        } else {
            stage.runOn(run_stream, stage_runtime_scalars);
        }
    }

    for (const auto& [name, _] : runtime_scalars) {
        if (!consumed_runtime_scalar_names.contains(name)) {
            throw std::runtime_error("Unexpected runtime scalar override for stamped execution plan: " + name);
        }
    }
}

void StampedExecutionPlan::run() { run({}); }

void StampedExecutionPlan::run(const std::unordered_map<std::string, float>& runtime_scalars) {
    if (steps.empty()) {
        return;
    }

    using StreamEvent = std::decay_t<decltype(std::declval<Stream&>().putEvent())>;

    std::vector<std::optional<StreamEvent>> completion_events(steps.size());

    std::vector<Stream> launch_streams;
    launch_streams.reserve(steps.size());

    std::vector<Stream> helper_streams_used;
    helper_streams_used.reserve(steps.size());

    std::unordered_set<std::string> consumed_runtime_scalar_names;

    auto rememberHelperStream = [&](Stream& helper_stream) {
        if (std::find(helper_streams_used.begin(), helper_streams_used.end(), helper_stream) == helper_streams_used.end()) {
            helper_streams_used.push_back(helper_stream);
        }
    };

    StreamEvent user_stream_ready;
    if (steps.size() > 1)
        user_stream_ready = stream.putEvent();

    for (uint32_t stage_idx = 0; stage_idx < steps.size(); ++stage_idx) {
        const bool use_helper_stream = (stage_idx != 0);
        const StampedExecutionStage& stage = steps[stage_idx];

        Stream& launch_stream_ref = use_helper_stream ? Expression::getNextHelperStream(stage.gpu_num) : stream;

        if (use_helper_stream) {
            rememberHelperStream(launch_stream_ref);
            launch_stream_ref.waitEvent(user_stream_ready);
        }

        for (uint32_t dep_stage_idx : stage.dependency_stage_indices) {
            if (dep_stage_idx >= stage_idx) {
                throw std::runtime_error("StampedExecutionPlan::run requires dependency_stage_indices to be topologically ordered.");
            }

            if (!completion_events[dep_stage_idx].has_value()) {
                throw std::runtime_error("StampedExecutionPlan::run missing completion event for dependency stage.");
            }

            if (!(launch_stream_ref == launch_streams[dep_stage_idx])) {
                launch_stream_ref.waitEvent(completion_events[dep_stage_idx].value());
            }
        }

        std::unordered_map<std::string, float> stage_runtime_scalars;

        if (!runtime_scalars.empty()) {
            std::unordered_set<std::string> needed_names;

            if (stage.kind == StampedExecutionStage::Kind::FusedKernel && stage.kernel != nullptr &&
                stage.kernel->requiresRuntimeScalars()) {
                needed_names = stage.kernel->runtimeScalarNames();
            } else if (stage.kind == StampedExecutionStage::Kind::CudaKernel && stage.cuda_kernel != nullptr &&
                       stage.cuda_kernel->requiresRuntimeScalars()) {
                needed_names = stage.cuda_kernel->runtimeScalarNames();
            } else if (stage.kind == StampedExecutionStage::Kind::Matmul && stage.matmul != nullptr) {
                if (stage.matmul->alphaRuntimeName().has_value()) {
                    needed_names.insert(*stage.matmul->alphaRuntimeName());
                }
                if (stage.matmul->betaRuntimeName().has_value()) {
                    needed_names.insert(*stage.matmul->betaRuntimeName());
                }
            } else if (stage.kind == StampedExecutionStage::Kind::Conditional && stage.conditional != nullptr &&
                       stage.conditional->requiresRuntimeScalars()) {
                needed_names = stage.conditional->runtimeScalarNames();
            }

            if (!needed_names.empty()) {
                stage_runtime_scalars.reserve(needed_names.size());

                for (const std::string& name : needed_names) {
                    auto it = runtime_scalars.find(name);
                    if (it == runtime_scalars.end()) {
                        throw std::runtime_error("Missing value for runtime scalar: " + name +
                                                 "  - if it was meant to be constant, use a constant scalar instead.");
                    }
                    stage_runtime_scalars.emplace(name, it->second);
                    consumed_runtime_scalar_names.insert(name);
                }
            }
        }

        if (stage_runtime_scalars.empty())
            stage.runOn(launch_stream_ref);
        else
            stage.runOn(launch_stream_ref, stage_runtime_scalars);

        completion_events[stage_idx] = launch_stream_ref.putEvent();
        launch_streams.push_back(launch_stream_ref);
    }

    for (const auto& [name, _] : runtime_scalars) {
        if (!consumed_runtime_scalar_names.contains(name)) {
            throw std::runtime_error("Unexpected runtime scalar override for stamped execution plan: " + name);
        }
    }

    for (Stream& helper_stream : helper_streams_used) {
        if (!(helper_stream == stream)) {
            stream.waitEvent(helper_stream.putEvent());
        }
    }
}

// static unordered_map<ReductionCacheKey, shared_ptr<BuiltReduction>> builtReductionCache;
static LruCacheThreadSafe<ReductionCacheKey, shared_ptr<BuiltReduction>> builtReductionCache(10'000);

static shared_ptr<BuiltReduction> cacheLookup(const ReductionCacheKey& key) {
    optional<shared_ptr<BuiltReduction>> hit = builtReductionCache.get(key);
    if (hit.has_value()) {
        return hit.value();
    }
    return nullptr;
}

static LruCacheThreadSafe<SoftmaxCacheKey, shared_ptr<BuiltSoftmax>> builtSoftmaxCache(10'000);

static shared_ptr<BuiltSoftmax> cacheLookup(const SoftmaxCacheKey& key) {
    optional<shared_ptr<BuiltSoftmax>> hit = builtSoftmaxCache.get(key);
    if (hit.has_value()) {
        return hit.value();
    }
    return nullptr;
}

static LruCacheThreadSafe<MatmulCacheKey, shared_ptr<BuiltMatmul>> builtMatmulCache(10'000);

static shared_ptr<BuiltMatmul> cacheLookup(const MatmulCacheKey& key) {
    optional<shared_ptr<BuiltMatmul>> hit = builtMatmulCache.get(key);
    if (hit.has_value()) {
        return hit.value();
    }
    return nullptr;
}

static cudnnDataType_t toCudnnDataType(DataType dtype) {
    switch (dtype) {
        case DataType::FP32:
            return CUDNN_DATA_FLOAT;

        case DataType::FP16:
            return CUDNN_DATA_HALF;

        case DataType::BF16:
            return CUDNN_DATA_BFLOAT16;

        case DataType::FP8_E4M3:
            return CUDNN_DATA_FP8_E4M3;

        case DataType::FP8_E5M2:
            return CUDNN_DATA_FP8_E5M2;

        default:
            throw std::runtime_error("toCudnnDataType: unsupported DataType value " +
                                     std::to_string(static_cast<int>(dtype)));
    }
}

static cudnnReduceTensorOp_t toCudnnReduceTensorOp(ExprOp op) {
    switch (op) {
        case ExprOp::REDUCE_SUM:
            return CUDNN_REDUCE_TENSOR_ADD;
        case ExprOp::REDUCE_PROD:
            return CUDNN_REDUCE_TENSOR_MUL;
        case ExprOp::REDUCE_MIN:
        case ExprOp::REDUCE_ARGMIN:
            return CUDNN_REDUCE_TENSOR_MIN;
        case ExprOp::REDUCE_MAX:
        case ExprOp::REDUCE_ARGMAX:
            return CUDNN_REDUCE_TENSOR_MAX;
        case ExprOp::REDUCE_AVG:
            return CUDNN_REDUCE_TENSOR_AVG;
        case ExprOp::REDUCE_NORM1:
            return CUDNN_REDUCE_TENSOR_NORM1;
        case ExprOp::REDUCE_NORM2:
            return CUDNN_REDUCE_TENSOR_NORM2;
        default:
            throw std::runtime_error("ExprOp is not a supported cuDNN reduction op.");
    }
}

std::vector<uint64_t> StampedEquation::computeReductionOutputDims(const std::vector<uint64_t>& input_dims,
                                                                  const std::vector<uint64_t>& reduction_axes,
                                                                  const std::vector<uint64_t>& squeeze_axes) {
    std::vector<uint64_t> output_dims = input_dims;

    for (uint64_t axis : reduction_axes) {
        if (axis >= output_dims.size())
            throw std::runtime_error("Reduction axis out of range.");
        output_dims[axis] = 1;
    }

    if (squeeze_axes.empty()) {
        return output_dims;
    }

    std::vector<uint64_t> squeezed;
    squeezed.reserve(output_dims.size());

    if (squeeze_axes.size() == 1 && squeeze_axes[0] == UINT64_MAX) {
        for (uint64_t d : output_dims) {
            if (d != 1)
                squeezed.push_back(d);
        }
    } else {
        uint64_t nextDimToSqueeze = squeeze_axes[0];
        uint64_t nextIndexInSqueezedDims = 1;

        for (uint64_t i = 0; i < output_dims.size(); ++i) {
            if (i == nextDimToSqueeze) {
                if (output_dims[i] != 1) {
                    throw runtime_error("Trying to squeeze axis " + to_string(nextDimToSqueeze) + " but it has size " +
                                        to_string(output_dims[i]) + ", can only squeeze dimensions of size 1.");
                }

                if (nextIndexInSqueezedDims < squeeze_axes.size()) {
                    nextDimToSqueeze = squeeze_axes[nextIndexInSqueezedDims];
                    nextIndexInSqueezedDims += 1;
                } else {
                    nextDimToSqueeze = UINT64_MAX;
                }
            } else {
                squeezed.push_back(output_dims[i]);
            }
        }

        if (nextIndexInSqueezedDims != squeeze_axes.size()) {
            throw runtime_error("Axis " + to_string(nextDimToSqueeze) + " was passed as a dimension to squeeze, but tensor has only " +
                                to_string(output_dims.size()) + " dimensions.");
        }
    }

    if (squeezed.empty())
        squeezed.push_back(1);

    return squeezed;
}

static cudnnTensorDescriptor_t createCudnnTensorDescriptor(std::vector<uint64_t> dims, DataType dtype) {
    while (dims.size() < 4)
        dims.push_back(1);
    if (dims.size() > 8)
        throw std::runtime_error("cuDNN reduction only supports rank <= 8.");

    std::vector<int> cudnn_dims(dims.begin(), dims.end());
    std::vector<int> strides(cudnn_dims.size());
    strides.back() = 1;
    for (int i = static_cast<int>(cudnn_dims.size()) - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * cudnn_dims[i + 1];

    cudnnTensorDescriptor_t desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
    CUDNN_CHECK(
        cudnnSetTensorNdDescriptor(desc, toCudnnDataType(dtype), static_cast<int>(cudnn_dims.size()), cudnn_dims.data(), strides.data()));
    return desc;
}

static std::vector<uint64_t> paddedCudnnReductionDims(std::vector<uint64_t> dims) {
    while (dims.size() < 4)
        dims.push_back(1);
    if (dims.size() > 8)
        throw std::runtime_error("cuDNN reduction only supports rank <= 8.");
    return dims;
}

static bool hasCudnnReductionDimension(const std::vector<uint64_t>& input_dims, const std::vector<uint64_t>& output_dims) {
    const std::vector<uint64_t> padded_input = paddedCudnnReductionDims(input_dims);
    const std::vector<uint64_t> padded_output = paddedCudnnReductionDims(output_dims);
    if (padded_input.size() != padded_output.size())
        return true;
    for (size_t i = 0; i < padded_input.size(); ++i) {
        if (padded_input[i] != padded_output[i])
            return true;
    }
    return false;
}

static bool singletonReductionCanReuseInputValue(ExprOp op) {
    switch (op) {
        case ExprOp::REDUCE_SUM:
        case ExprOp::REDUCE_PROD:
        case ExprOp::REDUCE_MIN:
        case ExprOp::REDUCE_MAX:
        case ExprOp::REDUCE_AVG:
            return true;
        default:
            return false;
    }
}

static fe::DataType_t toFrontendDataType(DataType dtype) {
    switch (dtype) {
        case DataType::FP32:
            return fe::DataType_t::FLOAT;
        case DataType::FP16:
            return fe::DataType_t::HALF;
        case DataType::BF16:
            return fe::DataType_t::BFLOAT16;
        case DataType::FP8_E4M3:
            return fe::DataType_t::FP8_E4M3;
        case DataType::FP8_E5M2:
            return fe::DataType_t::FP8_E5M2;
        case DataType::INT32:
            return fe::DataType_t::INT32;
        case DataType::INT64:
            return fe::DataType_t::INT64;
        default:
            throw std::runtime_error("Unsupported dtype for cuDNN Frontend convolution: " + TensorDescriptor::getElementTypeName(dtype));
    }
}

static std::vector<int64_t> toFrontendInt64Vector(const std::vector<uint64_t>& dims, const char* what) {
    std::vector<int64_t> out;
    out.reserve(dims.size());
    for (uint64_t dim : dims) {
        if (dim > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            throw std::runtime_error(std::string(what) + " dimension exceeds cuDNN Frontend int64 descriptor limit.");
        }
        out.push_back(static_cast<int64_t>(dim));
    }
    return out;
}

static std::vector<int64_t> packedFrontendStrides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (int64_t i = static_cast<int64_t>(dims.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * dims[static_cast<size_t>(i + 1)];
    }
    return strides;
}

static std::shared_ptr<fe::graph::Tensor_attributes> createFrontendConvolutionTensor(const std::shared_ptr<fe::graph::Graph>& graph,
                                                                                     const std::string& name,
                                                                                     int64_t uid,
                                                                                     const std::vector<uint64_t>& dims,
                                                                                     DataType dtype) {
    const std::vector<int64_t> frontend_dims = toFrontendInt64Vector(dims, name.c_str());
    return graph->tensor(fe::graph::Tensor_attributes()
                             .set_name(name)
                             .set_uid(uid)
                             .set_dim(frontend_dims)
                             .set_stride(packedFrontendStrides(frontend_dims))
                             .set_data_type(toFrontendDataType(dtype)));
}

static void setFrontendConvolutionOutputTensor(std::shared_ptr<fe::graph::Tensor_attributes>& tensor,
                                               const std::string& name,
                                               int64_t uid,
                                               const std::vector<uint64_t>& dims,
                                               DataType dtype) {
    const std::vector<int64_t> frontend_dims = toFrontendInt64Vector(dims, name.c_str());
    tensor->set_output(true)
        .set_name(name)
        .set_uid(uid)
        .set_dim(frontend_dims)
        .set_stride(packedFrontendStrides(frontend_dims))
        .set_data_type(toFrontendDataType(dtype));
}

static std::vector<int64_t> convolutionFrontendPadding(bool is_3d, int32_t pad_d, int32_t pad_h, int32_t pad_w) {
    if (is_3d) {
        return {static_cast<int64_t>(pad_d), static_cast<int64_t>(pad_h), static_cast<int64_t>(pad_w)};
    }
    return {static_cast<int64_t>(pad_h), static_cast<int64_t>(pad_w)};
}

static std::vector<int64_t> convolutionFrontendStrides(bool is_3d, int32_t stride_d, int32_t stride_h, int32_t stride_w) {
    if (is_3d) {
        return {static_cast<int64_t>(stride_d), static_cast<int64_t>(stride_h), static_cast<int64_t>(stride_w)};
    }
    return {static_cast<int64_t>(stride_h), static_cast<int64_t>(stride_w)};
}

static std::vector<int64_t> convolutionFrontendDilations(bool is_3d) {
    if (is_3d) {
        return {1, 1, 1};
    }
    return {1, 1};
}

static void checkFrontendStatus(cudnn_frontend::error_t status, const std::string& message) {
    if (!status.is_good()) {
        throw std::runtime_error(message + ": " + status.get_message());
    }
}

static int64_t checkedFrontendPlanWorkspaceBytes(const BuiltConvolution& built, int64_t plan_index, const char* op_name) {
    if (!built.frontend_graph) {
        throw std::runtime_error(std::string(op_name) + " missing cuDNN Frontend graph.");
    }
    const int64_t workspace_bytes = built.frontend_graph->get_workspace_size_plan_at_index(plan_index);
    if (workspace_bytes < 0) {
        throw std::runtime_error(std::string("cuDNN Frontend ") + op_name + " plan returned a negative workspace size.");
    }
    return workspace_bytes;
}

static void buildFrontendConvolutionCandidatePlans(BuiltConvolution& built, const Stream& stream, const char* op_name) {
    if (!built.frontend_graph) {
        throw std::runtime_error(std::string(op_name) + " missing cuDNN Frontend graph.");
    }

    ScopedGpu scopedGpu(stream.getGpuNum());
    checkFrontendStatus(built.frontend_graph->validate(), std::string("Failed to validate cuDNN Frontend ") + op_name + " graph");
    checkFrontendStatus(built.frontend_graph->build_operation_graph(stream.getCudnnHandle()),
                        std::string("Failed to build cuDNN Frontend ") + op_name + " operation graph");
    checkFrontendStatus(built.frontend_graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B, fe::HeurMode_t::FALLBACK}),
                        std::string("Failed to enumerate cuDNN Frontend ") + op_name + " execution plans");
    checkFrontendStatus(built.frontend_graph->check_support(stream.getCudnnHandle()),
                        std::string("Failed to check support for cuDNN Frontend ") + op_name + " execution plans");
}

namespace {
constexpr int kConvolutionAutotuneWarmupIterations = 2;
constexpr int kConvolutionAutotuneTimedIterations = 10;
constexpr int kConvolutionAutotuneMaxRotationSlots = kConvolutionAutotuneWarmupIterations + kConvolutionAutotuneTimedIterations;
constexpr int64_t kConvolutionAutotuneMaxCandidatePlans = 16;
constexpr uint64_t kConvolutionAutotuneTargetRotationBytes = 512ULL * 1024ULL * 1024ULL;
constexpr uint64_t kConvolutionAutotuneMinFreeMemReserveBytes = 512ULL * 1024ULL * 1024ULL;
constexpr uint64_t kConvolutionAutotuneMaxFreeMemFractionDivisor = 4;

struct FrontendConvolutionAutotuneBinding {
    int64_t uid;
    Tensor reference_tensor;
    bool rotate_for_timing = true;
};

struct FrontendConvolutionAutotuneTensorPool {
    std::vector<std::vector<Tensor>> rotating_tensors_by_binding;
    std::vector<std::unordered_map<int64_t, void*>> tensor_packs;
    std::optional<Tensor> workspace;
};

static uint64_t safeAddAutotuneBytes(uint64_t a, uint64_t b) {
    if (b > std::numeric_limits<uint64_t>::max() - a) {
        return std::numeric_limits<uint64_t>::max();
    }
    return a + b;
}

static int chooseFrontendConvolutionAutotuneRotationSlots(const std::vector<FrontendConvolutionAutotuneBinding>& bindings,
                                                          const TensorPlacement& placement,
                                                          int64_t workspace_bytes) {
    uint64_t rotating_slot_bytes = 0;
    uint64_t fixed_scratch_bytes = workspace_bytes > 0 ? static_cast<uint64_t>(workspace_bytes) : 0;
    for (const FrontendConvolutionAutotuneBinding& binding : bindings) {
        const uint64_t tensor_bytes = binding.reference_tensor.getArraySizeInBytes();
        if (binding.rotate_for_timing) {
            rotating_slot_bytes = safeAddAutotuneBytes(rotating_slot_bytes, tensor_bytes);
        }
    }

    if (rotating_slot_bytes == 0) {
        return 1;
    }

    uint64_t available_for_autotune = 0;
    if (placement.getMemDevice() == TensorPlacement::MemDevices::GPU) {
        const uint64_t free_bytes = static_cast<uint64_t>(MachineEvaluator::instance().getFreeMemBytes(placement.getDeviceNum()));
        const uint64_t reserve_bytes = std::min(free_bytes, kConvolutionAutotuneMinFreeMemReserveBytes);
        const uint64_t after_reserve = free_bytes > reserve_bytes ? free_bytes - reserve_bytes : 0;
        available_for_autotune = after_reserve / kConvolutionAutotuneMaxFreeMemFractionDivisor;
    } else {
        available_for_autotune = std::numeric_limits<uint64_t>::max();
    }

    if (available_for_autotune <= fixed_scratch_bytes) {
        return 1;
    }

    const uint64_t rotating_budget = available_for_autotune - fixed_scratch_bytes;
    const uint64_t max_affordable_slots = std::max<uint64_t>(1, rotating_budget / rotating_slot_bytes);
    const uint64_t target_slots = std::max<uint64_t>(1,
                                                     (kConvolutionAutotuneTargetRotationBytes + rotating_slot_bytes - 1) /
                                                         rotating_slot_bytes);
    const uint64_t bounded_slots = std::min<uint64_t>(
        static_cast<uint64_t>(kConvolutionAutotuneMaxRotationSlots), std::min(target_slots, max_affordable_slots));
    return static_cast<int>(std::max<uint64_t>(1, bounded_slots));
}

static FrontendConvolutionAutotuneTensorPool createFrontendConvolutionAutotuneTensorPool(
    const std::vector<FrontendConvolutionAutotuneBinding>& bindings,
    const TensorPlacement& workspace_placement,
    int64_t workspace_bytes) {
    FrontendConvolutionAutotuneTensorPool pool;
    const int rotation_slots = chooseFrontendConvolutionAutotuneRotationSlots(bindings, workspace_placement, workspace_bytes);

    pool.rotating_tensors_by_binding.resize(bindings.size());
    pool.tensor_packs.resize(rotation_slots);
    for (std::unordered_map<int64_t, void*>& tensor_pack : pool.tensor_packs) {
        tensor_pack.reserve(bindings.size());
    }

    for (size_t binding_index = 0; binding_index < bindings.size(); ++binding_index) {
        const FrontendConvolutionAutotuneBinding& binding = bindings[binding_index];
        if (binding.rotate_for_timing) {
            std::vector<Tensor>& rotating_tensors = pool.rotating_tensors_by_binding[binding_index];
            rotating_tensors.reserve(rotation_slots);
            for (int slot = 0; slot < rotation_slots; ++slot) {
                rotating_tensors.emplace_back(binding.reference_tensor.getPlacement(), binding.reference_tensor.getDescriptor());
                pool.tensor_packs[slot][binding.uid] =
                    const_cast<void*>(static_cast<const void*>(rotating_tensors.back().getMemPtr<void>()));
            }
        } else {
            void* ptr = const_cast<void*>(static_cast<const void*>(binding.reference_tensor.getMemPtr<void>()));
            for (std::unordered_map<int64_t, void*>& tensor_pack : pool.tensor_packs) {
                tensor_pack[binding.uid] = ptr;
            }
        }
    }

    if (workspace_bytes > 0) {
        pool.workspace = Tensor(workspace_placement, TensorDescriptor(DataType::UINT8, {static_cast<uint64_t>(workspace_bytes)}));
    }

    return pool;
}

static float timeFrontendConvolutionPlan(BuiltConvolution& built,
                                         const Stream& stream,
                                         FrontendConvolutionAutotuneTensorPool& pool,
                                         const char* op_name) {
    if (!built.frontend_graph) {
        throw std::runtime_error(std::string(op_name) + " missing cuDNN Frontend graph.");
    }
    if (pool.tensor_packs.empty()) {
        throw std::runtime_error(std::string(op_name) + " autotune tensor pack rotation pool is empty.");
    }

    Stream timing_stream = stream;
    auto run_once = [&](int iteration) {
        std::unordered_map<int64_t, void*>& tensor_pack = pool.tensor_packs[static_cast<size_t>(iteration) % pool.tensor_packs.size()];
        void* workspace_ptr = pool.workspace.has_value()
                                  ? const_cast<void*>(static_cast<const void*>(pool.workspace.value().getMemPtr<void>()))
                                  : nullptr;
        auto status = built.frontend_graph->execute(timing_stream.getCudnnHandle(), tensor_pack, workspace_ptr);
        if (!status.is_good()) {
            throw std::runtime_error(std::string("Failed to execute cuDNN Frontend ") + op_name + " plan during autotune: " +
                                     status.get_message());
        }
    };

    for (int iteration = 0; iteration < kConvolutionAutotuneWarmupIterations; ++iteration) {
        run_once(iteration);
    }
    timing_stream.synchronize();

    Event start(stream.getGpuNum(), true, true);
    Event stop(stream.getGpuNum(), true, true);
    start.record(timing_stream);
    for (int iteration = 0; iteration < kConvolutionAutotuneTimedIterations; ++iteration) {
        run_once(kConvolutionAutotuneWarmupIterations + iteration);
    }
    stop.record(timing_stream);
    return stop.synchronizeAndReportElapsedTimeInMilliseconds(start) / static_cast<float>(kConvolutionAutotuneTimedIterations);
}
}  // namespace

static void autotuneFrontendConvolutionGraph(BuiltConvolution& built,
                                             const Stream& stream,
                                             const std::vector<FrontendConvolutionAutotuneBinding>& autotune_bindings,
                                             const TensorPlacement& workspace_placement,
                                             const char* op_name) {
    buildFrontendConvolutionCandidatePlans(built, stream, op_name);

    const int64_t plan_count = built.frontend_graph->get_execution_plan_count();
    if (plan_count <= 0) {
        throw std::runtime_error(std::string("cuDNN Frontend ") + op_name + " produced no execution plans.");
    }

    // cuDNN Frontend returns execution plans in heuristic-ranked order for the requested modes.
    // Autotune only the front of that ordered pool so placement does not devolve into measuring
    // every possible engine/configuration. If this pool cannot produce a runnable plan, fail loudly.
    const int64_t candidate_count = std::min(plan_count, kConvolutionAutotuneMaxCandidatePlans);

    int64_t best_plan_index = -1;
    int64_t best_workspace_bytes = 0;
    float best_ms = std::numeric_limits<float>::infinity();

    for (int64_t plan_index = 0; plan_index < candidate_count; ++plan_index) {
        auto status = built.frontend_graph->build_plan_at_index(stream.getCudnnHandle(), plan_index);
        if (!status.is_good()) {
            continue;
        }

        const int64_t workspace_bytes = checkedFrontendPlanWorkspaceBytes(built, plan_index, op_name);

        try {
            FrontendConvolutionAutotuneTensorPool timing_pool =
                createFrontendConvolutionAutotuneTensorPool(autotune_bindings, workspace_placement, workspace_bytes);
            const float milliseconds = timeFrontendConvolutionPlan(built, stream, timing_pool, op_name);
            if (std::isfinite(milliseconds) && milliseconds < best_ms) {
                best_ms = milliseconds;
                best_plan_index = plan_index;
                best_workspace_bytes = workspace_bytes;
            }
        } catch (const std::exception&) {
            // Some plans can build but still fail under the concrete runtime tensor/workspace configuration.
            // Do not let one bad candidate become the selected convolution plan.
            continue;
        }
    }

    if (best_plan_index < 0) {
        throw std::runtime_error(std::string("cuDNN Frontend ") + op_name + " autotune could not build and run any of the top " +
                                 std::to_string(candidate_count) + " heuristic-ranked execution plans.");
    }

    checkFrontendStatus(built.frontend_graph->build_plan_at_index(stream.getCudnnHandle(), best_plan_index),
                        std::string("Failed to rebuild selected cuDNN Frontend ") + op_name + " execution plan");

    built.selected_plan_index = best_plan_index;
    built.workspace_bytes = static_cast<size_t>(best_workspace_bytes);
}

static void putFrontendTensorPointer(std::unordered_map<int64_t, void*>& pack, int64_t uid, const Tensor& tensor) {
    pack[uid] = const_cast<void*>(static_cast<const void*>(tensor.getMemPtr<void>()));
}

static void executeFrontendConvolutionGraph(const BuiltConvolution& built,
                                            const Stream& run_stream,
                                            std::unordered_map<int64_t, void*>& tensor_pack,
                                            const std::optional<Tensor>& workspace,
                                            const char* op_name) {
    if (!built.frontend_graph) {
        throw std::runtime_error(std::string(op_name) + " missing cuDNN Frontend graph.");
    }

    void* workspace_ptr = nullptr;
    if (built.workspace_bytes > 0) {
        if (!workspace.has_value()) {
            throw std::runtime_error(std::string(op_name) + " requires cuDNN Frontend workspace, but none was allocated.");
        }
        workspace_ptr = const_cast<void*>(static_cast<const void*>(workspace.value().getMemPtr<void>()));
    }

    if (built.selected_plan_index < 0) {
        throw std::runtime_error(std::string(op_name) + " has no autotuned cuDNN Frontend execution plan.");
    }

    auto status = built.frontend_graph->execute(run_stream.getCudnnHandle(), tensor_pack, workspace_ptr);
    if (!status.is_good()) {
        throw std::runtime_error(std::string("Failed to execute autotuned cuDNN Frontend ") + op_name + " graph: " + status.get_message());
    }
}

static cudnnReduceTensorDescriptor_t createCudnnReduceDescriptor(ExprOp op, DataType compute_dtype, bool output_indices) {
    cudnnReduceTensorDescriptor_t desc;
    CUDNN_CHECK(cudnnCreateReduceTensorDescriptor(&desc));
    CUDNN_CHECK(cudnnSetReduceTensorDescriptor(desc,
                                               toCudnnReduceTensorOp(op),
                                               toCudnnDataType(compute_dtype),
                                               CUDNN_PROPAGATE_NAN,
                                               output_indices ? CUDNN_REDUCE_TENSOR_FLATTENED_INDICES : CUDNN_REDUCE_TENSOR_NO_INDICES,
                                               CUDNN_32BIT_INDICES));
    return desc;
}

static size_t getReductionWorkspaceSize(int device_num,
                                        cudnnReduceTensorDescriptor_t reduce_desc,
                                        cudnnTensorDescriptor_t a_desc,
                                        cudnnTensorDescriptor_t c_desc) {
    Stream stream(device_num);
    size_t workspace_bytes = 0;
    CUDNN_CHECK(cudnnGetReductionWorkspaceSize(stream.getCudnnHandle(), reduce_desc, a_desc, c_desc, &workspace_bytes));
    return workspace_bytes;
}

static size_t getReductionIndicesSize(int device_num,
                                      cudnnReduceTensorDescriptor_t reduce_desc,
                                      cudnnTensorDescriptor_t a_desc,
                                      cudnnTensorDescriptor_t c_desc) {
    Stream stream(device_num);
    size_t indices_bytes = 0;
    CUDNN_CHECK(cudnnGetReductionIndicesSize(stream.getCudnnHandle(), reduce_desc, a_desc, c_desc, &indices_bytes));
    return indices_bytes;
}

static CublasMatrixMultiply::EpilogueFusion toCublasEpilogueFusion(MatmulEpilogue epilogue) {
    switch (epilogue) {
        case MatmulEpilogue::Default:
            return CublasMatrixMultiply::EpilogueFusion::Default;
        case MatmulEpilogue::Relu:
            return CublasMatrixMultiply::EpilogueFusion::Relu;
        case MatmulEpilogue::Gelu:
            return CublasMatrixMultiply::EpilogueFusion::Gelu;
    }
    throw std::runtime_error("Unknown MatmulEpilogue value.");
}

static CublasMatrixMultiply::BackwardEpilogueFusion toCublasBackwardEpilogueFusion(MatmulBackwardEpilogue epilogue) {
    switch (epilogue) {
        case MatmulBackwardEpilogue::DRelu:
            return CublasMatrixMultiply::BackwardEpilogueFusion::DRelu;
        case MatmulBackwardEpilogue::DGelu:
            return CublasMatrixMultiply::BackwardEpilogueFusion::DGelu;
        case MatmulBackwardEpilogue::Default:
            break;
    }
    throw std::runtime_error("Default or unknown MatmulBackwardEpilogue cannot be lowered to a cuBLASLt backward epilogue.");
}

static int32_t leadingDimensionForStoredMatrix(const Tensor& matrix) {
    const std::vector<uint64_t> dims = matrix.getDimensions();
    if (dims.size() != 2) {
        throw std::runtime_error("Matmul/gemm workspace planning currently only supports rank-2 tensors.");
    }
    return static_cast<int32_t>(dims[1]);
}

std::shared_ptr<BuiltMatmul> StampedEquation::buildMatmul(const std::shared_ptr<CompiledMatmul>& compiled_matmul,
                                                          const Tensor& lhs,
                                                          const Tensor& rhs,
                                                          const std::optional<Tensor>& addend,
                                                          const Tensor& output,
                                                          int device_num,
                                                          const std::optional<Tensor>& epilogue_aux,
                                                          const std::optional<Tensor>& bgrad_output) {
    if (!compiled_matmul) {
        throw std::runtime_error("buildMatmul requires non-null compiled payload.");
    }
    if (lhs.getDimensions().size() != 2 || rhs.getDimensions().size() != 2 || output.getDimensions().size() != 2) {
        throw std::runtime_error("buildMatmul currently only supports rank-2 tensors.");
    }
    if (compiled_matmul->backward_epilogue != MatmulBackwardEpilogue::Default && !epilogue_aux.has_value()) {
        throw std::runtime_error("buildMatmul backward cuBLASLt epilogue requires epilogue_aux.");
    }
    if (compiled_matmul->bgrad_output_dtype.has_value() && !bgrad_output.has_value()) {
        throw std::runtime_error("buildMatmul backward cuBLASLt bgrad epilogue requires bgrad_output.");
    }
    if (bgrad_output.has_value() && !compiled_matmul->bgrad_output_dtype.has_value()) {
        throw std::runtime_error("buildMatmul received bgrad_output but the compiled matmul does not declare one.");
    }
    if (compiled_matmul->backward_epilogue != MatmulBackwardEpilogue::Default && compiled_matmul->epilogue != MatmulEpilogue::Default) {
        throw std::runtime_error("buildMatmul cannot combine forward and backward cuBLASLt epilogues in one stage.");
    }
    bool use_bias_epilogue = false;
    if (compiled_matmul->op == ExprOp::GEMM) {
        if (!addend.has_value()) {
            throw std::runtime_error("buildMatmul requires an addend tensor for GEMM.");
        }
        const size_t addend_rank = addend.value().getDimensions().size();
        use_bias_epilogue = addend_rank == 1;
        if (addend_rank != 1 && addend_rank != 2) {
            throw std::runtime_error("buildMatmul currently supports rank-2 GEMM addends or rank-1 bias epilogue vectors.");
        }
        if (compiled_matmul->transpose_aux) {
            throw std::runtime_error("GEMM transpose_aux/transposeC is not supported by CublasMatrixMultiply in this staged path.");
        }
    }

    const std::vector<uint64_t> lhs_dims = lhs.getDimensions();
    const std::vector<uint64_t> rhs_dims = rhs.getDimensions();
    const int32_t a_rows = static_cast<int32_t>(lhs_dims[0]);
    const int32_t a_cols = static_cast<int32_t>(lhs_dims[1]);
    const int32_t b_rows = static_cast<int32_t>(rhs_dims[0]);
    const int32_t b_cols = static_cast<int32_t>(rhs_dims[1]);
    const int32_t ld_a = leadingDimensionForStoredMatrix(lhs);
    const int32_t ld_b = leadingDimensionForStoredMatrix(rhs);
    const int32_t ld_d = leadingDimensionForStoredMatrix(output);
    const int32_t ld_c = addend.has_value() ? (use_bias_epilogue ? ld_d : leadingDimensionForStoredMatrix(addend.value())) : ld_d;

    const CublasMatrixMultiply::MatmulDataTypes dataTypes{
        lhs.getDescriptor().getDataType(),
        rhs.getDescriptor().getDataType(),
        addend.has_value() ? (use_bias_epilogue ? output.getDescriptor().getDataType() : addend.value().getDescriptor().getDataType())
                           : output.getDescriptor().getDataType(),
        output.getDescriptor().getDataType(),
        compiled_matmul->compute_dtype};

    if (use_bias_epilogue && addend.value().getDescriptor().getDataType() != output.getDescriptor().getDataType()) {
        throw std::runtime_error("GEMM bias epilogue requires the bias dtype to match the output dtype.");
    }
    const bool use_backward_epilogue = compiled_matmul->backward_epilogue != MatmulBackwardEpilogue::Default;
    if ((compiled_matmul->epilogue != MatmulEpilogue::Default || use_bias_epilogue || use_backward_epilogue) &&
        (compiled_matmul->transpose_lhs || compiled_matmul->transpose_rhs || compiled_matmul->transpose_aux)) {
        throw std::runtime_error("cuBLASLt GEMM epilogue fusion currently supports only non-transposed row-major matmul/gemm stages.");
    }
    if (use_backward_epilogue && epilogue_aux.has_value() && compiled_matmul->epilogue_aux_dtype.has_value() &&
        epilogue_aux.value().getDescriptor().getDataType() != compiled_matmul->epilogue_aux_dtype.value()) {
        throw std::runtime_error("buildMatmul epilogue_aux dtype does not match the compiled matmul dtype plan.");
    }
    if (bgrad_output.has_value()) {
        if (bgrad_output.value().getDimensions().size() != 1 || bgrad_output.value().getDimensions()[0] != output.getDimensions()[1]) {
            throw std::runtime_error("buildMatmul bgrad_output must be a rank-1 tensor with one element per output column.");
        }
        if (bgrad_output.value().getDescriptor().getDataType() != compiled_matmul->bgrad_output_dtype.value()) {
            throw std::runtime_error("buildMatmul bgrad_output dtype does not match the compiled matmul dtype plan.");
        }
    }
    if (dataTypes.A != compiled_matmul->lhs_dtype || dataTypes.B != compiled_matmul->rhs_dtype ||
        dataTypes.C != (compiled_matmul->op == ExprOp::GEMM ? compiled_matmul->aux_dtype : compiled_matmul->output_dtype) ||
        dataTypes.D != compiled_matmul->output_dtype) {
        throw std::runtime_error("buildMatmul tensor dtypes do not match the compiled matmul dtype plan.");
    }

    MatmulCacheKey key(compiled_matmul->op,
                       a_rows,
                       a_cols,
                       b_rows,
                       b_cols,
                       ld_a,
                       ld_b,
                       ld_c,
                       ld_d,
                       compiled_matmul->transpose_lhs,
                       compiled_matmul->transpose_rhs,
                       compiled_matmul->transpose_aux,
                       use_bias_epilogue,
                       compiled_matmul->epilogue,
                       compiled_matmul->backward_epilogue,
                       bgrad_output.has_value(),
                       dataTypes.A,
                       dataTypes.B,
                       dataTypes.C,
                       dataTypes.D,
                       dataTypes.compute,
                       device_num);

    std::shared_ptr<BuiltMatmul> hit = cacheLookup(key);
    if (hit) {
        return hit;
    }

    auto built = std::make_shared<BuiltMatmul>(key);
    bool kernelWillRunOnGpu = false;
    const bool print_verbose_matmul_diagnostics = thorMatmulDiagnosticsVerbose();
    const char *diagnostic_path = "unknown";

    const bool use_cublaslt_epilogue_wrapper =
        use_bias_epilogue || compiled_matmul->epilogue != MatmulEpilogue::Default || use_backward_epilogue;

    if (compiled_matmul->op == ExprOp::MATMUL && !use_cublaslt_epilogue_wrapper) {
        CublasMatrixMultiply::instance().chooseOptimalMatrixMultiplyKernel(device_num,
                                                                           a_rows,
                                                                           a_cols,
                                                                           b_rows,
                                                                           b_cols,
                                                                           ld_a,
                                                                           ld_b,
                                                                           ld_d,
                                                                           compiled_matmul->transpose_lhs,
                                                                           compiled_matmul->transpose_rhs,
                                                                           dataTypes,
                                                                           print_verbose_matmul_diagnostics);
        diagnostic_path = "optimal-matmul-picker";
        built->workspace_bytes = CublasMatrixMultiply::instance().getMatrixMultiplyWorkspaceSizeInBytes(device_num,
                                                                                                        a_rows,
                                                                                                        a_cols,
                                                                                                        b_rows,
                                                                                                        b_cols,
                                                                                                        ld_a,
                                                                                                        ld_b,
                                                                                                        ld_d,
                                                                                                        compiled_matmul->transpose_lhs,
                                                                                                        compiled_matmul->transpose_rhs,
                                                                                                        dataTypes,
                                                                                                        kernelWillRunOnGpu);
    } else if (use_cublaslt_epilogue_wrapper) {
        diagnostic_path = "epilogue-workspace-wrapper";
        if (use_backward_epilogue) {
            if (!epilogue_aux.has_value()) {
                throw std::runtime_error("buildMatmul backward cuBLASLt epilogue requires epilogue_aux.");
            }
            const int64_t epilogue_aux_ld = static_cast<int64_t>(epilogue_aux.value().getDimensions()[1]);
            built->epilogue_algorithm = CublasMatrixMultiply::instance().selectGemmWithBackwardEpilogueAlgorithm(
                device_num,
                a_rows,
                a_cols,
                b_rows,
                b_cols,
                ld_a,
                ld_b,
                ld_c,
                ld_d,
                compiled_matmul->transpose_lhs,
                compiled_matmul->transpose_rhs,
                dataTypes,
                toCublasBackwardEpilogueFusion(compiled_matmul->backward_epilogue),
                addend.has_value(),
                bgrad_output.has_value(),
                epilogue_aux_ld);
            kernelWillRunOnGpu = built->epilogue_algorithm.has_value();
            built->workspace_bytes = kernelWillRunOnGpu ? built->epilogue_algorithm->workspace_size_in_bytes : 0;
        } else {
            built->epilogue_algorithm = CublasMatrixMultiply::instance().selectGemmWithEpilogueAlgorithm(
                device_num,
                a_rows,
                a_cols,
                b_rows,
                b_cols,
                ld_a,
                ld_b,
                ld_c,
                ld_d,
                compiled_matmul->transpose_lhs,
                compiled_matmul->transpose_rhs,
                dataTypes,
                toCublasEpilogueFusion(compiled_matmul->epilogue),
                addend.has_value(),
                use_bias_epilogue);
            kernelWillRunOnGpu = built->epilogue_algorithm.has_value();
            built->workspace_bytes = kernelWillRunOnGpu ? built->epilogue_algorithm->workspace_size_in_bytes : 0;
        }
    } else {
        CublasMatrixMultiply::instance().chooseOptimalGemmKernel(device_num,
                                                                 a_rows,
                                                                 a_cols,
                                                                 b_rows,
                                                                 b_cols,
                                                                 ld_a,
                                                                 ld_b,
                                                                 ld_c,
                                                                 ld_d,
                                                                 compiled_matmul->transpose_lhs,
                                                                 compiled_matmul->transpose_rhs,
                                                                 compiled_matmul->transpose_aux,
                                                                 dataTypes,
                                                                 print_verbose_matmul_diagnostics);
        diagnostic_path = "optimal-gemm-picker";
        built->workspace_bytes = CublasMatrixMultiply::instance().getGemmWorkspaceSizeInBytes(device_num,
                                                                                              a_rows,
                                                                                              a_cols,
                                                                                              b_rows,
                                                                                              b_cols,
                                                                                              ld_a,
                                                                                              ld_b,
                                                                                              ld_c,
                                                                                              ld_d,
                                                                                              compiled_matmul->transpose_lhs,
                                                                                              compiled_matmul->transpose_rhs,
                                                                                              compiled_matmul->transpose_aux,
                                                                                              dataTypes,
                                                                                              kernelWillRunOnGpu);
    }

    if (!kernelWillRunOnGpu) {
        throw std::runtime_error("No GPU kernel available for the staged matmul/gemm configuration.");
    }

    if (thorMatmulDiagnosticsEnabled()) {
        std::ostringstream diagnostic_key;
        diagnostic_key << "build:" << diagnostic_path << ':' << device_num << ':' << matmulExprOpName(compiled_matmul->op) << ':'
                       << a_rows << 'x' << a_cols << ':' << b_rows << 'x' << b_cols << ":ld=" << ld_a << ',' << ld_b << ','
                       << ld_c << ',' << ld_d << ":trans=" << static_cast<int>(compiled_matmul->transpose_lhs)
                       << static_cast<int>(compiled_matmul->transpose_rhs) << static_cast<int>(compiled_matmul->transpose_aux)
                       << ":bias=" << static_cast<int>(use_bias_epilogue) << ":epilogue="
                       << matmulEpilogueName(compiled_matmul->epilogue) << ":backward_epilogue="
                       << matmulBackwardEpilogueName(compiled_matmul->backward_epilogue) << ":bgrad="
                       << static_cast<int>(bgrad_output.has_value()) << ":dtypes=" << TensorDescriptor::getElementTypeName(dataTypes.A)
                       << ',' << TensorDescriptor::getElementTypeName(dataTypes.B) << ','
                       << TensorDescriptor::getElementTypeName(dataTypes.C) << ',' << TensorDescriptor::getElementTypeName(dataTypes.D)
                       << ',' << TensorDescriptor::getElementTypeName(dataTypes.compute);
        if (shouldPrintStampedMatmulDiagnosticOnce(diagnostic_key.str())) {
            std::fprintf(stderr,
                         "THOR_MATMUL_DIAGNOSTIC build path=%s op=%s gpu=%d A=%dx%d B=%dx%d ld=%d,%d,%d,%d "
                         "transpose=%d,%d,%d bias_epilogue=%d epilogue=%s backward_epilogue=%s bgrad_epilogue=%d "
                         "workspace_bytes=%zu dtypes=%s,%s,%s,%s compute=%s\n",
                         diagnostic_path,
                         matmulExprOpName(compiled_matmul->op),
                         device_num,
                         a_rows,
                         a_cols,
                         b_rows,
                         b_cols,
                         ld_a,
                         ld_b,
                         ld_c,
                         ld_d,
                         static_cast<int>(compiled_matmul->transpose_lhs),
                         static_cast<int>(compiled_matmul->transpose_rhs),
                         static_cast<int>(compiled_matmul->transpose_aux),
                         static_cast<int>(use_bias_epilogue),
                         matmulEpilogueName(compiled_matmul->epilogue),
                         matmulBackwardEpilogueName(compiled_matmul->backward_epilogue),
                         static_cast<int>(bgrad_output.has_value()),
                         built->workspace_bytes,
                         TensorDescriptor::getElementTypeName(dataTypes.A).c_str(),
                         TensorDescriptor::getElementTypeName(dataTypes.B).c_str(),
                         TensorDescriptor::getElementTypeName(dataTypes.C).c_str(),
                         TensorDescriptor::getElementTypeName(dataTypes.D).c_str(),
                         TensorDescriptor::getElementTypeName(dataTypes.compute).c_str());
        }
    }

    builtMatmulCache.put(key, built);
    return built;
}

std::shared_ptr<BuiltConvolution> StampedEquation::buildConvolution(const std::shared_ptr<CompiledConvolution>& compiled_convolution,
                                                                    const Tensor& input,
                                                                    const Tensor& filter,
                                                                    const Tensor& output,
                                                                    const Stream& stream,
                                                                    int device_num) {
    (void)device_num;
    if (!compiled_convolution) {
        throw std::runtime_error("buildConvolution requires non-null compiled payload.");
    }
    const bool is_3d = compiled_convolution->is_3d;
    const size_t expected_rank = is_3d ? 5 : 4;
    if (input.getDimensions().size() != expected_rank || filter.getDimensions().size() != expected_rank ||
        output.getDimensions().size() != expected_rank) {
        throw std::runtime_error(is_3d ? "buildConvolution expected rank-5 tensors for CONV3D."
                                       : "buildConvolution expected rank-4 tensors for CONV2D.");
    }

    auto built = std::make_shared<BuiltConvolution>();
    built->use_cudnn_frontend = true;
    built->frontend_graph = std::make_shared<fe::graph::Graph>();
    built->frontend_graph->set_io_data_type(toFrontendDataType(compiled_convolution->output_dtype))
        .set_intermediate_data_type(toFrontendDataType(compiled_convolution->compute_dtype))
        .set_compute_data_type(toFrontendDataType(compiled_convolution->compute_dtype));

    const char* prefix = is_3d ? "conv3d" : "conv2d";
    auto x = createFrontendConvolutionTensor(built->frontend_graph,
                                             std::string(prefix) + "_x",
                                             CUDNN_FRONTEND_CONV_X_UID,
                                             input.getDimensions(),
                                             compiled_convolution->input_dtype);
    auto w = createFrontendConvolutionTensor(built->frontend_graph,
                                             std::string(prefix) + "_w",
                                             CUDNN_FRONTEND_CONV_W_UID,
                                             filter.getDimensions(),
                                             compiled_convolution->filter_dtype);

    auto conv_attrs = fe::graph::Conv_fprop_attributes()
                          .set_name(std::string("thor_expr_") + prefix + "_fprop")
                          .set_padding(convolutionFrontendPadding(
                              is_3d, compiled_convolution->pad_d, compiled_convolution->pad_h, compiled_convolution->pad_w))
                          .set_stride(convolutionFrontendStrides(
                              is_3d, compiled_convolution->stride_d, compiled_convolution->stride_h, compiled_convolution->stride_w))
                          .set_dilation(convolutionFrontendDilations(is_3d))
                          .set_compute_data_type(toFrontendDataType(compiled_convolution->compute_dtype))
                          .set_convolution_mode(fe::ConvolutionMode_t::CROSS_CORRELATION);

    auto y = built->frontend_graph->conv_fprop(x, w, conv_attrs);
    setFrontendConvolutionOutputTensor(
        y, std::string(prefix) + "_y", CUDNN_FRONTEND_CONV_Y_UID, output.getDimensions(), compiled_convolution->output_dtype);

    std::vector<FrontendConvolutionAutotuneBinding> autotune_bindings = {{CUDNN_FRONTEND_CONV_X_UID, input, true},
                                                                         {CUDNN_FRONTEND_CONV_W_UID, filter, true},
                                                                         {CUDNN_FRONTEND_CONV_Y_UID, output, false}};
    autotuneFrontendConvolutionGraph(*built, stream, autotune_bindings, input.getPlacement(), is_3d ? "CONV3D forward" : "CONV2D forward");
    return built;
}

std::shared_ptr<BuiltConvolution> StampedEquation::buildConvolutionBackward(
    const std::shared_ptr<CompiledConvolutionBackward>& compiled_convolution_backward,
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& output,
    const Stream& stream,
    int device_num) {
    (void)device_num;
    if (!compiled_convolution_backward) {
        throw std::runtime_error("buildConvolutionBackward requires non-null compiled payload.");
    }

    const bool is_backward_data = compiled_convolution_backward->op == ExprOp::CONV2D_BACKWARD_DATA ||
                                  compiled_convolution_backward->op == ExprOp::CONV3D_BACKWARD_DATA;
    const bool is_backward_filter = compiled_convolution_backward->op == ExprOp::CONV2D_BACKWARD_FILTER ||
                                    compiled_convolution_backward->op == ExprOp::CONV3D_BACKWARD_FILTER;
    if (!is_backward_data && !is_backward_filter) {
        throw std::runtime_error("buildConvolutionBackward received unsupported convolution backward op.");
    }

    const bool is_3d = compiled_convolution_backward->op == ExprOp::CONV3D_BACKWARD_DATA ||
                       compiled_convolution_backward->op == ExprOp::CONV3D_BACKWARD_FILTER;
    const size_t expected_rank = is_3d ? 5 : 4;
    if (input.getDimensions().size() != expected_rank || grad_output.getDimensions().size() != expected_rank ||
        output.getDimensions().size() != expected_rank) {
        throw std::runtime_error(is_3d ? "buildConvolutionBackward expected rank-5 tensors for CONV3D backward."
                                       : "buildConvolutionBackward expected rank-4 tensors for CONV2D backward.");
    }

    auto built = std::make_shared<BuiltConvolution>();
    built->use_cudnn_frontend = true;
    built->frontend_graph = std::make_shared<fe::graph::Graph>();
    built->frontend_graph->set_io_data_type(toFrontendDataType(compiled_convolution_backward->output_dtype))
        .set_intermediate_data_type(toFrontendDataType(compiled_convolution_backward->compute_dtype))
        .set_compute_data_type(toFrontendDataType(compiled_convolution_backward->compute_dtype));

    const char* prefix = is_3d ? "conv3d" : "conv2d";
    const auto padding = convolutionFrontendPadding(
        is_3d, compiled_convolution_backward->pad_d, compiled_convolution_backward->pad_h, compiled_convolution_backward->pad_w);
    const auto strides = convolutionFrontendStrides(
        is_3d, compiled_convolution_backward->stride_d, compiled_convolution_backward->stride_h, compiled_convolution_backward->stride_w);
    const auto dilations = convolutionFrontendDilations(is_3d);
    const fe::DataType_t compute_dtype = toFrontendDataType(compiled_convolution_backward->compute_dtype);

    if (is_backward_data) {
        auto w = createFrontendConvolutionTensor(built->frontend_graph,
                                                 std::string(prefix) + "_bwd_data_w",
                                                 CUDNN_FRONTEND_CONV_W_UID,
                                                 input.getDimensions(),
                                                 compiled_convolution_backward->input_dtype);
        auto dy = createFrontendConvolutionTensor(built->frontend_graph,
                                                  std::string(prefix) + "_bwd_data_dy",
                                                  CUDNN_FRONTEND_CONV_Y_UID,
                                                  grad_output.getDimensions(),
                                                  compiled_convolution_backward->grad_output_dtype);
        auto conv_attrs = fe::graph::Conv_dgrad_attributes()
                              .set_name(std::string("thor_expr_") + prefix + "_dgrad")
                              .set_padding(padding)
                              .set_stride(strides)
                              .set_dilation(dilations)
                              .set_compute_data_type(compute_dtype)
                              .set_convolution_mode(fe::ConvolutionMode_t::CROSS_CORRELATION);

        auto dx = built->frontend_graph->conv_dgrad(dy, w, conv_attrs);
        setFrontendConvolutionOutputTensor(dx,
                                           std::string(prefix) + "_bwd_data_dx",
                                           CUDNN_FRONTEND_CONV_X_UID,
                                           output.getDimensions(),
                                           compiled_convolution_backward->output_dtype);

        std::vector<FrontendConvolutionAutotuneBinding> autotune_bindings = {{CUDNN_FRONTEND_CONV_W_UID, input, true},
                                                                             {CUDNN_FRONTEND_CONV_Y_UID, grad_output, true},
                                                                             {CUDNN_FRONTEND_CONV_X_UID, output, false}};
        autotuneFrontendConvolutionGraph(
            *built, stream, autotune_bindings, output.getPlacement(), is_3d ? "CONV3D backward-data" : "CONV2D backward-data");
        return built;
    }

    auto x = createFrontendConvolutionTensor(built->frontend_graph,
                                             std::string(prefix) + "_bwd_filter_x",
                                             CUDNN_FRONTEND_CONV_X_UID,
                                             input.getDimensions(),
                                             compiled_convolution_backward->input_dtype);
    auto dy = createFrontendConvolutionTensor(built->frontend_graph,
                                              std::string(prefix) + "_bwd_filter_dy",
                                              CUDNN_FRONTEND_CONV_Y_UID,
                                              grad_output.getDimensions(),
                                              compiled_convolution_backward->grad_output_dtype);
    auto conv_attrs = fe::graph::Conv_wgrad_attributes()
                          .set_name(std::string("thor_expr_") + prefix + "_wgrad")
                          .set_padding(padding)
                          .set_stride(strides)
                          .set_dilation(dilations)
                          .set_compute_data_type(compute_dtype)
                          .set_convolution_mode(fe::ConvolutionMode_t::CROSS_CORRELATION);

    auto dw = built->frontend_graph->conv_wgrad(dy, x, conv_attrs);
    setFrontendConvolutionOutputTensor(dw,
                                       std::string(prefix) + "_bwd_filter_dw",
                                       CUDNN_FRONTEND_CONV_W_UID,
                                       output.getDimensions(),
                                       compiled_convolution_backward->output_dtype);

    std::vector<FrontendConvolutionAutotuneBinding> autotune_bindings = {{CUDNN_FRONTEND_CONV_X_UID, input, true},
                                                                         {CUDNN_FRONTEND_CONV_Y_UID, grad_output, true},
                                                                         {CUDNN_FRONTEND_CONV_W_UID, output, false}};
    autotuneFrontendConvolutionGraph(
        *built, stream, autotune_bindings, output.getPlacement(), is_3d ? "CONV3D backward-filter" : "CONV2D backward-filter");
    return built;
}

std::shared_ptr<BuiltReduction> StampedEquation::buildReduction(const std::shared_ptr<CompiledReduction>& compiled_reduction,
                                                                const Tensor& input,
                                                                int device_num) {
    return buildReduction(compiled_reduction->op,
                          compiled_reduction->reduction_axes,
                          compiled_reduction->squeeze_axes,
                          compiled_reduction->input_dtype,
                          compiled_reduction->output_dtype,
                          compiled_reduction->compute_dtype,
                          /*output_indices=*/false,
                          input,
                          device_num);
}

std::shared_ptr<BuiltReduction> StampedEquation::buildReduction(ExprOp op,
                                                                const std::vector<uint64_t>& reduction_axes,
                                                                const std::vector<uint64_t>& squeeze_axes,
                                                                DataType input_dtype,
                                                                DataType output_dtype,
                                                                DataType compute_dtype,
                                                                bool output_indices,
                                                                const Tensor& input,
                                                                int device_num) {
    const std::vector<uint64_t> input_dims = input.getDimensions();

    ReductionCacheKey key(
        op, input_dims, reduction_axes, squeeze_axes, input_dtype, output_dtype, compute_dtype, output_indices, device_num);

    std::shared_ptr<BuiltReduction> hit = cacheLookup(key);
    if (hit)
        return hit;

    auto built = std::make_shared<BuiltReduction>(key);

    const std::vector<uint64_t> output_dims = computeReductionOutputDims(input_dims,
                                                                         built->key.reduction_axes,
                                                                         /*squeeze_axes=*/{});

    // cuDNN rejects reductions where every reduced dimension already has extent 1
    // because the padded source and destination descriptors are identical. For
    // singleton sum/prod/min/max/avg value reductions, the result is just the input
    // value, with only dtype conversion and optional squeeze/reshape at the Tensor
    // view level. Norm reductions are intentionally excluded because norm1/norm2
    // over a singleton is abs(x), not x.
    built->identity_reduction = !built->key.output_indices && singletonReductionCanReuseInputValue(built->key.op) &&
                                !hasCudnnReductionDimension(input_dims, output_dims);
    if (built->identity_reduction) {
        builtReductionCache.put(key, built);
        return built;
    }

    built->a_desc = createCudnnTensorDescriptor(input_dims, built->key.input_dtype);
    built->reduce_desc = createCudnnReduceDescriptor(built->key.op, built->key.compute_dtype, built->key.output_indices);
    built->c_desc = createCudnnTensorDescriptor(output_dims, built->key.output_dtype);

    built->workspace_bytes = getReductionWorkspaceSize(device_num, built->reduce_desc, built->a_desc, built->c_desc);
    if (built->key.output_indices) {
        built->indices_bytes = getReductionIndicesSize(device_num, built->reduce_desc, built->a_desc, built->c_desc);
    }

    builtReductionCache.put(key, built);
    return built;
}

std::shared_ptr<BuiltSoftmax> StampedEquation::buildSoftmax(const std::shared_ptr<CompiledSoftmax>& compiled_softmax,
                                                            const Tensor& input,
                                                            const Tensor& output,
                                                            int device_num) {
    if (!compiled_softmax) {
        throw std::runtime_error("buildSoftmax requires compiled_softmax.");
    }
    if (input.getDimensions() != output.getDimensions()) {
        throw std::runtime_error("Softmax input and output dimensions must match.");
    }
    if (input.getDataType() != compiled_softmax->input_dtype) {
        throw std::runtime_error("Softmax input dtype does not match compiled input dtype.");
    }
    if (output.getDataType() != compiled_softmax->output_dtype) {
        throw std::runtime_error("Softmax output dtype does not match compiled output dtype.");
    }

    SoftmaxCacheKey key{
        .input_dims = input.getDimensions(),
        .input_dtype = compiled_softmax->input_dtype,
        .output_dtype = compiled_softmax->output_dtype,
        .algorithm = compiled_softmax->algorithm,
        .mode = compiled_softmax->mode,
        .device_num = device_num,
    };

    std::shared_ptr<BuiltSoftmax> hit = cacheLookup(key);
    if (hit)
        return hit;

    auto built = std::make_shared<BuiltSoftmax>(key);
    built->x_desc = createCudnnTensorDescriptor(input.getDimensions(), built->key.input_dtype);
    built->y_desc = createCudnnTensorDescriptor(output.getDimensions(), built->key.output_dtype);

    builtSoftmaxCache.put(key, built);
    return built;
}

bool StampedEquation::requiresRuntimeScalars() const {
    if (!compiledEquation) {
        return false;
    }

    for (size_t i = 0; i < compiledEquation->input_kinds.size(); ++i) {
        if (compiledEquation->input_kinds[i] == NamedInput::Kind::RuntimeScalarFp32) {
            return true;
        }
    }
    return false;
}

std::unordered_set<std::string> StampedEquation::runtimeScalarNames() const {
    std::unordered_set<std::string> names;
    if (!compiledEquation) {
        return names;
    }

    for (size_t i = 0; i < compiledEquation->input_names.size(); ++i) {
        if (compiledEquation->input_kinds[i] == NamedInput::Kind::RuntimeScalarFp32) {
            names.insert(compiledEquation->input_names[i]);
        }
    }
    return names;
}

}  // namespace ThorImplementation
