#include "Utilities/Expression/StampedEquation.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/Expression/EquationRunner.h"
#include "Utilities/Expression/FusedEquation.h"
#include "Utilities/Expression/MatmulScalarKernel.h"
#include "Utilities/Expression/ReduceMinMaxBackwardKernel.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.h"

#include <cudnn_frontend.h>

#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
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

static int64_t checkedDim(const std::vector<uint64_t>& dims, size_t idx, const char* tensor_name) {
    if (idx >= dims.size()) {
        throw std::runtime_error(std::string("Attention tensor '") + tensor_name + "' must have rank 4 [B,H,S,D].");
    }
    if (dims[idx] > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::runtime_error(std::string("Attention tensor '") + tensor_name + "' dimension exceeds int64_t range.");
    }
    return static_cast<int64_t>(dims[idx]);
}

}  // namespace

static void putFrontendTensorPointer(std::unordered_map<int64_t, void*>& pack, int64_t uid, const Tensor& tensor);
static void executeFrontendConvolutionGraph(const BuiltConvolution& built,
                                            const Stream& run_stream,
                                            std::unordered_map<int64_t, void*>& tensor_pack,
                                            const std::optional<Tensor>& workspace,
                                            const char* op_name);


CudnnRmsNormDescriptor CompiledRmsNorm::descriptorFor(const Tensor& inputTensor, const Tensor& scaleTensor, const Tensor& outputTensor) const {
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
    const std::vector<uint64_t> qDims = qTensor.getDimensions();
    const std::vector<uint64_t> kDims = kTensor.getDimensions();
    const std::vector<uint64_t> vDims = vTensor.getDimensions();
    const std::vector<uint64_t> oDims = oTensor.getDimensions();
    if (qDims.size() != 4 || kDims.size() != 4 || vDims.size() != 4 || oDims.size() != 4) {
        throw std::runtime_error("Thor cuDNN attention expression stage requires rank-4 tensors in logical [B,H,S,D] order.");
    }

    CudnnAttentionDescriptor descriptor;
    descriptor.q = AttentionTensorSpec::fromLayout(q_layout,
                                                   checkedDim(qDims, 0, "q"),
                                                   checkedDim(qDims, 1, "q"),
                                                   checkedDim(qDims, 2, "q"),
                                                   checkedDim(qDims, 3, "q"),
                                                   qTensor.getDataType());
    descriptor.k = AttentionTensorSpec::fromLayout(k_layout,
                                                   checkedDim(kDims, 0, "k"),
                                                   checkedDim(kDims, 1, "k"),
                                                   checkedDim(kDims, 2, "k"),
                                                   checkedDim(kDims, 3, "k"),
                                                   kTensor.getDataType());
    descriptor.v = AttentionTensorSpec::fromLayout(v_layout,
                                                   checkedDim(vDims, 0, "v"),
                                                   checkedDim(vDims, 1, "v"),
                                                   checkedDim(vDims, 2, "v"),
                                                   checkedDim(vDims, 3, "v"),
                                                   vTensor.getDataType());
    descriptor.o = AttentionTensorSpec::fromLayout(o_layout,
                                                   checkedDim(oDims, 0, "o"),
                                                   checkedDim(oDims, 1, "o"),
                                                   checkedDim(oDims, 2, "o"),
                                                   checkedDim(oDims, 3, "o"),
                                                   oTensor.getDataType());
    if (use_ragged_offsets) {
        descriptor.q.ragged = true;
        descriptor.k.ragged = true;
        descriptor.v.ragged = true;
        descriptor.o.ragged = true;
    }
    descriptor.computeDataType = compute_dtype;
    descriptor.intermediateDataType = TensorDescriptor::DataType::FP32;
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
        qTensor.getDataType() == TensorDescriptor::DataType::FP8_E4M3 || qTensor.getDataType() == TensorDescriptor::DataType::FP8_E5M2 ||
        kTensor.getDataType() == TensorDescriptor::DataType::FP8_E4M3 || kTensor.getDataType() == TensorDescriptor::DataType::FP8_E5M2 ||
        vTensor.getDataType() == TensorDescriptor::DataType::FP8_E4M3 || vTensor.getDataType() == TensorDescriptor::DataType::FP8_E5M2 ||
        oTensor.getDataType() == TensorDescriptor::DataType::FP8_E4M3 || oTensor.getDataType() == TensorDescriptor::DataType::FP8_E5M2;
    descriptor.validateForward();
    return descriptor;
}


CudnnAttentionDescriptor CompiledAttentionBackward::descriptorFor(const Tensor& qTensor,
                                                                  const Tensor& kTensor,
                                                                  const Tensor& vTensor,
                                                                  const Tensor& oTensor) const {
    CudnnAttentionDescriptor descriptor;
    const std::vector<uint64_t> qDims = qTensor.getDimensions();
    const std::vector<uint64_t> kDims = kTensor.getDimensions();
    const std::vector<uint64_t> vDims = vTensor.getDimensions();
    const std::vector<uint64_t> oDims = oTensor.getDimensions();
    if (qDims.size() != 4 || kDims.size() != 4 || vDims.size() != 4 || oDims.size() != 4) {
        throw std::runtime_error("Thor cuDNN attention-backward expression stage requires rank-4 tensors in logical [B,H,S,D] order.");
    }

    descriptor.q = AttentionTensorSpec::fromLayout(q_layout,
                                                   checkedDim(qDims, 0, "q"),
                                                   checkedDim(qDims, 1, "q"),
                                                   checkedDim(qDims, 2, "q"),
                                                   checkedDim(qDims, 3, "q"),
                                                   qTensor.getDataType());
    descriptor.k = AttentionTensorSpec::fromLayout(k_layout,
                                                   checkedDim(kDims, 0, "k"),
                                                   checkedDim(kDims, 1, "k"),
                                                   checkedDim(kDims, 2, "k"),
                                                   checkedDim(kDims, 3, "k"),
                                                   kTensor.getDataType());
    descriptor.v = AttentionTensorSpec::fromLayout(v_layout,
                                                   checkedDim(vDims, 0, "v"),
                                                   checkedDim(vDims, 1, "v"),
                                                   checkedDim(vDims, 2, "v"),
                                                   checkedDim(vDims, 3, "v"),
                                                   vTensor.getDataType());
    descriptor.o = AttentionTensorSpec::fromLayout(o_layout,
                                                   checkedDim(oDims, 0, "o"),
                                                   checkedDim(oDims, 1, "o"),
                                                   checkedDim(oDims, 2, "o"),
                                                   checkedDim(oDims, 3, "o"),
                                                   oTensor.getDataType());
    if (use_ragged_offsets) {
        descriptor.q.ragged = true;
        descriptor.k.ragged = true;
        descriptor.v.ragged = true;
        descriptor.o.ragged = true;
    }
    descriptor.computeDataType = compute_dtype;
    descriptor.intermediateDataType = TensorDescriptor::DataType::FP32;
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
        qTensor.getDataType() == TensorDescriptor::DataType::FP8_E4M3 || qTensor.getDataType() == TensorDescriptor::DataType::FP8_E5M2 ||
        kTensor.getDataType() == TensorDescriptor::DataType::FP8_E4M3 || kTensor.getDataType() == TensorDescriptor::DataType::FP8_E5M2 ||
        vTensor.getDataType() == TensorDescriptor::DataType::FP8_E4M3 || vTensor.getDataType() == TensorDescriptor::DataType::FP8_E5M2 ||
        oTensor.getDataType() == TensorDescriptor::DataType::FP8_E4M3 || oTensor.getDataType() == TensorDescriptor::DataType::FP8_E5M2;
    descriptor.validateBackward();
    return descriptor;
}

TensorDescriptor::DataType CompiledAttentionBackward::outputDTypeFor(ExprOp op) const {
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
                                    TensorDescriptor::DataType output_dtype) {
    return forward.q_layout == backward.q_layout && forward.k_layout == backward.k_layout &&
           forward.v_layout == backward.v_layout && forward.o_layout == backward.o_layout &&
           forward.mask_kind == backward.mask_kind && forward.diagonal_left_bound == backward.diagonal_left_bound &&
           forward.diagonal_right_bound == backward.diagonal_right_bound &&
           sameOptionalFloat(forward.attention_scale, backward.attention_scale) &&
           forward.use_alibi_mask == backward.use_alibi_mask && forward.use_bias == backward.use_bias &&
           forward.use_padding_mask == backward.use_padding_mask && forward.use_ragged_offsets == backward.use_ragged_offsets &&
           forward.use_paged_kv_cache == backward.use_paged_kv_cache &&
           forward.paged_kv_max_sequence_length == backward.paged_kv_max_sequence_length &&
           forward.dropout_probability == backward.dropout_probability &&
           forward.compute_dtype == backward.compute_dtype &&
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
    CudnnAttentionForwardArgs args{.q = q, .k = k, .v = v, .o = output};
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
                "Attention-backward expected same-plan retained cuDNN forward stats, but the matching forward stage did not populate them.");
        }
        if (forwardOutput.getDimensions() != dO.getDimensions() || forwardOutput.getDataType() != dO.getDataType() ||
            forwardOutput.getPlacement() != dO.getPlacement()) {
            throw std::runtime_error("Retained attention forward output is incompatible with attention-backward dO.");
        }
    }

    CudnnAttentionDescriptor descriptor = compiled_attention_backward->descriptorFor(q, k, v, forwardOutput);
    descriptor.generateStats = true;

    if (!use_saved_forward) {
        CudnnAttentionForwardArgs fwdArgs{.q = q, .k = k, .v = v, .o = oScratch, .stats = stats};
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

    CudnnAttentionBackwardArgs bwdArgs{
        .q = q, .k = k, .v = v, .o = forwardOutput, .dO = dO, .stats = forwardStats, .dQ = dQ, .dK = dK, .dV = dV};
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
        executeFrontendConvolutionGraph(*built_convolution,
                                        run_stream,
                                        tensor_pack,
                                        workspace,
                                        compiled_convolution->is_3d ? "CONV3D forward" : "CONV2D forward");
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
            executeFrontendConvolutionGraph(*built_convolution,
                                            run_stream,
                                            tensor_pack,
                                            workspace,
                                            compiled_convolution_backward->op == ExprOp::CONV3D_BACKWARD_DATA
                                                ? "CONV3D backward-data"
                                                : "CONV2D backward-data");
            return;
        }
        if (compiled_convolution_backward->op == ExprOp::CONV2D_BACKWARD_FILTER ||
            compiled_convolution_backward->op == ExprOp::CONV3D_BACKWARD_FILTER) {
            putFrontendTensorPointer(tensor_pack, CUDNN_FRONTEND_CONV_X_UID, input);
            putFrontendTensorPointer(tensor_pack, CUDNN_FRONTEND_CONV_Y_UID, grad_output);
            putFrontendTensorPointer(tensor_pack, CUDNN_FRONTEND_CONV_W_UID, output);
            executeFrontendConvolutionGraph(*built_convolution,
                                            run_stream,
                                            tensor_pack,
                                            workspace,
                                            compiled_convolution_backward->op == ExprOp::CONV3D_BACKWARD_FILTER
                                                ? "CONV3D backward-filter"
                                                : "CONV2D backward-filter");
            return;
        }
        throw std::runtime_error("StampedConvolutionBackward received unsupported cuDNN Frontend convolution backward op.");
    }

    throw std::runtime_error("StampedConvolutionBackward received non-frontend convolution payload unexpectedly.");
}


StampedRmsNorm::StampedRmsNorm(std::shared_ptr<CompiledRmsNorm> compiled,
                               const Tensor& input,
                               const Tensor& scale,
                               const Tensor& output,
                               const Stream& stream)
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
        if (binding.sourceDType != TensorDescriptor::DataType::FP32) {
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
        if (tensor.getDataType() == TensorDescriptor::DataType::FP32) {
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
    if (binding.sourceDType != TensorDescriptor::DataType::FP32) {
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
        if (tensor.getDataType() == TensorDescriptor::DataType::FP32 && resolved.host_value == 1.0f) {
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
                throw std::runtime_error("cuBLASLt MATMUL backward epilogue fusion currently supports only non-transposed row-major stages.");
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
                run_stream);
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
                                                                                    run_stream);
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

    const CublasMatrixMultiply::MatmulDataTypes dataTypes{lhs.getDescriptor().getDataType(),
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
            throw std::runtime_error("Stamped GEMM backward epilogue requires a rank-2 addend or no addend; rank-1 bias addends are forward epilogues.");
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
            resolved_scales.pointer_mode);
        return;
    }

    const bool use_cublaslt_epilogue_wrapper = use_bias_epilogue || compiled_matmul->epilogue != MatmulEpilogue::Default || use_backward_epilogue;
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
                                                                                    resolved_scales.pointer_mode);
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
            } else if (stage.kind == StampedExecutionStage::Kind::Matmul && stage.matmul != nullptr) {
                if (stage.matmul->alphaRuntimeName().has_value()) {
                    needed_names.insert(*stage.matmul->alphaRuntimeName());
                }
                if (stage.matmul->betaRuntimeName().has_value()) {
                    needed_names.insert(*stage.matmul->betaRuntimeName());
                }
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

static cudnnDataType_t toCudnnDataType(TensorDescriptor::DataType dtype) {
    switch (dtype) {
        case TensorDescriptor::DataType::FP32:
            return CUDNN_DATA_FLOAT;

        case TensorDescriptor::DataType::FP16:
            return CUDNN_DATA_HALF;

        case TensorDescriptor::DataType::BF16:
            return CUDNN_DATA_BFLOAT16;

        case TensorDescriptor::DataType::FP8_E4M3:
            return CUDNN_DATA_FP8_E4M3;

        case TensorDescriptor::DataType::FP8_E5M2:
            return CUDNN_DATA_FP8_E5M2;

        default:
            throw std::runtime_error("toCudnnDataType: unsupported TensorDescriptor::DataType value " +
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

static cudnnTensorDescriptor_t createCudnnTensorDescriptor(std::vector<uint64_t> dims, TensorDescriptor::DataType dtype) {
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

static fe::DataType_t toFrontendDataType(TensorDescriptor::DataType dtype) {
    switch (dtype) {
        case TensorDescriptor::DataType::FP32:
            return fe::DataType_t::FLOAT;
        case TensorDescriptor::DataType::FP16:
            return fe::DataType_t::HALF;
        case TensorDescriptor::DataType::BF16:
            return fe::DataType_t::BFLOAT16;
        case TensorDescriptor::DataType::FP8_E4M3:
            return fe::DataType_t::FP8_E4M3;
        case TensorDescriptor::DataType::FP8_E5M2:
            return fe::DataType_t::FP8_E5M2;
        case TensorDescriptor::DataType::INT32:
            return fe::DataType_t::INT32;
        case TensorDescriptor::DataType::INT64:
            return fe::DataType_t::INT64;
        default:
            throw std::runtime_error("Unsupported dtype for cuDNN Frontend convolution: " +
                                     TensorDescriptor::getElementTypeName(dtype));
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

static std::shared_ptr<fe::graph::Tensor_attributes> createFrontendConvolutionTensor(
    const std::shared_ptr<fe::graph::Graph>& graph,
    const std::string& name,
    int64_t uid,
    const std::vector<uint64_t>& dims,
    TensorDescriptor::DataType dtype) {
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
                                               TensorDescriptor::DataType dtype) {
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

static void buildFrontendConvolutionGraph(BuiltConvolution& built, const Stream& stream, const char* op_name) {
    if (!built.frontend_graph) {
        throw std::runtime_error(std::string(op_name) + " missing cuDNN Frontend graph.");
    }

    ScopedGpu scopedGpu(stream.getGpuNum());
    auto status = built.frontend_graph->build(stream.getCudnnHandle(), {fe::HeurMode_t::A});
    if (!status.is_good()) {
        throw std::runtime_error(std::string("Failed to build cuDNN Frontend ") + op_name + " graph: " + status.get_message());
    }

    int64_t workspace_bytes = 0;
    status = built.frontend_graph->get_workspace_size(workspace_bytes);
    if (!status.is_good()) {
        throw std::runtime_error(std::string("Failed to query cuDNN Frontend ") + op_name +
                                 " workspace size: " + status.get_message());
    }
    if (workspace_bytes < 0) {
        throw std::runtime_error(std::string("cuDNN Frontend ") + op_name + " returned a negative workspace size.");
    }
    built.workspace_bytes = static_cast<size_t>(workspace_bytes);
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

    auto status = built.frontend_graph->execute(run_stream.getCudnnHandle(), tensor_pack, workspace_ptr);
    if (!status.is_good()) {
        throw std::runtime_error(std::string("Failed to execute cuDNN Frontend ") + op_name + " graph: " + status.get_message());
    }
}

static cudnnReduceTensorDescriptor_t createCudnnReduceDescriptor(ExprOp op, TensorDescriptor::DataType compute_dtype, bool output_indices) {
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
        addend.has_value() ? (use_bias_epilogue ? output.getDescriptor().getDataType() : addend.value().getDescriptor().getDataType()) : output.getDescriptor().getDataType(),
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
    if (use_backward_epilogue && epilogue_aux.has_value() &&
        compiled_matmul->epilogue_aux_dtype.has_value() &&
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

    const bool use_cublaslt_epilogue_wrapper = use_bias_epilogue || compiled_matmul->epilogue != MatmulEpilogue::Default || use_backward_epilogue;

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
                                                                           dataTypes);
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
        // Fused cuBLASLt epilogues use the heuristic path directly for now.
        // They have no workspace requirement in this staged wrapper; the stage-kind
        // optimization still removes the separate expression FusedKernel.
        kernelWillRunOnGpu = true;
        built->workspace_bytes = 0;
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
                                                                 dataTypes);
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
                          .set_padding(convolutionFrontendPadding(is_3d,
                                                                  compiled_convolution->pad_d,
                                                                  compiled_convolution->pad_h,
                                                                  compiled_convolution->pad_w))
                          .set_stride(convolutionFrontendStrides(is_3d,
                                                                 compiled_convolution->stride_d,
                                                                 compiled_convolution->stride_h,
                                                                 compiled_convolution->stride_w))
                          .set_dilation(convolutionFrontendDilations(is_3d))
                          .set_compute_data_type(toFrontendDataType(compiled_convolution->compute_dtype))
                          .set_convolution_mode(fe::ConvolutionMode_t::CROSS_CORRELATION);

    auto y = built->frontend_graph->conv_fprop(x, w, conv_attrs);
    setFrontendConvolutionOutputTensor(y,
                                       std::string(prefix) + "_y",
                                       CUDNN_FRONTEND_CONV_Y_UID,
                                       output.getDimensions(),
                                       compiled_convolution->output_dtype);

    buildFrontendConvolutionGraph(*built, stream, is_3d ? "CONV3D forward" : "CONV2D forward");
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
    const auto padding = convolutionFrontendPadding(is_3d,
                                                    compiled_convolution_backward->pad_d,
                                                    compiled_convolution_backward->pad_h,
                                                    compiled_convolution_backward->pad_w);
    const auto strides = convolutionFrontendStrides(is_3d,
                                                    compiled_convolution_backward->stride_d,
                                                    compiled_convolution_backward->stride_h,
                                                    compiled_convolution_backward->stride_w);
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

        buildFrontendConvolutionGraph(*built, stream, is_3d ? "CONV3D backward-data" : "CONV2D backward-data");
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

    buildFrontendConvolutionGraph(*built, stream, is_3d ? "CONV3D backward-filter" : "CONV2D backward-filter");
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
                                                                TensorDescriptor::DataType input_dtype,
                                                                TensorDescriptor::DataType output_dtype,
                                                                TensorDescriptor::DataType compute_dtype,
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
