#include "Utilities/Expression/StampedEquation.h"
#include "Utilities/ComputeTopology/MachineEvaluator.h"
#include "Utilities/CudaDriver/CudaGraphConditional.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/Expression/BatchedMatmulPlan.h"
#include "Utilities/Expression/EquationRunner.h"
#include "Utilities/Expression/ExpressionDTypeResolution.h"
#include "Utilities/Expression/FusedEquation.h"
#include "Utilities/Expression/MatmulScalarKernel.h"
#include "Utilities/Expression/ReduceMinMaxBackwardKernel.h"
#include "Utilities/Expression/SegmentedBroadcastKernel.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/RaggedMatmulCapacityBuckets.h"
#include "Utilities/TensorOperations/Copy/StridedCopy.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"

#include <cudnn_frontend.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <tuple>
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
    const char* value = std::getenv("THOR_MATMUL_DIAGNOSTICS");
    return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}

static bool thorMatmulDiagnosticsVerbose() {
    const char* value = std::getenv("THOR_MATMUL_DIAGNOSTICS");
    if (value == nullptr) {
        return false;
    }
    const std::string mode(value);
    return mode == "2" || mode == "verbose" || mode == "VERBOSE" || mode == "full" || mode == "FULL";
}

static const char* matmulExprOpName(ExprOp op) {
    switch (op) {
        case ExprOp::MATMUL:
            return "MATMUL";
        case ExprOp::GEMM:
            return "GEMM";
        default:
            return "OTHER";
    }
}

static const char* matmulEpilogueName(MatmulEpilogue epilogue) {
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

static const char* matmulBackwardEpilogueName(MatmulBackwardEpilogue epilogue) {
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

static bool shouldPrintStampedMatmulDiagnosticOnce(const std::string& key) {
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

static CubScanDirection toCubScanDirection(bool reverse) { return reverse ? CubScanDirection::Reverse : CubScanDirection::Forward; }

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

static AttentionTensorSpec packedRaggedAttentionSpecForTensor(const Tensor& tensor,
                                                              AttentionTensorLayout layout,
                                                              uint64_t batchSize,
                                                              const char* tensor_name) {
    if (layout != AttentionTensorLayout::BSHD) {
        throw std::runtime_error(std::string("Canonical ragged attention tensor '") + tensor_name +
                                 "' requires BSHD/token-major layout.");
    }
    const std::vector<uint64_t> dims = tensor.getDimensions();
    if (dims.size() != 3 || batchSize == 0 || dims[0] == 0 || dims[1] == 0 || dims[2] == 0) {
        throw std::runtime_error(std::string("Canonical ragged attention tensor '") + tensor_name +
                                 "' must use packed [T,H,D] storage with a non-zero logical batch.");
    }
    AttentionTensorSpec spec;
    spec.dimensions = {static_cast<int64_t>(batchSize),
                       static_cast<int64_t>(dims[1]),
                       static_cast<int64_t>(dims[0]),
                       static_cast<int64_t>(dims[2])};
    const uint64_t elementsPerToken = dims[1] * dims[2];
    spec.strides = {static_cast<int64_t>(dims[0] * elementsPerToken),
                    static_cast<int64_t>(dims[2]),
                    static_cast<int64_t>(elementsPerToken),
                    1};
    spec.dataType = tensor.getDataType();
    spec.ragged = true;
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

DataType CompiledRmsNormBackward::outputDTypeFor(ExprOp op) const {
    switch (op) {
        case ExprOp::RMSNORM_BACKWARD_X:
            return dx_dtype;
        case ExprOp::RMSNORM_BACKWARD_SCALE:
            return dscale_dtype;
        default:
            throw std::runtime_error("CompiledRmsNormBackward::outputDTypeFor received a non-RMSNorm-backward op.");
    }
}

CudnnRmsNormDescriptor CompiledRmsNormBackward::descriptorFor(const Tensor& inputTensor,
                                                              const Tensor& scaleTensor,
                                                              const Tensor& dyTensor,
                                                              const Tensor& dxTensor,
                                                              const Tensor& dscaleTensor) const {
    const std::vector<uint64_t> inputDims = inputTensor.getDimensions();
    if (inputDims.size() != 2 || dyTensor.getDimensions() != inputDims || dxTensor.getDimensions() != inputDims) {
        throw std::runtime_error("Thor RMSNorm backward stage requires rank-2 x/dy/dx tensors with identical dimensions.");
    }
    if (scaleTensor.getDimensions() != std::vector<uint64_t>{normalized_feature_count} ||
        dscaleTensor.getDimensions() != std::vector<uint64_t>{normalized_feature_count}) {
        throw std::runtime_error("Thor RMSNorm backward stage requires rank-1 scale/dscale tensors matching hidden size.");
    }
    if (inputDims[1] != normalized_feature_count) {
        throw std::runtime_error("Thor RMSNorm backward hidden dimension does not match the compiled descriptor.");
    }
    if (inputTensor.getDataType() != input_dtype || scaleTensor.getDataType() != scale_dtype ||
        dyTensor.getDataType() != dy_dtype || dxTensor.getDataType() != dx_dtype || dscaleTensor.getDataType() != dscale_dtype) {
        throw std::runtime_error("Thor RMSNorm backward tensor dtype does not match the compiled descriptor.");
    }

    CudnnRmsNormDescriptor descriptor;
    descriptor.outerSize = inputDims[0];
    descriptor.normalizedFeatureCount = normalized_feature_count;
    descriptor.inputDataType = input_dtype;
    descriptor.parameterDataType = scale_dtype;
    descriptor.outputDataType = dy_dtype;
    descriptor.computeDataType = compute_dtype;
    descriptor.epsilon = static_cast<float>(epsilon);
    descriptor.training = true;
    descriptor.fusedActivation = CudnnRmsNormFusedActivation::NONE;
    descriptor.debugName = debug_name;
    descriptor.validateBackward();
    return descriptor;
}

CudnnAttentionDescriptor CompiledAttention::descriptorFor(const Tensor& qTensor,
                                                          const Tensor& kTensor,
                                                          const Tensor& vTensor,
                                                          const Tensor& oTensor,
                                                          uint64_t raggedBatchSize) const {
    CudnnAttentionDescriptor descriptor;
    const bool queryPackedRagged = use_ragged_offsets && qTensor.getDimensions().size() == 3;
    const bool kvPackedRagged = use_ragged_offsets && kTensor.getDimensions().size() == 3;
    // The original Expression ragged-attention API keeps packed token payloads in
    // rank-4 [B,S,H,D] storage and uses the supplied row partitions to define the
    // logical sequence starts/lengths.  High-level mixed Attention, added later,
    // uses rank-3 [T,H,D] storage for the genuinely ragged domain and a synthetic
    // row partition for the dense rank-4 domain.  Therefore rank only disambiguates
    // dense vs ragged when at least one sequence domain is rank 3.  If both domains
    // are rank 4 and ragged offsets were explicitly supplied, preserve the legacy
    // Expression contract and mark both domains ragged.
    const bool legacyRank4FullyRagged = use_ragged_offsets && !queryPackedRagged && !kvPackedRagged;
    const bool queryRagged = queryPackedRagged || legacyRank4FullyRagged;
    const bool kvRagged = kvPackedRagged || legacyRank4FullyRagged;
    if (use_ragged_offsets) {
        if ((vTensor.getDimensions().size() == 3) != kvPackedRagged ||
            (oTensor.getDimensions().size() == 3) != queryPackedRagged) {
            throw std::runtime_error("Attention mixed ragged storage requires Q/O and K/V to agree within each sequence domain.");
        }
        descriptor.q = queryPackedRagged ? packedRaggedAttentionSpecForTensor(qTensor, q_layout, raggedBatchSize, "q")
                                         : attentionSpecForTensor(qTensor, q_layout, "q");
        descriptor.o = queryPackedRagged ? packedRaggedAttentionSpecForTensor(oTensor, o_layout, raggedBatchSize, "o")
                                         : attentionSpecForTensor(oTensor, o_layout, "o");
        descriptor.k = kvPackedRagged ? packedRaggedAttentionSpecForTensor(kTensor, k_layout, raggedBatchSize, "k")
                                      : attentionSpecForTensor(kTensor, k_layout, "k");
        descriptor.v = kvPackedRagged ? packedRaggedAttentionSpecForTensor(vTensor, v_layout, raggedBatchSize, "v")
                                      : attentionSpecForTensor(vTensor, v_layout, "v");
        descriptor.q.ragged = queryRagged;
        descriptor.o.ragged = queryRagged;
        descriptor.k.ragged = kvRagged;
        descriptor.v.ragged = kvRagged;
    } else {
        descriptor.q = attentionSpecForTensor(qTensor, q_layout, "q");
        descriptor.k = attentionSpecForTensor(kTensor, k_layout, "k");
        descriptor.v = attentionSpecForTensor(vTensor, v_layout, "v");
        descriptor.o = attentionSpecForTensor(oTensor, o_layout, "o");
    }
    descriptor.computeDataType = compute_dtype;
    descriptor.intermediateDataType = DataType::FP32;
    descriptor.attentionScale = attention_scale;
    descriptor.maskKind = mask_kind;
    descriptor.diagonalLeftBound = diagonal_left_bound;
    descriptor.diagonalRightBound = diagonal_right_bound;
    descriptor.useAlibiMask = use_alibi_mask;
    descriptor.useBias = use_bias;
    descriptor.usePaddingMask = use_padding_mask || use_ragged_offsets;
    descriptor.usePagedKvCache = use_paged_kv_cache;
    descriptor.pagedKv.maxSequenceLengthKv = paged_kv_max_sequence_length;
    descriptor.dropout.probability = dropout_probability;
    descriptor.dropout.usePhilox = true;
    descriptor.debugName = debug_name;
    descriptor.useFp8 = qTensor.getDataType() == DataType::FP8_E4M3 || qTensor.getDataType() == DataType::FP8_E5M2 ||
                        kTensor.getDataType() == DataType::FP8_E4M3 || kTensor.getDataType() == DataType::FP8_E5M2 ||
                        vTensor.getDataType() == DataType::FP8_E4M3 || vTensor.getDataType() == DataType::FP8_E5M2 ||
                        oTensor.getDataType() == DataType::FP8_E4M3 || oTensor.getDataType() == DataType::FP8_E5M2;
    descriptor.validateForward();
    return descriptor;
}

CudnnAttentionDescriptor CompiledAttentionBackward::descriptorFor(const Tensor& qTensor,
                                                                  const Tensor& kTensor,
                                                                  const Tensor& vTensor,
                                                                  const Tensor& oTensor,
                                                                  uint64_t raggedBatchSize) const {
    CudnnAttentionDescriptor descriptor;
    const bool queryPackedRagged = use_ragged_offsets && qTensor.getDimensions().size() == 3;
    const bool kvPackedRagged = use_ragged_offsets && kTensor.getDimensions().size() == 3;
    // The original Expression ragged-attention API keeps packed token payloads in
    // rank-4 [B,S,H,D] storage and uses the supplied row partitions to define the
    // logical sequence starts/lengths.  High-level mixed Attention, added later,
    // uses rank-3 [T,H,D] storage for the genuinely ragged domain and a synthetic
    // row partition for the dense rank-4 domain.  Therefore rank only disambiguates
    // dense vs ragged when at least one sequence domain is rank 3.  If both domains
    // are rank 4 and ragged offsets were explicitly supplied, preserve the legacy
    // Expression contract and mark both domains ragged.
    const bool legacyRank4FullyRagged = use_ragged_offsets && !queryPackedRagged && !kvPackedRagged;
    const bool queryRagged = queryPackedRagged || legacyRank4FullyRagged;
    const bool kvRagged = kvPackedRagged || legacyRank4FullyRagged;
    if (use_ragged_offsets) {
        if ((vTensor.getDimensions().size() == 3) != kvPackedRagged ||
            (oTensor.getDimensions().size() == 3) != queryPackedRagged) {
            throw std::runtime_error("Attention mixed ragged storage requires Q/O and K/V to agree within each sequence domain.");
        }
        descriptor.q = queryPackedRagged ? packedRaggedAttentionSpecForTensor(qTensor, q_layout, raggedBatchSize, "q")
                                         : attentionSpecForTensor(qTensor, q_layout, "q");
        descriptor.o = queryPackedRagged ? packedRaggedAttentionSpecForTensor(oTensor, o_layout, raggedBatchSize, "o")
                                         : attentionSpecForTensor(oTensor, o_layout, "o");
        descriptor.k = kvPackedRagged ? packedRaggedAttentionSpecForTensor(kTensor, k_layout, raggedBatchSize, "k")
                                      : attentionSpecForTensor(kTensor, k_layout, "k");
        descriptor.v = kvPackedRagged ? packedRaggedAttentionSpecForTensor(vTensor, v_layout, raggedBatchSize, "v")
                                      : attentionSpecForTensor(vTensor, v_layout, "v");
        descriptor.q.ragged = queryRagged;
        descriptor.o.ragged = queryRagged;
        descriptor.k.ragged = kvRagged;
        descriptor.v.ragged = kvRagged;
    } else {
        descriptor.q = attentionSpecForTensor(qTensor, q_layout, "q");
        descriptor.k = attentionSpecForTensor(kTensor, k_layout, "k");
        descriptor.v = attentionSpecForTensor(vTensor, v_layout, "v");
        descriptor.o = attentionSpecForTensor(oTensor, o_layout, "o");
    }
    descriptor.computeDataType = compute_dtype;
    descriptor.intermediateDataType = DataType::FP32;
    descriptor.attentionScale = attention_scale;
    descriptor.maskKind = mask_kind;
    descriptor.diagonalLeftBound = diagonal_left_bound;
    descriptor.diagonalRightBound = diagonal_right_bound;
    descriptor.useAlibiMask = use_alibi_mask;
    descriptor.useBias = use_bias;
    descriptor.usePaddingMask = use_padding_mask || use_ragged_offsets;
    descriptor.usePagedKvCache = use_paged_kv_cache;
    descriptor.pagedKv.maxSequenceLengthKv = paged_kv_max_sequence_length;
    descriptor.dropout.probability = dropout_probability;
    descriptor.dropout.usePhilox = true;
    descriptor.generateStats = true;
    descriptor.deterministicBackward = deterministic_backward;
    descriptor.debugName = debug_name;
    descriptor.useFp8 = qTensor.getDataType() == DataType::FP8_E4M3 || qTensor.getDataType() == DataType::FP8_E5M2 ||
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

bool attentionConfigMatchesBackward(const CompiledAttention& forward, const CompiledAttentionBackward& backward, DataType output_dtype) {
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

    uint64_t raggedBatchSize = 0;
    if (compiled_attention->use_ragged_offsets) {
        if (!q_ragged_offsets.has_value() || !kv_ragged_offsets.has_value()) {
            throw std::runtime_error("StampedAttention requires canonical q/kv row partitions for ragged attention.");
        }
        const auto qOffsetDims = q_ragged_offsets->getDimensions();
        const auto kvOffsetDims = kv_ragged_offsets->getDimensions();
        if (qOffsetDims.size() != 1 || qOffsetDims.size() != kvOffsetDims.size() || qOffsetDims != kvOffsetDims || qOffsetDims[0] < 2) {
            throw std::runtime_error("StampedAttention ragged q/kv row partitions must both have shape [B+1].");
        }
        raggedBatchSize = qOffsetDims[0] - 1;
    }
    CudnnAttentionDescriptor descriptor = compiled_attention->descriptorFor(q, k, v, output, raggedBatchSize);
    const bool queryPackedRagged = compiled_attention->use_ragged_offsets && q.getDimensions().size() == 3;
    const bool kvPackedRagged = compiled_attention->use_ragged_offsets && k.getDimensions().size() == 3;
    Tensor cudnnQ = queryPackedRagged ? q : cudnnSemanticTensorView(q, compiled_attention->q_layout, "q");
    Tensor cudnnK = kvPackedRagged ? k : cudnnSemanticTensorView(k, compiled_attention->k_layout, "k");
    Tensor cudnnV = kvPackedRagged ? v : cudnnSemanticTensorView(v, compiled_attention->v_layout, "v");
    Tensor cudnnO = queryPackedRagged ? output : cudnnSemanticTensorView(output, compiled_attention->o_layout, "o");
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
        if (!q_ragged_offsets.has_value() || !kv_ragged_offsets.has_value() || !ragged_scratch.has_value()) {
            throw std::runtime_error("StampedAttention requires canonical q/kv row partitions and ragged metadata scratch.");
        }
        args.qRowPartitionOffsets = q_ragged_offsets.value();
        args.kvRowPartitionOffsets = kv_ragged_offsets.value();
        args.raggedScratch = ragged_scratch.value();
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
                                   const std::optional<CudnnRaggedAttentionScratch>& ragged_scratch,
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
      ragged_scratch(ragged_scratch),
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

    uint64_t raggedBatchSize = 0;
    if (compiled_attention_backward->use_ragged_offsets) {
        if (!q_ragged_offsets.has_value() || !kv_ragged_offsets.has_value()) {
            throw std::runtime_error("StampedAttentionBackward requires canonical q/kv row partitions for ragged attention.");
        }
        const auto qOffsetDims = q_ragged_offsets->getDimensions();
        const auto kvOffsetDims = kv_ragged_offsets->getDimensions();
        if (qOffsetDims.size() != 1 || qOffsetDims != kvOffsetDims || qOffsetDims[0] < 2) {
            throw std::runtime_error("StampedAttentionBackward ragged q/kv row partitions must both have shape [B+1].");
        }
        raggedBatchSize = qOffsetDims[0] - 1;
    }
    CudnnAttentionDescriptor descriptor = compiled_attention_backward->descriptorFor(q, k, v, forwardOutput, raggedBatchSize);
    descriptor.generateStats = true;

    const bool queryPackedRagged = compiled_attention_backward->use_ragged_offsets && q.getDimensions().size() == 3;
    const bool kvPackedRagged = compiled_attention_backward->use_ragged_offsets && k.getDimensions().size() == 3;
    Tensor cudnnQ = queryPackedRagged ? q : cudnnSemanticTensorView(q, compiled_attention_backward->q_layout, "q");
    Tensor cudnnK = kvPackedRagged ? k : cudnnSemanticTensorView(k, compiled_attention_backward->k_layout, "k");
    Tensor cudnnV = kvPackedRagged ? v : cudnnSemanticTensorView(v, compiled_attention_backward->v_layout, "v");
    Tensor cudnnO = queryPackedRagged ? forwardOutput : cudnnSemanticTensorView(forwardOutput, compiled_attention_backward->o_layout, "o");
    Tensor cudnnDO = queryPackedRagged ? dO : cudnnSemanticTensorView(dO, compiled_attention_backward->o_layout, "dO");
    Tensor cudnnDQ = queryPackedRagged ? dQ : cudnnSemanticTensorView(dQ, compiled_attention_backward->q_layout, "dQ");
    Tensor cudnnDK = kvPackedRagged ? dK : cudnnSemanticTensorView(dK, compiled_attention_backward->k_layout, "dK");
    Tensor cudnnDV = kvPackedRagged ? dV : cudnnSemanticTensorView(dV, compiled_attention_backward->v_layout, "dV");

    if (!use_saved_forward) {
        Tensor cudnnOScratch = queryPackedRagged ? oScratch : cudnnSemanticTensorView(oScratch, compiled_attention_backward->o_layout, "oScratch");
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
            if (!q_ragged_offsets.has_value() || !kv_ragged_offsets.has_value() || !ragged_scratch.has_value()) {
                throw std::runtime_error(
                    "StampedAttentionBackward requires canonical q/kv row partitions and ragged metadata scratch.");
            }
            fwdArgs.qRowPartitionOffsets = q_ragged_offsets.value();
            fwdArgs.kvRowPartitionOffsets = kv_ragged_offsets.value();
            fwdArgs.raggedScratch = ragged_scratch.value();
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
        if (!q_ragged_offsets.has_value() || !kv_ragged_offsets.has_value() || !ragged_scratch.has_value()) {
            throw std::runtime_error(
                "StampedAttentionBackward requires canonical q/kv row partitions and ragged metadata scratch.");
        }
        bwdArgs.qRowPartitionOffsets = q_ragged_offsets.value();
        bwdArgs.kvRowPartitionOffsets = kv_ragged_offsets.value();
        bwdArgs.raggedScratch = ragged_scratch.value();
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
                                                   const std::optional<CudnnRaggedAttentionScratch>& ragged_scratch,
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
      ragged_scratch(ragged_scratch),
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

    for (size_t i = 0; i < inputNames.size(); ++i) {
        if (compiledEquation->input_kinds[i] != NamedInput::Kind::RuntimeScalarFp32) {
            continue;
        }

        const std::string& name = inputNames[i];
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

static void refreshCudnnSoftmaxInputAdapter(const Tensor& source_input, Tensor& input, Stream& run_stream) {
    if (source_input.getPlacement() != input.getPlacement() || source_input.getDimensions() != input.getDimensions()) {
        throw std::runtime_error("cuDNN softmax input adapter must preserve the source tensor placement and dimensions.");
    }

    if (source_input.getTensorId() == input.getTensorId()) {
        if (source_input.getDataType() != input.getDataType()) {
            throw std::runtime_error("Aliased cuDNN softmax input cannot have mismatched source and operation dtypes.");
        }
        return;
    }

    input.copyFromAsync(source_input, run_stream);
}

StampedReduction::StampedReduction(std::shared_ptr<BuiltReduction> built,
                                   const Tensor& input,
                                   const Tensor& output,
                                   const Stream& stream)
    : built_reduction(std::move(built)), output(output), stream(stream) {
    THOR_THROW_IF_FALSE(built_reduction->key.result_kind == ReductionResultKind::Value);
    THOR_THROW_IF_FALSE(built_reduction->value_op.has_value());
    THOR_THROW_IF_FALSE(built_reduction->geometry.has_value());
    THOR_THROW_IF_FALSE(input.getDataType() == built_reduction->key.input_dtype);
    THOR_THROW_IF_FALSE(output.getDataType() == built_reduction->key.output_dtype);

    std::vector<uint32_t> axes;
    axes.reserve(built_reduction->key.reduction_axes.size());
    for (uint64_t axis : built_reduction->key.reduction_axes) {
        THOR_THROW_IF_FALSE(axis <= UINT32_MAX);
        axes.push_back(static_cast<uint32_t>(axis));
    }
    cub_reduction = CubReduction(built_reduction->value_op.value(), std::move(axes), output.getDataType())
                        .stamp(input, output, stream);
    THOR_THROW_IF_FALSE(cub_reduction->getGeometry().path == built_reduction->geometry->path);
}

void StampedReduction::run() { runOn(stream); }

void StampedReduction::runOn(Stream& run_stream) const {
    THOR_THROW_IF_FALSE(cub_reduction != nullptr);
    cub_reduction->runOn(run_stream);
}

StampedArgMinMax::StampedArgMinMax(std::shared_ptr<BuiltReduction> built,
                                   const Tensor& input,
                                   const Tensor& output,
                                   const Stream& stream)
    : built_reduction(std::move(built)), output(output), stream(stream) {
    if (built_reduction->key.result_kind != ReductionResultKind::Indices || !built_reduction->arg_op.has_value()
        || !built_reduction->geometry.has_value()) {
        throw std::runtime_error("StampedArgMinMax requires an index-producing reduction plan.");
    }

    std::vector<uint32_t> axes;
    axes.reserve(built_reduction->key.reduction_axes.size());
    for (uint64_t axis : built_reduction->key.reduction_axes) {
        THOR_THROW_IF_FALSE(axis <= UINT32_MAX);
        axes.push_back(static_cast<uint32_t>(axis));
    }

    CubArgReductionOutputOptions outputs;
    outputs.produce_value = false;
    outputs.produce_index = true;
    outputs.index_output_dtype = output.getDataType();
    cub_arg_reduction = CubArgReduction(built_reduction->arg_op.value(), std::move(axes), outputs)
                            .stamp(input, std::nullopt, output, stream);
    THOR_THROW_IF_FALSE(cub_arg_reduction->getGeometry().path == built_reduction->geometry->path);
}

void StampedArgMinMax::run() { runOn(stream); }

void StampedArgMinMax::runOn(Stream& run_stream) const {
    THOR_THROW_IF_FALSE(cub_arg_reduction != nullptr);
    cub_arg_reduction->runOn(run_stream);
}

StampedSegmentedReduction::StampedSegmentedReduction(std::shared_ptr<CompiledSegmentedReduction> compiled,
                                                     const Tensor& input,
                                                     const Tensor& output,
                                                     const Tensor& segment_offsets,
                                                     const Stream& stream)
    : compiled_segmented_reduction(std::move(compiled)), input(input), segment_offsets(segment_offsets), output(output), stream(stream) {
    if (!compiled_segmented_reduction) {
        throw std::runtime_error("StampedSegmentedReduction requires a compiled segmented reduction descriptor.");
    }
    if (input.getDataType() != compiled_segmented_reduction->input_dtype
        || output.getDataType() != compiled_segmented_reduction->output_dtype
        || segment_offsets.getDataType() != compiled_segmented_reduction->offset_dtype) {
        throw std::runtime_error("Segmented-reduction tensor dtypes do not match the compiled descriptor.");
    }

    CubReductionOp cub_op;
    switch (compiled_segmented_reduction->op) {
        case ExprOp::SEGMENTED_REDUCE_SUM:
            cub_op = CubReductionOp::Sum;
            break;
        case ExprOp::SEGMENTED_REDUCE_MIN:
            cub_op = CubReductionOp::Min;
            break;
        case ExprOp::SEGMENTED_REDUCE_MAX:
            cub_op = CubReductionOp::Max;
            break;
        case ExprOp::SEGMENTED_REDUCE_MEAN:
            cub_op = CubReductionOp::Mean;
            break;
        default:
            throw std::runtime_error("Unsupported segmented-reduction op.");
    }

    if (input.getDimensions().empty() || input.getDimensions()[0] == 0 ||
        input.getTotalNumElements() % input.getDimensions()[0] != 0 ||
        input.getTotalNumElements() / input.getDimensions()[0] != compiled_segmented_reduction->elements_per_value) {
        throw std::runtime_error("Segmented-reduction elements-per-value metadata does not match the input tensor shape.");
    }

    cub_segmented_reduction = CubSegmentedReduction(cub_op, output.getDataType())
                                  .stampRuntimeOffsets(input, output, segment_offsets, stream);
    THOR_THROW_IF_FALSE(cub_segmented_reduction->getPath() == CubReductionPath::OffsetSegmented);
}

void StampedSegmentedReduction::run() { runOn(stream); }

void StampedSegmentedReduction::runOn(Stream& run_stream) const {
    THOR_THROW_IF_FALSE(cub_segmented_reduction != nullptr);
    cub_segmented_reduction->runOn(run_stream);
}

StampedSegmentedBroadcast::StampedSegmentedBroadcast(std::shared_ptr<CompiledSegmentedBroadcast> compiled,
                                                     const Tensor& per_segment_values,
                                                     const Tensor& segment_offsets,
                                                     const Tensor& output,
                                                     const Stream& stream)
    : compiled_segmented_broadcast(std::move(compiled)),
      per_segment_values(per_segment_values),
      segment_offsets(segment_offsets),
      output(output),
      stream(stream) {
    if (!compiled_segmented_broadcast) {
        throw std::runtime_error("StampedSegmentedBroadcast requires a compiled descriptor.");
    }
    if (per_segment_values.getDataType() != compiled_segmented_broadcast->input_dtype ||
        output.getDataType() != compiled_segmented_broadcast->output_dtype ||
        segment_offsets.getDataType() != compiled_segmented_broadcast->offset_dtype) {
        throw std::runtime_error("StampedSegmentedBroadcast tensor dtypes do not match the compiled descriptor.");
    }
    if (per_segment_values.getDimensions().empty()) {
        throw std::runtime_error("StampedSegmentedBroadcast requires per-segment values with rank >= 1.");
    }
    std::vector<uint64_t> expected_output_dims = per_segment_values.getDimensions();
    expected_output_dims[0] = compiled_segmented_broadcast->max_output_values;
    if (output.getDimensions() != expected_output_dims) {
        throw std::runtime_error("StampedSegmentedBroadcast output shape does not match [max_output_values,D...].");
    }
    if (per_segment_values.getDimensions()[0] == 0 || compiled_segmented_broadcast->max_output_values == 0 ||
        per_segment_values.getTotalNumElements() % per_segment_values.getDimensions()[0] != 0 ||
        per_segment_values.getTotalNumElements() / per_segment_values.getDimensions()[0] !=
            compiled_segmented_broadcast->elements_per_value ||
        output.getTotalNumElements() % compiled_segmented_broadcast->max_output_values != 0 ||
        output.getTotalNumElements() / compiled_segmented_broadcast->max_output_values !=
            compiled_segmented_broadcast->elements_per_value) {
        throw std::runtime_error("StampedSegmentedBroadcast elements-per-value metadata does not match tensor shapes.");
    }
}

void StampedSegmentedBroadcast::run() { runOn(stream); }

void StampedSegmentedBroadcast::runOn(Stream& run_stream) const {
    launchSegmentedBroadcast(per_segment_values,
                             segment_offsets,
                             output,
                             compiled_segmented_broadcast->normalize_by_segment_length,
                             run_stream);
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
                arg_scan_plan =
                    prepareCubDeviceArgScan(input, output, num_items, toCubArgScanOp(compiled_scan->op), cub_mode, cub_direction);
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
                               const Tensor& source_input,
                               const Tensor& input,
                               const Tensor& output,
                               const Stream& stream)
    : compiled_softmax(std::move(compiled)),
      built_softmax(std::move(built)),
      source_input(source_input),
      input(input),
      output(output),
      stream(stream) {
    if (!compiled_softmax || !built_softmax) {
        throw std::runtime_error("StampedSoftmax requires compiled and built softmax payloads.");
    }
    THOR_THROW_IF_FALSE(input.getDataType() == built_softmax->key.input_dtype);
    THOR_THROW_IF_FALSE(output.getDataType() == built_softmax->key.output_dtype);
}

void StampedSoftmax::run() { runOn(stream); }

void StampedSoftmax::runOn(Stream& run_stream) const {
    refreshCudnnSoftmaxInputAdapter(source_input, input, run_stream);

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

// Consumer-side pre-read sanitation for bucketed cuDNN RMSNorm. cuDNN is
// deliberately dispatched over selected_rows, so only [active_rows, selected_rows)
// must be made safe. Storage in [selected_rows, full_capacity_rows) is outside this
// consumer's physical read and remains undefined.
static void sanitizePackedRmsNormOverreadRows(Tensor tensor,
                                                uint64_t active_rows,
                                                uint64_t selected_rows,
                                                uint64_t full_capacity_rows,
                                                uint64_t outer_per_packed_row,
                                                Stream& stream) {
    if (selected_rows > full_capacity_rows || active_rows > selected_rows || outer_per_packed_row == 0) {
        throw std::runtime_error("Packed-row RMSNorm selected extent is incompatible with its logical active extent.");
    }
    if (active_rows == selected_rows) {
        return;
    }
    const std::vector<uint64_t> dims = tensor.getDimensions();
    if (dims.size() != 2 || dims[0] != full_capacity_rows * outer_per_packed_row || dims[1] == 0) {
        throw std::runtime_error("Packed-row RMSNorm operand must be a contiguous rank-2 full-capacity tensor.");
    }
    const uint64_t hidden = dims[1];
    const uint64_t active_outer = active_rows * outer_per_packed_row;
    const uint64_t selected_outer = selected_rows * outer_per_packed_row;
    Tensor overread = tensor.aliasView({selected_outer - active_outer, hidden},
                                       {hidden, 1},
                                       active_outer * hidden);
    overread.memsetAsync(stream, 0);
}

StampedRmsNorm::StampedRmsNorm(std::shared_ptr<CompiledRmsNorm> compiled,
                               const Tensor& input,
                               const Tensor& scale,
                               const Tensor& output,
                               const Stream& stream,
                               std::optional<Tensor> row_partition_offsets,
                               std::shared_ptr<RmsNormForwardState> forward_state)
    : compiled_rms_norm(std::move(compiled)),
      input(input),
      scale(scale),
      row_partition_offsets(std::move(row_partition_offsets)),
      output(output),
      stream(stream),
      forward_state(forward_state ? std::move(forward_state) : std::make_shared<RmsNormForwardState>()) {
    if (!compiled_rms_norm) {
        throw std::runtime_error("StampedRmsNorm requires a compiled RMSNorm payload.");
    }

    if (compiled_rms_norm->packed_row_capacity != 0) {
        if (!this->row_partition_offsets.has_value() || compiled_rms_norm->ragged_batch_size == 0) {
            throw std::runtime_error("Packed-row RMSNorm requires an explicit row-partition runtime binding.");
        }
        const TensorDescriptor offsets_descriptor = this->row_partition_offsets->getDescriptor();
        RowPartitionRuntime(this->row_partition_offsets.value(),
                            RowPartitionDescriptor(compiled_rms_norm->ragged_batch_size,
                                                   compiled_rms_norm->packed_row_capacity,
                                                   offsets_descriptor.getDataType()));
        const std::vector<uint64_t> input_dims = input.getDimensions();
        if (input_dims.size() != 2 || input_dims[0] % compiled_rms_norm->packed_row_capacity != 0) {
            throw std::runtime_error(
                "Packed-row RMSNorm requires logical [outer, hidden] input whose outer dimension is divisible by packed_row_capacity.");
        }
        const uint64_t outer_per_packed_row = input_dims[0] / compiled_rms_norm->packed_row_capacity;
        if (outer_per_packed_row == 0) {
            throw std::runtime_error("Packed-row RMSNorm outer samples per packed row must be non-zero.");
        }

        // Pre-build the same finite row-capacity family used by ragged FC so
        // runtime active-row changes never force a new cuDNN graph build.
        for (const uint64_t capacity_rows : makeRaggedMatmulCapacityBuckets(compiled_rms_norm->packed_row_capacity)) {
            const uint64_t bucket_outer = capacity_rows * outer_per_packed_row;
            Tensor bucket_input = input.aliasView({bucket_outer, input_dims[1]}, {input_dims[1], 1}, 0);
            Tensor bucket_output = output.aliasView({bucket_outer, input_dims[1]}, {input_dims[1], 1}, 0);
            const CudnnRmsNormDescriptor descriptor = compiled_rms_norm->descriptorFor(bucket_input, scale, bucket_output);
            CudnnRmsNorm::instance().warmForward(descriptor, stream.getGpuNum());
        }
    }
}

void StampedRmsNorm::run() { runOn(stream); }

void StampedRmsNorm::runOn(Stream& run_stream) const {
    if (compiled_rms_norm->packed_row_capacity == 0) {
        CudnnRmsNormDescriptor descriptor = compiled_rms_norm->descriptorFor(input, scale, output);
        CudnnRmsNormForwardArgs args;
        args.x = input;
        args.scale = scale;
        args.y = output;
        if (forward_state != nullptr && forward_state->retain_for_backward) {
            descriptor.training = true;
            if (!forward_state->inv_variance.isInitialized()) {
                forward_state->inv_variance =
                    Tensor(input.getPlacement(), TensorDescriptor(DataType::FP32, {input.getDimensions().at(0)}));
            }
            args.invVariance = forward_state->inv_variance;
            forward_state->has_valid_state = false;
        }
        CudnnRmsNorm::instance().forward(descriptor, args, run_stream);
        if (descriptor.training) {
            forward_state->has_valid_state = true;
            forward_state->packed_active_rows = 0;
            forward_state->packed_selected_rows = 0;
        }
        return;
    }

    if (!row_partition_offsets.has_value()) {
        throw std::runtime_error("Packed-row RMSNorm is missing its row-partition runtime binding.");
    }
    const TensorDescriptor offsets_descriptor = row_partition_offsets->getDescriptor();
    RowPartitionRuntime row_partition(row_partition_offsets.value(),
                                      RowPartitionDescriptor(compiled_rms_norm->ragged_batch_size,
                                                             compiled_rms_norm->packed_row_capacity,
                                                             offsets_descriptor.getDataType()));
    const uint64_t active_rows = row_partition.requireHostActiveValueCount();

    const std::vector<uint64_t> input_dims = input.getDimensions();
    const uint64_t hidden = input_dims[1];
    const uint64_t outer_per_packed_row = input_dims[0] / compiled_rms_norm->packed_row_capacity;
    const std::vector<uint64_t> buckets = makeRaggedMatmulCapacityBuckets(compiled_rms_norm->packed_row_capacity);
    const uint64_t selected_rows =
        active_rows == 0 ? buckets.front() : chooseRaggedMatmulCapacityBucket(active_rows, buckets);
    const uint64_t selected_outer = selected_rows * outer_per_packed_row;

    // cuDNN deliberately executes the selected physical bucket. Sanitize exactly
    // the rows that this consumer will over-read and leave all later capacity undefined.
    Tensor mutable_input = input;
    sanitizePackedRmsNormOverreadRows(mutable_input,
                                      active_rows,
                                      selected_rows,
                                      compiled_rms_norm->packed_row_capacity,
                                      outer_per_packed_row,
                                      run_stream);

    Tensor bucket_input = input.aliasView({selected_outer, hidden}, {hidden, 1}, 0);
    Tensor bucket_output = output.aliasView({selected_outer, hidden}, {hidden, 1}, 0);
    CudnnRmsNormDescriptor descriptor = compiled_rms_norm->descriptorFor(bucket_input, scale, bucket_output);
    CudnnRmsNormForwardArgs args;
    args.x = bucket_input;
    args.scale = scale;
    args.y = bucket_output;
    if (forward_state != nullptr && forward_state->retain_for_backward) {
        descriptor.training = true;
        if (!forward_state->inv_variance.isInitialized()) {
            forward_state->inv_variance =
                Tensor(input.getPlacement(), TensorDescriptor(DataType::FP32, {input_dims[0]}));
        }
        args.invVariance = forward_state->inv_variance.aliasView({selected_outer}, {1}, 0);
        forward_state->has_valid_state = false;
    }
    CudnnRmsNorm::instance().forward(descriptor, args, run_stream);
    if (descriptor.training) {
        forward_state->packed_active_rows = active_rows;
        forward_state->packed_selected_rows = selected_rows;
        forward_state->has_valid_state = true;
    }
}

void StampedRmsNorm::retainForwardStateForBackward() {
    if (compiled_rms_norm->fused_activation != CudnnRmsNormFusedActivation::NONE) {
        throw std::runtime_error("RMSNorm fused activation backward is not supported.");
    }
    if (!forward_state->retain_for_backward) {
        // A state produced before retention was requested would have come from an
        // inference forward and therefore cannot contain training invVariance.
        forward_state->has_valid_state = false;
        if (compiled_rms_norm->packed_row_capacity != 0) {
            const std::vector<uint64_t> input_dims = input.getDimensions();
            const uint64_t outer_per_packed_row = input_dims[0] / compiled_rms_norm->packed_row_capacity;
            for (const uint64_t capacity_rows : makeRaggedMatmulCapacityBuckets(compiled_rms_norm->packed_row_capacity)) {
                const uint64_t bucket_outer = capacity_rows * outer_per_packed_row;
                Tensor bucket_input = input.aliasView({bucket_outer, input_dims[1]}, {input_dims[1], 1}, 0);
                Tensor bucket_output = output.aliasView({bucket_outer, input_dims[1]}, {input_dims[1], 1}, 0);
                CudnnRmsNormDescriptor descriptor = compiled_rms_norm->descriptorFor(bucket_input, scale, bucket_output);
                descriptor.training = true;
                CudnnRmsNorm::instance().warmForward(descriptor, stream.getGpuNum());
            }
        }
    }
    forward_state->retain_for_backward = true;
    if (!forward_state->inv_variance.isInitialized()) {
        forward_state->inv_variance =
            Tensor(input.getPlacement(), TensorDescriptor(DataType::FP32, {input.getDimensions().at(0)}));
    }
}

std::optional<PackedRowConsumerDiagnostic> StampedRmsNorm::packedRowConsumerDiagnostic() const {
    if (compiled_rms_norm->packed_row_capacity == 0) {
        return std::nullopt;
    }
    if (!row_partition_offsets.has_value()) {
        throw std::runtime_error("Packed-row RMSNorm diagnostic is missing its row-partition runtime binding.");
    }
    const TensorDescriptor offsets_descriptor = row_partition_offsets->getDescriptor();
    RowPartitionRuntime row_partition(row_partition_offsets.value(),
                                      RowPartitionDescriptor(compiled_rms_norm->ragged_batch_size,
                                                             compiled_rms_norm->packed_row_capacity,
                                                             offsets_descriptor.getDataType()));
    const uint64_t active_rows = row_partition.requireHostActiveValueCount();
    const std::vector<uint64_t> buckets = makeRaggedMatmulCapacityBuckets(compiled_rms_norm->packed_row_capacity);
    const uint64_t selected_rows =
        active_rows == 0 ? buckets.front() : chooseRaggedMatmulCapacityBucket(active_rows, buckets);
    if (selected_rows < active_rows || selected_rows > compiled_rms_norm->packed_row_capacity) {
        throw std::runtime_error("Packed-row RMSNorm diagnostic selected an invalid physical row extent.");
    }

    const std::vector<uint64_t> dims = input.getDimensions();
    if (dims.size() != 2 || dims[0] % compiled_rms_norm->packed_row_capacity != 0 || dims[1] == 0) {
        throw std::runtime_error("Packed-row RMSNorm diagnostic requires a contiguous rank-2 full-capacity input.");
    }
    const uint64_t outer_per_packed_row = dims[0] / compiled_rms_norm->packed_row_capacity;
    const uint64_t elements_per_packed_row = outer_per_packed_row * dims[1];
    const uint64_t bytes_per_packed_row = input.getDescriptor().getArraySizeInBytes(elements_per_packed_row);

    PackedRowConsumerDiagnostic diagnostic;
    diagnostic.kind = PackedRowConsumerKind::RmsNorm;
    diagnostic.active_rows = active_rows;
    diagnostic.selected_rows = selected_rows;
    diagnostic.full_capacity_rows = compiled_rms_norm->packed_row_capacity;
    diagnostic.sanitized_rows = selected_rows - active_rows;
    if (diagnostic.sanitized_rows != 0) {
        diagnostic.sanitized_operand_count = 1;
        diagnostic.sanitized_bytes = diagnostic.sanitized_rows * bytes_per_packed_row;
    }
    diagnostic.full_tail_bytes =
        (compiled_rms_norm->packed_row_capacity - active_rows) * bytes_per_packed_row;
    return diagnostic;
}

bool StampedRmsNorm::canProvideForwardStateFor(const CompiledRmsNormBackward& backward,
                                              const Tensor& input_tensor,
                                              const Tensor& scale_tensor,
                                              const Tensor& dy_tensor,
                                              const std::optional<Tensor>& backward_row_partition_offsets) const {
    if (compiled_rms_norm->fused_activation != CudnnRmsNormFusedActivation::NONE) {
        return false;
    }
    const bool packed_offsets_match =
        compiled_rms_norm->packed_row_capacity == 0
            ? !row_partition_offsets.has_value() && !backward_row_partition_offsets.has_value()
            : row_partition_offsets.has_value() && backward_row_partition_offsets.has_value() &&
                  tensorMatches(row_partition_offsets.value(), backward_row_partition_offsets.value());
    return packed_offsets_match && tensorMatches(input, input_tensor) && tensorMatches(scale, scale_tensor) &&
           dy_tensor.getDimensions() == output.getDimensions() && dy_tensor.getDataType() == backward.dy_dtype &&
           compiled_rms_norm->normalized_feature_count == backward.normalized_feature_count &&
           compiled_rms_norm->packed_row_capacity == backward.packed_row_capacity &&
           compiled_rms_norm->ragged_batch_size == backward.ragged_batch_size &&
           compiled_rms_norm->epsilon == backward.epsilon && compiled_rms_norm->input_dtype == backward.input_dtype &&
           compiled_rms_norm->scale_dtype == backward.scale_dtype && compiled_rms_norm->output_dtype == backward.dy_dtype &&
           compiled_rms_norm->compute_dtype == backward.compute_dtype;
}

StampedRmsNormBackward::StampedRmsNormBackward(std::shared_ptr<CompiledRmsNormBackward> compiled,
                                               const Tensor& input,
                                               const Tensor& scale,
                                               const Tensor& dY,
                                               const Tensor& dX,
                                               const Tensor& dScale,
                                               const Stream& stream,
                                               std::optional<Tensor> row_partition_offsets,
                                               std::shared_ptr<RmsNormForwardState> saved_forward_state)
    : compiled_rms_norm_backward(std::move(compiled)),
      input(input),
      scale(scale),
      dY(dY),
      dX(dX),
      dScale(dScale),
      row_partition_offsets(std::move(row_partition_offsets)),
      stream(stream),
      outputs({dX, dScale}),
      saved_forward_state(std::move(saved_forward_state)) {
    if (!compiled_rms_norm_backward) {
        throw std::runtime_error("StampedRmsNormBackward requires a compiled RMSNorm backward payload.");
    }
    if (compiled_rms_norm_backward->packed_row_capacity != 0) {
        if (!this->row_partition_offsets.has_value() || compiled_rms_norm_backward->ragged_batch_size == 0) {
            throw std::runtime_error("Packed-row RMSNorm backward requires an explicit row-partition runtime binding.");
        }
        const TensorDescriptor offsets_descriptor = this->row_partition_offsets->getDescriptor();
        RowPartitionRuntime(this->row_partition_offsets.value(),
                            RowPartitionDescriptor(compiled_rms_norm_backward->ragged_batch_size,
                                                   compiled_rms_norm_backward->packed_row_capacity,
                                                   offsets_descriptor.getDataType()));
        const std::vector<uint64_t> input_dims = input.getDimensions();
        if (input_dims.size() != 2 || input_dims[0] % compiled_rms_norm_backward->packed_row_capacity != 0) {
            throw std::runtime_error(
                "Packed-row RMSNorm backward requires [outer, hidden] input divisible by packed row capacity.");
        }
        const uint64_t outer_per_packed_row = input_dims[0] / compiled_rms_norm_backward->packed_row_capacity;
        for (const uint64_t capacity_rows : makeRaggedMatmulCapacityBuckets(compiled_rms_norm_backward->packed_row_capacity)) {
            const uint64_t bucket_outer = capacity_rows * outer_per_packed_row;
            Tensor bucket_input = input.aliasView({bucket_outer, input_dims[1]}, {input_dims[1], 1}, 0);
            Tensor bucket_dy = dY.aliasView({bucket_outer, input_dims[1]}, {input_dims[1], 1}, 0);
            Tensor bucket_dx = dX.aliasView({bucket_outer, input_dims[1]}, {input_dims[1], 1}, 0);
            CudnnRmsNormDescriptor descriptor =
                compiled_rms_norm_backward->descriptorFor(bucket_input, scale, bucket_dy, bucket_dx, dScale);
            CudnnRmsNorm::instance().warmBackward(descriptor, stream.getGpuNum());
        }
    }
}

bool StampedRmsNormBackward::tryLinkForwardStateFrom(const std::shared_ptr<StampedRmsNorm>& forward) {
    if (!forward ||
        !forward->canProvideForwardStateFor(*compiled_rms_norm_backward, input, scale, dY, row_partition_offsets)) {
        return false;
    }
    forward->retainForwardStateForBackward();
    saved_forward_state = forward->getForwardState();
    return true;
}

void StampedRmsNormBackward::run() { runOn(stream); }

void StampedRmsNormBackward::runOn(Stream& run_stream) const {
    if (compiled_rms_norm_backward->packed_row_capacity == 0) {
        std::shared_ptr<RmsNormForwardState> state = saved_forward_state;
        if (state != nullptr) {
            if (!state->has_valid_state || !state->inv_variance.isInitialized()) {
                throw std::runtime_error(
                    "Dense RMSNorm backward was linked to forward state that has not been populated by a training forward pass.");
            }
        } else {
            // A standalone differentiated equation has no separately stamped forward execution plan.
            // Generate the exact cuDNN training statistic locally rather than falling back to
            // the old algebraic RMSNorm derivative. CustomLayer/network training never takes
            // this path: its backward plan is linked to the retained forward state above.
            if (fallback_forward_state == nullptr) {
                fallback_forward_state = std::make_shared<RmsNormForwardState>();
            }
            if (!fallback_forward_state->inv_variance.isInitialized()) {
                fallback_forward_state->inv_variance =
                    Tensor(input.getPlacement(), TensorDescriptor(DataType::FP32, {input.getDimensions().at(0)}));
            }
            if (!fallback_output.isInitialized()) {
                fallback_output = Tensor(input.getPlacement(), TensorDescriptor(dY.getDataType(), input.getDimensions()));
            }
            CudnnRmsNormDescriptor forward_descriptor =
                compiled_rms_norm_backward->descriptorFor(input, scale, dY, dX, dScale);
            forward_descriptor.training = true;
            CudnnRmsNormForwardArgs forward_args;
            forward_args.x = input;
            forward_args.scale = scale;
            forward_args.y = fallback_output;
            forward_args.invVariance = fallback_forward_state->inv_variance;
            CudnnRmsNorm::instance().forward(forward_descriptor, forward_args, run_stream);
            fallback_forward_state->has_valid_state = true;
            state = fallback_forward_state;
        }

        CudnnRmsNormDescriptor descriptor = compiled_rms_norm_backward->descriptorFor(input, scale, dY, dX, dScale);
        CudnnRmsNormBackwardArgs args;
        args.dy = dY;
        args.x = input;
        args.scale = scale;
        args.invVariance = state->inv_variance;
        args.dx = dX;
        args.dscale = dScale;
        CudnnRmsNorm::instance().backward(descriptor, args, run_stream);
        return;
    }

    if (!row_partition_offsets.has_value()) {
        throw std::runtime_error("Packed-row RMSNorm backward is missing its row-partition runtime binding.");
    }
    const TensorDescriptor offsets_descriptor = row_partition_offsets->getDescriptor();
    RowPartitionRuntime row_partition(row_partition_offsets.value(),
                                      RowPartitionDescriptor(compiled_rms_norm_backward->ragged_batch_size,
                                                             compiled_rms_norm_backward->packed_row_capacity,
                                                             offsets_descriptor.getDataType()));
    const uint64_t active_rows = row_partition.requireHostActiveValueCount();
    const std::vector<uint64_t> input_dims = input.getDimensions();
    const uint64_t hidden = input_dims[1];
    const uint64_t outer_per_packed_row = input_dims[0] / compiled_rms_norm_backward->packed_row_capacity;
    const std::vector<uint64_t> buckets = makeRaggedMatmulCapacityBuckets(compiled_rms_norm_backward->packed_row_capacity);
    const uint64_t selected_rows =
        active_rows == 0 ? buckets.front() : chooseRaggedMatmulCapacityBucket(active_rows, buckets);
    const uint64_t selected_outer = selected_rows * outer_per_packed_row;

    // dY is the backward consumer's newly supplied row-bound operand. Sanitize only
    // the bucket slack cuDNN will read. x was sanitized by the matching forward
    // consumer and remains the same retained forward input storage.
    Tensor mutable_dy = dY;
    sanitizePackedRmsNormOverreadRows(mutable_dy,
                                      active_rows,
                                      selected_rows,
                                      compiled_rms_norm_backward->packed_row_capacity,
                                      outer_per_packed_row,
                                      run_stream);

    Tensor bucket_input = input.aliasView({selected_outer, hidden}, {hidden, 1}, 0);
    Tensor bucket_dy = dY.aliasView({selected_outer, hidden}, {hidden, 1}, 0);
    Tensor bucket_dx = dX.aliasView({selected_outer, hidden}, {hidden, 1}, 0);

    std::shared_ptr<RmsNormForwardState> state = saved_forward_state;
    if (state != nullptr) {
        if (!state->has_valid_state || !state->inv_variance.isInitialized() ||
            state->packed_active_rows != active_rows || state->packed_selected_rows != selected_rows) {
            throw std::runtime_error(
                "Packed-row RMSNorm backward forward state does not match the current logical/bucket extent.");
        }
    } else {
        // Standalone differentiated packed equations have no separately stamped
        // forward plan. Recreate the cuDNN training statistic over the same selected
        // bucket while honoring the consumer-responsibility contract.
        Tensor mutable_input = input;
        sanitizePackedRmsNormOverreadRows(mutable_input,
                                          active_rows,
                                          selected_rows,
                                          compiled_rms_norm_backward->packed_row_capacity,
                                          outer_per_packed_row,
                                          run_stream);
        if (fallback_forward_state == nullptr) {
            fallback_forward_state = std::make_shared<RmsNormForwardState>();
        }
        if (!fallback_forward_state->inv_variance.isInitialized()) {
            fallback_forward_state->inv_variance =
                Tensor(input.getPlacement(), TensorDescriptor(DataType::FP32, {input_dims[0]}));
        }
        if (!fallback_output.isInitialized()) {
            fallback_output = Tensor(input.getPlacement(), TensorDescriptor(dY.getDataType(), input_dims));
        }
        Tensor bucket_fallback_output = fallback_output.aliasView({selected_outer, hidden}, {hidden, 1}, 0);
        CudnnRmsNormDescriptor forward_descriptor =
            compiled_rms_norm_backward->descriptorFor(bucket_input, scale, bucket_dy, bucket_dx, dScale);
        forward_descriptor.training = true;
        CudnnRmsNormForwardArgs forward_args;
        forward_args.x = bucket_input;
        forward_args.scale = scale;
        forward_args.y = bucket_fallback_output;
        forward_args.invVariance = fallback_forward_state->inv_variance.aliasView({selected_outer}, {1}, 0);
        CudnnRmsNorm::instance().forward(forward_descriptor, forward_args, run_stream);
        fallback_forward_state->packed_active_rows = active_rows;
        fallback_forward_state->packed_selected_rows = selected_rows;
        fallback_forward_state->has_valid_state = true;
        state = fallback_forward_state;
    }

    CudnnRmsNormDescriptor descriptor =
        compiled_rms_norm_backward->descriptorFor(bucket_input, scale, bucket_dy, bucket_dx, dScale);
    CudnnRmsNormBackwardArgs args;
    args.dy = bucket_dy;
    args.x = bucket_input;
    args.scale = scale;
    args.invVariance = state->inv_variance.aliasView({selected_outer}, {1}, 0);
    args.dx = bucket_dx;
    args.dscale = dScale;
    CudnnRmsNorm::instance().backward(descriptor, args, run_stream);
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
    prepared_forward = prepareEmbeddingForward(
        indices,
        weights,
        output,
        compiled_embedding_lookup->has_padding_index ? std::optional<uint64_t>(compiled_embedding_lookup->padding_index) : std::nullopt,
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
                             std::optional<Tensor> bgrad_output,
                             std::optional<Tensor> row_partition_offsets)
    : compiled_matmul(std::move(compiled)),
      built_matmul(std::move(built)),
      lhs(lhs),
      rhs(rhs),
      addend(addend),
      output(output),
      epilogue_aux(epilogue_aux),
      bgrad_output(bgrad_output),
      row_partition_offsets(std::move(row_partition_offsets)),
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
    if (compiled_matmul->packed_row_binding != MatmulPackedRowBinding::None) {
        if (!this->row_partition_offsets.has_value() || compiled_matmul->ragged_batch_size == 0) {
            throw std::runtime_error("Packed-row MATMUL requires an explicit row-partition runtime binding.");
        }
        const TensorDescriptor offsets_descriptor = this->row_partition_offsets->getDescriptor();
        RowPartitionRuntime(this->row_partition_offsets.value(),
                            RowPartitionDescriptor(compiled_matmul->ragged_batch_size,
                                                   compiled_matmul->packed_row_capacity,
                                                   offsets_descriptor.getDataType()));
    } else if (this->row_partition_offsets.has_value()) {
        throw std::runtime_error("Dense MATMUL unexpectedly received a row-partition runtime binding.");
    }
    if (built_matmul->workspace_bytes != 0) {
        if (!workspace.has_value()) {
            throw std::runtime_error("StampedMatmul requires workspace for the chosen optimal kernel.");
        }
        THOR_THROW_IF_FALSE(workspace.value().getArraySizeInBytes() >= built_matmul->workspace_bytes);
    }
}

StampedMatmulKernelDiagnostic StampedMatmul::kernelDiagnostic() const {
    const MatmulCacheKey& key = built_matmul->key;
    StampedMatmulKernelDiagnostic diagnostic;
    diagnostic.m = key.transpose_a ? key.a_cols : key.a_rows;
    diagnostic.k = key.transpose_a ? key.a_rows : key.a_cols;
    const int32_t rhs_k = key.transpose_b ? key.b_cols : key.b_rows;
    diagnostic.n = key.transpose_b ? key.b_rows : key.b_cols;
    if (diagnostic.k != rhs_k) {
        throw std::runtime_error("StampedMatmul kernel diagnostic found incompatible effective GEMM dimensions.");
    }
    diagnostic.batch_count = key.batch_config.batchCount;
    const uint64_t m = static_cast<uint64_t>(diagnostic.m);
    const uint64_t n = static_cast<uint64_t>(diagnostic.n);
    const uint64_t k = static_cast<uint64_t>(diagnostic.k);
    const uint64_t batch = static_cast<uint64_t>(diagnostic.batch_count);
    if (m != 0 && n > std::numeric_limits<uint64_t>::max() / m) {
        throw std::runtime_error("StampedMatmul kernel diagnostic FLOP count overflow.");
    }
    const uint64_t mn = m * n;
    if (mn != 0 && k > std::numeric_limits<uint64_t>::max() / mn) {
        throw std::runtime_error("StampedMatmul kernel diagnostic FLOP count overflow.");
    }
    const uint64_t mnk = mn * k;
    if (mnk != 0 && batch > std::numeric_limits<uint64_t>::max() / mnk) {
        throw std::runtime_error("StampedMatmul kernel diagnostic FLOP count overflow.");
    }
    const uint64_t fma_count = mnk * batch;
    if (fma_count > std::numeric_limits<uint64_t>::max() / 2) {
        throw std::runtime_error("StampedMatmul kernel diagnostic FLOP count overflow.");
    }
    diagnostic.flop_count = fma_count * 2;
    diagnostic.workspace_bytes = built_matmul->workspace_bytes;

    if (built_matmul->cublas_kernel.has_value()) {
        const CublasKernel& kernel = built_matmul->cublas_kernel.value();
        diagnostic.has_measured_kernel = kernel.getMeasuredRunCount() > 0;
        diagnostic.waves_count = kernel.getWavesCount(static_cast<int>(gpuNum()));
        if (diagnostic.has_measured_kernel) {
            diagnostic.picker_runtime_ms = kernel.getAverageRunTimeMilliseconds();
        }
        diagnostic.algorithm_id = kernel.getAlgorithmId();
    }
    return diagnostic;
}

std::optional<PackedRowConsumerDiagnostic> StampedMatmul::packedRowConsumerDiagnostic() const {
    if (!built_matmul->bucketed_cublas_gemm.has_value()) {
        return std::nullopt;
    }
    if (!row_partition_offsets.has_value()) {
        throw std::runtime_error("Packed-row MATMUL diagnostic is missing its row-partition runtime binding.");
    }
    const TensorDescriptor offsets_descriptor = row_partition_offsets->getDescriptor();
    RowPartitionRuntime row_partition(row_partition_offsets.value(),
                                      RowPartitionDescriptor(compiled_matmul->ragged_batch_size,
                                                             compiled_matmul->packed_row_capacity,
                                                             offsets_descriptor.getDataType()));
    const uint64_t active_rows = row_partition.requireHostActiveValueCount();
    const uint64_t selected_rows = built_matmul->bucketed_cublas_gemm->getSelectedCapacityRows(active_rows);
    if (selected_rows < active_rows || selected_rows > compiled_matmul->packed_row_capacity) {
        throw std::runtime_error("Packed-row MATMUL diagnostic selected an invalid physical row extent.");
    }

    PackedRowConsumerDiagnostic diagnostic;
    diagnostic.kind = PackedRowConsumerKind::Matmul;
    diagnostic.active_rows = active_rows;
    diagnostic.selected_rows = selected_rows;
    diagnostic.full_capacity_rows = compiled_matmul->packed_row_capacity;
    diagnostic.sanitized_rows = selected_rows - active_rows;

    auto account_row_bound_operand = [&](const Tensor& tensor) {
        const std::vector<uint64_t> dims = tensor.getDimensions();
        if (dims.size() != 2 || dims[0] != compiled_matmul->packed_row_capacity || dims[1] == 0) {
            throw std::runtime_error("Packed-row MATMUL diagnostic requires contiguous rank-2 row-bound operands.");
        }
        const uint64_t bytes_per_row = tensor.getDescriptor().getArraySizeInBytes(dims[1]);
        diagnostic.full_tail_bytes += (compiled_matmul->packed_row_capacity - active_rows) * bytes_per_row;
        if (diagnostic.sanitized_rows != 0) {
            ++diagnostic.sanitized_operand_count;
            diagnostic.sanitized_bytes += diagnostic.sanitized_rows * bytes_per_row;
        }
    };

    const bool binds_lhs = compiled_matmul->packed_row_binding == MatmulPackedRowBinding::RowsA ||
                           compiled_matmul->packed_row_binding == MatmulPackedRowBinding::RowsAAndRowsB;
    const bool binds_rhs = compiled_matmul->packed_row_binding == MatmulPackedRowBinding::RowsB ||
                           compiled_matmul->packed_row_binding == MatmulPackedRowBinding::RowsAAndRowsB;
    if (binds_lhs) {
        account_row_bound_operand(lhs);
    }
    if (binds_rhs) {
        account_row_bound_operand(rhs);
    }
    return diagnostic;
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

static bool packedMatmulBindsRowsLhs(MatmulPackedRowBinding binding) {
    return binding == MatmulPackedRowBinding::RowsA || binding == MatmulPackedRowBinding::RowsAAndRowsB;
}

static bool packedMatmulBindsRowsRhs(MatmulPackedRowBinding binding) {
    return binding == MatmulPackedRowBinding::RowsB || binding == MatmulPackedRowBinding::RowsAAndRowsB;
}

// Consumer-side pre-read sanitation for bucketed cuBLASLt MATMUL. The selected
// GEMM reads selected_rows from every row-bound operand, so sanitize exactly
// [active_rows, selected_rows) immediately before that read. Never canonicalize
// [selected_rows, full_capacity_rows) merely because the tensor is ragged.
static void sanitizePackedMatmulOverreadRows(Tensor tensor,
                                             uint64_t active_rows,
                                             uint64_t selected_rows,
                                             uint64_t full_capacity_rows,
                                             Stream& stream) {
    if (selected_rows > full_capacity_rows || active_rows > selected_rows) {
        throw std::runtime_error("Packed-row MATMUL selected extent is incompatible with its logical active extent.");
    }
    if (active_rows == selected_rows) {
        return;
    }

    const std::vector<uint64_t> dims = tensor.getDimensions();
    if (dims.size() != 2 || dims[0] != full_capacity_rows || dims[1] == 0) {
        throw std::runtime_error("Packed-row MATMUL row-bound operand must be a contiguous rank-2 full-capacity tensor.");
    }

    const uint64_t row_width = dims[1];
    Tensor overread = tensor.aliasView({selected_rows - active_rows, row_width},
                                       {row_width, 1},
                                       active_rows * row_width);
    overread.memsetAsync(stream, 0);
}

void StampedMatmul::run() { runOn(stream); }

void StampedMatmul::runOn(Stream& run_stream) const { runOn(run_stream, {}); }

void StampedMatmul::runOn(Stream& run_stream, const std::unordered_map<std::string, float>& runtime_scalars) const {
    if (compiled_matmul->op == ExprOp::MATMUL) {
        if (lhs.getDimensions().size() < 2 || rhs.getDimensions().size() < 2 || output.getDimensions().size() < 2) {
            throw std::runtime_error("Stamped MATMUL requires rank >= 2 tensors.");
        }
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
            if (!built_matmul->epilogue_plan) {
                throw std::runtime_error("Stamped MATMUL backward epilogue runtime missing compile-time cuBLASLt plan.");
            }
            built_matmul->epilogue_plan->runGemmWithBackwardEpilogue(
                lhs, rhs, std::nullopt, output, &alphaOne, &betaZero, run_stream, CublasScalarPointerMode::Host, workspace);
            return;
        }

        if (compiled_matmul->epilogue == MatmulEpilogue::Default) {
            if (built_matmul->bucketed_cublas_gemm.has_value()) {
                if (!row_partition_offsets.has_value()) {
                    throw std::runtime_error("Packed-row expression MATMUL is missing its row-partition runtime binding.");
                }
                const TensorDescriptor offsets_descriptor = row_partition_offsets->getDescriptor();
                RowPartitionRuntime row_partition(row_partition_offsets.value(),
                                                  RowPartitionDescriptor(compiled_matmul->ragged_batch_size,
                                                                         compiled_matmul->packed_row_capacity,
                                                                         offsets_descriptor.getDataType()));
                const uint64_t active_rows = row_partition.requireHostActiveValueCount();
                // cuBLASLt executes a pre-tuned row bucket rather than an arbitrary logical
                // row count. The MATMUL is therefore the physical consumer responsible for
                // sanitizing exactly the bucket slack it will read. Producer tails remain
                // undefined, including when another active-aware ragged stage feeds us. An
                // all-empty batch selects the smallest cached bucket rather than full capacity.
                const uint64_t selected_rows = built_matmul->bucketed_cublas_gemm->getSelectedCapacityRows(active_rows);
                if (packedMatmulBindsRowsLhs(compiled_matmul->packed_row_binding)) {
                    sanitizePackedMatmulOverreadRows(
                        lhs, active_rows, selected_rows, compiled_matmul->packed_row_capacity, run_stream);
                }
                if (packedMatmulBindsRowsRhs(compiled_matmul->packed_row_binding)) {
                    sanitizePackedMatmulOverreadRows(
                        rhs, active_rows, selected_rows, compiled_matmul->packed_row_capacity, run_stream);
                }
                const float alphaOne = 1.0f;
                const float betaZero = 0.0f;
                CHECK_CUBLAS(built_matmul->bucketed_cublas_gemm->launchUncheckedPrevalidated(active_rows,
                                                                                              lhs,
                                                                                              rhs,
                                                                                              output,
                                                                                              output,
                                                                                              workspace,
                                                                                              &alphaOne,
                                                                                              &betaZero,
                                                                                              run_stream,
                                                                                              CublasScalarPointerMode::Host));
                // The selected bucket's output slack and all rows beyond it are outside
                // the logical ragged tensor. Do not canonicalize either region after the
                // consumer finishes; a later over-reading consumer must sanitize its own
                // selected over-read immediately before use.
                return;
            }
            if (!built_matmul->cublas_kernel.has_value()) {
                throw std::runtime_error("Stamped MATMUL runtime missing compile-time cuBLAS kernel artifact.");
            }
            const float alphaOne = 1.0f;
            const float betaZero = 0.0f;
            CHECK_CUBLAS(built_matmul->cublas_kernel->launchUncheckedPrevalidated(lhs,
                                                                        rhs,
                                                                        output,
                                                                        output,
                                                                        workspace,
                                                                        &alphaOne,
                                                                        &betaZero,
                                                                        run_stream,
                                                                        CublasScalarPointerMode::Host));
            return;
        }

        if (compiled_matmul->transpose_lhs || compiled_matmul->transpose_rhs) {
            throw std::runtime_error("cuBLASLt MATMUL activation epilogue fusion currently supports only non-transposed row-major stages.");
        }
        const float alphaOne = 1.0f;
        const float betaZero = 0.0f;
        if (!built_matmul->epilogue_plan) {
            throw std::runtime_error("Stamped MATMUL epilogue runtime missing compile-time cuBLASLt plan.");
        }
        built_matmul->epilogue_plan->runGemmWithEpilogue(
            lhs, rhs, std::nullopt, output, &alphaOne, &betaZero, run_stream, CublasScalarPointerMode::Host, workspace, false);
        return;
    }

    if (lhs.getDimensions().size() != 2 || rhs.getDimensions().size() != 2 || output.getDimensions().size() != 2) {
        throw std::runtime_error("Stamped GEMM currently requires rank-2 matrix tensors.");
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
        if (!built_matmul->epilogue_plan) {
            throw std::runtime_error("Stamped GEMM backward epilogue runtime missing compile-time cuBLASLt plan.");
        }
        built_matmul->epilogue_plan->runGemmWithBackwardEpilogue(lhs,
                                                                 rhs,
                                                                 addend,
                                                                 output,
                                                                 resolved_scales.alpha.ptr,
                                                                 resolved_scales.beta.ptr,
                                                                 run_stream,
                                                                 resolved_scales.pointer_mode,
                                                                 workspace);
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
        if (!built_matmul->epilogue_plan) {
            throw std::runtime_error("Stamped GEMM epilogue runtime missing compile-time cuBLASLt plan.");
        }
        built_matmul->epilogue_plan->runGemmWithEpilogue(lhs,
                                                         rhs,
                                                         addend.value(),
                                                         output,
                                                         resolved_scales.alpha.ptr,
                                                         resolved_scales.beta.ptr,
                                                         run_stream,
                                                         resolved_scales.pointer_mode,
                                                         workspace,
                                                         use_bias_epilogue);
        return;
    }

    if (!built_matmul->cublas_kernel.has_value()) {
        throw std::runtime_error("Stamped GEMM runtime missing compile-time cuBLAS kernel artifact.");
    }
    CHECK_CUBLAS(built_matmul->cublas_kernel->launchUncheckedPrevalidated(lhs,
                                                                rhs,
                                                                addend.value(),
                                                                output,
                                                                workspace,
                                                                resolved_scales.alpha.ptr,
                                                                resolved_scales.beta.ptr,
                                                                run_stream,
                                                                resolved_scales.pointer_mode));
}

void StampedMatmul::runOnConditionalGraphCapture(Stream& run_stream) const {
    if (!alpha_runtime_name.has_value() && !beta_runtime_name.has_value()) {
        runOn(run_stream);
        return;
    }
    if (compiled_matmul->op != ExprOp::GEMM) {
        throw std::runtime_error("Conditional graph host runtime scalar capture is only expected for GEMM scale inputs.");
    }
    if (lhs.getDimensions().size() != 2 || rhs.getDimensions().size() != 2 || output.getDimensions().size() != 2) {
        throw std::runtime_error("Stamped GEMM currently requires rank-2 matrix tensors.");
    }
    if (!addend.has_value()) {
        throw std::runtime_error("Stamped GEMM requires an addend tensor.");
    }

    auto resolve_scale = [&](const std::optional<RuntimeInputValue>& bound_input,
                             const std::optional<std::string>& runtime_name,
                             double base_scale,
                             const std::optional<Tensor>& device_scratch,
                             const std::optional<Tensor>& host_scratch) {
        if (runtime_name.has_value()) {
            if (!device_scratch.has_value()) {
                throw std::runtime_error("Conditional graph dynamic GEMM scale is missing device scalar scratch.");
            }
            ResolvedMatmulScale resolved(device_scratch, host_scratch);
            resolved.host_value = static_cast<float>(base_scale);
            resolved.setDevicePointer(static_cast<const float*>(device_scratch->getMemPtr()));
            return resolved;
        }
        return resolveMatmulRuntimeScale(
            bound_input, runtime_name, base_scale, {}, device_scratch, host_scratch, run_stream);
    };

    ResolvedMatmulScales resolved_scales;
    resolved_scales.alpha = resolve_scale(alpha_input,
                                          alpha_runtime_name,
                                          compiled_matmul->alpha,
                                          alpha_device_scratch,
                                          alpha_host_scratch);
    resolved_scales.beta = resolve_scale(beta_input,
                                         beta_runtime_name,
                                         compiled_matmul->beta,
                                         beta_device_scratch,
                                         beta_host_scratch);
    resolved_scales.alpha.refreshHostPointer();
    resolved_scales.beta.refreshHostPointer();
    if (resolved_scales.alpha.is_device_pointer || resolved_scales.beta.is_device_pointer) {
        resolved_scales.pointer_mode = CublasScalarPointerMode::Device;
        if (!resolved_scales.alpha.is_device_pointer) {
            resolved_scales.alpha.copyHostValueToDevice(run_stream);
        }
        if (!resolved_scales.beta.is_device_pointer) {
            resolved_scales.beta.copyHostValueToDevice(run_stream);
        }
    }

    const bool use_bias_epilogue = addend.value().getDimensions().size() == 1;
    if (!use_bias_epilogue && addend.value().getDimensions().size() != 2) {
        throw std::runtime_error("Stamped GEMM currently supports rank-2 addend tensors or rank-1 bias epilogue vectors.");
    }
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
        if (!built_matmul->epilogue_plan) {
            throw std::runtime_error("Stamped GEMM backward epilogue runtime missing compile-time cuBLASLt plan.");
        }
        built_matmul->epilogue_plan->runGemmWithBackwardEpilogue(lhs,
                                                                 rhs,
                                                                 addend,
                                                                 output,
                                                                 resolved_scales.alpha.ptr,
                                                                 resolved_scales.beta.ptr,
                                                                 run_stream,
                                                                 resolved_scales.pointer_mode,
                                                                 workspace);
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
            if (beta_runtime_name.has_value() || resolved_scales.beta.host_value != 1.0f) {
                throw std::runtime_error("GEMM bias epilogue currently requires an unscaled +bias addend.");
            }
        }
        if (!built_matmul->epilogue_plan) {
            throw std::runtime_error("Stamped GEMM epilogue runtime missing compile-time cuBLASLt plan.");
        }
        built_matmul->epilogue_plan->runGemmWithEpilogue(lhs,
                                                         rhs,
                                                         addend.value(),
                                                         output,
                                                         resolved_scales.alpha.ptr,
                                                         resolved_scales.beta.ptr,
                                                         run_stream,
                                                         resolved_scales.pointer_mode,
                                                         workspace,
                                                         use_bias_epilogue);
        return;
    }

    if (!built_matmul->cublas_kernel.has_value()) {
        throw std::runtime_error("Stamped GEMM runtime missing compile-time cuBLAS kernel artifact.");
    }
    CHECK_CUBLAS(built_matmul->cublas_kernel->launchUncheckedPrevalidated(lhs,
                                                                          rhs,
                                                                          addend.value(),
                                                                          output,
                                                                          workspace,
                                                                          resolved_scales.alpha.ptr,
                                                                          resolved_scales.beta.ptr,
                                                                          run_stream,
                                                                          resolved_scales.pointer_mode));
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
                                                         const Stream& stream)
    : built_reduction(std::move(built)),
      input(input),
      grad_output(grad_output),
      output(output),
      indices(indices),
      stream(stream) {
    if (built_reduction->key.result_kind != ReductionResultKind::Indices || !built_reduction->arg_op.has_value()
        || !built_reduction->geometry.has_value()) {
        throw std::runtime_error("StampedReduceMinMaxBackward requires an index-producing reduction plan.");
    }
    if (indices.getDataType() != DataType::UINT32) {
        throw std::runtime_error("StampedReduceMinMaxBackward requires UINT32 local winner indices.");
    }

    std::vector<uint32_t> axes;
    axes.reserve(built_reduction->key.reduction_axes.size());
    for (uint64_t axis : built_reduction->key.reduction_axes) {
        THOR_THROW_IF_FALSE(axis <= UINT32_MAX);
        axes.push_back(static_cast<uint32_t>(axis));
    }

    CubArgReductionOutputOptions outputs;
    outputs.produce_value = false;
    outputs.produce_index = true;
    outputs.index_output_dtype = DataType::UINT32;
    cub_arg_reduction = CubArgReduction(built_reduction->arg_op.value(), std::move(axes), outputs)
                            .stamp(input, std::nullopt, indices, stream);
    THOR_THROW_IF_FALSE(cub_arg_reduction->getGeometry().path == built_reduction->geometry->path);
    scatter_plan = prepareReduceMinMaxBackwardScatter(input.getDimensions(),
                                                       built_reduction->key.reduction_axes,
                                                       built_reduction->key.squeeze_axes,
                                                       input.getPlacement(),
                                                       stream);
}

StampedReduceMinMaxBackward::StampedReduceMinMaxBackward(CubArgReductionOp segmented_op,
                                                         const Tensor& input,
                                                         const Tensor& grad_output,
                                                         const Tensor& output,
                                                         const Tensor& indices,
                                                         const Tensor& segment_offsets,
                                                         const Stream& stream)
    : built_reduction(nullptr),
      input(input),
      grad_output(grad_output),
      output(output),
      indices(indices),
      segment_offsets(segment_offsets),
      stream(stream) {
    if (input.getPlacement() != grad_output.getPlacement() || input.getPlacement() != output.getPlacement() ||
        input.getPlacement() != segment_offsets.getPlacement()) {
        throw std::runtime_error("Segmented reduce-min/max backward tensors must share one GPU placement.");
    }
    if (input.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU ||
        input.getPlacement().getDeviceNum() != stream.getGpuNum()) {
        throw std::runtime_error("Segmented reduce-min/max backward must be stamped on the input tensor's GPU.");
    }
    if (grad_output.getDataType() != output.getDataType()) {
        throw std::runtime_error("Segmented reduce-min/max backward output dtype must match upstream gradient dtype.");
    }
    if (input.getDimensions().empty() || grad_output.getDimensions().empty() || segment_offsets.getDimensions().size() != 1) {
        throw std::runtime_error(
            "Segmented reduce-min/max backward requires input [N,D...], grad [B,D...], and rank-1 offsets tensors.");
    }
    const uint64_t batch_size = segment_offsets.getDimensions()[0] - 1;
    if (batch_size == 0 || input.getDimensions()[0] == 0 || grad_output.getDimensions()[0] != batch_size) {
        throw std::runtime_error("Segmented reduce-min/max backward received inconsistent batch dimensions.");
    }
    if (output.getDimensions() != input.getDimensions()) {
        throw std::runtime_error("Segmented reduce-min/max backward output dimensions must match packed input dimensions.");
    }
    if (input.getTotalNumElements() % input.getDimensions()[0] != 0 ||
        grad_output.getTotalNumElements() % batch_size != 0) {
        throw std::runtime_error("Segmented reduce-min/max backward tensor extents are not divisible by their leading dimensions.");
    }
    segmented_elements_per_value = input.getTotalNumElements() / input.getDimensions()[0];
    if (segmented_elements_per_value == 0 || grad_output.getTotalNumElements() / batch_size != segmented_elements_per_value) {
        throw std::runtime_error("Segmented reduce-min/max backward input and upstream gradient trailing extents must match.");
    }
    if (!indices.isInitialized()
        || (indices.getDataType() != DataType::UINT32 && indices.getDataType() != DataType::UINT64)
        || indices.getPlacement() != input.getPlacement()
        || indices.getDimensions() != grad_output.getDimensions()) {
        throw std::runtime_error(
            "Segmented reduce-min/max backward requires UINT32/UINT64 winner indices shaped like the upstream gradient.");
    }
    cub_segmented_arg_reduction = CubSegmentedArgReduction(segmented_op, indices.getDataType())
                                      .stampRuntimeOffsets(input, indices, segment_offsets, stream);
}

void StampedReduceMinMaxBackward::run() { runOn(stream); }

void StampedReduceMinMaxBackward::runOn(Stream& run_stream) {
    if (segment_offsets.has_value()) {
        THOR_THROW_IF_FALSE(cub_segmented_arg_reduction != nullptr);
        cub_segmented_arg_reduction->runOn(run_stream);

        const Tensor& offsets = segment_offsets.value();
        const uint64_t num_segments = offsets.getDimensions()[0] - 1;
        launchSegmentedReduceMinMaxBackwardActivePrefixZero(offsets.getMemPtr(),
                                                            offsets.getDataType(),
                                                            num_segments,
                                                            output.getMemPtr(),
                                                            output.getTotalNumElements(),
                                                            segmented_elements_per_value,
                                                            output.getDataType(),
                                                            run_stream.getStream());
        launchSegmentedReduceMinMaxBackwardScatter(grad_output.getMemPtr(),
                                                   indices.getMemPtr(),
                                                   indices.getDataType(),
                                                   output.getMemPtr(),
                                                   grad_output.getTotalNumElements(),
                                                   output.getTotalNumElements(),
                                                   grad_output.getDataType(),
                                                   output.getDataType(),
                                                   run_stream.getStream());
        return;
    }

    THOR_THROW_IF_FALSE(cub_arg_reduction != nullptr);
    cub_arg_reduction->runOn(run_stream);
    output.memsetAsync(run_stream, 0);

    launchReduceMinMaxBackwardScatter(grad_output.getMemPtr(),
                                      static_cast<const uint32_t*>(indices.getMemPtr()),
                                      (void*)output.getMemPtr(),
                                      scatter_plan,
                                      grad_output.getDataType(),
                                      output.getDataType(),
                                      run_stream.getStream());
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

static std::vector<cudaGraphNode_t> graphNodes(cudaGraph_t graph) {
    size_t node_count = 0;
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &node_count));

    std::vector<cudaGraphNode_t> nodes(node_count);
    if (node_count != 0) {
        CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &node_count));
        nodes.resize(node_count);
    }
    return nodes;
}

static std::vector<cudaGraphNode_t> graphLeafNodes(cudaGraph_t graph) {
    std::vector<cudaGraphNode_t> nodes = graphNodes(graph);
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

namespace detail {
struct ConditionalGraphCaptureAccess {
    static const std::vector<StampedExecutionStage>& steps(const StampedExecutionPlan& plan) { return plan.steps; }

    static bool hasOutputMaterializations(const StampedExecutionPlan& plan) { return !plan.output_materializations.empty(); }

    static void materializeOutputs(const StampedExecutionPlan& plan, Stream& capture_stream) {
        plan.materializeOutputsOn(capture_stream);
    }

    static const StampedExecutionPlan& predicatePlan(const StampedConditional& conditional) {
        return *conditional.predicate_plan;
    }

    static const StampedExecutionPlan& thenPlan(const StampedConditional& conditional) { return *conditional.then_plan; }

    static const StampedExecutionPlan& elsePlan(const StampedConditional& conditional) { return *conditional.else_plan; }

    static std::vector<ConditionalRuntimeScalarKernelArgument> fusedRuntimeScalarArguments(const StampedEquation& equation) {
        std::vector<ConditionalRuntimeScalarKernelArgument> arguments;
        THOR_THROW_IF_FALSE(equation.compiledEquation != nullptr);
        for (uint32_t i = 0; i < equation.compiledEquation->input_kinds.size(); ++i) {
            if (equation.compiledEquation->input_kinds[i] == NamedInput::Kind::RuntimeScalarFp32) {
                arguments.push_back(ConditionalRuntimeScalarKernelArgument{
                    .kernel_argument_index = i,
                    .name = equation.inputNames.at(i),
                    .multiplier = 1.0f,
                });
            }
        }
        return arguments;
    }

    static uint32_t fusedKernelArgumentCount(const StampedEquation& equation) {
        THOR_THROW_IF_FALSE(equation.compiledEquation != nullptr);
        uint32_t count = static_cast<uint32_t>(equation.compiledEquation->numInputs() + equation.outputs.size());
        count += equation.compiledEquation->launch_kind == CompiledEquation::LaunchKind::FusedTiledTranspose ? 3U : 1U;
        return count;
    }

    static std::vector<ConditionalRuntimeScalarKernelArgument> cudaRuntimeScalarArguments(const StampedCudaKernel& kernel) {
        std::vector<ConditionalRuntimeScalarKernelArgument> arguments;
        for (uint32_t i = 0; i < kernel.params.size(); ++i) {
            if (kernel.params[i].kind == StampedCudaKernelParam::Kind::HostRuntimeScalar) {
                arguments.push_back(ConditionalRuntimeScalarKernelArgument{
                    .kernel_argument_index = i,
                    .name = kernel.params[i].name,
                    .multiplier = 1.0f,
                });
            }
        }
        return arguments;
    }

    static uint32_t cudaKernelArgumentCount(const StampedCudaKernel& kernel) {
        return static_cast<uint32_t>(kernel.params.size());
    }

    static std::optional<std::tuple<std::string, float, float*>> matmulAlphaHostRuntimeScalar(StampedMatmul& matmul) {
        if (!matmul.alpha_runtime_name.has_value()) {
            return std::nullopt;
        }
        if (!matmul.alpha_device_scratch.has_value()) {
            throw std::runtime_error("Conditional graph dynamic GEMM alpha is missing device scalar scratch.");
        }
        return std::make_tuple(*matmul.alpha_runtime_name,
                               static_cast<float>(matmul.compiled_matmul->alpha),
                               const_cast<float*>(matmul.alpha_device_scratch->getMemPtr<float>()));
    }

    static std::optional<std::tuple<std::string, float, float*>> matmulBetaHostRuntimeScalar(StampedMatmul& matmul) {
        if (!matmul.beta_runtime_name.has_value()) {
            return std::nullopt;
        }
        if (!matmul.beta_device_scratch.has_value()) {
            throw std::runtime_error("Conditional graph dynamic GEMM beta is missing device scalar scratch.");
        }
        return std::make_tuple(*matmul.beta_runtime_name,
                               static_cast<float>(matmul.compiled_matmul->beta),
                               const_cast<float*>(matmul.beta_device_scratch->getMemPtr<float>()));
    }

    static void runMatmulForConditionalGraphCapture(const StampedMatmul& matmul, Stream& stream) {
        matmul.runOnConditionalGraphCapture(stream);
    }
};
}  // namespace detail

template <typename CaptureFn>
static void captureIntoExistingGraph(cudaGraph_t graph,
                                     Stream& capture_stream,
                                     const std::vector<cudaGraphNode_t>& dependencies,
                                     CaptureFn&& capture_fn) {
    cudaGraph_t captured_graph = nullptr;
    const cudaGraphNode_t* deps = dependencies.empty() ? nullptr : dependencies.data();
    CUDA_CHECK(
        cudaStreamBeginCaptureToGraph(capture_stream.getStream(), graph, deps, nullptr, dependencies.size(), cudaStreamCaptureModeGlobal));

    try {
        capture_fn();
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
        throw std::runtime_error("CUDA graph capture did not return the requested target graph.");
    }
}

static cudaGraphNode_t uniqueNewKernelNode(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& before_nodes) {
    std::unordered_set<cudaGraphNode_t> before(before_nodes.begin(), before_nodes.end());
    cudaGraphNode_t kernel_node = nullptr;
    for (cudaGraphNode_t node : graphNodes(graph)) {
        if (before.contains(node)) {
            continue;
        }
        cudaGraphNodeType type{};
        CUDA_CHECK(cudaGraphNodeGetType(node, &type));
        if (type != cudaGraphNodeTypeKernel) {
            continue;
        }
        if (kernel_node != nullptr) {
            throw std::runtime_error("Conditional runtime scalar capture produced multiple kernel nodes for one scalar-bound stage.");
        }
        kernel_node = node;
    }
    if (kernel_node == nullptr) {
        throw std::runtime_error("Conditional runtime scalar capture did not produce a CUDA kernel node.");
    }
    return kernel_node;
}

static void registerConditionalRuntimeScalarKernelBinding(
    cudaGraphNode_t kernel_node,
    uint32_t kernel_argument_count,
    std::vector<detail::ConditionalRuntimeScalarKernelArgument> runtime_arguments,
    bool use_driver_api,
    std::vector<detail::ConditionalRuntimeScalarKernelBinding>& bindings) {
    if (runtime_arguments.empty()) {
        return;
    }

    detail::ConditionalRuntimeScalarKernelBinding binding;
    binding.source_node = kernel_node;
    binding.use_driver_api = use_driver_api;
    if (use_driver_api) {
        CUDA_KERNEL_NODE_PARAMS params{};
        CU_CHECK(graphKernelNodeGetParams(reinterpret_cast<CUgraphNode>(kernel_node), &params));
        if (params.kernelParams == nullptr || params.extra != nullptr) {
            throw std::runtime_error(
                "Conditional host runtime scalar updates require CUDA driver kernel nodes captured with kernelParams arguments.");
        }
        binding.driver_template_params = params;
        binding.template_kernel_params.assign(params.kernelParams, params.kernelParams + kernel_argument_count);
    } else {
        cudaKernelNodeParams params{};
        CUDA_CHECK(cudaGraphKernelNodeGetParams(kernel_node, &params));
        if (params.kernelParams == nullptr || params.extra != nullptr) {
            throw std::runtime_error(
                "Conditional host runtime scalar updates require CUDA runtime kernel nodes captured with kernelParams arguments.");
        }
        binding.runtime_template_params = params;
        binding.template_kernel_params.assign(params.kernelParams, params.kernelParams + kernel_argument_count);
    }
    binding.launch_kernel_params = binding.template_kernel_params;
    binding.runtime_arguments = std::move(runtime_arguments);
    binding.runtime_values.resize(binding.runtime_arguments.size());
    for (const auto& argument : binding.runtime_arguments) {
        if (argument.kernel_argument_index >= binding.template_kernel_params.size()) {
            throw std::runtime_error("Conditional runtime scalar kernel argument index is out of range.");
        }
    }
    bindings.push_back(std::move(binding));
}

static std::unordered_set<std::string> runtimeScalarNamesForStage(const StampedExecutionStage& stage);

static std::unordered_map<std::string, float> placeholderRuntimeScalarsForStage(const StampedExecutionStage& stage) {
    std::unordered_map<std::string, float> placeholders;
    std::unordered_set<std::string> names = runtimeScalarNamesForStage(stage);
    placeholders.reserve(names.size());
    for (const std::string& name : names) {
        placeholders.emplace(name, 1.0f);
    }
    return placeholders;
}

static cudaGraphNode_t appendConditionalPlansIntoGraph(
    const StampedExecutionPlan& predicate_plan,
    const StampedExecutionPlan& then_plan,
    const StampedExecutionPlan& else_plan,
    cudaGraph_t graph,
    Stream& capture_stream,
    const std::vector<cudaGraphNode_t>& dependencies,
    std::vector<detail::ConditionalRuntimeScalarKernelBinding>& runtime_scalar_bindings);

static std::vector<cudaGraphNode_t> appendPlanSequentiallyIntoGraph(
    const StampedExecutionPlan& plan,
    cudaGraph_t graph,
    Stream& capture_stream,
    std::vector<detail::ConditionalRuntimeScalarKernelBinding>& runtime_scalar_bindings,
    const std::vector<cudaGraphNode_t>& dependencies = {}) {
    std::vector<cudaGraphNode_t> current_dependencies = dependencies;
    for (const StampedExecutionStage& stage : detail::ConditionalGraphCaptureAccess::steps(plan)) {
        if (stage.kind == StampedExecutionStage::Kind::Conditional) {
            THOR_THROW_IF_FALSE(stage.conditional != nullptr);
            const StampedConditional& conditional = *stage.conditional;
            cudaGraphNode_t conditional_node = appendConditionalPlansIntoGraph(
                detail::ConditionalGraphCaptureAccess::predicatePlan(conditional),
                detail::ConditionalGraphCaptureAccess::thenPlan(conditional),
                detail::ConditionalGraphCaptureAccess::elsePlan(conditional),
                graph,
                capture_stream,
                current_dependencies,
                runtime_scalar_bindings);
            current_dependencies = {conditional_node};
            continue;
        }

        const std::unordered_set<std::string> stage_runtime_scalar_names = runtimeScalarNamesForStage(stage);
        if (stage_runtime_scalar_names.empty()) {
            captureIntoExistingGraph(graph, capture_stream, current_dependencies, [&]() { stage.runOn(capture_stream); });
            current_dependencies = graphLeafNodes(graph);
            continue;
        }

        if (stage.kind == StampedExecutionStage::Kind::FusedKernel) {
            THOR_THROW_IF_FALSE(stage.kernel != nullptr);
            const std::vector<cudaGraphNode_t> before_nodes = graphNodes(graph);
            const std::unordered_map<std::string, float> placeholders = placeholderRuntimeScalarsForStage(stage);
            captureIntoExistingGraph(graph, capture_stream, current_dependencies, [&]() {
                stage.kernel->runOn(capture_stream, placeholders);
            });
            cudaGraphNode_t kernel_node = uniqueNewKernelNode(graph, before_nodes);
            registerConditionalRuntimeScalarKernelBinding(
                kernel_node,
                detail::ConditionalGraphCaptureAccess::fusedKernelArgumentCount(*stage.kernel),
                detail::ConditionalGraphCaptureAccess::fusedRuntimeScalarArguments(*stage.kernel),
                /*use_driver_api=*/true,
                runtime_scalar_bindings);
            current_dependencies = graphLeafNodes(graph);
            continue;
        }

        if (stage.kind == StampedExecutionStage::Kind::CudaKernel) {
            THOR_THROW_IF_FALSE(stage.cuda_kernel != nullptr);
            const std::vector<cudaGraphNode_t> before_nodes = graphNodes(graph);
            const std::unordered_map<std::string, float> placeholders = placeholderRuntimeScalarsForStage(stage);
            captureIntoExistingGraph(graph, capture_stream, current_dependencies, [&]() {
                stage.cuda_kernel->runOn(capture_stream, placeholders);
            });
            cudaGraphNode_t kernel_node = uniqueNewKernelNode(graph, before_nodes);
            registerConditionalRuntimeScalarKernelBinding(
                kernel_node,
                detail::ConditionalGraphCaptureAccess::cudaKernelArgumentCount(*stage.cuda_kernel),
                detail::ConditionalGraphCaptureAccess::cudaRuntimeScalarArguments(*stage.cuda_kernel),
                /*use_driver_api=*/true,
                runtime_scalar_bindings);
            current_dependencies = graphLeafNodes(graph);
            continue;
        }

        if (stage.kind == StampedExecutionStage::Kind::Matmul) {
            THOR_THROW_IF_FALSE(stage.matmul != nullptr);
            StampedMatmul& matmul = *stage.matmul;

            auto capture_runtime_scalar_write = [&](const std::optional<std::tuple<std::string, float, float*>>& spec) {
                if (!spec.has_value()) {
                    return;
                }
                const auto& [name, multiplier, destination] = spec.value();
                const std::vector<cudaGraphNode_t> before_nodes = graphNodes(graph);
                captureIntoExistingGraph(graph, capture_stream, current_dependencies, [&]() {
                    launchWriteFp32DeviceScalar(destination, multiplier, capture_stream.getStream());
                });
                cudaGraphNode_t kernel_node = uniqueNewKernelNode(graph, before_nodes);
                registerConditionalRuntimeScalarKernelBinding(
                    kernel_node,
                    2,
                    {detail::ConditionalRuntimeScalarKernelArgument{
                        .kernel_argument_index = 1,
                        .name = name,
                        .multiplier = multiplier,
                    }},
                    /*use_driver_api=*/false,
                    runtime_scalar_bindings);
                current_dependencies = graphLeafNodes(graph);
            };

            capture_runtime_scalar_write(detail::ConditionalGraphCaptureAccess::matmulAlphaHostRuntimeScalar(matmul));
            capture_runtime_scalar_write(detail::ConditionalGraphCaptureAccess::matmulBetaHostRuntimeScalar(matmul));
            captureIntoExistingGraph(graph, capture_stream, current_dependencies, [&]() {
                detail::ConditionalGraphCaptureAccess::runMatmulForConditionalGraphCapture(matmul, capture_stream);
            });
            current_dependencies = graphLeafNodes(graph);
            continue;
        }

        throw std::runtime_error(
            "Conditional CUDA graph capture encountered a host runtime scalar on an unsupported execution stage kind: " +
            StampedExecutionStage::kindToString(stage.kind));
    }

    if (detail::ConditionalGraphCaptureAccess::hasOutputMaterializations(plan)) {
        captureIntoExistingGraph(graph, capture_stream, current_dependencies, [&]() {
            detail::ConditionalGraphCaptureAccess::materializeOutputs(plan, capture_stream);
        });
        current_dependencies = graphLeafNodes(graph);
    }

    return current_dependencies;
}

static std::vector<cudaGraphNode_t> captureConditionalSetterIntoGraph(cudaGraphConditionalHandle conditional_handle,
                                                                     const Tensor& predicate,
                                                                     cudaGraph_t graph,
                                                                     Stream& capture_stream,
                                                                     const std::vector<cudaGraphNode_t>& dependencies) {
    captureIntoExistingGraph(graph, capture_stream, dependencies, [&]() {
        launchSetCudaGraphConditionalFromBool(conditional_handle, predicate, capture_stream);
    });
    return graphLeafNodes(graph);
}

static cudaGraphNode_t appendConditionalPlansIntoGraph(
    const StampedExecutionPlan& predicate_plan,
    const StampedExecutionPlan& then_plan,
    const StampedExecutionPlan& else_plan,
    cudaGraph_t graph,
    Stream& capture_stream,
    const std::vector<cudaGraphNode_t>& dependencies,
    std::vector<detail::ConditionalRuntimeScalarKernelBinding>& runtime_scalar_bindings) {
    Tensor predicate = predicate_plan.output();
    validateConditionalPredicateTensor(predicate);

    std::vector<cudaGraphNode_t> predicate_leaves =
        appendPlanSequentiallyIntoGraph(predicate_plan, graph, capture_stream, runtime_scalar_bindings, dependencies);
    if (predicate_leaves.empty()) {
        throw std::runtime_error("Graph-level conditional predicate graph produced no CUDA graph nodes.");
    }

    cudaGraphConditionalHandle conditional_handle{};
    CUDA_CHECK(cudaGraphConditionalHandleCreate(&conditional_handle, graph, 0, 0));

    std::vector<cudaGraphNode_t> setter_leaves =
        captureConditionalSetterIntoGraph(conditional_handle, predicate, graph, capture_stream, predicate_leaves);
    if (setter_leaves.empty()) {
        throw std::runtime_error("Graph-level conditional setter graph produced no CUDA graph leaf nodes.");
    }

    cudaGraphNodeParams conditional_params{};
    conditional_params.type = cudaGraphNodeTypeConditional;
    conditional_params.conditional.handle = conditional_handle;
    conditional_params.conditional.type = cudaGraphCondTypeIf;
    conditional_params.conditional.size = 2;

    cudaGraphNode_t conditional_node = nullptr;
    CUDA_CHECK(cudaGraphAddNode(
        &conditional_node, graph, setter_leaves.data(), nullptr, setter_leaves.size(), &conditional_params));

    if (conditional_params.conditional.phGraph_out == nullptr) {
        throw std::runtime_error("CUDA did not return body graphs for graph-level conditional node.");
    }

    (void)appendPlanSequentiallyIntoGraph(
        then_plan, conditional_params.conditional.phGraph_out[0], capture_stream, runtime_scalar_bindings);
    (void)appendPlanSequentiallyIntoGraph(
        else_plan, conditional_params.conditional.phGraph_out[1], capture_stream, runtime_scalar_bindings);
    return conditional_node;
}

struct BuiltConditionalCudaGraph {
    CudaGraphExecutable executable;
    std::vector<detail::ConditionalRuntimeScalarKernelBinding> runtime_scalar_bindings;
};

static BuiltConditionalCudaGraph buildConditionalCudaGraph(const StampedExecutionPlan& predicate_plan,
                                                            const StampedExecutionPlan& then_plan,
                                                            const StampedExecutionPlan& else_plan,
                                                            const Stream& stream) {
    Stream capture_stream(stream.getGpuNum());

    // Pre-create common library handles outside capture. Some stage types lazily create
    // cuDNN/cuBLAS handles from the stream; doing that while capture is active would make
    // the conditional graph path fragile.
    (void)capture_stream.getCudnnHandle();
    (void)capture_stream.getCublasHandle();

    cudaGraph_t root_graph = nullptr;
    CUDA_CHECK(cudaGraphCreate(&root_graph, 0));

    try {
        std::vector<detail::ConditionalRuntimeScalarKernelBinding> runtime_scalar_bindings;
        (void)appendConditionalPlansIntoGraph(
            predicate_plan, then_plan, else_plan, root_graph, capture_stream, {}, runtime_scalar_bindings);

        CudaGraph graph(root_graph, stream.getGpuNum(), false);
        root_graph = nullptr;
        // Runtime scalar bindings retain source-node handles, including handles from nested
        // conditional body graphs. Keep the captured source graph alive with the executable
        // for as long as those handles may be used by cudaGraphExecKernelNodeSetParams().
        CudaGraphExecutable executable = graph.instantiate(/*retainSourceGraph=*/true);
        executable.upload(capture_stream);
        return BuiltConditionalCudaGraph{
            .executable = std::move(executable),
            .runtime_scalar_bindings = std::move(runtime_scalar_bindings),
        };
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

    BuiltConditionalCudaGraph built =
        buildConditionalCudaGraph(*this->predicate_plan, *this->then_plan, *this->else_plan, this->stream);
    conditional_graph = std::move(built.executable);
    runtime_scalar_kernel_bindings = std::move(built.runtime_scalar_bindings);
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
    const std::unordered_set<std::string> expected_names = runtimeScalarNames();
    for (const std::string& name : expected_names) {
        if (!runtime_scalars.contains(name)) {
            throw std::runtime_error("Missing value for runtime scalar: " + name +
                                     "  - if it was meant to be constant, use a constant scalar instead.");
        }
    }
    for (const auto& [name, _] : runtime_scalars) {
        if (!expected_names.contains(name)) {
            throw std::runtime_error("Unexpected runtime scalar override for graph-level conditional: " + name);
        }
    }

    for (detail::ConditionalRuntimeScalarKernelBinding& binding : runtime_scalar_kernel_bindings) {
        binding.launch_kernel_params = binding.template_kernel_params;
        for (uint32_t i = 0; i < binding.runtime_arguments.size(); ++i) {
            const detail::ConditionalRuntimeScalarKernelArgument& argument = binding.runtime_arguments[i];
            auto it = runtime_scalars.find(argument.name);
            if (it == runtime_scalars.end()) {
                throw std::runtime_error("Missing value for conditional runtime scalar kernel binding: " + argument.name);
            }
            binding.runtime_values[i] = it->second * argument.multiplier;
            binding.launch_kernel_params[argument.kernel_argument_index] = &binding.runtime_values[i];
        }

        if (binding.use_driver_api) {
            CUDA_KERNEL_NODE_PARAMS params = binding.driver_template_params;
            params.kernelParams = binding.launch_kernel_params.data();
            params.extra = nullptr;
            conditional_graph.setDriverKernelNodeParams(reinterpret_cast<CUgraphNode>(binding.source_node), params);
        } else {
            cudaKernelNodeParams params = binding.runtime_template_params;
            params.kernelParams = binding.launch_kernel_params.data();
            params.extra = nullptr;
            conditional_graph.setKernelNodeParams(binding.source_node, params);
        }
    }

    // CUDA snapshots executable-node parameters for each cudaGraphLaunch. Thor's network scheduler
    // calls this method from a single host thread, so patching and enqueueing the launch are one
    // ordered scheduling operation while previously enqueued launches retain their own scalar values.
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

void StampedExecutionPlan::linkRmsNormBackwardStatesFrom(const StampedExecutionPlan& forward_plan) {
    for (const StampedExecutionStage& backward_stage : steps) {
        if (backward_stage.kind != StampedExecutionStage::Kind::RmsNormBackward || backward_stage.rms_norm_backward == nullptr) {
            continue;
        }
        bool linked = false;
        for (const StampedExecutionStage& forward_stage : forward_plan.steps) {
            if (forward_stage.kind != StampedExecutionStage::Kind::RmsNorm || forward_stage.rms_norm == nullptr) {
                continue;
            }
            if (backward_stage.rms_norm_backward->tryLinkForwardStateFrom(forward_stage.rms_norm)) {
                linked = true;
                break;
            }
        }
        if (!linked) {
            throw std::runtime_error(
                "RMSNorm backward stage could not find a matching forward RMSNorm state provider in the linked forward plan.");
        }
    }
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

void StampedExecutionPlan::materializeOutputsOn(Stream& run_stream) const {
    for (const StampedOutputMaterialization& materialization : output_materializations) {
        Tensor destination = materialization.destination;
        materializeTensorViewAsync(materialization.source, destination, run_stream);
    }
}

namespace {

class StampedExecutionEventPool {
   public:
    Event acquire(uint32_t gpu_num) {
        auto& gpu_events = free_events_[gpu_num];
        if (gpu_events.empty()) {
            return Event(static_cast<int32_t>(gpu_num), /*enableTiming=*/false, /*expectingHostToWaitOnThisOne=*/false);
        }

        Event event = gpu_events.back();
        gpu_events.pop_back();
        return event;
    }

    void release(const Event& event) { free_events_[static_cast<uint32_t>(event.getGpuNum())].push_back(event); }

   private:
    std::unordered_map<uint32_t, std::vector<Event>> free_events_;
};

// Submission is already host-thread local. A thread-local pool avoids both CUDA event create/destroy calls and a
// contended global lease lock in the hot path. Events are returned after all record/wait API calls for this submission
// have been issued; CUDA stream waits retain the captured event generation even if the handle is recorded again later.
StampedExecutionEventPool& stampedExecutionEventPool() {
    thread_local StampedExecutionEventPool pool;
    return pool;
}

class StampedExecutionEventLeases {
   public:
    explicit StampedExecutionEventLeases(size_t expected_event_count) { leased_events_.reserve(expected_event_count); }

    Event acquire(uint32_t gpu_num) {
        Event event = stampedExecutionEventPool().acquire(gpu_num);
        leased_events_.push_back(event);
        return event;
    }

    ~StampedExecutionEventLeases() {
        for (const Event& event : leased_events_) {
            stampedExecutionEventPool().release(event);
        }
    }

   private:
    std::vector<Event> leased_events_;
};

}  // namespace

detail::StampedExecutionSchedule detail::buildStampedExecutionSchedule(const std::vector<StampedExecutionStage>& steps,
                                                                        uint32_t caller_gpu_num) {
    StampedExecutionSchedule schedule;
    schedule.stage_lane_indices.resize(steps.size());
    schedule.stage_has_downstream_dependency.assign(steps.size(), false);
    schedule.stage_needs_completion_event.assign(steps.size(), false);
    schedule.lane_gpu_nums.push_back(caller_gpu_num);

    if (steps.empty()) {
        return schedule;
    }

    std::vector<std::vector<uint32_t>> children(steps.size());
    for (uint32_t stage_idx = 0; stage_idx < steps.size(); ++stage_idx) {
        for (uint32_t dep_stage_idx : steps[stage_idx].dependency_stage_indices) {
            if (dep_stage_idx >= stage_idx) {
                throw std::runtime_error(
                    "StampedExecutionPlan::runOn requires dependency_stage_indices to be topologically ordered.");
            }
            children[dep_stage_idx].push_back(stage_idx);
        }
    }

    // At a fork, exactly one child continues on the producer's lane. Other children receive new lanes and therefore
    // may execute concurrently. Prefer the first same-GPU child so a normal linear chain never leaves its stream.
    std::vector<std::optional<uint32_t>> continuation_child(steps.size());
    for (uint32_t stage_idx = 0; stage_idx < steps.size(); ++stage_idx) {
        for (uint32_t child_idx : children[stage_idx]) {
            if (steps[child_idx].gpu_num == steps[stage_idx].gpu_num) {
                continuation_child[stage_idx] = child_idx;
                break;
            }
        }
    }

    bool caller_root_claimed = false;
    for (uint32_t stage_idx = 0; stage_idx < steps.size(); ++stage_idx) {
        const StampedExecutionStage& stage = steps[stage_idx];

        std::optional<uint32_t> inherited_lane;
        for (uint32_t dep_stage_idx : stage.dependency_stage_indices) {
            if (!continuation_child[dep_stage_idx].has_value() ||
                continuation_child[dep_stage_idx].value() != stage_idx ||
                steps[dep_stage_idx].gpu_num != stage.gpu_num) {
                continue;
            }

            const uint32_t candidate_lane = schedule.stage_lane_indices[dep_stage_idx];
            if (!inherited_lane.has_value() || candidate_lane == 0) {
                inherited_lane = candidate_lane;
            }
            if (candidate_lane == 0) {
                break;
            }
        }

        if (inherited_lane.has_value()) {
            schedule.stage_lane_indices[stage_idx] = inherited_lane.value();
            continue;
        }

        if (stage.dependency_stage_indices.empty() && !caller_root_claimed && stage.gpu_num == caller_gpu_num) {
            schedule.stage_lane_indices[stage_idx] = 0;
            caller_root_claimed = true;
            continue;
        }

        schedule.stage_lane_indices[stage_idx] = static_cast<uint32_t>(schedule.lane_gpu_nums.size());
        schedule.lane_gpu_nums.push_back(stage.gpu_num);
    }

    for (uint32_t stage_idx = 0; stage_idx < steps.size(); ++stage_idx) {
        if (steps[stage_idx].dependency_stage_indices.empty() && schedule.stage_lane_indices[stage_idx] != 0) {
            schedule.needs_caller_ready_event = true;
        }
        for (uint32_t dep_stage_idx : steps[stage_idx].dependency_stage_indices) {
            schedule.stage_has_downstream_dependency[dep_stage_idx] = true;
            if (schedule.stage_lane_indices[dep_stage_idx] != schedule.stage_lane_indices[stage_idx]) {
                schedule.stage_needs_completion_event[dep_stage_idx] = true;
            }
        }
    }
    for (uint32_t stage_idx = 0; stage_idx < steps.size(); ++stage_idx) {
        if (!schedule.stage_has_downstream_dependency[stage_idx] && schedule.stage_lane_indices[stage_idx] != 0) {
            schedule.stage_needs_completion_event[stage_idx] = true;
        }
    }

    return schedule;
}

void StampedExecutionPlan::run() { runOn(stream); }

void StampedExecutionPlan::run(const std::unordered_map<std::string, float>& runtime_scalars) { runOn(stream, runtime_scalars); }

void StampedExecutionPlan::runOn(Stream& run_stream) const { runOn(run_stream, {}); }

void StampedExecutionPlan::runOn(Stream& run_stream, const std::unordered_map<std::string, float>& runtime_scalars) const {
    if (steps.empty()) {
        materializeOutputsOn(run_stream);
        return;
    }

    // The overwhelmingly common single-stage plan needs no DAG scheduler at runtime. Runtime-scalar plans retain the
    // general path so their override validation semantics remain unchanged.
    if (steps.size() == 1 && steps.front().dependency_stage_indices.empty() && runtime_scalars.empty()) {
        steps.front().runOn(run_stream);
        materializeOutputsOn(run_stream);
        return;
    }

    std::optional<detail::StampedExecutionSchedule> alternate_schedule;
    const detail::StampedExecutionSchedule* schedule_ptr = &execution_schedule;
    if (run_stream.getGpuNum() != stream.getGpuNum()) {
        alternate_schedule = detail::buildStampedExecutionSchedule(steps, static_cast<uint32_t>(run_stream.getGpuNum()));
        schedule_ptr = &alternate_schedule.value();
    }
    const detail::StampedExecutionSchedule& schedule = *schedule_ptr;

    // No branch means no scheduler: CUDA stream ordering is the entire dependency mechanism. This is the normal path
    // for a linear multi-stage expression and intentionally performs no helper-stream lookup or event bookkeeping.
    if (schedule.lane_gpu_nums.size() == 1) {
        std::unordered_set<std::string> consumed_runtime_scalar_names;
        for (const StampedExecutionStage& stage : steps) {
            std::unordered_map<std::string, float> stage_runtime_scalars;
            if (!runtime_scalars.empty()) {
                std::unordered_set<std::string> needed_names = runtimeScalarNamesForStage(stage);
                if (!needed_names.empty()) {
                    stage_runtime_scalars.reserve(needed_names.size());
                    for (const std::string& name : needed_names) {
                        auto it = runtime_scalars.find(name);
                        if (it == runtime_scalars.end()) {
                            throw std::runtime_error(
                                "Missing value for runtime scalar: " + name +
                                "  - if it was meant to be constant, use a constant scalar instead.");
                        }
                        stage_runtime_scalars.emplace(name, it->second);
                        consumed_runtime_scalar_names.insert(name);
                    }
                }
            }

            if (stage_runtime_scalars.empty())
                stage.runOn(run_stream);
            else
                stage.runOn(run_stream, stage_runtime_scalars);
        }

        for (const auto& [name, _] : runtime_scalars) {
            if (!consumed_runtime_scalar_names.contains(name)) {
                throw std::runtime_error("Unexpected runtime scalar override for stamped execution plan: " + name);
            }
        }
        materializeOutputsOn(run_stream);
        return;
    }

    std::vector<Stream> lane_streams;
    lane_streams.reserve(schedule.lane_gpu_nums.size());
    lane_streams.push_back(run_stream);
    for (uint32_t lane_idx = 1; lane_idx < schedule.lane_gpu_nums.size(); ++lane_idx) {
        lane_streams.push_back(Expression::getNextHelperStream(schedule.lane_gpu_nums[lane_idx]));
    }

    size_t expected_event_count = schedule.needs_caller_ready_event ? 1 : 0;
    for (bool needs_event : schedule.stage_needs_completion_event) {
        expected_event_count += needs_event ? 1 : 0;
    }

    StampedExecutionEventLeases event_leases(expected_event_count);
    std::optional<Event> caller_stream_ready;
    if (schedule.needs_caller_ready_event) {
        caller_stream_ready = event_leases.acquire(static_cast<uint32_t>(run_stream.getGpuNum()));
        run_stream.putEvent(caller_stream_ready.value());
    }

    std::vector<std::optional<Event>> completion_events(steps.size());
    std::unordered_set<std::string> consumed_runtime_scalar_names;

    for (uint32_t stage_idx = 0; stage_idx < steps.size(); ++stage_idx) {
        const StampedExecutionStage& stage = steps[stage_idx];
        Stream& launch_stream_ref = lane_streams[schedule.stage_lane_indices[stage_idx]];

        if (stage.dependency_stage_indices.empty() && schedule.stage_lane_indices[stage_idx] != 0) {
            THOR_THROW_IF_FALSE(caller_stream_ready.has_value());
            launch_stream_ref.waitEvent(caller_stream_ready.value());
        }

        for (uint32_t dep_stage_idx : stage.dependency_stage_indices) {
            Stream& dependency_stream = lane_streams[schedule.stage_lane_indices[dep_stage_idx]];
            if (launch_stream_ref == dependency_stream) {
                continue;
            }
            if (!completion_events[dep_stage_idx].has_value()) {
                throw std::runtime_error("StampedExecutionPlan::runOn missing completion event for cross-stream dependency stage.");
            }
            launch_stream_ref.waitEvent(completion_events[dep_stage_idx].value());
        }

        std::unordered_map<std::string, float> stage_runtime_scalars;
        if (!runtime_scalars.empty()) {
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
        }

        if (stage_runtime_scalars.empty())
            stage.runOn(launch_stream_ref);
        else
            stage.runOn(launch_stream_ref, stage_runtime_scalars);

        if (schedule.stage_needs_completion_event[stage_idx]) {
            completion_events[stage_idx] = event_leases.acquire(stage.gpu_num);
            launch_stream_ref.putEvent(completion_events[stage_idx].value());
        }
    }

    for (const auto& [name, _] : runtime_scalars) {
        if (!consumed_runtime_scalar_names.contains(name)) {
            throw std::runtime_error("Unexpected runtime scalar override for stamped execution plan: " + name);
        }
    }

    // Only terminal helper work needs an explicit final join. Helper branches that rejoin a caller-stream stage have
    // already been transitively joined by that stage's cross-stream dependency wait.
    for (uint32_t stage_idx = 0; stage_idx < steps.size(); ++stage_idx) {
        Stream& launch_stream_ref = lane_streams[schedule.stage_lane_indices[stage_idx]];
        if (schedule.stage_has_downstream_dependency[stage_idx] || launch_stream_ref == run_stream) {
            continue;
        }
        THOR_THROW_IF_FALSE(completion_events[stage_idx].has_value());
        run_stream.waitEvent(completion_events[stage_idx].value());
    }

    materializeOutputsOn(run_stream);
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

static cudnnDataType_t toCudnnSoftmaxDataType(DataType dtype) {
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
            throw std::runtime_error("toCudnnSoftmaxDataType: unsupported DataType value " + std::to_string(static_cast<int>(dtype)));
    }
}

static CubReductionOp toCubReductionOp(ExprOp op) {
    switch (op) {
        case ExprOp::REDUCE_SUM:
            return CubReductionOp::Sum;
        case ExprOp::REDUCE_PROD:
            return CubReductionOp::Product;
        case ExprOp::REDUCE_MIN:
            return CubReductionOp::Min;
        case ExprOp::REDUCE_MAX:
            return CubReductionOp::Max;
        case ExprOp::REDUCE_AVG:
            return CubReductionOp::Mean;
        case ExprOp::REDUCE_NORM1:
            return CubReductionOp::L1Norm;
        case ExprOp::REDUCE_NORM2:
            return CubReductionOp::L2Norm;
        default:
            throw std::runtime_error("ExprOp is not a supported CUB value reduction op.");
    }
}

static CubArgReductionOp toCubArgReductionOp(ExprOp op) {
    switch (op) {
        case ExprOp::REDUCE_MIN:
        case ExprOp::REDUCE_ARGMIN:
        case ExprOp::REDUCE_MIN_BACKWARD:
            return CubArgReductionOp::ArgMin;
        case ExprOp::REDUCE_MAX:
        case ExprOp::REDUCE_ARGMAX:
        case ExprOp::REDUCE_MAX_BACKWARD:
            return CubArgReductionOp::ArgMax;
        default:
            throw std::runtime_error("ExprOp is not a supported CUB arg reduction op.");
    }
}

static std::vector<uint32_t> narrowReductionAxes(const std::vector<uint64_t>& axes) {
    std::vector<uint32_t> narrowed;
    narrowed.reserve(axes.size());
    for (uint64_t axis : axes) {
        if (axis > UINT32_MAX) {
            throw std::runtime_error("Reduction axis exceeds UINT32_MAX.");
        }
        narrowed.push_back(static_cast<uint32_t>(axis));
    }
    return narrowed;
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

static cudnnTensorDescriptor_t createCudnnSoftmaxTensorDescriptor(std::vector<uint64_t> dims, DataType dtype) {
    while (dims.size() < 4)
        dims.push_back(1);
    if (dims.size() > 8)
        throw std::runtime_error("cuDNN softmax tensor descriptors support rank <= 8.");

    std::vector<int> cudnn_dims(dims.begin(), dims.end());
    std::vector<int> strides(cudnn_dims.size());
    strides.back() = 1;
    for (int i = static_cast<int>(cudnn_dims.size()) - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * cudnn_dims[i + 1];

    cudnnTensorDescriptor_t desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
    CUDNN_CHECK(
        cudnnSetTensorNdDescriptor(desc, toCudnnSoftmaxDataType(dtype), static_cast<int>(cudnn_dims.size()), cudnn_dims.data(), strides.data()));
    return desc;
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

struct FrontendConvolutionAutotuneCandidate {
    int64_t plan_index = -1;
    int64_t workspace_bytes = 0;
    bool failed = false;
    std::vector<float> samples_ms;
};

struct FrontendConvolutionAutotuneTensorPool {
    std::vector<std::vector<Tensor>> rotating_tensors_by_binding;
    std::vector<std::unordered_map<int64_t, void*>> tensor_packs;
    std::vector<Tensor> workspaces;
};

constexpr float kConvolutionAutotuneTieRelativeTolerance = 0.001f;

static uint64_t safeAddAutotuneBytes(uint64_t a, uint64_t b) {
    if (b > std::numeric_limits<uint64_t>::max() - a) {
        return std::numeric_limits<uint64_t>::max();
    }
    return a + b;
}

static int chooseFrontendConvolutionAutotuneRotationSlots(const std::vector<FrontendConvolutionAutotuneBinding>& bindings,
                                                          const TensorPlacement& placement,
                                                          int64_t workspace_bytes) {
    uint64_t rotating_slot_bytes = workspace_bytes > 0 ? static_cast<uint64_t>(workspace_bytes) : 0;
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

    const uint64_t max_affordable_slots = std::max<uint64_t>(1, available_for_autotune / rotating_slot_bytes);
    const uint64_t target_slots =
        std::max<uint64_t>(1, (kConvolutionAutotuneTargetRotationBytes + rotating_slot_bytes - 1) / rotating_slot_bytes);
    const uint64_t bounded_slots =
        std::min<uint64_t>(static_cast<uint64_t>(kConvolutionAutotuneMaxRotationSlots), std::min(target_slots, max_affordable_slots));
    return static_cast<int>(std::max<uint64_t>(1, bounded_slots));
}

static FrontendConvolutionAutotuneTensorPool createFrontendConvolutionAutotuneTensorPool(
    const std::vector<FrontendConvolutionAutotuneBinding>& bindings, const TensorPlacement& workspace_placement, int64_t workspace_bytes) {
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
        pool.workspaces.reserve(rotation_slots);
        for (int slot = 0; slot < rotation_slots; ++slot) {
            pool.workspaces.emplace_back(workspace_placement, TensorDescriptor(DataType::UINT8, {static_cast<uint64_t>(workspace_bytes)}));
        }
    }

    return pool;
}

static void touchFrontendConvolutionAutotuneTensorPool(FrontendConvolutionAutotuneTensorPool& pool, const Stream& stream) {
    Stream touch_stream = stream;
    for (std::vector<Tensor>& tensors : pool.rotating_tensors_by_binding) {
        for (Tensor& tensor : tensors) {
            tensor.memsetAsync(touch_stream, 0);
        }
    }
    for (Tensor& workspace : pool.workspaces) {
        workspace.memsetAsync(touch_stream, 0);
    }
    touch_stream.synchronize();
}

static void* frontendConvolutionAutotuneWorkspacePointer(FrontendConvolutionAutotuneTensorPool& pool, int iteration) {
    if (pool.workspaces.empty()) {
        return nullptr;
    }
    Tensor& workspace = pool.workspaces[static_cast<size_t>(iteration) % pool.workspaces.size()];
    return const_cast<void*>(static_cast<const void*>(workspace.getMemPtr<void>()));
}

static void selectFrontendConvolutionPlan(BuiltConvolution& built, const Stream& stream, int64_t plan_index, const char* op_name) {
    auto status = built.frontend_graph->build_plan_at_index(stream.getCudnnHandle(), plan_index);
    checkFrontendStatus(status, std::string("Failed to build cuDNN Frontend ") + op_name + " execution plan during autotune");
}

static void executeFrontendConvolutionPlanOnce(BuiltConvolution& built,
                                               const Stream& stream,
                                               FrontendConvolutionAutotuneTensorPool& pool,
                                               int iteration,
                                               const char* op_name) {
    if (!built.frontend_graph) {
        throw std::runtime_error(std::string(op_name) + " missing cuDNN Frontend graph.");
    }
    if (pool.tensor_packs.empty()) {
        throw std::runtime_error(std::string(op_name) + " autotune tensor pack rotation pool is empty.");
    }

    std::unordered_map<int64_t, void*>& tensor_pack = pool.tensor_packs[static_cast<size_t>(iteration) % pool.tensor_packs.size()];
    void* workspace_ptr = frontendConvolutionAutotuneWorkspacePointer(pool, iteration);
    auto status = built.frontend_graph->execute(stream.getCudnnHandle(), tensor_pack, workspace_ptr);
    if (!status.is_good()) {
        throw std::runtime_error(std::string("Failed to execute cuDNN Frontend ") + op_name +
                                 " plan during autotune: " + status.get_message());
    }
}

static float timeFrontendConvolutionPlanOnce(BuiltConvolution& built,
                                             const Stream& stream,
                                             FrontendConvolutionAutotuneTensorPool& pool,
                                             int iteration,
                                             const char* op_name) {
    Event start(stream.getGpuNum(), true, true);
    Event stop(stream.getGpuNum(), true, true);
    start.record(stream);
    executeFrontendConvolutionPlanOnce(built, stream, pool, iteration, op_name);
    stop.record(stream);
    return stop.synchronizeAndReportElapsedTimeInMilliseconds(start);
}

static float scoreFrontendConvolutionAutotuneSamples(const std::vector<float>& samples_ms) {
    if (samples_ms.empty()) {
        return std::numeric_limits<float>::infinity();
    }
    std::vector<float> sorted = samples_ms;
    std::sort(sorted.begin(), sorted.end());
    const size_t drop_each_side = sorted.size() >= 5 ? 1 : 0;
    const size_t begin = drop_each_side;
    const size_t end = sorted.size() - drop_each_side;
    double total_ms = 0.0;
    size_t count = 0;
    for (size_t i = begin; i < end; ++i) {
        if (!std::isfinite(sorted[i])) {
            continue;
        }
        total_ms += static_cast<double>(sorted[i]);
        ++count;
    }
    if (count == 0) {
        return std::numeric_limits<float>::infinity();
    }
    return static_cast<float>(total_ms / static_cast<double>(count));
}

static bool isBetterFrontendConvolutionCandidate(float candidate_score_ms,
                                                 int64_t candidate_plan_index,
                                                 float best_score_ms,
                                                 int64_t best_plan_index) {
    if (!std::isfinite(candidate_score_ms)) {
        return false;
    }
    if (!std::isfinite(best_score_ms) || best_plan_index < 0) {
        return true;
    }
    const float decisive_delta_ms = std::max(best_score_ms * kConvolutionAutotuneTieRelativeTolerance, 1.0e-6f);
    if (candidate_score_ms + decisive_delta_ms < best_score_ms) {
        return true;
    }
    if (std::fabs(candidate_score_ms - best_score_ms) <= decisive_delta_ms && candidate_plan_index < best_plan_index) {
        return true;
    }
    return false;
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

    std::vector<FrontendConvolutionAutotuneCandidate> candidates;
    candidates.reserve(static_cast<size_t>(candidate_count));
    int64_t max_workspace_bytes = 0;

    for (int64_t plan_index = 0; plan_index < candidate_count; ++plan_index) {
        auto status = built.frontend_graph->build_plan_at_index(stream.getCudnnHandle(), plan_index);
        if (!status.is_good()) {
            continue;
        }

        const int64_t workspace_bytes = checkedFrontendPlanWorkspaceBytes(built, plan_index, op_name);
        FrontendConvolutionAutotuneCandidate candidate;
        candidate.plan_index = plan_index;
        candidate.workspace_bytes = workspace_bytes;
        candidate.samples_ms.reserve(kConvolutionAutotuneTimedIterations);
        candidates.push_back(std::move(candidate));
        max_workspace_bytes = std::max(max_workspace_bytes, workspace_bytes);
    }

    if (candidates.empty()) {
        throw std::runtime_error(std::string("cuDNN Frontend ") + op_name + " autotune could not build any of the top " +
                                 std::to_string(candidate_count) + " heuristic-ranked execution plans.");
    }

    FrontendConvolutionAutotuneTensorPool timing_pool =
        createFrontendConvolutionAutotuneTensorPool(autotune_bindings, workspace_placement, max_workspace_bytes);
    touchFrontendConvolutionAutotuneTensorPool(timing_pool, stream);

    auto mark_candidate_failed = [&](FrontendConvolutionAutotuneCandidate& candidate) {
        candidate.failed = true;
        candidate.samples_ms.clear();
        Stream recovery_stream = stream;
        try {
            recovery_stream.synchronize();
        } catch (const std::exception&) {
            // If the failed cuDNN plan left an asynchronous runtime error behind, the next concrete execution
            // will report it. Do not let one autotune candidate select a bad plan for this convolution.
        }
    };

    int iteration = 0;
    for (int warmup = 0; warmup < kConvolutionAutotuneWarmupIterations; ++warmup) {
        for (size_t offset = 0; offset < candidates.size(); ++offset) {
            const size_t candidate_index = (offset + static_cast<size_t>(warmup)) % candidates.size();
            FrontendConvolutionAutotuneCandidate& candidate = candidates[candidate_index];
            if (candidate.failed) {
                continue;
            }

            try {
                selectFrontendConvolutionPlan(built, stream, candidate.plan_index, op_name);
                executeFrontendConvolutionPlanOnce(built, stream, timing_pool, iteration++, op_name);
            } catch (const std::exception&) {
                mark_candidate_failed(candidate);
            }
        }
    }

    Stream timing_stream = stream;
    timing_stream.synchronize();

    for (int timed_round = 0; timed_round < kConvolutionAutotuneTimedIterations; ++timed_round) {
        for (size_t offset = 0; offset < candidates.size(); ++offset) {
            const size_t candidate_index = (offset + static_cast<size_t>(timed_round)) % candidates.size();
            FrontendConvolutionAutotuneCandidate& candidate = candidates[candidate_index];
            if (candidate.failed) {
                continue;
            }

            try {
                selectFrontendConvolutionPlan(built, stream, candidate.plan_index, op_name);
                const float milliseconds = timeFrontendConvolutionPlanOnce(built, stream, timing_pool, iteration++, op_name);
                if (std::isfinite(milliseconds)) {
                    candidate.samples_ms.push_back(milliseconds);
                } else {
                    mark_candidate_failed(candidate);
                }
            } catch (const std::exception&) {
                mark_candidate_failed(candidate);
            }
        }
    }

    int64_t best_plan_index = -1;
    int64_t best_workspace_bytes = 0;
    float best_score_ms = std::numeric_limits<float>::infinity();

    for (const FrontendConvolutionAutotuneCandidate& candidate : candidates) {
        if (candidate.failed || candidate.samples_ms.empty()) {
            continue;
        }
        const float score_ms = scoreFrontendConvolutionAutotuneSamples(candidate.samples_ms);
        if (isBetterFrontendConvolutionCandidate(score_ms, candidate.plan_index, best_score_ms, best_plan_index)) {
            best_score_ms = score_ms;
            best_plan_index = candidate.plan_index;
            best_workspace_bytes = candidate.workspace_bytes;
        }
    }

    if (best_plan_index < 0) {
        throw std::runtime_error(std::string("cuDNN Frontend ") + op_name + " autotune could not run any of the top " +
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
        throw std::runtime_error("GEMM/epilogue workspace planning currently only supports rank-2 tensors.");
    }
    return static_cast<int32_t>(dims[1]);
}

static int32_t checkedMatmulInt32(uint64_t value, const char* role) {
    if (value > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error(std::string("Expression matmul ") + role + " exceeds the cuBLASLt int32 limit.");
    }
    return static_cast<int32_t>(value);
}

static int64_t checkedMatmulInt64(uint64_t value, const char* role) {
    if (value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::runtime_error(std::string("Expression matmul ") + role + " exceeds the cuBLASLt int64 stride limit.");
    }
    return static_cast<int64_t>(value);
}

static BucketedCublasGemmRowBinding toBucketedRowBinding(MatmulPackedRowBinding binding) {
    switch (binding) {
        case MatmulPackedRowBinding::RowsA: return BucketedCublasGemmRowBinding::RowsA;
        case MatmulPackedRowBinding::RowsB: return BucketedCublasGemmRowBinding::RowsB;
        case MatmulPackedRowBinding::RowsAAndRowsB: return BucketedCublasGemmRowBinding::RowsAAndRowsB;
        case MatmulPackedRowBinding::None: break;
    }
    throw std::invalid_argument("Expression packed-row matmul requires a non-empty row binding.");
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

    const bool plain_matmul = compiled_matmul->op == ExprOp::MATMUL && compiled_matmul->epilogue == MatmulEpilogue::Default &&
                              compiled_matmul->backward_epilogue == MatmulBackwardEpilogue::Default &&
                              !compiled_matmul->bgrad_output_dtype.has_value();
    const bool bucketed_packed_rows = compiled_matmul->packed_row_binding != MatmulPackedRowBinding::None;
    if (bucketed_packed_rows && (!plain_matmul || compiled_matmul->packed_row_capacity == 0)) {
        throw std::runtime_error("Packed-row bucketed expression matmul currently requires a plain MATMUL with non-zero capacity.");
    }
    if (plain_matmul) {
        if (lhs.getDimensions().size() < 2 || rhs.getDimensions().size() < 2 || output.getDimensions().size() < 2) {
            throw std::runtime_error("buildMatmul requires rank >= 2 tensors for MATMUL.");
        }
    } else if (lhs.getDimensions().size() != 2 || rhs.getDimensions().size() != 2 || output.getDimensions().size() != 2) {
        throw std::runtime_error("GEMM and fused MATMUL epilogues currently require rank-2 tensors.");
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

    int32_t a_rows = 0;
    int32_t a_cols = 0;
    int32_t b_rows = 0;
    int32_t b_cols = 0;
    int32_t ld_a = 0;
    int32_t ld_b = 0;
    int32_t ld_c = 0;
    int32_t ld_d = 0;
    bool backend_transpose_a = compiled_matmul->transpose_lhs;
    bool backend_transpose_b = compiled_matmul->transpose_rhs;
    bool backend_transpose_c = compiled_matmul->transpose_aux;
    CublasStridedBatchConfig batch_config = CublasStridedBatchConfig::single();

    if (plain_matmul) {
        const BatchedMatmulLayoutPlan layout_plan =
            planBatchedMatmulLayout(lhs, rhs, output, compiled_matmul->transpose_lhs, compiled_matmul->transpose_rhs);
        if (!layout_plan.canLowerWithoutMaterialization()) {
            throw std::runtime_error(
                "Expression MATMUL layout cannot be consumed directly by cuBLASLt without materialization/postprocessing. "
                "Plain MATMUL lowering does not materialize operands; use a directly addressable view/layout.");
        }
        if (!layout_plan.grouping.isSingleStridedBatch()) {
            throw std::runtime_error(
                "Expression MATMUL buildMatmul received a layout that requires multiple regular strided-batched groups. "
                "The Expression lowering layer must decompose it into single-group views before building cuBLASLt kernels.");
        }

        a_rows = checkedMatmulInt32(layout_plan.lhs_matrix.stored_rows, "lhs rows");
        a_cols = checkedMatmulInt32(layout_plan.lhs_matrix.stored_cols, "lhs columns");
        b_rows = checkedMatmulInt32(layout_plan.rhs_matrix.stored_rows, "rhs rows");
        b_cols = checkedMatmulInt32(layout_plan.rhs_matrix.stored_cols, "rhs columns");
        ld_a = checkedMatmulInt32(layout_plan.lhs_matrix.leading_dimension, "lhs leading dimension");
        ld_b = checkedMatmulInt32(layout_plan.rhs_matrix.leading_dimension, "rhs leading dimension");
        ld_d = checkedMatmulInt32(layout_plan.output_matrix.leading_dimension, "output leading dimension");
        ld_c = ld_d;
        backend_transpose_a = layout_plan.lhs_matrix.backend_transpose;
        backend_transpose_b = layout_plan.rhs_matrix.backend_transpose;
        backend_transpose_c = false;

        if (layout_plan.grouping.batch_count > 1) {
            batch_config = CublasStridedBatchConfig::strided(
                checkedMatmulInt32(layout_plan.grouping.batch_count, "batch count"),
                checkedMatmulInt64(layout_plan.grouping.lhs_batch_stride_elements, "lhs batch stride"),
                checkedMatmulInt64(layout_plan.grouping.rhs_batch_stride_elements, "rhs batch stride"),
                checkedMatmulInt64(layout_plan.grouping.output_batch_stride_elements, "output C batch stride"),
                checkedMatmulInt64(layout_plan.grouping.output_batch_stride_elements, "output D batch stride"));
        }
    } else {
        const std::vector<uint64_t> lhs_dims = lhs.getDimensions();
        const std::vector<uint64_t> rhs_dims = rhs.getDimensions();
        a_rows = checkedMatmulInt32(lhs_dims[0], "lhs rows");
        a_cols = checkedMatmulInt32(lhs_dims[1], "lhs columns");
        b_rows = checkedMatmulInt32(rhs_dims[0], "rhs rows");
        b_cols = checkedMatmulInt32(rhs_dims[1], "rhs columns");
        ld_a = leadingDimensionForStoredMatrix(lhs);
        ld_b = leadingDimensionForStoredMatrix(rhs);
        ld_d = leadingDimensionForStoredMatrix(output);
        ld_c = addend.has_value() ? (use_bias_epilogue ? ld_d : leadingDimensionForStoredMatrix(addend.value())) : ld_d;
    }

    if (bucketed_packed_rows) {
        if (batch_config.isBatched()) {
            throw std::runtime_error("Packed-row bucketed expression matmul does not support strided-batched layouts.");
        }
        const int32_t capacity = checkedMatmulInt32(compiled_matmul->packed_row_capacity, "packed row capacity");
        if ((compiled_matmul->packed_row_binding == MatmulPackedRowBinding::RowsA ||
             compiled_matmul->packed_row_binding == MatmulPackedRowBinding::RowsAAndRowsB) && a_rows != capacity) {
            throw std::runtime_error("Packed-row expression matmul rowsA does not match its declared full capacity.");
        }
        if ((compiled_matmul->packed_row_binding == MatmulPackedRowBinding::RowsB ||
             compiled_matmul->packed_row_binding == MatmulPackedRowBinding::RowsAAndRowsB) && b_rows != capacity) {
            throw std::runtime_error("Packed-row expression matmul rowsB does not match its declared full capacity.");
        }
    }

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

    const bool use_cublaslt_epilogue_wrapper =
        use_bias_epilogue || compiled_matmul->epilogue != MatmulEpilogue::Default || use_backward_epilogue;

    MatmulCacheKey key(compiled_matmul->op,
                       a_rows,
                       a_cols,
                       b_rows,
                       b_cols,
                       ld_a,
                       ld_b,
                       ld_c,
                       ld_d,
                       backend_transpose_a,
                       backend_transpose_b,
                       backend_transpose_c,
                       batch_config,
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

    std::shared_ptr<BuiltMatmul> hit = (use_cublaslt_epilogue_wrapper || bucketed_packed_rows) ? nullptr : cacheLookup(key);
    if (hit) {
        return hit;
    }

    auto built = std::make_shared<BuiltMatmul>(key);
    bool kernelWillRunOnGpu = false;
    const bool print_verbose_matmul_diagnostics = thorMatmulDiagnosticsVerbose();
    const char* diagnostic_path = "unknown";

    if (bucketed_packed_rows) {
        BucketedCublasGemmShape shape{a_rows, a_cols, b_rows, b_cols, ld_a, ld_b, ld_c, ld_d,
                                      backend_transpose_a, backend_transpose_b, backend_transpose_c};
        built->bucketed_cublas_gemm = BucketedCublasGemm::build(device_num,
                                                                 compiled_matmul->packed_row_capacity,
                                                                 shape,
                                                                 toBucketedRowBinding(compiled_matmul->packed_row_binding),
                                                                 dataTypes,
                                                                 print_verbose_matmul_diagnostics);
        built->workspace_bytes = built->bucketed_cublas_gemm->getWorkspaceSizeInBytes();
        kernelWillRunOnGpu = true;
        diagnostic_path = "packed-row-bucketed-matmul";
    } else if (compiled_matmul->op == ExprOp::MATMUL && !use_cublaslt_epilogue_wrapper) {
        if (batch_config.isBatched()) {
            if (dataTypes.A == DataType::FP8_E4M3 || dataTypes.A == DataType::FP8_E5M2 || dataTypes.B == DataType::FP8_E4M3 ||
                dataTypes.B == DataType::FP8_E5M2) {
                throw std::runtime_error(
                    "Expression strided-batched MATMUL does not yet support FP8 because the centralized FP8 transpose-workspace path "
                    "is not batch-aware.");
            }
            CublasMatrixMultiply::instance().chooseOptimalStridedBatchedGemmKernel(device_num,
                                                                                    a_rows,
                                                                                    a_cols,
                                                                                    b_rows,
                                                                                    b_cols,
                                                                                    ld_a,
                                                                                    ld_b,
                                                                                    ld_c,
                                                                                    ld_d,
                                                                                    backend_transpose_a,
                                                                                    backend_transpose_b,
                                                                                    backend_transpose_c,
                                                                                    dataTypes,
                                                                                    batch_config,
                                                                                    print_verbose_matmul_diagnostics);
            diagnostic_path = "optimal-strided-batched-matmul-picker";
        } else {
            CublasMatrixMultiply::instance().chooseOptimalMatrixMultiplyKernel(device_num,
                                                                               a_rows,
                                                                               a_cols,
                                                                               b_rows,
                                                                               b_cols,
                                                                               ld_a,
                                                                               ld_b,
                                                                               ld_d,
                                                                               backend_transpose_a,
                                                                               backend_transpose_b,
                                                                               dataTypes,
                                                                               print_verbose_matmul_diagnostics);
            diagnostic_path = "optimal-matmul-picker";
        }
        built->cublas_kernel = CublasMatrixMultiply::instance().getCachedGemmKernel(device_num,
                                                                                    a_rows,
                                                                                    a_cols,
                                                                                    b_rows,
                                                                                    b_cols,
                                                                                    ld_a,
                                                                                    ld_b,
                                                                                    ld_c,
                                                                                    ld_d,
                                                                                    backend_transpose_a,
                                                                                    backend_transpose_b,
                                                                                    backend_transpose_c,
                                                                                    dataTypes,
                                                                                    true,
                                                                                    batch_config);
        built->workspace_bytes = built->cublas_kernel->getWorkspaceSizeInBytes(device_num);
        kernelWillRunOnGpu = true;
    } else if (use_cublaslt_epilogue_wrapper) {
        diagnostic_path = "epilogue-workspace-wrapper";
        if (use_backward_epilogue) {
            if (!epilogue_aux.has_value()) {
                throw std::runtime_error("buildMatmul backward cuBLASLt epilogue requires epilogue_aux.");
            }
            built->epilogue_plan = CublasMatrixMultiply::instance().buildGemmWithBackwardEpiloguePlan(
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
                epilogue_aux.value(),
                bgrad_output);
            kernelWillRunOnGpu = static_cast<bool>(built->epilogue_plan);
            if (kernelWillRunOnGpu) {
                built->epilogue_algorithm = built->epilogue_plan->algorithm;
                built->workspace_bytes = built->epilogue_plan->algorithm.workspace_size_in_bytes;
            }
        } else {
            built->epilogue_plan =
                CublasMatrixMultiply::instance().buildGemmWithEpiloguePlan(device_num,
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
                                                                           addend,
                                                                           use_bias_epilogue);
            kernelWillRunOnGpu = static_cast<bool>(built->epilogue_plan);
            if (kernelWillRunOnGpu) {
                built->epilogue_algorithm = built->epilogue_plan->algorithm;
                built->workspace_bytes = built->epilogue_plan->algorithm.workspace_size_in_bytes;
            }
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
        built->cublas_kernel = CublasMatrixMultiply::instance().getCachedGemmKernel(device_num,
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
                                                                                    true);
        built->workspace_bytes = built->cublas_kernel->getWorkspaceSizeInBytes(device_num);
        kernelWillRunOnGpu = true;
    }

    if (!kernelWillRunOnGpu) {
        throw std::runtime_error("No GPU kernel available for the staged matmul/gemm configuration.");
    }

    if (thorMatmulDiagnosticsEnabled()) {
        std::ostringstream diagnostic_key;
        diagnostic_key << "build:" << diagnostic_path << ':' << device_num << ':' << matmulExprOpName(compiled_matmul->op) << ':' << a_rows
                       << 'x' << a_cols << ':' << b_rows << 'x' << b_cols << ":ld=" << ld_a << ',' << ld_b << ',' << ld_c << ',' << ld_d
                       << ":trans=" << static_cast<int>(compiled_matmul->transpose_lhs) << static_cast<int>(compiled_matmul->transpose_rhs)
                       << static_cast<int>(compiled_matmul->transpose_aux) << ":bias=" << static_cast<int>(use_bias_epilogue)
                       << ":epilogue=" << matmulEpilogueName(compiled_matmul->epilogue)
                       << ":backward_epilogue=" << matmulBackwardEpilogueName(compiled_matmul->backward_epilogue)
                       << ":bgrad=" << static_cast<int>(bgrad_output.has_value())
                       << ":dtypes=" << TensorDescriptor::getElementTypeName(dataTypes.A) << ','
                       << TensorDescriptor::getElementTypeName(dataTypes.B) << ',' << TensorDescriptor::getElementTypeName(dataTypes.C)
                       << ',' << TensorDescriptor::getElementTypeName(dataTypes.D) << ','
                       << TensorDescriptor::getElementTypeName(dataTypes.compute);
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

    if (!use_cublaslt_epilogue_wrapper && !bucketed_packed_rows) {
        builtMatmulCache.put(key, built);
    }
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

    std::vector<FrontendConvolutionAutotuneBinding> autotune_bindings = {
        {CUDNN_FRONTEND_CONV_X_UID, input, true}, {CUDNN_FRONTEND_CONV_W_UID, filter, true}, {CUDNN_FRONTEND_CONV_Y_UID, output, true}};
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
                                                                             {CUDNN_FRONTEND_CONV_X_UID, output, true}};
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
                                                                         {CUDNN_FRONTEND_CONV_W_UID, output, true}};
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
                          ReductionResultKind::Value,
                          input,
                          device_num);
}

std::shared_ptr<BuiltReduction> StampedEquation::buildReduction(ExprOp op,
                                                                const std::vector<uint64_t>& reduction_axes,
                                                                const std::vector<uint64_t>& squeeze_axes,
                                                                DataType input_dtype,
                                                                DataType output_dtype,
                                                                DataType compute_dtype,
                                                                ReductionResultKind result_kind,
                                                                const Tensor& input,
                                                                int device_num) {
    if (result_kind == ReductionResultKind::Value && !isSupportedFusionFloatingType(output_dtype)) {
        throw std::runtime_error("Thor reduction stage requested unsupported floating-point output dtype.");
    }
    if (compute_dtype != DataType::FP32) {
        throw std::runtime_error("Thor reduction stages require FP32 compute.");
    }

    const std::vector<uint64_t> input_dims = input.getDimensions();

    ReductionCacheKey key(op,
                          input_dims,
                          input.getStridesElements(),
                          reduction_axes,
                          squeeze_axes,
                          input_dtype,
                          output_dtype,
                          compute_dtype,
                          result_kind,
                          device_num);

    std::shared_ptr<BuiltReduction> hit = cacheLookup(key);
    if (hit)
        return hit;

    auto built = std::make_shared<BuiltReduction>(key);
    if (input.getDataType() != built->key.input_dtype) {
        throw std::runtime_error("Reduction input tensor dtype does not match the compiled input dtype.");
    }

    const std::vector<uint32_t> axes = narrowReductionAxes(built->key.reduction_axes);
    built->geometry = CubReduction::analyzeGeometry(input_dims, input.getStridesElements(), axes);

    switch (built->key.result_kind) {
        case ReductionResultKind::Value:
            if (!isValueReductionOp(built->key.op)) {
                throw std::runtime_error("Value-reduction planning received a non-value reduction op.");
            }
            built->value_op = toCubReductionOp(built->key.op);
            break;
        case ReductionResultKind::Indices:
            built->arg_op = toCubArgReductionOp(built->key.op);
            break;
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
    built->x_desc = createCudnnSoftmaxTensorDescriptor(input.getDimensions(), built->key.input_dtype);
    built->y_desc = createCudnnSoftmaxTensorDescriptor(output.getDimensions(), built->key.output_dtype);

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

    for (size_t i = 0; i < inputNames.size(); ++i) {
        if (compiledEquation->input_kinds[i] == NamedInput::Kind::RuntimeScalarFp32) {
            names.insert(inputNames[i]);
        }
    }
    return names;
}

}  // namespace ThorImplementation
