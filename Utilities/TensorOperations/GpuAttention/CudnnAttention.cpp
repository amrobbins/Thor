#include "Utilities/TensorOperations/GpuAttention/CudnnAttention.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/ScopedGpu.h"

#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

#include <cudnn_frontend.h>

using namespace ThorImplementation;
using namespace std;

namespace {

constexpr int64_t UID_Q = 10;
constexpr int64_t UID_K = 11;
constexpr int64_t UID_V = 12;
constexpr int64_t UID_O = 13;
constexpr int64_t UID_STATS = 14;
constexpr int64_t UID_BIAS = 15;
constexpr int64_t UID_DBIA = 16;
constexpr int64_t UID_DO = 17;
constexpr int64_t UID_DQ = 18;
constexpr int64_t UID_DK = 19;
constexpr int64_t UID_DV = 20;
constexpr int64_t UID_SEQ_Q = 21;
constexpr int64_t UID_SEQ_KV = 22;
constexpr int64_t UID_RAGGED_Q = 23;
constexpr int64_t UID_RAGGED_K = 24;
constexpr int64_t UID_RAGGED_V = 25;
constexpr int64_t UID_RAGGED_O = 26;
constexpr int64_t UID_RAGGED_DO = 27;
constexpr int64_t UID_RAGGED_DQ = 28;
constexpr int64_t UID_RAGGED_DK = 29;
constexpr int64_t UID_RAGGED_DV = 30;
constexpr int64_t UID_DROPOUT_SEED = 31;
constexpr int64_t UID_DROPOUT_OFFSET = 32;
constexpr int64_t UID_DROPOUT_MASK = 33;
constexpr int64_t UID_DROPOUT_SCALE = 34;
constexpr int64_t UID_PAGE_TABLE_K = 35;
constexpr int64_t UID_PAGE_TABLE_V = 36;
constexpr int64_t UID_DESCALE_Q = 40;
constexpr int64_t UID_DESCALE_K = 41;
constexpr int64_t UID_DESCALE_V = 42;
constexpr int64_t UID_DESCALE_S = 43;
constexpr int64_t UID_DESCALE_O = 44;
constexpr int64_t UID_DESCALE_DO = 45;
constexpr int64_t UID_DESCALE_DP = 46;
constexpr int64_t UID_SCALE_S = 50;
constexpr int64_t UID_SCALE_O = 51;
constexpr int64_t UID_SCALE_DQ = 52;
constexpr int64_t UID_SCALE_DK = 53;
constexpr int64_t UID_SCALE_DV = 54;
constexpr int64_t UID_SCALE_DP = 55;
constexpr int64_t UID_AMAX_S = 60;
constexpr int64_t UID_AMAX_O = 61;
constexpr int64_t UID_AMAX_DQ = 62;
constexpr int64_t UID_AMAX_DK = 63;
constexpr int64_t UID_AMAX_DV = 64;
constexpr int64_t UID_AMAX_DP = 65;

void throwInvalidAttention(const string& message) { throw invalid_argument("Invalid cuDNN attention descriptor: " + message); }

bool envFlagEnabled(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && std::string_view(value) == "1";
}

bool experimentalCudnnAttentionSupportSurfaceProbeEnabled() {
    return envFlagEnabled("THOR_EXPERIMENTAL_CUDNN_ATTENTION_SUPPORT_SURFACE") ||
           envFlagEnabled("THOR_RUN_EXPERIMENTAL_CUDNN_ATTENTION_SUPPORT_SURFACE") ||
           envFlagEnabled("THOR_RUN_EXPERIMENTAL_CUDNN_FP8_ATTENTION_SUPPORT_SURFACE");
}

bool experimentalCudnnRaggedBiasBackwardProbeEnabled() {
    const char* value = std::getenv("THOR_EXPERIMENTAL_CUDNN_RAGGED_BIAS_BACKWARD");
    return (value != nullptr && std::string_view(value) == "1") || experimentalCudnnAttentionSupportSurfaceProbeEnabled();
}

vector<int64_t> asInt64(const vector<uint64_t>& values) {
    vector<int64_t> converted;
    converted.reserve(values.size());
    for (uint64_t value : values) {
        if (value > static_cast<uint64_t>(numeric_limits<int64_t>::max()))
            throwInvalidAttention("tensor dimension is too large for cuDNN Frontend int64 dimensions");
        converted.push_back(static_cast<int64_t>(value));
    }
    return converted;
}

string joinInts(const vector<int64_t>& values) {
    ostringstream out;
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0)
            out << ',';
        out << values[i];
    }
    out << ']';
    return out.str();
}

void appendSpec(ostringstream& out, string_view name, const AttentionTensorSpec& spec) {
    out << name << ".dim=" << joinInts(spec.dimensions) << ';';
    out << name << ".stride=" << joinInts(spec.strides) << ';';
    out << name << ".dtype=" << TensorDescriptor::getElementTypeName(spec.dataType) << ';';
    out << name << ".ragged=" << spec.ragged << ';';
}

bool isFp8(DataType dtype) {
    return dtype == DataType::FP8_E4M3 || dtype == DataType::FP8_E5M2;
}

bool isFp16OrBf16(DataType dtype) {
    return dtype == DataType::FP16 || dtype == DataType::BF16;
}

bool hasBshdPackedStrides(const AttentionTensorSpec& spec) {
    if (spec.dimensions.size() != 4 || spec.strides.size() != 4)
        return false;
    const int64_t heads = spec.dimensions.at(1);
    const int64_t sequenceLength = spec.dimensions.at(2);
    const int64_t headDim = spec.dimensions.at(3);
    return spec.strides == vector<int64_t>{sequenceLength * heads * headDim, headDim, heads * headDim, 1};
}

void requireInitialized(const Tensor& tensor, string_view name) {
    if (!tensor.isInitialized())
        throw invalid_argument(string("cuDNN attention tensor '") + string(name) + "' is uninitialized");
}

void requireGpuTensor(const Tensor& tensor, string_view name, int gpuNum) {
    requireInitialized(tensor, name);
    const TensorPlacement placement = tensor.getPlacement();
    if (placement.getMemDevice() != TensorPlacement::MemDevices::GPU)
        throw invalid_argument(string("cuDNN attention tensor '") + string(name) + "' must be on GPU memory");
    if (placement.getDeviceNum() != gpuNum)
        throw invalid_argument(string("cuDNN attention tensor '") + string(name) + "' is on GPU " + to_string(placement.getDeviceNum()) +
                               " but stream is on GPU " + to_string(gpuNum));
}

void requireOptionalGpuTensor(const optional<Tensor>& tensor, string_view name, int gpuNum) {
    if (!tensor.has_value())
        throw invalid_argument(string("cuDNN attention requires optional tensor '") + string(name) + "' but it was not provided");
    requireGpuTensor(tensor.value(), name, gpuNum);
}

void requireTensorMatchesSpec(const Tensor& tensor, const AttentionTensorSpec& spec, string_view name) {
    requireInitialized(tensor, name);
    if (tensor.getDataType() != spec.dataType) {
        throw invalid_argument(string("cuDNN attention tensor '") + string(name) + "' dtype mismatch. Expected " +
                               TensorDescriptor::getElementTypeName(spec.dataType) + ", got " +
                               TensorDescriptor::getElementTypeName(tensor.getDataType()));
    }
    const vector<int64_t> dims = asInt64(tensor.getDimensions());
    if (dims != spec.dimensions) {
        throw invalid_argument(string("cuDNN attention tensor '") + string(name) + "' dimension mismatch. Expected " +
                               joinInts(spec.dimensions) + ", got " + joinInts(dims));
    }
}

void requireFp8ScaleScalarMatchesDescriptor(const Tensor& scalar, string_view name) {
    requireInitialized(scalar, name);
    if (scalar.getDataType() != DataType::FP32) {
        throw invalid_argument(string("cuDNN FP8 attention scalar '") + string(name) + "' dtype mismatch. Expected FP32, got " +
                               TensorDescriptor::getElementTypeName(scalar.getDataType()));
    }
    const vector<int64_t> expected{1, 1, 1, 1};
    const vector<int64_t> dims = asInt64(scalar.getDimensions());
    if (dims != expected) {
        throw invalid_argument(string("cuDNN FP8 attention scalar '") + string(name) + "' dimension mismatch. Expected " +
                               joinInts(expected) + ", got " + joinInts(dims));
    }
}

// Additive bias is a score-space tensor, not a packed/ragged tensor. Even when Q/K/V/O use ragged THD storage,
// the bias is indexed by local logical sequence coordinates over the descriptor's max sequence domain.
// cuDNN's primary SDPA forward path supports broadcasting each score-bias input dimension independently.
// For backward, production Thor only sends dense or batch/head-broadcast bias directly to cuDNN. Biases
// broadcast across Sq and/or Skv are materialized as dense score-space tensors before cuDNN backward, then
// dense dBias is reduced explicitly back to the original public bias shape.
vector<int64_t> denseScoreBiasDimensions(const CudnnAttentionDescriptor& descriptor) {
    return {descriptor.batchSize(), descriptor.queryHeads(), descriptor.queryLength(), descriptor.keyValueLength()};
}

vector<int64_t> contiguousStrides(const vector<int64_t>& dims) {
    vector<int64_t> strides(dims.size(), 1);
    for (int64_t i = static_cast<int64_t>(dims.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * dims[static_cast<size_t>(i + 1)];
    }
    return strides;
}

vector<int64_t> denseScoreBiasStrides(const CudnnAttentionDescriptor& descriptor) { return contiguousStrides(denseScoreBiasDimensions(descriptor)); }

AttentionTensorSpec fullDenseScoreBiasSpec(const CudnnAttentionDescriptor& descriptor, DataType dataType) {
    AttentionTensorSpec spec;
    spec.dimensions = denseScoreBiasDimensions(descriptor);
    spec.strides = denseScoreBiasStrides(descriptor);
    spec.dataType = dataType;
    spec.ragged = false;
    return spec;
}

AttentionTensorSpec tensorSpecForBiasTensor(const Tensor& tensor) {
    AttentionTensorSpec spec;
    spec.dimensions = asInt64(tensor.getDimensions());
    spec.strides = asInt64(tensor.getStridesElements());
    spec.dataType = tensor.getDataType();
    spec.ragged = false;
    return spec;
}

const AttentionTensorSpec& scoreBiasSpecOrDefault(const CudnnAttentionDescriptor& descriptor,
                                                  DataType dataType,
                                                  AttentionTensorSpec& fallbackStorage) {
    if (descriptor.bias.has_value()) {
        return descriptor.bias.value();
    }
    fallbackStorage = fullDenseScoreBiasSpec(descriptor, dataType);
    return fallbackStorage;
}

const AttentionTensorSpec& scoreDBiasSpecOrDefault(const CudnnAttentionDescriptor& descriptor, AttentionTensorSpec& fallbackStorage) {
    if (descriptor.dBias.has_value()) {
        return descriptor.dBias.value();
    }
    fallbackStorage = fullDenseScoreBiasSpec(descriptor, descriptor.q.dataType);
    return fallbackStorage;
}

bool isSupportedAttentionDBiasDataType(DataType dtype, const CudnnAttentionDescriptor& descriptor) {
    return dtype == descriptor.q.dataType || dtype == descriptor.computeDataType;
}

DataType attentionForwardBiasDataType(const CudnnAttentionDescriptor& descriptor) {
    // cuDNN FP16/BF16 SDPA takes additive score bias in the compute dtype, while FP8 SDPA takes
    // additive score bias in the FP8 IO dtype.  Keeping this as an explicit helper makes the
    // support-surface probes exercise the same descriptor contract that production code would use.
    return descriptor.useFp8 ? descriptor.q.dataType : descriptor.computeDataType;
}

bool scoreBiasUsesSequenceBroadcast(const AttentionTensorSpec& spec, const CudnnAttentionDescriptor& descriptor) {
    const vector<int64_t> full = denseScoreBiasDimensions(descriptor);
    return spec.dimensions.size() == 4 &&
           ((spec.dimensions[2] == 1 && full[2] != 1) || (spec.dimensions[3] == 1 && full[3] != 1));
}

void validateScoreBiasSpec(const AttentionTensorSpec& spec,
                           const CudnnAttentionDescriptor& descriptor,
                           string_view name,
                           DataType expectedDataType,
                           bool allowBroadcast) {
    if (spec.dataType != expectedDataType) {
        throw invalid_argument(string("cuDNN attention tensor '") + string(name) + "' dtype mismatch. Expected " +
                               TensorDescriptor::getElementTypeName(expectedDataType) + ", got " +
                               TensorDescriptor::getElementTypeName(spec.dataType));
    }
    if (spec.dimensions.size() != 4 || spec.strides.size() != 4) {
        throw invalid_argument(string("cuDNN attention tensor '") + string(name) +
                               "' must be a rank-4 score-space tensor in [B,Hq,Sq,Skv] semantic order");
    }

    const vector<int64_t> full = denseScoreBiasDimensions(descriptor);
    const bool batchOk = spec.dimensions[0] == full[0] || (allowBroadcast && spec.dimensions[0] == 1);
    const bool headsOk = spec.dimensions[1] == full[1] || (allowBroadcast && spec.dimensions[1] == 1);
    const bool queryOk = spec.dimensions[2] == full[2] || (allowBroadcast && spec.dimensions[2] == 1);
    const bool keyValueOk = spec.dimensions[3] == full[3] || (allowBroadcast && spec.dimensions[3] == 1);
    if (!batchOk || !headsOk || !queryOk || !keyValueOk) {
        const string expected = allowBroadcast ? "[1|B,1|Hq,1|Sq,1|Skv]" : "[B,Hq,Sq,Skv]";
        throw invalid_argument(string("cuDNN attention tensor '") + string(name) +
                               "' dimension mismatch. Expected additive-bias shape " + expected + " for B/Hq/Sq/Skv " +
                               joinInts(full) + ", got " + joinInts(spec.dimensions));
    }

    const vector<int64_t> expectedStrides = contiguousStrides(spec.dimensions);
    if (spec.strides != expectedStrides) {
        throw invalid_argument(string("cuDNN attention tensor '") + string(name) +
                               "' stride mismatch. Expected contiguous additive-bias strides " + joinInts(expectedStrides) +
                               ", got " + joinInts(spec.strides));
    }
}

void requireBiasMatchesDescriptor(const Tensor& bias,
                                  const CudnnAttentionDescriptor& descriptor,
                                  string_view name,
                                  DataType expectedDataType,
                                  bool allowBroadcast) {
    requireInitialized(bias, name);
    const AttentionTensorSpec actual = tensorSpecForBiasTensor(bias);
    AttentionTensorSpec fallback;
    const AttentionTensorSpec& expected = scoreBiasSpecOrDefault(descriptor, expectedDataType, fallback);
    validateScoreBiasSpec(expected, descriptor, name, expectedDataType, allowBroadcast);
    if (actual.dataType != expected.dataType) {
        throw invalid_argument(string("cuDNN attention tensor '") + string(name) + "' dtype mismatch. Expected " +
                               TensorDescriptor::getElementTypeName(expected.dataType) + ", got " +
                               TensorDescriptor::getElementTypeName(actual.dataType));
    }
    if (actual.dimensions != expected.dimensions) {
        throw invalid_argument(string("cuDNN attention tensor '") + string(name) + "' dimension mismatch. Expected " +
                               joinInts(expected.dimensions) + ", got " + joinInts(actual.dimensions));
    }
    if (actual.strides != expected.strides) {
        throw invalid_argument(string("cuDNN attention tensor '") + string(name) + "' stride mismatch. Expected " +
                               joinInts(expected.strides) + ", got " + joinInts(actual.strides));
    }
}

void requireAttentionBiasMatchesDescriptor(const Tensor& bias, const CudnnAttentionDescriptor& descriptor, string_view name) {
    requireBiasMatchesDescriptor(bias, descriptor, name, attentionForwardBiasDataType(descriptor), true);
}

void requireAttentionDBiasMatchesDescriptor(const Tensor& bias, const CudnnAttentionDescriptor& descriptor, string_view name) {
    // Thor currently exposes dBias only for the full dense score tensor; broadcast dBias reduction
    // should be added as a separate explicit reduction stage.  The dtype is part of the runtime
    // backward graph signature because some expression plans materialize native dBias in the IO dtype
    // while others route it through an FP32 terminal gradient path.
    requireInitialized(bias, name);
    const AttentionTensorSpec actual = tensorSpecForBiasTensor(bias);
    AttentionTensorSpec fallback;
    const AttentionTensorSpec& expected = scoreDBiasSpecOrDefault(descriptor, fallback);
    validateScoreBiasSpec(expected, descriptor, name, expected.dataType, false);
    if (!isSupportedAttentionDBiasDataType(expected.dataType, descriptor)) {
        throw invalid_argument(string("cuDNN attention tensor '") + string(name) +
                               "' dtype mismatch. Expected dBias dtype to be either the q tensor dtype (" +
                               TensorDescriptor::getElementTypeName(descriptor.q.dataType) + ") or compute dtype (" +
                               TensorDescriptor::getElementTypeName(descriptor.computeDataType) + "), got " +
                               TensorDescriptor::getElementTypeName(expected.dataType));
    }
    if (actual.dataType != expected.dataType) {
        throw invalid_argument(string("cuDNN attention tensor '") + string(name) + "' dtype mismatch. Expected " +
                               TensorDescriptor::getElementTypeName(expected.dataType) + ", got " +
                               TensorDescriptor::getElementTypeName(actual.dataType));
    }
    if (actual.dimensions != expected.dimensions) {
        throw invalid_argument(string("cuDNN attention tensor '") + string(name) + "' dimension mismatch. Expected " +
                               joinInts(expected.dimensions) + ", got " + joinInts(actual.dimensions));
    }
    if (actual.strides != expected.strides) {
        throw invalid_argument(string("cuDNN attention tensor '") + string(name) + "' stride mismatch. Expected " +
                               joinInts(expected.strides) + ", got " + joinInts(actual.strides));
    }
}

CudnnAttentionDescriptor descriptorWithRuntimeBiasSpec(const CudnnAttentionDescriptor& descriptor,
                                                       const optional<Tensor>& bias,
                                                       DataType expectedDataType) {
    CudnnAttentionDescriptor withBiasSpec = descriptor;
    if (withBiasSpec.useBias && !withBiasSpec.bias.has_value() && bias.has_value()) {
        withBiasSpec.bias = tensorSpecForBiasTensor(bias.value());
        validateScoreBiasSpec(withBiasSpec.bias.value(), withBiasSpec, "bias", expectedDataType, true);
    }
    return withBiasSpec;
}

CudnnAttentionDescriptor descriptorWithRuntimeDBiasSpec(const CudnnAttentionDescriptor& descriptor, const optional<Tensor>& dBias) {
    CudnnAttentionDescriptor withDBiasSpec = descriptor;
    if (withDBiasSpec.useBias && !withDBiasSpec.dBias.has_value() && dBias.has_value()) {
        withDBiasSpec.dBias = tensorSpecForBiasTensor(dBias.value());
        validateScoreBiasSpec(withDBiasSpec.dBias.value(), withDBiasSpec, "dBias", withDBiasSpec.dBias->dataType, false);
    }
    return withDBiasSpec;
}

void requireSeqLenMatchesDescriptor(const Tensor& seq_len, const CudnnAttentionDescriptor& descriptor, string_view name) {
    requireInitialized(seq_len, name);
    if (seq_len.getDataType() != DataType::INT32) {
        throw invalid_argument(string("cuDNN attention tensor '") + string(name) + "' dtype mismatch. Expected INT32, got " +
                               TensorDescriptor::getElementTypeName(seq_len.getDataType()));
    }
    const vector<int64_t> expected{descriptor.batchSize()};
    const vector<int64_t> dims = asInt64(seq_len.getDimensions());
    if (dims != expected) {
        throw invalid_argument(string("cuDNN attention tensor '") + string(name) + "' dimension mismatch. Expected " +
                               joinInts(expected) + ", got " + joinInts(dims));
    }
}

void requireRaggedOffsetMatchesDescriptor(const Tensor& offset, const CudnnAttentionDescriptor& descriptor, string_view name) {
    requireInitialized(offset, name);
    if (offset.getDataType() != DataType::INT32) {
        throw invalid_argument(string("cuDNN attention tensor '") + string(name) + "' dtype mismatch. Expected INT32, got " +
                               TensorDescriptor::getElementTypeName(offset.getDataType()));
    }
    const vector<int64_t> expected{descriptor.batchSize() + 1};
    const vector<int64_t> dims = asInt64(offset.getDimensions());
    if (dims != expected) {
        throw invalid_argument(string("cuDNN attention tensor '") + string(name) + "' dimension mismatch. Expected " +
                               joinInts(expected) + ", got " + joinInts(dims));
    }
}

void requireDropoutScalarMatchesDescriptor(const Tensor& scalar, string_view name) {
    requireInitialized(scalar, name);
    if (scalar.getDataType() != DataType::INT64) {
        throw invalid_argument(string("cuDNN attention tensor '") + string(name) + "' dtype mismatch. Expected INT64, got " +
                               TensorDescriptor::getElementTypeName(scalar.getDataType()));
    }
    const vector<int64_t> expected{1, 1, 1, 1};
    const vector<int64_t> dims = asInt64(scalar.getDimensions());
    if (dims != expected) {
        throw invalid_argument(string("cuDNN attention tensor '") + string(name) + "' dimension mismatch. Expected " +
                               joinInts(expected) + ", got " + joinInts(dims));
    }
}

int64_t ceilDivPositive(int64_t numerator, int64_t denominator) {
    if (numerator <= 0 || denominator <= 0)
        throw invalid_argument("ceilDivPositive requires positive operands");
    return (numerator + denominator - 1) / denominator;
}

int64_t pagedKvBlockSizeForTable(const CudnnAttentionDescriptor& descriptor, string_view name) {
    const string label(name);
    if (label.find('V') != string::npos || label.find("_v") != string::npos || label.find("v_") != string::npos)
        return descriptor.v.dimensions.at(2);
    return descriptor.k.dimensions.at(2);
}

int64_t pagedKvPageCountForTable(const CudnnAttentionDescriptor& descriptor, string_view name) {
    return ceilDivPositive(descriptor.pagedKv.maxSequenceLengthKv, pagedKvBlockSizeForTable(descriptor, name));
}

void requirePagedKvTableMatchesDescriptor(const Tensor& table, const CudnnAttentionDescriptor& descriptor, string_view name) {
    requireInitialized(table, name);
    if (table.getDataType() != DataType::INT32) {
        throw invalid_argument(string("cuDNN attention tensor '") + string(name) + "' dtype mismatch. Expected INT32, got " +
                               TensorDescriptor::getElementTypeName(table.getDataType()));
    }
    const vector<int64_t> expected{descriptor.batchSize(), 1, pagedKvPageCountForTable(descriptor, name), 1};
    const vector<int64_t> dims = asInt64(table.getDimensions());
    if (dims != expected) {
        throw invalid_argument(string("cuDNN attention tensor '") + string(name) + "' dimension mismatch. Expected " +
                               joinInts(expected) + ", got " + joinInts(dims));
    }
}

namespace fe = cudnn_frontend;

fe::DataType_t toFrontendDataType(DataType dtype) {
    switch (dtype) {
        case DataType::FP16:
            return fe::DataType_t::HALF;
        case DataType::BF16:
            return fe::DataType_t::BFLOAT16;
        case DataType::FP32:
            return fe::DataType_t::FLOAT;
        case DataType::FP8_E4M3:
            return fe::DataType_t::FP8_E4M3;
        case DataType::FP8_E5M2:
            return fe::DataType_t::FP8_E5M2;
        case DataType::INT32:
            return fe::DataType_t::INT32;
        case DataType::INT64:
            return fe::DataType_t::INT64;
        default:
            throw invalid_argument("Unsupported cuDNN Frontend attention dtype: " + TensorDescriptor::getElementTypeName(dtype));
    }
}

struct BuiltGraph {
    shared_ptr<fe::graph::Graph> graph;
    Tensor workspace;
    int64_t workspaceBytes = 0;
};

class AttentionGraphCache {
   public:
    BuiltGraph& getOrBuildForward(const CudnnAttentionDescriptor& descriptor, int gpuNum) {
        const string key = descriptor.cacheKey("forward", gpuNum);
        unique_lock<mutex> lock(mtx);
        auto iter = graphs.find(key);
        if (iter != graphs.end())
            return iter->second;
        BuiltGraph graph = buildForwardGraph(descriptor, gpuNum);
        auto [inserted, _] = graphs.emplace(key, std::move(graph));
        return inserted->second;
    }

    BuiltGraph& getOrBuildBackward(const CudnnAttentionDescriptor& descriptor, int gpuNum) {
        const string key = descriptor.cacheKey("backward", gpuNum);
        unique_lock<mutex> lock(mtx);
        auto iter = graphs.find(key);
        if (iter != graphs.end())
            return iter->second;
        BuiltGraph graph = buildBackwardGraph(descriptor, gpuNum);
        auto [inserted, _] = graphs.emplace(key, std::move(graph));
        return inserted->second;
    }

    void clear() {
        unique_lock<mutex> lock(mtx);
        graphs.clear();
    }

    size_t size() const {
        unique_lock<mutex> lock(mtx);
        return graphs.size();
    }

   private:
    shared_ptr<fe::graph::Tensor_attributes> tensor(shared_ptr<fe::graph::Graph>& graph,
                                                    string_view name,
                                                    int64_t uid,
                                                    const vector<int64_t>& dim,
                                                    const vector<int64_t>& stride,
                                                    DataType dtype) {
        return graph->tensor(fe::graph::Tensor_attributes()
                                 .set_name(string(name))
                                 .set_uid(uid)
                                 .set_dim(dim)
                                 .set_stride(stride)
                                 .set_data_type(toFrontendDataType(dtype)));
    }

    shared_ptr<fe::graph::Tensor_attributes> ioTensor(shared_ptr<fe::graph::Graph>& graph,
                                                      string_view name,
                                                      int64_t uid,
                                                      const vector<int64_t>& dim,
                                                      const vector<int64_t>& stride) {
        // Match the cuDNN Frontend SDPA backward sample: IO tensors inherit the graph
        // io_data_type instead of redundantly setting per-tensor dtypes.  With
        // cudnn-frontend 1.23 + cuDNN 9.21, explicitly re-setting the IO tensor
        // dtype on the backward graph can trigger an internal reshape-mode
        // attribute BAD_PARAM during graph construction.  Non-IO tensors such as
        // stats and sequence lengths still use explicit dtypes below.
        return graph->tensor(fe::graph::Tensor_attributes()
                                 .set_name(string(name))
                                 .set_uid(uid)
                                 .set_dim(dim)
                                 .set_stride(stride));
    }

    shared_ptr<fe::graph::Tensor_attributes> scalar(shared_ptr<fe::graph::Graph>& graph,
                                                    string_view name,
                                                    int64_t uid,
                                                    DataType dtype = DataType::FP32) {
        return tensor(graph, name, uid, {1, 1, 1, 1}, {1, 1, 1, 1}, dtype);
    }

    shared_ptr<fe::graph::Tensor_attributes> seqLen(shared_ptr<fe::graph::Graph>& graph, string_view name, int64_t uid, int64_t batch) {
        return tensor(graph, name, uid, {batch, 1, 1, 1}, {1, 1, 1, 1}, DataType::INT32);
    }

    shared_ptr<fe::graph::Tensor_attributes> raggedOffset(shared_ptr<fe::graph::Graph>& graph,
                                                          string_view name,
                                                          int64_t uid,
                                                          int64_t batch) {
        return tensor(graph, name, uid, {batch + 1, 1, 1, 1}, {1, 1, 1, 1}, DataType::INT32);
    }

    shared_ptr<fe::graph::Tensor_attributes> makeAttentionTensor(shared_ptr<fe::graph::Graph>& graph,
                                                                 string_view name,
                                                                 int64_t uid,
                                                                 const AttentionTensorSpec& spec,
                                                                 shared_ptr<fe::graph::Tensor_attributes> raggedOffsetTensor) {
        auto attr = tensor(graph, name, uid, spec.dimensions, spec.strides, spec.dataType);
        if (spec.ragged)
            attr->set_ragged_offset(raggedOffsetTensor);
        return attr;
    }

    shared_ptr<fe::graph::Tensor_attributes> makeAttentionIoTensor(shared_ptr<fe::graph::Graph>& graph,
                                                                   string_view name,
                                                                   int64_t uid,
                                                                   const AttentionTensorSpec& spec,
                                                                   shared_ptr<fe::graph::Tensor_attributes> raggedOffsetTensor) {
        auto attr = ioTensor(graph, name, uid, spec.dimensions, spec.strides);
        if (spec.ragged)
            attr->set_ragged_offset(raggedOffsetTensor);
        return attr;
    }

    void applyMaskOptions(fe::graph::SDPA_attributes& attrs, const CudnnAttentionDescriptor& descriptor) {
        switch (descriptor.maskKind) {
            case AttentionMaskKind::None:
                break;
            case AttentionMaskKind::CausalTopLeft:
                attrs.set_diagonal_alignment(fe::DiagonalAlignment_t::TOP_LEFT)
                    .set_diagonal_band_right_bound(descriptor.diagonalRightBound);
                break;
            case AttentionMaskKind::CausalBottomRight:
                attrs.set_diagonal_alignment(fe::DiagonalAlignment_t::BOTTOM_RIGHT).set_diagonal_band_right_bound(0);
                break;
            case AttentionMaskKind::SlidingWindowTopLeft:
                attrs.set_diagonal_alignment(fe::DiagonalAlignment_t::TOP_LEFT)
                    .set_diagonal_band_left_bound(descriptor.diagonalLeftBound)
                    .set_diagonal_band_right_bound(descriptor.diagonalRightBound);
                break;
            case AttentionMaskKind::SlidingWindowBottomRight:
                attrs.set_diagonal_alignment(fe::DiagonalAlignment_t::BOTTOM_RIGHT)
                    .set_diagonal_band_left_bound(descriptor.diagonalLeftBound)
                    .set_diagonal_band_right_bound(descriptor.diagonalRightBound);
                break;
        }
        if (descriptor.useAlibiMask)
            attrs.set_alibi_mask(true);
    }

    void applyMaskOptions(fe::graph::SDPA_backward_attributes& attrs, const CudnnAttentionDescriptor& descriptor) {
        switch (descriptor.maskKind) {
            case AttentionMaskKind::None:
                break;
            case AttentionMaskKind::CausalTopLeft:
                attrs.set_diagonal_alignment(fe::DiagonalAlignment_t::TOP_LEFT)
                    .set_diagonal_band_right_bound(descriptor.diagonalRightBound);
                break;
            case AttentionMaskKind::CausalBottomRight:
                attrs.set_diagonal_alignment(fe::DiagonalAlignment_t::BOTTOM_RIGHT).set_diagonal_band_right_bound(0);
                break;
            case AttentionMaskKind::SlidingWindowTopLeft:
                attrs.set_diagonal_alignment(fe::DiagonalAlignment_t::TOP_LEFT)
                    .set_diagonal_band_left_bound(descriptor.diagonalLeftBound)
                    .set_diagonal_band_right_bound(descriptor.diagonalRightBound);
                break;
            case AttentionMaskKind::SlidingWindowBottomRight:
                attrs.set_diagonal_alignment(fe::DiagonalAlignment_t::BOTTOM_RIGHT)
                    .set_diagonal_band_left_bound(descriptor.diagonalLeftBound)
                    .set_diagonal_band_right_bound(descriptor.diagonalRightBound);
                break;
        }
        if (descriptor.useAlibiMask)
            attrs.set_alibi_mask(true);
    }

    void finalize(BuiltGraph& built, int gpuNum) {
        ScopedGpu scopedGpu(gpuNum);
        Stream temporaryStream(gpuNum);
        auto status = built.graph->build(temporaryStream.getCudnnHandle(), {fe::HeurMode_t::A});
        if (!status.is_good())
            throw runtime_error(
                "Failed to build cuDNN Frontend SDPA graph with primary heuristics only "
                "(Thor attention does not permit cuDNN fallback engines): " +
                status.get_message());

        int64_t workspaceBytes = 0;
        status = built.graph->get_workspace_size(workspaceBytes);
        if (!status.is_good())
            throw runtime_error("Failed to query cuDNN Frontend SDPA workspace: " + status.get_message());

        built.workspaceBytes = workspaceBytes;
        if (workspaceBytes > 0) {
            built.workspace = Tensor(TensorPlacement(TensorPlacement::MemDevices::GPU, gpuNum),
                                     TensorDescriptor(DataType::UINT8, {static_cast<uint64_t>(workspaceBytes)}),
                                     256);
        }
    }

    BuiltGraph buildForwardGraph(const CudnnAttentionDescriptor& descriptor, int gpuNum) {
        descriptor.validateForward();

        BuiltGraph built;
        built.graph = make_shared<fe::graph::Graph>();
        built.graph->set_io_data_type(toFrontendDataType(descriptor.q.dataType))
            .set_intermediate_data_type(toFrontendDataType(descriptor.intermediateDataType))
            .set_compute_data_type(toFrontendDataType(descriptor.computeDataType));

        auto qRagged = descriptor.q.ragged ? raggedOffset(built.graph, "ragged_offset_q", UID_RAGGED_Q, descriptor.batchSize()) : nullptr;
        auto kRagged = descriptor.k.ragged ? raggedOffset(built.graph, "ragged_offset_k", UID_RAGGED_K, descriptor.batchSize()) : nullptr;
        auto vRagged = descriptor.v.ragged ? raggedOffset(built.graph, "ragged_offset_v", UID_RAGGED_V, descriptor.batchSize()) : nullptr;
        auto oRagged = descriptor.o.ragged ? raggedOffset(built.graph, "ragged_offset_o", UID_RAGGED_O, descriptor.batchSize()) : nullptr;

        auto q = makeAttentionTensor(built.graph, "q", UID_Q, descriptor.q, qRagged);
        auto k = makeAttentionTensor(built.graph, "k", UID_K, descriptor.k, kRagged);
        auto v = makeAttentionTensor(built.graph, "v", UID_V, descriptor.v, vRagged);

        if (!descriptor.useFp8) {
            auto attrs = fe::graph::SDPA_attributes().set_name(descriptor.debugName).set_generate_stats(descriptor.generateStats);
            attrs.set_attn_scale(descriptor.attentionScale.value_or(1.0f / sqrtf(static_cast<float>(descriptor.qkHeadDim()))));
            applyMaskOptions(attrs, descriptor);

            if (descriptor.usePaddingMask) {
                attrs.set_padding_mask(true)
                    .set_seq_len_q(seqLen(built.graph, "seq_len_q", UID_SEQ_Q, descriptor.batchSize()))
                    .set_seq_len_kv(seqLen(built.graph, "seq_len_kv", UID_SEQ_KV, descriptor.batchSize()));
            }
            if (descriptor.useBias) {
                AttentionTensorSpec fallback;
                const AttentionTensorSpec& biasSpec = scoreBiasSpecOrDefault(descriptor, descriptor.computeDataType, fallback);
                attrs.set_bias(tensor(built.graph, "bias", UID_BIAS, biasSpec.dimensions, biasSpec.strides, descriptor.computeDataType));
            }
            if (descriptor.usePagedKvCache) {
                const int64_t k_pages = pagedKvPageCountForTable(descriptor, "page_table_k");
                const int64_t v_pages = pagedKvPageCountForTable(descriptor, "page_table_v");
                attrs
                    .set_paged_attention_k_table(
                        tensor(built.graph,
                               "page_table_k",
                               UID_PAGE_TABLE_K,
                               {descriptor.batchSize(), 1, k_pages, 1},
                               {k_pages, k_pages, 1, 1},
                               DataType::INT32))
                    .set_paged_attention_v_table(
                        tensor(built.graph,
                               "page_table_v",
                               UID_PAGE_TABLE_V,
                               {descriptor.batchSize(), 1, v_pages, 1},
                               {v_pages, v_pages, 1, 1},
                               DataType::INT32))
                    .set_paged_attention_max_seq_len_kv(static_cast<int>(descriptor.pagedKv.maxSequenceLengthKv));
            }
            if (descriptor.dropout.probability > 0.0f) {
                if (descriptor.dropout.usePhilox) {
                    attrs.set_dropout(descriptor.dropout.probability,
                                      scalar(built.graph, "dropout_seed", UID_DROPOUT_SEED, DataType::INT64),
                                      scalar(built.graph, "dropout_offset", UID_DROPOUT_OFFSET, DataType::INT64));
                } else {
                    attrs.set_dropout(
                        tensor(built.graph,
                               "dropout_mask",
                               UID_DROPOUT_MASK,
                               {descriptor.batchSize(), descriptor.queryHeads(), descriptor.queryLength(), descriptor.keyValueLength()},
                               {descriptor.queryHeads() * descriptor.queryLength() * descriptor.keyValueLength(),
                                descriptor.queryLength() * descriptor.keyValueLength(),
                                descriptor.keyValueLength(),
                                1},
                               DataType::BOOLEAN),
                        scalar(built.graph, "dropout_scale", UID_DROPOUT_SCALE));
                }
            }

            auto [o, stats] = built.graph->sdpa(q, k, v, attrs);
            o->set_output(true)
                .set_uid(UID_O)
                .set_dim(descriptor.o.dimensions)
                .set_stride(descriptor.o.strides)
                .set_data_type(toFrontendDataType(descriptor.o.dataType));
            if (descriptor.o.ragged)
                o->set_ragged_offset(oRagged);
            if (descriptor.generateStats) {
                stats->set_output(true)
                    .set_uid(UID_STATS)
                    .set_dim({descriptor.batchSize(), descriptor.queryHeads(), descriptor.queryLength(), 1})
                    .set_stride({descriptor.queryHeads() * descriptor.queryLength(), descriptor.queryLength(), 1, 1})
                    .set_data_type(fe::DataType_t::FLOAT);
            }
        } else {
            auto attrs = fe::graph::SDPA_fp8_attributes().set_name(descriptor.debugName).set_generate_stats(descriptor.generateStats);
            attrs.set_attn_scale(descriptor.attentionScale.value_or(1.0f / sqrtf(static_cast<float>(descriptor.qkHeadDim()))));
            if (descriptor.maskKind == AttentionMaskKind::CausalTopLeft || descriptor.maskKind == AttentionMaskKind::CausalBottomRight) {
                // The FP8 frontend API currently exposes only set_causal_mask(bool), with no diagonal-alignment knob.
                // Keep bottom-right as a probe label only; it verifies that cuDNN still accepts the causal FP8 API for
                // uneven decode-like shapes, not that a bottom-right diagonal alignment was encoded.
                attrs.set_causal_mask(true);
            } else if (descriptor.maskKind != AttentionMaskKind::None) {
                throwInvalidAttention("FP8 cuDNN SDPA currently supports no mask or causal mask in this wrapper");
            }
            if (descriptor.usePaddingMask) {
                attrs.set_padding_mask(true)
                    .set_seq_len_q(seqLen(built.graph, "seq_len_q", UID_SEQ_Q, descriptor.batchSize()))
                    .set_seq_len_kv(seqLen(built.graph, "seq_len_kv", UID_SEQ_KV, descriptor.batchSize()));
            }
            if (descriptor.useBias) {
                AttentionTensorSpec fallback;
                const DataType biasDataType = attentionForwardBiasDataType(descriptor);
                const AttentionTensorSpec& biasSpec = scoreBiasSpecOrDefault(descriptor, biasDataType, fallback);
                attrs.set_bias(tensor(built.graph, "bias", UID_BIAS, biasSpec.dimensions, biasSpec.strides, biasDataType));
            }
            if (descriptor.dropout.probability > 0.0f) {
                if (descriptor.dropout.usePhilox) {
                    attrs.set_dropout(descriptor.dropout.probability,
                                      scalar(built.graph, "dropout_seed", UID_DROPOUT_SEED, DataType::INT64),
                                      scalar(built.graph, "dropout_offset", UID_DROPOUT_OFFSET, DataType::INT64));
                } else {
                    attrs.set_dropout(
                        tensor(built.graph,
                               "dropout_mask",
                               UID_DROPOUT_MASK,
                               {descriptor.batchSize(), descriptor.queryHeads(), descriptor.queryLength(), descriptor.keyValueLength()},
                               {descriptor.queryHeads() * descriptor.queryLength() * descriptor.keyValueLength(),
                                descriptor.queryLength() * descriptor.keyValueLength(),
                                descriptor.keyValueLength(),
                                1},
                               descriptor.q.dataType),
                        scalar(built.graph, "dropout_scale", UID_DROPOUT_SCALE));
                }
            }

            auto [o, stats, amaxS, amaxO] = built.graph->sdpa_fp8(q,
                                                                  k,
                                                                  v,
                                                                  scalar(built.graph, "descale_q", UID_DESCALE_Q),
                                                                  scalar(built.graph, "descale_k", UID_DESCALE_K),
                                                                  scalar(built.graph, "descale_v", UID_DESCALE_V),
                                                                  scalar(built.graph, "descale_s", UID_DESCALE_S),
                                                                  scalar(built.graph, "scale_s", UID_SCALE_S),
                                                                  scalar(built.graph, "scale_o", UID_SCALE_O),
                                                                  attrs);
            o->set_output(true)
                .set_uid(UID_O)
                .set_dim(descriptor.o.dimensions)
                .set_stride(descriptor.o.strides)
                .set_data_type(toFrontendDataType(descriptor.o.dataType));
            if (descriptor.generateStats)
                stats->set_output(true)
                    .set_uid(UID_STATS)
                    .set_dim({descriptor.batchSize(), descriptor.queryHeads(), descriptor.queryLength(), 1})
                    .set_stride({descriptor.queryHeads() * descriptor.queryLength(), descriptor.queryLength(), 1, 1})
                    .set_data_type(fe::DataType_t::FLOAT);
            amaxS->set_output(true)
                .set_uid(UID_AMAX_S)
                .set_dim({1, 1, 1, 1})
                .set_stride({1, 1, 1, 1})
                .set_data_type(fe::DataType_t::FLOAT);
            amaxO->set_output(true)
                .set_uid(UID_AMAX_O)
                .set_dim({1, 1, 1, 1})
                .set_stride({1, 1, 1, 1})
                .set_data_type(fe::DataType_t::FLOAT);
        }

        finalize(built, gpuNum);
        return built;
    }

    BuiltGraph buildBackwardGraph(const CudnnAttentionDescriptor& descriptor, int gpuNum) {
        descriptor.validateBackward();

        BuiltGraph built;
        built.graph = make_shared<fe::graph::Graph>();
        built.graph->set_io_data_type(toFrontendDataType(descriptor.q.dataType))
            .set_intermediate_data_type(toFrontendDataType(descriptor.intermediateDataType))
            .set_compute_data_type(toFrontendDataType(descriptor.computeDataType));

        auto qRagged = descriptor.q.ragged ? raggedOffset(built.graph, "ragged_offset_q", UID_RAGGED_Q, descriptor.batchSize()) : nullptr;
        auto kRagged = descriptor.k.ragged ? raggedOffset(built.graph, "ragged_offset_k", UID_RAGGED_K, descriptor.batchSize()) : nullptr;
        auto vRagged = descriptor.v.ragged ? raggedOffset(built.graph, "ragged_offset_v", UID_RAGGED_V, descriptor.batchSize()) : nullptr;
        auto oRagged = descriptor.o.ragged ? raggedOffset(built.graph, "ragged_offset_o", UID_RAGGED_O, descriptor.batchSize()) : nullptr;

        auto q = makeAttentionIoTensor(built.graph, "q", UID_Q, descriptor.q, qRagged);
        auto k = makeAttentionIoTensor(built.graph, "k", UID_K, descriptor.k, kRagged);
        auto v = makeAttentionIoTensor(built.graph, "v", UID_V, descriptor.v, vRagged);
        auto o = makeAttentionIoTensor(built.graph, "o", UID_O, descriptor.o, oRagged);
        auto dO = makeAttentionIoTensor(built.graph, "dO", UID_DO, descriptor.o, oRagged);
        auto stats = tensor(built.graph,
                            "stats",
                            UID_STATS,
                            {descriptor.batchSize(), descriptor.queryHeads(), descriptor.queryLength(), 1},
                            {descriptor.queryHeads() * descriptor.queryLength(), descriptor.queryLength(), 1, 1},
                            DataType::FP32);

        if (descriptor.useFp8) {
            auto attrs = fe::graph::SDPA_fp8_backward_attributes().set_name(descriptor.debugName + "_fp8_backward");
            attrs.set_attn_scale(descriptor.attentionScale.value_or(1.0f / sqrtf(static_cast<float>(descriptor.qkHeadDim()))));
            if (descriptor.maskKind == AttentionMaskKind::CausalTopLeft || descriptor.maskKind == AttentionMaskKind::CausalBottomRight) {
                attrs.set_causal_mask(true);
            } else if (descriptor.maskKind != AttentionMaskKind::None) {
                throwInvalidAttention("FP8 cuDNN SDPA backward currently supports no mask or causal mask in this wrapper");
            }

            auto [dQ, dK, dV, amaxDQ, amaxDK, amaxDV, amaxDP] =
                built.graph->sdpa_fp8_backward(q,
                                               k,
                                               v,
                                               o,
                                               dO,
                                               stats,
                                               scalar(built.graph, "descale_q", UID_DESCALE_Q),
                                               scalar(built.graph, "descale_k", UID_DESCALE_K),
                                               scalar(built.graph, "descale_v", UID_DESCALE_V),
                                               scalar(built.graph, "descale_o", UID_DESCALE_O),
                                               scalar(built.graph, "descale_do", UID_DESCALE_DO),
                                               scalar(built.graph, "descale_s", UID_DESCALE_S),
                                               scalar(built.graph, "descale_dp", UID_DESCALE_DP),
                                               scalar(built.graph, "scale_s", UID_SCALE_S),
                                               scalar(built.graph, "scale_dq", UID_SCALE_DQ),
                                               scalar(built.graph, "scale_dk", UID_SCALE_DK),
                                               scalar(built.graph, "scale_dv", UID_SCALE_DV),
                                               scalar(built.graph, "scale_dp", UID_SCALE_DP),
                                               attrs);
            dQ->set_output(true)
                .set_uid(UID_DQ)
                .set_dim(descriptor.q.dimensions)
                .set_stride(descriptor.q.strides)
                .set_data_type(toFrontendDataType(descriptor.q.dataType));
            dK->set_output(true)
                .set_uid(UID_DK)
                .set_dim(descriptor.k.dimensions)
                .set_stride(descriptor.k.strides)
                .set_data_type(toFrontendDataType(descriptor.k.dataType));
            dV->set_output(true)
                .set_uid(UID_DV)
                .set_dim(descriptor.v.dimensions)
                .set_stride(descriptor.v.strides)
                .set_data_type(toFrontendDataType(descriptor.v.dataType));
            amaxDQ->set_output(true)
                .set_uid(UID_AMAX_DQ)
                .set_dim({1, 1, 1, 1})
                .set_stride({1, 1, 1, 1})
                .set_data_type(fe::DataType_t::FLOAT);
            amaxDK->set_output(true)
                .set_uid(UID_AMAX_DK)
                .set_dim({1, 1, 1, 1})
                .set_stride({1, 1, 1, 1})
                .set_data_type(fe::DataType_t::FLOAT);
            amaxDV->set_output(true)
                .set_uid(UID_AMAX_DV)
                .set_dim({1, 1, 1, 1})
                .set_stride({1, 1, 1, 1})
                .set_data_type(fe::DataType_t::FLOAT);
            amaxDP->set_output(true)
                .set_uid(UID_AMAX_DP)
                .set_dim({1, 1, 1, 1})
                .set_stride({1, 1, 1, 1})
                .set_data_type(fe::DataType_t::FLOAT);

            finalize(built, gpuNum);
            return built;
        }

        auto attrs = fe::graph::SDPA_backward_attributes()
                         .set_name(descriptor.debugName + "_backward")
                         .set_attn_scale(descriptor.attentionScale.value_or(1.0f / sqrtf(static_cast<float>(descriptor.qkHeadDim()))))
                         .set_compute_data_type(toFrontendDataType(descriptor.computeDataType));
        if (descriptor.deterministicBackward) {
            attrs.set_deterministic_algorithm(true);
        }
        applyMaskOptions(attrs, descriptor);
        if (descriptor.usePaddingMask) {
            attrs.set_padding_mask(true)
                .set_seq_len_q(seqLen(built.graph, "seq_len_q", UID_SEQ_Q, descriptor.batchSize()))
                .set_seq_len_kv(seqLen(built.graph, "seq_len_kv", UID_SEQ_KV, descriptor.batchSize()));
        }
        if (descriptor.useBias) {
            AttentionTensorSpec fallback;
            const AttentionTensorSpec& biasSpec = scoreBiasSpecOrDefault(descriptor, descriptor.computeDataType, fallback);
            attrs.set_bias(tensor(built.graph, "bias", UID_BIAS, biasSpec.dimensions, biasSpec.strides, descriptor.computeDataType));
            AttentionTensorSpec dBiasFallback;
            const AttentionTensorSpec& dBiasSpec = scoreDBiasSpecOrDefault(descriptor, dBiasFallback);
            attrs.set_dbias(tensor(built.graph, "dBias", UID_DBIA, dBiasSpec.dimensions, dBiasSpec.strides, dBiasSpec.dataType));
        }
        if (descriptor.dropout.probability > 0.0f) {
            if (!descriptor.dropout.usePhilox) {
                throwInvalidAttention("attention backward currently supports only Philox dropout seed/offset");
            }
            attrs.set_dropout(descriptor.dropout.probability,
                              scalar(built.graph, "dropout_seed", UID_DROPOUT_SEED, DataType::INT64),
                              scalar(built.graph, "dropout_offset", UID_DROPOUT_OFFSET, DataType::INT64));
        }

        auto [dQ, dK, dV] = built.graph->sdpa_backward(q, k, v, o, dO, stats, attrs);
        dQ->set_output(true).set_uid(UID_DQ).set_dim(descriptor.q.dimensions).set_stride(descriptor.q.strides);
        if (descriptor.q.ragged)
            dQ->set_ragged_offset(qRagged);
        dK->set_output(true).set_uid(UID_DK).set_dim(descriptor.k.dimensions).set_stride(descriptor.k.strides);
        if (descriptor.k.ragged)
            dK->set_ragged_offset(kRagged);
        dV->set_output(true).set_uid(UID_DV).set_dim(descriptor.v.dimensions).set_stride(descriptor.v.strides);
        if (descriptor.v.ragged)
            dV->set_ragged_offset(vRagged);

        finalize(built, gpuNum);
        return built;
    }

    mutable mutex mtx;
    unordered_map<string, BuiltGraph> graphs;
};

AttentionGraphCache& cache() {
    static AttentionGraphCache c;
    return c;
}

void insertTensor(unordered_map<int64_t, void*>& pack, int64_t uid, const Tensor& tensor) {
    pack[uid] = const_cast<void*>(static_cast<const void*>(tensor.getMemPtr<void>()));
}

void insertOptionalTensor(unordered_map<int64_t, void*>& pack, int64_t uid, const optional<Tensor>& tensor) {
    if (tensor.has_value())
        pack[uid] = const_cast<void*>(static_cast<const void*>(tensor.value().getMemPtr<void>()));
}

void executeGraph(BuiltGraph& built, unordered_map<int64_t, void*>& pack, Stream stream) {
    void* workspace = built.workspaceBytes > 0 ? built.workspace.getMemPtr<void>() : nullptr;
    auto status = built.graph->execute(stream.getCudnnHandle(), pack, workspace);
    if (!status.is_good())
        throw runtime_error("Failed to execute cuDNN Frontend SDPA graph: " + status.get_message());
}

}  // namespace

AttentionTensorSpec AttentionTensorSpec::bhsd(
    int64_t batch, int64_t heads, int64_t sequenceLength, int64_t headDim, DataType dataType) {
    return AttentionTensorSpec{
        {batch, heads, sequenceLength, headDim}, {heads * sequenceLength * headDim, sequenceLength * headDim, headDim, 1}, dataType, false};
}

AttentionTensorSpec AttentionTensorSpec::bshd(
    int64_t batch, int64_t heads, int64_t sequenceLength, int64_t headDim, DataType dataType) {
    // Semantic dimension order remains [B,H,S,D]; this stride maps those indices to BSHD physical storage.
    return AttentionTensorSpec{
        {batch, heads, sequenceLength, headDim}, {sequenceLength * heads * headDim, headDim, heads * headDim, 1}, dataType, false};
}

AttentionTensorSpec AttentionTensorSpec::fromLayout(AttentionTensorLayout layout,
                                                    int64_t batch,
                                                    int64_t heads,
                                                    int64_t sequenceLength,
                                                    int64_t headDim,
                                                    DataType dataType) {
    switch (layout) {
        case AttentionTensorLayout::BHSD:
            return bhsd(batch, heads, sequenceLength, headDim, dataType);
        case AttentionTensorLayout::BSHD:
            return bshd(batch, heads, sequenceLength, headDim, dataType);
    }
    throw std::invalid_argument("Unknown AttentionTensorLayout.");
}

string AttentionTensorSpec::toString() const {
    ostringstream out;
    out << "dim=" << joinInts(dimensions) << " stride=" << joinInts(strides) << " dtype=" << TensorDescriptor::getElementTypeName(dataType)
        << " ragged=" << ragged;
    return out.str();
}

int64_t CudnnAttentionDescriptor::batchSize() const { return q.dimensions.at(0); }
int64_t CudnnAttentionDescriptor::queryHeads() const { return q.dimensions.at(1); }
int64_t CudnnAttentionDescriptor::keyValueHeads() const { return k.dimensions.at(1); }
int64_t CudnnAttentionDescriptor::queryLength() const { return q.dimensions.at(2); }
int64_t CudnnAttentionDescriptor::keyValueLength() const { return usePagedKvCache ? pagedKv.maxSequenceLengthKv : k.dimensions.at(2); }
int64_t CudnnAttentionDescriptor::qkHeadDim() const { return q.dimensions.at(3); }
int64_t CudnnAttentionDescriptor::vHeadDim() const { return v.dimensions.at(3); }

void CudnnAttentionDescriptor::validateForward() const {
    auto checkSpec = [](const AttentionTensorSpec& spec, string_view name) {
        if (spec.dimensions.size() != 4 || spec.strides.size() != 4)
            throwInvalidAttention(string(name) + " must have 4 dims and 4 strides in semantic [B,H,S,D] order");
        for (int64_t dim : spec.dimensions) {
            if (dim <= 0)
                throwInvalidAttention(string(name) + " dimensions must be positive: " + joinInts(spec.dimensions));
        }
        for (int64_t stride : spec.strides) {
            if (stride < 0)
                throwInvalidAttention(string(name) + " strides must be non-negative: " + joinInts(spec.strides));
        }
        if (spec.strides[3] != 1)
            throwInvalidAttention(string(name) + " head dimension must be stride-1 for cuDNN SDPA fast kernels");
    };

    checkSpec(q, "q");
    checkSpec(k, "k");
    checkSpec(v, "v");
    checkSpec(o, "o");

    if (q.dimensions[0] != o.dimensions[0])
        throwInvalidAttention("q/o batch dimensions must match");
    if (!usePagedKvCache && (q.dimensions[0] != k.dimensions[0] || q.dimensions[0] != v.dimensions[0]))
        throwInvalidAttention("non-paged attention requires q/k/v batch dimensions to match");
    if (k.dimensions[1] != v.dimensions[1])
        throwInvalidAttention("k/v head counts must match");
    if (q.dimensions[1] % k.dimensions[1] != 0)
        throwInvalidAttention("query heads must be an integer multiple of key/value heads for MHA/MQA/GQA");
    if (q.dimensions[2] != o.dimensions[2])
        throwInvalidAttention("q and o sequence lengths must match");
    if (!usePagedKvCache && k.dimensions[2] != v.dimensions[2])
        throwInvalidAttention("k/v sequence lengths must match");
    if (q.dimensions[3] != k.dimensions[3])
        throwInvalidAttention("q/k head dimensions must match");
    if (v.dimensions[3] != o.dimensions[3])
        throwInvalidAttention("v/o head dimensions must match");
    if (q.dimensions[1] != o.dimensions[1])
        throwInvalidAttention("q/o head counts must match");

    const bool fp8Mode = useFp8 || isFp8(q.dataType) || isFp8(k.dataType) || isFp8(v.dataType) || isFp8(o.dataType);
    if (fp8Mode) {
        if (!useFp8)
            throwInvalidAttention("FP8 tensor dtypes require descriptor.useFp8=true so the FP8 cuDNN SDPA API is used");
        if (!isFp8(q.dataType) || !isFp8(k.dataType) || !isFp8(v.dataType) || !isFp8(o.dataType))
            throwInvalidAttention("FP8 attention requires q/k/v/o to all be FP8 tensors in this wrapper");
        if (q.dataType != k.dataType || q.dataType != v.dataType || q.dataType != o.dataType)
            throwInvalidAttention("FP8 attention requires q/k/v/o to use the same FP8 format in this wrapper");
        if (q.dimensions[3] % 16 != 0 || v.dimensions[3] % 16 != 0)
            throwInvalidAttention("FP8 attention head dimensions must be multiples of 16");
        if ((q.dimensions[3] > 128 || v.dimensions[3] > 128) && !experimentalCudnnAttentionSupportSurfaceProbeEnabled())
            throwInvalidAttention("FP8 cuDNN SDPA forward is enabled only for qk/v head dimensions <= 128 on the validated support surface");
        if ((maskKind == AttentionMaskKind::SlidingWindowTopLeft || maskKind == AttentionMaskKind::SlidingWindowBottomRight) &&
            !experimentalCudnnAttentionSupportSurfaceProbeEnabled())
            throwInvalidAttention("FP8 cuDNN SDPA forward is enabled only for no-mask or causal-mask cases on the validated support surface");
        if (maskKind == AttentionMaskKind::CausalBottomRight && !experimentalCudnnAttentionSupportSurfaceProbeEnabled())
            throwInvalidAttention(
                "FP8 cuDNN SDPA exposes only a boolean causal mask in this wrapper; bottom-right/decode diagonal semantics are not enabled");
        if (useAlibiMask && !experimentalCudnnAttentionSupportSurfaceProbeEnabled())
            throwInvalidAttention("FP8 cuDNN SDPA ALiBi is not enabled on the validated support surface");
        if (useBias && !experimentalCudnnAttentionSupportSurfaceProbeEnabled())
            throwInvalidAttention("cuDNN FP8 SDPA does not support additive score bias on the validated support surface");
        if (dropout.probability != 0.0f && !experimentalCudnnAttentionSupportSurfaceProbeEnabled())
            throwInvalidAttention("cuDNN FP8 SDPA dropout is not supported on the validated support surface");
        if (usePagedKvCache && !experimentalCudnnAttentionSupportSurfaceProbeEnabled())
            throwInvalidAttention("FP8 cuDNN SDPA paged KV cache is not enabled on the validated support surface");
        if (q.dimensions[2] == 1 && keyValueLength() > 1 && !experimentalCudnnAttentionSupportSurfaceProbeEnabled())
            throwInvalidAttention("FP8 cuDNN SDPA decode-style Sq=1 forward is not enabled on the validated support surface");
    } else {
        if (!isFp16OrBf16(q.dataType) || q.dataType != k.dataType || q.dataType != v.dataType || q.dataType != o.dataType)
            throwInvalidAttention("non-FP8 attention requires q/k/v/o to be the same FP16 or BF16 dtype");
        if (q.dimensions[3] % 8 != 0 || v.dimensions[3] % 8 != 0)
            throwInvalidAttention("FP16/BF16 attention head dimensions must be multiples of 8");
    }

    if (computeDataType != DataType::FP32)
        throwInvalidAttention("computeDataType should be FP32 for numerically stable cuDNN SDPA");
    if (intermediateDataType != DataType::FP32)
        throwInvalidAttention("intermediateDataType should be FP32 for numerically stable cuDNN SDPA");
    const bool anyRagged = q.ragged || k.ragged || v.ragged || o.ragged;
    if (anyRagged && !usePaddingMask)
        throwInvalidAttention("ragged attention requires usePaddingMask=true so cuDNN receives q/kv sequence lengths for THD padding-mask semantics");
    if (anyRagged) {
        if (!hasBshdPackedStrides(q) || !hasBshdPackedStrides(k) || !hasBshdPackedStrides(v) || !hasBshdPackedStrides(o))
            throwInvalidAttention(
                "ragged attention requires BSHD physical layouts for q/k/v/o because ragged offsets index packed token-contiguous THD storage");
        if (q.ragged != o.ragged)
            throwInvalidAttention("ragged attention requires q and o to either both use ragged offsets or both be dense");
        if (k.ragged != v.ragged)
            throwInvalidAttention("ragged attention requires k and v to either both use ragged offsets or both be dense");
        if (q.dimensions[3] != v.dimensions[3])
            throwInvalidAttention(
                "ragged attention requires value head_dim to match query/key head_dim because Thor uses shared Q/O and K/V element offsets");
        if (usePagedKvCache && !experimentalCudnnAttentionSupportSurfaceProbeEnabled())
            throwInvalidAttention("ragged attention and paged KV cache are separate variable-length modes and cannot be combined");
        if (useFp8 && !experimentalCudnnAttentionSupportSurfaceProbeEnabled())
            throwInvalidAttention("ragged FP8 attention is not enabled until FP8 forward is validated for packed variable-length layouts");
    }
    if (useBias) {
        AttentionTensorSpec fallback;
        const DataType biasDataType = attentionForwardBiasDataType(*this);
        const AttentionTensorSpec& biasSpec = scoreBiasSpecOrDefault(*this, biasDataType, fallback);
        validateScoreBiasSpec(biasSpec, *this, "bias", biasDataType, true);
    }

    if (useAlibiMask) {
        const bool usesCausalDiagonal = maskKind == AttentionMaskKind::CausalTopLeft ||
                                        maskKind == AttentionMaskKind::CausalBottomRight ||
                                        maskKind == AttentionMaskKind::SlidingWindowTopLeft ||
                                        maskKind == AttentionMaskKind::SlidingWindowBottomRight;
        if ((!usesCausalDiagonal || diagonalRightBound != 0) && !experimentalCudnnAttentionSupportSurfaceProbeEnabled())
            throwInvalidAttention(
                "ALiBi requires causal diagonal masking with diagonalRightBound == 0 because cuDNN rejects ALiBi with positive right bounds");
    }
    if ((maskKind == AttentionMaskKind::CausalBottomRight || maskKind == AttentionMaskKind::SlidingWindowBottomRight) &&
        (useBias || useAlibiMask || dropout.probability > 0.0f) && !experimentalCudnnAttentionSupportSurfaceProbeEnabled())
        throwInvalidAttention(
            "bottom-right/decode attention currently requires additive bias, ALiBi, and dropout to be disabled in the cuDNN primary SDPA path");
    if (usePagedKvCache) {
        if (pagedKv.maxSequenceLengthKv <= 0)
            throwInvalidAttention("paged KV attention requires pagedKv.maxSequenceLengthKv > 0");
        if (!usePaddingMask)
            throwInvalidAttention("paged KV attention requires usePaddingMask=true with q/kv sequence length tensors");
        if (useBias && !experimentalCudnnAttentionSupportSurfaceProbeEnabled())
            throwInvalidAttention("paged KV attention with additive bias is disabled until the paged-bias layout is defined");
        if (dropout.probability > 0.0f && !experimentalCudnnAttentionSupportSurfaceProbeEnabled())
            throwInvalidAttention("paged KV attention is inference-only and cannot currently be combined with dropout");
        if (k.dimensions[2] <= 0 || v.dimensions[2] <= 0)
            throwInvalidAttention("paged KV K/V container block sizes must be positive");
    }
    if (dropout.probability < 0.0f || dropout.probability >= 1.0f)
        throwInvalidAttention("dropout probability must be in [0, 1)");
}

void CudnnAttentionDescriptor::validateBackward() const {
    validateForward();
    if (usePagedKvCache && !experimentalCudnnAttentionSupportSurfaceProbeEnabled())
        throwInvalidAttention("paged KV attention backward is not enabled; the paged KV path is inference-only until training semantics are defined");
    const bool anyRagged = q.ragged || k.ragged || v.ragged || o.ragged;
    if (useFp8 && !experimentalCudnnAttentionSupportSurfaceProbeEnabled()) {
        throwInvalidAttention(
            "cuDNN FP8 SDPA backward is not supported on the validated support surface; FP8 attention is forward-only in Thor");
    }
    if (useFp8) {
        if (useBias)
            throwInvalidAttention("FP8 attention backward does not expose additive bias/dBias in the current cuDNN Frontend API");
        if (dropout.probability > 0.0f)
            throwInvalidAttention("FP8 attention backward does not support dropout in the current cuDNN Frontend API");
        if (usePaddingMask || anyRagged)
            throwInvalidAttention("FP8 attention backward does not expose padding-mask or ragged sequence-length tensors in the current wrapper");
        if (usePagedKvCache)
            throwInvalidAttention("FP8 attention backward does not support paged KV cache in the current wrapper");
        if (maskKind != AttentionMaskKind::None && maskKind != AttentionMaskKind::CausalTopLeft && maskKind != AttentionMaskKind::CausalBottomRight)
            throwInvalidAttention("FP8 attention backward currently supports only no mask or causal mask in the current wrapper");
    }
    if (anyRagged && useBias && !experimentalCudnnRaggedBiasBackwardProbeEnabled())
        throwInvalidAttention(
            "cuDNN primary SDPA backward does not support ragged offsets with additive bias; ragged additive bias is forward-only "
            "until a supported dBias/backward path is implemented. Set THOR_EXPERIMENTAL_CUDNN_RAGGED_BIAS_BACKWARD=1 "
            "to bypass this guard for cuDNN support-surface probing only.");
    if (useBias) {
        AttentionTensorSpec fallback;
        const DataType biasDataType = attentionForwardBiasDataType(*this);
        const AttentionTensorSpec& biasSpec = scoreBiasSpecOrDefault(*this, biasDataType, fallback);
        // cuDNN backward is reliable for dense bias and batch/head-broadcast bias. Sequence-broadcast
        // bias is lowered by production autodiff through an explicit dense materialization; keep the native
        // cuDNN sequence-broadcast path behind the support-surface probe because some shapes are rejected on
        // SM120 and some accepted Skv-vector shapes produce incorrect gradients.
        validateScoreBiasSpec(biasSpec, *this, "bias", biasDataType, true);
        if (scoreBiasUsesSequenceBroadcast(biasSpec, *this) && !experimentalCudnnAttentionSupportSurfaceProbeEnabled()) {
            throwInvalidAttention(
                "cuDNN SDPA backward with additive bias broadcast across Sq or Skv is not enabled directly; Thor production "
                "autodiff must materialize a dense [B,Hq,Sq,Skv] score bias before attention backward and then reduce dBias "
                "back to the public bias shape");
        }

        AttentionTensorSpec dBiasFallback;
        const AttentionTensorSpec& dBiasSpec = scoreDBiasSpecOrDefault(*this, dBiasFallback);
        validateScoreBiasSpec(dBiasSpec, *this, "dBias", dBiasSpec.dataType, false);
        if (!isSupportedAttentionDBiasDataType(dBiasSpec.dataType, *this)) {
            throwInvalidAttention("additive-bias backward dBias dtype must be either the q tensor dtype or the attention compute dtype");
        }
    }
    if (!generateStats)
        throwInvalidAttention("backward requires descriptor.generateStats=true in the corresponding forward pass");
}

string CudnnAttentionDescriptor::cacheKey(string_view passName, int gpuNum) const {
    ostringstream out;
    out << "gpu=" << gpuNum << ";pass=" << passName << ';';
    appendSpec(out, "q", q);
    appendSpec(out, "k", k);
    appendSpec(out, "v", v);
    appendSpec(out, "o", o);
    out << "compute=" << TensorDescriptor::getElementTypeName(computeDataType) << ';';
    out << "intermediate=" << TensorDescriptor::getElementTypeName(intermediateDataType) << ';';
    out << "scale=" << (attentionScale.has_value() ? to_string(attentionScale.value()) : string("default")) << ';';
    out << "mask=" << static_cast<int>(maskKind) << ";left=" << diagonalLeftBound << ";right=" << diagonalRightBound << ';';
    out << "stats=" << generateStats << ";detBwd=" << deterministicBackward << ";pad=" << usePaddingMask << ";alibi=" << useAlibiMask
        << ";bias=" << useBias << ";paged=" << usePagedKvCache << ";fp8=" << useFp8 << ';';
    if (useBias) {
        AttentionTensorSpec fallback;
        const DataType biasDataType = attentionForwardBiasDataType(*this);
        const AttentionTensorSpec& biasSpec = scoreBiasSpecOrDefault(*this, biasDataType, fallback);
        appendSpec(out, "bias", biasSpec);
        AttentionTensorSpec dBiasFallback;
        const AttentionTensorSpec& dBiasSpec = scoreDBiasSpecOrDefault(*this, dBiasFallback);
        appendSpec(out, "dBias", dBiasSpec);
    }
    out << "dropP=" << dropout.probability << ";dropPhilox=" << dropout.usePhilox << ';';
    out << "pagedMax=" << pagedKv.maxSequenceLengthKv << ';';
    return out.str();
}

CudnnScaledDotProductAttention& CudnnScaledDotProductAttention::instance() {
    static CudnnScaledDotProductAttention executor;
    return executor;
}

void CudnnScaledDotProductAttention::forward(const CudnnAttentionDescriptor& descriptor,
                                             const CudnnAttentionForwardArgs& args,
                                             Stream stream) {
    CudnnAttentionDescriptor runtimeDescriptor = descriptorWithRuntimeBiasSpec(descriptor, args.bias, attentionForwardBiasDataType(descriptor));
    runtimeDescriptor.validateForward();
    const int gpuNum = stream.getGpuNum();
    requireGpuTensor(args.q, "q", gpuNum);
    requireGpuTensor(args.k, "k", gpuNum);
    requireGpuTensor(args.v, "v", gpuNum);
    requireGpuTensor(args.o, "o", gpuNum);
    requireTensorMatchesSpec(args.q, runtimeDescriptor.q, "q");
    requireTensorMatchesSpec(args.k, runtimeDescriptor.k, "k");
    requireTensorMatchesSpec(args.v, runtimeDescriptor.v, "v");
    requireTensorMatchesSpec(args.o, runtimeDescriptor.o, "o");

    BuiltGraph& graph = cache().getOrBuildForward(runtimeDescriptor, gpuNum);
    unordered_map<int64_t, void*> pack;
    insertTensor(pack, UID_Q, args.q);
    insertTensor(pack, UID_K, args.k);
    insertTensor(pack, UID_V, args.v);
    insertTensor(pack, UID_O, args.o);

    if (runtimeDescriptor.generateStats) {
        requireOptionalGpuTensor(args.stats, "stats", gpuNum);
        insertTensor(pack, UID_STATS, args.stats.value());
    }
    if (runtimeDescriptor.useBias) {
        requireOptionalGpuTensor(args.bias, "bias", gpuNum);
        requireAttentionBiasMatchesDescriptor(args.bias.value(), runtimeDescriptor, "bias");
        insertTensor(pack, UID_BIAS, args.bias.value());
    }
    if (runtimeDescriptor.usePaddingMask) {
        requireOptionalGpuTensor(args.seqLenQ, "seqLenQ", gpuNum);
        requireOptionalGpuTensor(args.seqLenKv, "seqLenKv", gpuNum);
        requireSeqLenMatchesDescriptor(args.seqLenQ.value(), runtimeDescriptor, "seqLenQ");
        requireSeqLenMatchesDescriptor(args.seqLenKv.value(), runtimeDescriptor, "seqLenKv");
        insertTensor(pack, UID_SEQ_Q, args.seqLenQ.value());
        insertTensor(pack, UID_SEQ_KV, args.seqLenKv.value());
    }
    if (runtimeDescriptor.q.ragged) {
        requireOptionalGpuTensor(args.raggedOffsetQ, "raggedOffsetQ", gpuNum);
        requireRaggedOffsetMatchesDescriptor(args.raggedOffsetQ.value(), runtimeDescriptor, "raggedOffsetQ");
        insertTensor(pack, UID_RAGGED_Q, args.raggedOffsetQ.value());
    }
    if (runtimeDescriptor.k.ragged) {
        requireOptionalGpuTensor(args.raggedOffsetK, "raggedOffsetK", gpuNum);
        requireRaggedOffsetMatchesDescriptor(args.raggedOffsetK.value(), runtimeDescriptor, "raggedOffsetK");
        insertTensor(pack, UID_RAGGED_K, args.raggedOffsetK.value());
    }
    if (runtimeDescriptor.v.ragged) {
        requireOptionalGpuTensor(args.raggedOffsetV, "raggedOffsetV", gpuNum);
        requireRaggedOffsetMatchesDescriptor(args.raggedOffsetV.value(), runtimeDescriptor, "raggedOffsetV");
        insertTensor(pack, UID_RAGGED_V, args.raggedOffsetV.value());
    }
    if (runtimeDescriptor.o.ragged) {
        requireOptionalGpuTensor(args.raggedOffsetO, "raggedOffsetO", gpuNum);
        requireRaggedOffsetMatchesDescriptor(args.raggedOffsetO.value(), runtimeDescriptor, "raggedOffsetO");
        insertTensor(pack, UID_RAGGED_O, args.raggedOffsetO.value());
    }
    if (runtimeDescriptor.dropout.probability > 0.0f) {
        if (descriptor.dropout.usePhilox) {
            requireOptionalGpuTensor(args.dropoutSeed, "dropoutSeed", gpuNum);
            requireOptionalGpuTensor(args.dropoutOffset, "dropoutOffset", gpuNum);
            requireDropoutScalarMatchesDescriptor(args.dropoutSeed.value(), "dropoutSeed");
            requireDropoutScalarMatchesDescriptor(args.dropoutOffset.value(), "dropoutOffset");
            insertTensor(pack, UID_DROPOUT_SEED, args.dropoutSeed.value());
            insertTensor(pack, UID_DROPOUT_OFFSET, args.dropoutOffset.value());
        } else {
            requireOptionalGpuTensor(args.dropoutMask, "dropoutMask", gpuNum);
            requireOptionalGpuTensor(args.dropoutScale, "dropoutScale", gpuNum);
            insertTensor(pack, UID_DROPOUT_MASK, args.dropoutMask.value());
            insertTensor(pack, UID_DROPOUT_SCALE, args.dropoutScale.value());
        }
    }
    if (runtimeDescriptor.usePagedKvCache) {
        requireOptionalGpuTensor(args.pageTableK, "pageTableK", gpuNum);
        requireOptionalGpuTensor(args.pageTableV, "pageTableV", gpuNum);
        requirePagedKvTableMatchesDescriptor(args.pageTableK.value(), runtimeDescriptor, "pageTableK");
        requirePagedKvTableMatchesDescriptor(args.pageTableV.value(), runtimeDescriptor, "pageTableV");
        insertTensor(pack, UID_PAGE_TABLE_K, args.pageTableK.value());
        insertTensor(pack, UID_PAGE_TABLE_V, args.pageTableV.value());
    }
    if (descriptor.useFp8) {
        requireOptionalGpuTensor(args.descaleQ, "descaleQ", gpuNum);
        requireOptionalGpuTensor(args.descaleK, "descaleK", gpuNum);
        requireOptionalGpuTensor(args.descaleV, "descaleV", gpuNum);
        requireOptionalGpuTensor(args.descaleS, "descaleS", gpuNum);
        requireOptionalGpuTensor(args.scaleS, "scaleS", gpuNum);
        requireOptionalGpuTensor(args.scaleO, "scaleO", gpuNum);
        requireOptionalGpuTensor(args.amaxS, "amaxS", gpuNum);
        requireOptionalGpuTensor(args.amaxO, "amaxO", gpuNum);
        requireFp8ScaleScalarMatchesDescriptor(args.descaleQ.value(), "descaleQ");
        requireFp8ScaleScalarMatchesDescriptor(args.descaleK.value(), "descaleK");
        requireFp8ScaleScalarMatchesDescriptor(args.descaleV.value(), "descaleV");
        requireFp8ScaleScalarMatchesDescriptor(args.descaleS.value(), "descaleS");
        requireFp8ScaleScalarMatchesDescriptor(args.scaleS.value(), "scaleS");
        requireFp8ScaleScalarMatchesDescriptor(args.scaleO.value(), "scaleO");
        requireFp8ScaleScalarMatchesDescriptor(args.amaxS.value(), "amaxS");
        requireFp8ScaleScalarMatchesDescriptor(args.amaxO.value(), "amaxO");
        insertTensor(pack, UID_DESCALE_Q, args.descaleQ.value());
        insertTensor(pack, UID_DESCALE_K, args.descaleK.value());
        insertTensor(pack, UID_DESCALE_V, args.descaleV.value());
        insertTensor(pack, UID_DESCALE_S, args.descaleS.value());
        insertTensor(pack, UID_SCALE_S, args.scaleS.value());
        insertTensor(pack, UID_SCALE_O, args.scaleO.value());
        insertTensor(pack, UID_AMAX_S, args.amaxS.value());
        insertTensor(pack, UID_AMAX_O, args.amaxO.value());
    }

    executeGraph(graph, pack, stream);
}

void CudnnScaledDotProductAttention::backward(const CudnnAttentionDescriptor& descriptor,
                                              const CudnnAttentionBackwardArgs& args,
                                              Stream stream) {
    CudnnAttentionDescriptor runtimeDescriptor = descriptorWithRuntimeBiasSpec(descriptor, args.bias, attentionForwardBiasDataType(descriptor));
    runtimeDescriptor = descriptorWithRuntimeDBiasSpec(runtimeDescriptor, args.dBias);
    runtimeDescriptor.validateBackward();
    const int gpuNum = stream.getGpuNum();
    requireGpuTensor(args.q, "q", gpuNum);
    requireGpuTensor(args.k, "k", gpuNum);
    requireGpuTensor(args.v, "v", gpuNum);
    requireGpuTensor(args.o, "o", gpuNum);
    requireGpuTensor(args.dO, "dO", gpuNum);
    requireGpuTensor(args.stats, "stats", gpuNum);
    requireGpuTensor(args.dQ, "dQ", gpuNum);
    requireGpuTensor(args.dK, "dK", gpuNum);
    requireGpuTensor(args.dV, "dV", gpuNum);

    BuiltGraph& graph = cache().getOrBuildBackward(runtimeDescriptor, gpuNum);
    unordered_map<int64_t, void*> pack;
    insertTensor(pack, UID_Q, args.q);
    insertTensor(pack, UID_K, args.k);
    insertTensor(pack, UID_V, args.v);
    insertTensor(pack, UID_O, args.o);
    insertTensor(pack, UID_DO, args.dO);
    insertTensor(pack, UID_STATS, args.stats);
    insertTensor(pack, UID_DQ, args.dQ);
    insertTensor(pack, UID_DK, args.dK);
    insertTensor(pack, UID_DV, args.dV);

    if (runtimeDescriptor.useBias) {
        requireOptionalGpuTensor(args.bias, "bias", gpuNum);
        requireAttentionBiasMatchesDescriptor(args.bias.value(), runtimeDescriptor, "bias");
        requireOptionalGpuTensor(args.dBias, "dBias", gpuNum);
        requireAttentionDBiasMatchesDescriptor(args.dBias.value(), runtimeDescriptor, "dBias");
        insertTensor(pack, UID_BIAS, args.bias.value());
        insertTensor(pack, UID_DBIA, args.dBias.value());
    }
    if (runtimeDescriptor.usePaddingMask) {
        requireOptionalGpuTensor(args.seqLenQ, "seqLenQ", gpuNum);
        requireOptionalGpuTensor(args.seqLenKv, "seqLenKv", gpuNum);
        requireSeqLenMatchesDescriptor(args.seqLenQ.value(), runtimeDescriptor, "seqLenQ");
        requireSeqLenMatchesDescriptor(args.seqLenKv.value(), runtimeDescriptor, "seqLenKv");
        insertTensor(pack, UID_SEQ_Q, args.seqLenQ.value());
        insertTensor(pack, UID_SEQ_KV, args.seqLenKv.value());
    }
    if (runtimeDescriptor.q.ragged) {
        requireOptionalGpuTensor(args.raggedOffsetQ, "raggedOffsetQ", gpuNum);
        requireRaggedOffsetMatchesDescriptor(args.raggedOffsetQ.value(), runtimeDescriptor, "raggedOffsetQ");
        insertTensor(pack, UID_RAGGED_Q, args.raggedOffsetQ.value());
    }
    if (runtimeDescriptor.k.ragged) {
        requireOptionalGpuTensor(args.raggedOffsetK, "raggedOffsetK", gpuNum);
        requireRaggedOffsetMatchesDescriptor(args.raggedOffsetK.value(), runtimeDescriptor, "raggedOffsetK");
        insertTensor(pack, UID_RAGGED_K, args.raggedOffsetK.value());
    }
    if (runtimeDescriptor.v.ragged) {
        requireOptionalGpuTensor(args.raggedOffsetV, "raggedOffsetV", gpuNum);
        requireRaggedOffsetMatchesDescriptor(args.raggedOffsetV.value(), runtimeDescriptor, "raggedOffsetV");
        insertTensor(pack, UID_RAGGED_V, args.raggedOffsetV.value());
    }
    if (runtimeDescriptor.o.ragged) {
        requireOptionalGpuTensor(args.raggedOffsetO, "raggedOffsetO", gpuNum);
        requireRaggedOffsetMatchesDescriptor(args.raggedOffsetO.value(), runtimeDescriptor, "raggedOffsetO");
        insertTensor(pack, UID_RAGGED_O, args.raggedOffsetO.value());
    }
    if (descriptor.dropout.probability > 0.0f && descriptor.dropout.usePhilox) {
        requireOptionalGpuTensor(args.dropoutSeed, "dropoutSeed", gpuNum);
        requireOptionalGpuTensor(args.dropoutOffset, "dropoutOffset", gpuNum);
        requireDropoutScalarMatchesDescriptor(args.dropoutSeed.value(), "dropoutSeed");
        requireDropoutScalarMatchesDescriptor(args.dropoutOffset.value(), "dropoutOffset");
        insertTensor(pack, UID_DROPOUT_SEED, args.dropoutSeed.value());
        insertTensor(pack, UID_DROPOUT_OFFSET, args.dropoutOffset.value());
    }
    if (descriptor.useFp8) {
        requireOptionalGpuTensor(args.descaleQ, "descaleQ", gpuNum);
        requireOptionalGpuTensor(args.descaleK, "descaleK", gpuNum);
        requireOptionalGpuTensor(args.descaleV, "descaleV", gpuNum);
        requireOptionalGpuTensor(args.descaleO, "descaleO", gpuNum);
        requireOptionalGpuTensor(args.descaleDO, "descaleDO", gpuNum);
        requireOptionalGpuTensor(args.descaleS, "descaleS", gpuNum);
        requireOptionalGpuTensor(args.descaleDP, "descaleDP", gpuNum);
        requireOptionalGpuTensor(args.scaleS, "scaleS", gpuNum);
        requireOptionalGpuTensor(args.scaleDQ, "scaleDQ", gpuNum);
        requireOptionalGpuTensor(args.scaleDK, "scaleDK", gpuNum);
        requireOptionalGpuTensor(args.scaleDV, "scaleDV", gpuNum);
        requireOptionalGpuTensor(args.scaleDP, "scaleDP", gpuNum);
        requireOptionalGpuTensor(args.amaxDQ, "amaxDQ", gpuNum);
        requireOptionalGpuTensor(args.amaxDK, "amaxDK", gpuNum);
        requireOptionalGpuTensor(args.amaxDV, "amaxDV", gpuNum);
        requireOptionalGpuTensor(args.amaxDP, "amaxDP", gpuNum);
        requireFp8ScaleScalarMatchesDescriptor(args.descaleQ.value(), "descaleQ");
        requireFp8ScaleScalarMatchesDescriptor(args.descaleK.value(), "descaleK");
        requireFp8ScaleScalarMatchesDescriptor(args.descaleV.value(), "descaleV");
        requireFp8ScaleScalarMatchesDescriptor(args.descaleO.value(), "descaleO");
        requireFp8ScaleScalarMatchesDescriptor(args.descaleDO.value(), "descaleDO");
        requireFp8ScaleScalarMatchesDescriptor(args.descaleS.value(), "descaleS");
        requireFp8ScaleScalarMatchesDescriptor(args.descaleDP.value(), "descaleDP");
        requireFp8ScaleScalarMatchesDescriptor(args.scaleS.value(), "scaleS");
        requireFp8ScaleScalarMatchesDescriptor(args.scaleDQ.value(), "scaleDQ");
        requireFp8ScaleScalarMatchesDescriptor(args.scaleDK.value(), "scaleDK");
        requireFp8ScaleScalarMatchesDescriptor(args.scaleDV.value(), "scaleDV");
        requireFp8ScaleScalarMatchesDescriptor(args.scaleDP.value(), "scaleDP");
        requireFp8ScaleScalarMatchesDescriptor(args.amaxDQ.value(), "amaxDQ");
        requireFp8ScaleScalarMatchesDescriptor(args.amaxDK.value(), "amaxDK");
        requireFp8ScaleScalarMatchesDescriptor(args.amaxDV.value(), "amaxDV");
        requireFp8ScaleScalarMatchesDescriptor(args.amaxDP.value(), "amaxDP");
        insertTensor(pack, UID_DESCALE_Q, args.descaleQ.value());
        insertTensor(pack, UID_DESCALE_K, args.descaleK.value());
        insertTensor(pack, UID_DESCALE_V, args.descaleV.value());
        insertTensor(pack, UID_DESCALE_O, args.descaleO.value());
        insertTensor(pack, UID_DESCALE_DO, args.descaleDO.value());
        insertTensor(pack, UID_DESCALE_S, args.descaleS.value());
        insertTensor(pack, UID_DESCALE_DP, args.descaleDP.value());
        insertTensor(pack, UID_SCALE_S, args.scaleS.value());
        insertTensor(pack, UID_SCALE_DQ, args.scaleDQ.value());
        insertTensor(pack, UID_SCALE_DK, args.scaleDK.value());
        insertTensor(pack, UID_SCALE_DV, args.scaleDV.value());
        insertTensor(pack, UID_SCALE_DP, args.scaleDP.value());
        insertTensor(pack, UID_AMAX_DQ, args.amaxDQ.value());
        insertTensor(pack, UID_AMAX_DK, args.amaxDK.value());
        insertTensor(pack, UID_AMAX_DV, args.amaxDV.value());
        insertTensor(pack, UID_AMAX_DP, args.amaxDP.value());
    }

    executeGraph(graph, pack, stream);
}

void CudnnScaledDotProductAttention::warmForward(const CudnnAttentionDescriptor& descriptor, int gpuNum) {
    (void)cache().getOrBuildForward(descriptor, gpuNum);
}

void CudnnScaledDotProductAttention::warmBackward(const CudnnAttentionDescriptor& descriptor, int gpuNum) {
    (void)cache().getOrBuildBackward(descriptor, gpuNum);
}

void CudnnScaledDotProductAttention::clearCache() { cache().clear(); }

size_t CudnnScaledDotProductAttention::cachedGraphCount() const { return cache().size(); }

bool CudnnScaledDotProductAttention::frontendAvailable() { return true; }
