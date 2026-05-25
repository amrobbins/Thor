#include "Utilities/TensorOperations/DeepLearning/CudnnRmsNorm.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/ScopedGpu.h"

#include <cstddef>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <cudnn_frontend.h>

using namespace ThorImplementation;
using namespace std;

namespace {

namespace fe = cudnn_frontend;

constexpr int64_t UID_X = 30;
constexpr int64_t UID_SCALE = 31;
constexpr int64_t UID_Y = 32;
constexpr int64_t UID_INV_VARIANCE = 33;
constexpr int64_t UID_DY = 34;
constexpr int64_t UID_DX = 35;
constexpr int64_t UID_DSCALE = 36;

[[noreturn]] void throwInvalidRmsNorm(const string& message) { throw invalid_argument("Invalid cuDNN RMSNorm descriptor: " + message); }

bool isSupportedRmsNormIoDtype(DataType dtype) {
    switch (dtype) {
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
            return true;
        default:
            return false;
    }
}

fe::DataType_t toFrontendDataType(DataType dtype) {
    switch (dtype) {
        case DataType::FP16:
            return fe::DataType_t::HALF;
        case DataType::BF16:
            return fe::DataType_t::BFLOAT16;
        case DataType::FP32:
            return fe::DataType_t::FLOAT;
        default:
            throw invalid_argument("Unsupported cuDNN Frontend RMSNorm dtype: " + TensorDescriptor::getElementTypeName(dtype));
    }
}

int64_t checkedI64(uint64_t value, string_view what) {
    if (value == 0) {
        throwInvalidRmsNorm(string(what) + " must be non-zero");
    }
    if (value > static_cast<uint64_t>(numeric_limits<int64_t>::max())) {
        throwInvalidRmsNorm(string(what) + " is too large for cuDNN Frontend int64 dimensions");
    }
    return static_cast<int64_t>(value);
}

string dtypeName(DataType dtype) { return TensorDescriptor::getElementTypeName(dtype); }

[[noreturn]] void throwInvalidFusedActivation(string_view value) {
    throw invalid_argument("Invalid cuDNN RMSNorm fused activation: " + string(value) +
                           ". Supported values are 'none' and 'swish'.");
}

uint64_t checkedMul(uint64_t a, uint64_t b, string_view what) {
    if (a != 0 && b > numeric_limits<uint64_t>::max() / a) {
        throw invalid_argument(string("cuDNN RMSNorm ") + string(what) + " element count overflows uint64_t");
    }
    return a * b;
}

void requireInitialized(const Tensor& tensor, string_view name) {
    if (!tensor.isInitialized()) {
        throw invalid_argument(string("cuDNN RMSNorm tensor '") + string(name) + "' is not initialized.");
    }
}

void requireGpuTensor(const Tensor& tensor, string_view name) {
    requireInitialized(tensor, name);
    if (tensor.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw invalid_argument(string("cuDNN RMSNorm tensor '") + string(name) + "' must be a GPU tensor.");
    }
}

void requireSameGpu(const Tensor& tensor, int gpuNum, string_view name) {
    requireGpuTensor(tensor, name);
    if (tensor.getPlacement().getDeviceNum() != gpuNum) {
        throw invalid_argument(string("cuDNN RMSNorm tensor '") + string(name) + "' is on GPU " +
                               to_string(tensor.getPlacement().getDeviceNum()) + ", expected GPU " + to_string(gpuNum) + ".");
    }
}

void requireDtype(const Tensor& tensor, DataType expected, string_view name) {
    if (tensor.getDataType() != expected) {
        throw invalid_argument(string("cuDNN RMSNorm tensor '") + string(name) + "' dtype mismatch. Expected " + dtypeName(expected) +
                               ", got " + dtypeName(tensor.getDataType()) + ".");
    }
}

void requireNumElements(const Tensor& tensor, uint64_t expected, string_view name) {
    const uint64_t actual = tensor.getTotalNumElements();
    if (actual != expected) {
        throw invalid_argument(string("cuDNN RMSNorm tensor '") + string(name) + "' element-count mismatch. Expected " +
                               to_string(expected) + ", got " + to_string(actual) + ".");
    }
}

void requireIoTensor(const Tensor& tensor,
                     const CudnnRmsNormDescriptor& descriptor,
                     DataType expectedDtype,
                     int gpuNum,
                     string_view name) {
    requireSameGpu(tensor, gpuNum, name);
    requireDtype(tensor, expectedDtype, name);
    requireNumElements(tensor, checkedMul(descriptor.outerSize, descriptor.normalizedFeatureCount, "IO"), name);
}

void requireParameterTensor(const Tensor& tensor, const CudnnRmsNormDescriptor& descriptor, int gpuNum, string_view name) {
    requireSameGpu(tensor, gpuNum, name);
    requireDtype(tensor, descriptor.parameterDataType, name);
    requireNumElements(tensor, descriptor.normalizedFeatureCount, name);
}

void requireStatsTensor(const Tensor& tensor, const CudnnRmsNormDescriptor& descriptor, int gpuNum, string_view name) {
    requireSameGpu(tensor, gpuNum, name);
    requireDtype(tensor, DataType::FP32, name);
    requireNumElements(tensor, descriptor.outerSize, name);
}

void insertTensor(unordered_map<int64_t, void*>& pack, int64_t uid, const Tensor& tensor) {
    pack[uid] = const_cast<void*>(static_cast<const void*>(tensor.getMemPtr<void>()));
}

vector<int64_t> ioDims(const CudnnRmsNormDescriptor& descriptor) {
    return {checkedI64(descriptor.outerSize, "outerSize"), checkedI64(descriptor.normalizedFeatureCount, "normalizedFeatureCount"), 1, 1};
}

vector<int64_t> ioStrides(const CudnnRmsNormDescriptor& descriptor) {
    const int64_t hidden = checkedI64(descriptor.normalizedFeatureCount, "normalizedFeatureCount");
    return {hidden, 1, hidden, hidden};
}

vector<int64_t> parameterDims(const CudnnRmsNormDescriptor& descriptor) {
    return {1, checkedI64(descriptor.normalizedFeatureCount, "normalizedFeatureCount"), 1, 1};
}

vector<int64_t> parameterStrides(const CudnnRmsNormDescriptor& descriptor) {
    const int64_t hidden = checkedI64(descriptor.normalizedFeatureCount, "normalizedFeatureCount");
    return {hidden, 1, hidden, hidden};
}

vector<int64_t> statsDims(const CudnnRmsNormDescriptor& descriptor) {
    return {checkedI64(descriptor.outerSize, "outerSize"), 1, 1, 1};
}

vector<int64_t> statsStrides(const CudnnRmsNormDescriptor&) { return {1, 1, 1, 1}; }

struct BuiltGraph {
    shared_ptr<fe::graph::Graph> graph;
    Tensor workspace;
    int64_t workspaceBytes = 0;
};

class RmsNormGraphCache {
   public:
    BuiltGraph& getOrBuildForward(const CudnnRmsNormDescriptor& descriptor, int gpuNum) {
        const string key = descriptor.cacheKey("forward", gpuNum);
        unique_lock<mutex> lock(mtx);
        auto iter = graphs.find(key);
        if (iter != graphs.end())
            return iter->second;
        BuiltGraph graph = buildForwardGraph(descriptor, gpuNum);
        auto [inserted, _] = graphs.emplace(key, std::move(graph));
        return inserted->second;
    }

    BuiltGraph& getOrBuildBackward(const CudnnRmsNormDescriptor& descriptor, int gpuNum) {
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
        return graph->tensor(fe::graph::Tensor_attributes().set_name(string(name)).set_uid(uid).set_dim(dim).set_stride(stride));
    }

    void finalize(BuiltGraph& built, int gpuNum) {
        ScopedGpu scopedGpu(gpuNum);
        Stream temporaryStream(gpuNum);
        auto status = built.graph->build(temporaryStream.getCudnnHandle(), {fe::HeurMode_t::A});
        if (!status.is_good()) {
            throw runtime_error(
                "Failed to build cuDNN Frontend RMSNorm graph with primary heuristics only "
                "(Thor RMSNorm does not permit cuDNN fallback engines): " +
                status.get_message());
        }

        int64_t workspaceBytes = 0;
        status = built.graph->get_workspace_size(workspaceBytes);
        if (!status.is_good()) {
            throw runtime_error("Failed to query cuDNN Frontend RMSNorm workspace: " + status.get_message());
        }

        built.workspaceBytes = workspaceBytes;
        if (workspaceBytes > 0) {
            built.workspace = Tensor(TensorPlacement(TensorPlacement::MemDevices::GPU, gpuNum),
                                     TensorDescriptor(DataType::UINT8, {static_cast<uint64_t>(workspaceBytes)}),
                                     256);
        }
    }

    BuiltGraph buildForwardGraph(const CudnnRmsNormDescriptor& descriptor, int gpuNum) {
        descriptor.validateForward();

        ScopedGpu scopedGpu(gpuNum);
        BuiltGraph built;
        built.graph = make_shared<fe::graph::Graph>();
        built.graph->set_io_data_type(toFrontendDataType(descriptor.inputDataType))
            .set_intermediate_data_type(toFrontendDataType(descriptor.computeDataType))
            .set_compute_data_type(toFrontendDataType(descriptor.computeDataType));

        const vector<int64_t> dims = ioDims(descriptor);
        const vector<int64_t> strides = ioStrides(descriptor);
        auto x = ioTensor(built.graph, descriptor.debugName + "_x", UID_X, dims, strides);
        auto scale = tensor(built.graph, descriptor.debugName + "_scale", UID_SCALE, parameterDims(descriptor), parameterStrides(descriptor), descriptor.parameterDataType);
        auto epsilon = built.graph->tensor(descriptor.epsilon);

        auto attrs = fe::graph::Rmsnorm_attributes()
                         .set_name(descriptor.debugName + "_forward")
                         .set_forward_phase(descriptor.training ? fe::NormFwdPhase_t::TRAINING : fe::NormFwdPhase_t::INFERENCE)
                         .set_epsilon(epsilon)
                         .set_compute_data_type(toFrontendDataType(descriptor.computeDataType));

        auto [rmsOutput, invVariance] = built.graph->rmsnorm(x, scale, attrs);
        rmsOutput->set_dim(dims).set_stride(strides);

        shared_ptr<fe::graph::Tensor_attributes> y = rmsOutput;
        if (descriptor.fusedActivation == CudnnRmsNormFusedActivation::SWISH) {
            auto swishAttrs = fe::graph::Pointwise_attributes()
                                  .set_name(descriptor.debugName + "_swish")
                                  .set_mode(fe::PointwiseMode_t::SWISH_FWD)
                                  .set_compute_data_type(toFrontendDataType(descriptor.computeDataType));
            y = built.graph->pointwise(rmsOutput, swishAttrs);
            y->set_dim(dims).set_stride(strides);
        }

        y->set_output(true).set_uid(UID_Y).set_dim(dims).set_stride(strides);
        if (descriptor.outputDataType != descriptor.inputDataType) {
            y->set_data_type(toFrontendDataType(descriptor.outputDataType));
        }

        if (descriptor.training) {
            THOR_THROW_IF_FALSE(invVariance != nullptr);
            invVariance->set_output(true)
                .set_uid(UID_INV_VARIANCE)
                .set_dim(statsDims(descriptor))
                .set_stride(statsStrides(descriptor))
                .set_data_type(toFrontendDataType(DataType::FP32));
        }

        finalize(built, gpuNum);
        return built;
    }

    BuiltGraph buildBackwardGraph(const CudnnRmsNormDescriptor& descriptor, int gpuNum) {
        descriptor.validateBackward();

        ScopedGpu scopedGpu(gpuNum);
        BuiltGraph built;
        built.graph = make_shared<fe::graph::Graph>();
        built.graph->set_io_data_type(toFrontendDataType(descriptor.inputDataType))
            .set_intermediate_data_type(toFrontendDataType(descriptor.computeDataType))
            .set_compute_data_type(toFrontendDataType(descriptor.computeDataType));

        const vector<int64_t> dims = ioDims(descriptor);
        const vector<int64_t> strides = ioStrides(descriptor);
        auto dy = ioTensor(built.graph, descriptor.debugName + "_dy", UID_DY, dims, strides);
        auto x = ioTensor(built.graph, descriptor.debugName + "_x", UID_X, dims, strides);
        auto scale = tensor(built.graph, descriptor.debugName + "_scale", UID_SCALE, parameterDims(descriptor), parameterStrides(descriptor), descriptor.parameterDataType);
        auto invVariance = tensor(built.graph,
                                  descriptor.debugName + "_inv_variance",
                                  UID_INV_VARIANCE,
                                  statsDims(descriptor),
                                  statsStrides(descriptor),
                                  DataType::FP32);

        auto attrs = fe::graph::Rmsnorm_backward_attributes().set_name(descriptor.debugName + "_backward").has_dbias(false);

        auto [dx, dscale, dbias] = built.graph->rmsnorm_backward(dy, x, scale, invVariance, attrs);
        THOR_THROW_IF_FALSE(dbias == nullptr);
        dx->set_output(true).set_uid(UID_DX).set_dim(dims).set_stride(strides);
        dscale->set_output(true)
            .set_uid(UID_DSCALE)
            .set_dim(parameterDims(descriptor))
            .set_stride(parameterStrides(descriptor))
            .set_data_type(toFrontendDataType(descriptor.parameterDataType));

        finalize(built, gpuNum);
        return built;
    }

    mutable mutex mtx;
    unordered_map<string, BuiltGraph> graphs;
};

RmsNormGraphCache& cache() {
    static RmsNormGraphCache instance;
    return instance;
}

}  // namespace

const char* ThorImplementation::toString(CudnnRmsNormFusedActivation activation) {
    switch (activation) {
        case CudnnRmsNormFusedActivation::NONE:
            return "none";
        case CudnnRmsNormFusedActivation::SWISH:
            return "swish";
    }
    return "unknown";
}

CudnnRmsNormFusedActivation ThorImplementation::cudnnRmsNormFusedActivationFromString(string_view value) {
    if (value == "none")
        return CudnnRmsNormFusedActivation::NONE;
    if (value == "swish" || value == "silu")
        return CudnnRmsNormFusedActivation::SWISH;
    throwInvalidFusedActivation(value);
}

void CudnnRmsNormDescriptor::validateForward() const {
    checkedI64(outerSize, "outerSize");
    checkedI64(normalizedFeatureCount, "normalizedFeatureCount");
    (void)checkedMul(outerSize, normalizedFeatureCount, "IO");
    if (!isSupportedRmsNormIoDtype(inputDataType)) {
        throwInvalidRmsNorm("inputDataType must be fp16, bf16, or fp32; got " + dtypeName(inputDataType));
    }
    if (!isSupportedRmsNormIoDtype(outputDataType)) {
        throwInvalidRmsNorm("outputDataType must be fp16, bf16, or fp32; got " + dtypeName(outputDataType));
    }
    if (fusedActivation == CudnnRmsNormFusedActivation::SWISH) {
        if (training) {
            throwInvalidRmsNorm(
                "fused SWISH is inference-only because cuDNN Frontend only detects the RMSNorm + SiLU fusion when RMSNorm is in inference phase");
        }
        if (inputDataType != DataType::BF16 || outputDataType != DataType::BF16 ||
            parameterDataType != DataType::BF16) {
            throwInvalidRmsNorm("fused SWISH currently requires bf16 input, bf16 output, and bf16 scale parameters; got input " +
                                dtypeName(inputDataType) + ", output " + dtypeName(outputDataType) + ", scale " + dtypeName(parameterDataType));
        }
    } else if (parameterDataType != DataType::FP32) {
        throwInvalidRmsNorm("scale parameters are currently required to be fp32; got " + dtypeName(parameterDataType));
    }
    if (computeDataType != DataType::FP32) {
        throwInvalidRmsNorm("computeDataType is currently required to be fp32; got " + dtypeName(computeDataType));
    }
    if (!(epsilon > 0.0f)) {
        throwInvalidRmsNorm("epsilon must be > 0");
    }
}

void CudnnRmsNormDescriptor::validateBackward() const {
    validateForward();
    if (fusedActivation != CudnnRmsNormFusedActivation::NONE) {
        throwInvalidRmsNorm("fused activation backward is not supported; RMSNorm + SiLU fusion is an inference forward-only cuDNN Frontend path");
    }
}

string CudnnRmsNormDescriptor::cacheKey(string_view passName, int gpuNum) const {
    ostringstream out;
    out << "rmsnorm:" << passName << ":gpu=" << gpuNum << ":outer=" << outerSize << ":hidden=" << normalizedFeatureCount
        << ":in=" << static_cast<int>(inputDataType) << ":out=" << static_cast<int>(outputDataType)
        << ":param=" << static_cast<int>(parameterDataType) << ":compute=" << static_cast<int>(computeDataType)
        << ":eps=" << epsilon << ":training=" << training << ":fused=" << toString(fusedActivation);
    return out.str();
}

CudnnRmsNorm& CudnnRmsNorm::instance() {
    static CudnnRmsNorm singleton;
    return singleton;
}

void CudnnRmsNorm::forward(const CudnnRmsNormDescriptor& descriptor, const CudnnRmsNormForwardArgs& args, Stream stream) {
    descriptor.validateForward();
    const int gpuNum = stream.getGpuNum();
    requireIoTensor(args.x, descriptor, descriptor.inputDataType, gpuNum, "x");
    requireIoTensor(args.y, descriptor, descriptor.outputDataType, gpuNum, "y");
    requireParameterTensor(args.scale, descriptor, gpuNum, "scale");
    if (descriptor.training) {
        if (!args.invVariance.has_value()) {
            throw invalid_argument("cuDNN RMSNorm forward training requires an invVariance output tensor.");
        }
        requireStatsTensor(args.invVariance.value(), descriptor, gpuNum, "invVariance");
    }

    ScopedGpu scopedGpu(gpuNum);
    BuiltGraph& built = cache().getOrBuildForward(descriptor, gpuNum);
    unordered_map<int64_t, void*> variantPack;
    insertTensor(variantPack, UID_X, args.x);
    insertTensor(variantPack, UID_SCALE, args.scale);
    insertTensor(variantPack, UID_Y, args.y);
    if (descriptor.training) {
        insertTensor(variantPack, UID_INV_VARIANCE, args.invVariance.value());
    }

    void* workspace = built.workspaceBytes > 0 ? built.workspace.getMemPtr<void>() : nullptr;
    auto status = built.graph->execute(stream.getCudnnHandle(), variantPack, workspace);
    if (!status.is_good()) {
        throw runtime_error("Failed to execute cuDNN Frontend RMSNorm graph: " + status.get_message());
    }
}

void CudnnRmsNorm::backward(const CudnnRmsNormDescriptor& descriptor, const CudnnRmsNormBackwardArgs& args, Stream stream) {
    descriptor.validateBackward();
    const int gpuNum = stream.getGpuNum();
    requireIoTensor(args.dy, descriptor, descriptor.outputDataType, gpuNum, "dy");
    requireIoTensor(args.x, descriptor, descriptor.inputDataType, gpuNum, "x");
    requireIoTensor(args.dx, descriptor, descriptor.inputDataType, gpuNum, "dx");
    requireParameterTensor(args.scale, descriptor, gpuNum, "scale");
    requireStatsTensor(args.invVariance, descriptor, gpuNum, "invVariance");
    requireParameterTensor(args.dscale, descriptor, gpuNum, "dscale");

    ScopedGpu scopedGpu(gpuNum);
    BuiltGraph& built = cache().getOrBuildBackward(descriptor, gpuNum);
    unordered_map<int64_t, void*> variantPack;
    insertTensor(variantPack, UID_DY, args.dy);
    insertTensor(variantPack, UID_X, args.x);
    insertTensor(variantPack, UID_SCALE, args.scale);
    insertTensor(variantPack, UID_INV_VARIANCE, args.invVariance);
    insertTensor(variantPack, UID_DX, args.dx);
    insertTensor(variantPack, UID_DSCALE, args.dscale);

    void* workspace = built.workspaceBytes > 0 ? built.workspace.getMemPtr<void>() : nullptr;
    auto status = built.graph->execute(stream.getCudnnHandle(), variantPack, workspace);
    if (!status.is_good()) {
        throw runtime_error("Failed to execute cuDNN Frontend RMSNorm graph: " + status.get_message());
    }
}

void CudnnRmsNorm::warmForward(const CudnnRmsNormDescriptor& descriptor, int gpuNum) { (void)cache().getOrBuildForward(descriptor, gpuNum); }

void CudnnRmsNorm::warmBackward(const CudnnRmsNormDescriptor& descriptor, int gpuNum) { (void)cache().getOrBuildBackward(descriptor, gpuNum); }

void CudnnRmsNorm::clearCache() { cache().clear(); }

size_t CudnnRmsNorm::cachedGraphCount() const { return cache().size(); }

bool CudnnRmsNorm::frontendAvailable() { return true; }
