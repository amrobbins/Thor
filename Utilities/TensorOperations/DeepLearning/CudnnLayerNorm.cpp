#include "Utilities/TensorOperations/DeepLearning/CudnnLayerNorm.h"

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

constexpr int64_t UID_X = 10;
constexpr int64_t UID_SCALE = 11;
constexpr int64_t UID_BIAS = 12;
constexpr int64_t UID_Y = 13;
constexpr int64_t UID_MEAN = 14;
constexpr int64_t UID_INV_VARIANCE = 15;
constexpr int64_t UID_DY = 16;
constexpr int64_t UID_DX = 17;
constexpr int64_t UID_DSCALE = 18;
constexpr int64_t UID_DBIAS = 19;

[[noreturn]] void throwInvalidLayerNorm(const string& message) { throw invalid_argument("Invalid cuDNN LayerNorm descriptor: " + message); }

bool isSupportedLayerNormIoDtype(DataType dtype) {
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
            throw invalid_argument("Unsupported cuDNN Frontend LayerNorm dtype: " + TensorDescriptor::getElementTypeName(dtype));
    }
}

int64_t checkedI64(uint64_t value, string_view what) {
    if (value == 0) {
        throwInvalidLayerNorm(string(what) + " must be non-zero");
    }
    if (value > static_cast<uint64_t>(numeric_limits<int64_t>::max())) {
        throwInvalidLayerNorm(string(what) + " is too large for cuDNN Frontend int64 dimensions");
    }
    return static_cast<int64_t>(value);
}

string dtypeName(DataType dtype) { return TensorDescriptor::getElementTypeName(dtype); }

uint64_t checkedMul(uint64_t a, uint64_t b, string_view what) {
    if (a != 0 && b > numeric_limits<uint64_t>::max() / a) {
        throw invalid_argument(string("cuDNN LayerNorm ") + string(what) + " element count overflows uint64_t");
    }
    return a * b;
}

void requireInitialized(const Tensor& tensor, string_view name) {
    if (!tensor.isInitialized()) {
        throw invalid_argument(string("cuDNN LayerNorm tensor '") + string(name) + "' is not initialized.");
    }
}

void requireGpuTensor(const Tensor& tensor, string_view name) {
    requireInitialized(tensor, name);
    if (tensor.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw invalid_argument(string("cuDNN LayerNorm tensor '") + string(name) + "' must be a GPU tensor.");
    }
}

void requireSameGpu(const Tensor& tensor, int gpuNum, string_view name) {
    requireGpuTensor(tensor, name);
    if (tensor.getPlacement().getDeviceNum() != gpuNum) {
        throw invalid_argument(string("cuDNN LayerNorm tensor '") + string(name) + "' is on GPU " +
                               to_string(tensor.getPlacement().getDeviceNum()) + ", expected GPU " + to_string(gpuNum) + ".");
    }
}

void requireDtype(const Tensor& tensor, DataType expected, string_view name) {
    if (tensor.getDataType() != expected) {
        throw invalid_argument(string("cuDNN LayerNorm tensor '") + string(name) + "' dtype mismatch. Expected " + dtypeName(expected) +
                               ", got " + dtypeName(tensor.getDataType()) + ".");
    }
}

void requireNumElements(const Tensor& tensor, uint64_t expected, string_view name) {
    const uint64_t actual = tensor.getTotalNumElements();
    if (actual != expected) {
        throw invalid_argument(string("cuDNN LayerNorm tensor '") + string(name) + "' element-count mismatch. Expected " +
                               to_string(expected) + ", got " + to_string(actual) + ".");
    }
}

void requireIoTensor(const Tensor& tensor,
                     const CudnnLayerNormDescriptor& descriptor,
                     DataType expectedDtype,
                     int gpuNum,
                     string_view name) {
    requireSameGpu(tensor, gpuNum, name);
    requireDtype(tensor, expectedDtype, name);
    requireNumElements(tensor, checkedMul(descriptor.outerSize, descriptor.normalizedFeatureCount, "IO"), name);
}

void requireParameterTensor(const Tensor& tensor, const CudnnLayerNormDescriptor& descriptor, int gpuNum, string_view name) {
    requireSameGpu(tensor, gpuNum, name);
    requireDtype(tensor, descriptor.parameterDataType, name);
    requireNumElements(tensor, descriptor.normalizedFeatureCount, name);
}

void requireStatsTensor(const Tensor& tensor, const CudnnLayerNormDescriptor& descriptor, int gpuNum, string_view name) {
    requireSameGpu(tensor, gpuNum, name);
    requireDtype(tensor, DataType::FP32, name);
    requireNumElements(tensor, descriptor.outerSize, name);
}

void insertTensor(unordered_map<int64_t, void*>& pack, int64_t uid, const Tensor& tensor) {
    pack[uid] = const_cast<void*>(static_cast<const void*>(tensor.getMemPtr<void>()));
}

vector<int64_t> ioDims(const CudnnLayerNormDescriptor& descriptor) {
    return {checkedI64(descriptor.outerSize, "outerSize"), checkedI64(descriptor.normalizedFeatureCount, "normalizedFeatureCount"), 1, 1};
}

vector<int64_t> ioStrides(const CudnnLayerNormDescriptor& descriptor) {
    const int64_t hidden = checkedI64(descriptor.normalizedFeatureCount, "normalizedFeatureCount");
    return {hidden, 1, hidden, hidden};
}

vector<int64_t> parameterDims(const CudnnLayerNormDescriptor& descriptor) {
    return {1, checkedI64(descriptor.normalizedFeatureCount, "normalizedFeatureCount"), 1, 1};
}

vector<int64_t> parameterStrides(const CudnnLayerNormDescriptor& descriptor) {
    const int64_t hidden = checkedI64(descriptor.normalizedFeatureCount, "normalizedFeatureCount");
    return {hidden, 1, hidden, hidden};
}

vector<int64_t> statsDims(const CudnnLayerNormDescriptor& descriptor) {
    return {checkedI64(descriptor.outerSize, "outerSize"), 1, 1, 1};
}

vector<int64_t> statsStrides(const CudnnLayerNormDescriptor&) { return {1, 1, 1, 1}; }

struct BuiltGraph {
    shared_ptr<fe::graph::Graph> graph;
    Tensor workspace;
    int64_t workspaceBytes = 0;
};

class LayerNormGraphCache {
   public:
    BuiltGraph& getOrBuildForward(const CudnnLayerNormDescriptor& descriptor, int gpuNum) {
        const string key = descriptor.cacheKey("forward", gpuNum);
        unique_lock<mutex> lock(mtx);
        auto iter = graphs.find(key);
        if (iter != graphs.end())
            return iter->second;
        BuiltGraph graph = buildForwardGraph(descriptor, gpuNum);
        auto [inserted, _] = graphs.emplace(key, std::move(graph));
        return inserted->second;
    }

    BuiltGraph& getOrBuildBackward(const CudnnLayerNormDescriptor& descriptor, int gpuNum) {
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
                "Failed to build cuDNN Frontend LayerNorm graph with primary heuristics only "
                "(Thor LayerNorm does not permit cuDNN fallback engines): " +
                status.get_message());
        }

        int64_t workspaceBytes = 0;
        status = built.graph->get_workspace_size(workspaceBytes);
        if (!status.is_good()) {
            throw runtime_error("Failed to query cuDNN Frontend LayerNorm workspace: " + status.get_message());
        }

        built.workspaceBytes = workspaceBytes;
        if (workspaceBytes > 0) {
            built.workspace = Tensor(TensorPlacement(TensorPlacement::MemDevices::GPU, gpuNum),
                                     TensorDescriptor(DataType::UINT8, {static_cast<uint64_t>(workspaceBytes)}),
                                     256);
        }
    }

    BuiltGraph buildForwardGraph(const CudnnLayerNormDescriptor& descriptor, int gpuNum) {
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
        auto bias = tensor(built.graph, descriptor.debugName + "_bias", UID_BIAS, parameterDims(descriptor), parameterStrides(descriptor), descriptor.parameterDataType);
        auto epsilon = built.graph->tensor(descriptor.epsilon);

        auto attrs = fe::graph::Layernorm_attributes()
                         .set_name(descriptor.debugName + "_forward")
                         .set_forward_phase(descriptor.training ? fe::NormFwdPhase_t::TRAINING : fe::NormFwdPhase_t::INFERENCE)
                         .set_epsilon(epsilon)
                         .set_compute_data_type(toFrontendDataType(descriptor.computeDataType));

        auto [y, mean, invVariance] = built.graph->layernorm(x, scale, bias, attrs);
        y->set_output(true).set_uid(UID_Y).set_dim(dims).set_stride(strides);
        if (descriptor.outputDataType != descriptor.inputDataType) {
            y->set_data_type(toFrontendDataType(descriptor.outputDataType));
        }

        if (descriptor.training) {
            THOR_THROW_IF_FALSE(mean != nullptr);
            THOR_THROW_IF_FALSE(invVariance != nullptr);
            mean->set_output(true)
                .set_uid(UID_MEAN)
                .set_dim(statsDims(descriptor))
                .set_stride(statsStrides(descriptor))
                .set_data_type(toFrontendDataType(DataType::FP32));
            invVariance->set_output(true)
                .set_uid(UID_INV_VARIANCE)
                .set_dim(statsDims(descriptor))
                .set_stride(statsStrides(descriptor))
                .set_data_type(toFrontendDataType(DataType::FP32));
        }

        finalize(built, gpuNum);
        return built;
    }

    BuiltGraph buildBackwardGraph(const CudnnLayerNormDescriptor& descriptor, int gpuNum) {
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
        auto mean = tensor(built.graph, descriptor.debugName + "_mean", UID_MEAN, statsDims(descriptor), statsStrides(descriptor), DataType::FP32);
        auto invVariance = tensor(built.graph,
                                  descriptor.debugName + "_inv_variance",
                                  UID_INV_VARIANCE,
                                  statsDims(descriptor),
                                  statsStrides(descriptor),
                                  DataType::FP32);

        auto attrs = fe::graph::Layernorm_backward_attributes()
                         .set_name(descriptor.debugName + "_backward")
                         .set_saved_mean_and_inv_variance(mean, invVariance)
                         .set_compute_data_type(toFrontendDataType(descriptor.computeDataType));

        auto [dx, dscale, dbias] = built.graph->layernorm_backward(dy, x, scale, attrs);
        dx->set_output(true).set_uid(UID_DX).set_dim(dims).set_stride(strides);
        dscale->set_output(true)
            .set_uid(UID_DSCALE)
            .set_dim(parameterDims(descriptor))
            .set_stride(parameterStrides(descriptor))
            .set_data_type(toFrontendDataType(descriptor.parameterDataType));
        dbias->set_output(true)
            .set_uid(UID_DBIAS)
            .set_dim(parameterDims(descriptor))
            .set_stride(parameterStrides(descriptor))
            .set_data_type(toFrontendDataType(descriptor.parameterDataType));

        finalize(built, gpuNum);
        return built;
    }

    mutable mutex mtx;
    unordered_map<string, BuiltGraph> graphs;
};

LayerNormGraphCache& cache() {
    static LayerNormGraphCache instance;
    return instance;
}

}  // namespace

void CudnnLayerNormDescriptor::validateForward() const {
    checkedI64(outerSize, "outerSize");
    checkedI64(normalizedFeatureCount, "normalizedFeatureCount");
    (void)checkedMul(outerSize, normalizedFeatureCount, "IO");
    if (!isSupportedLayerNormIoDtype(inputDataType)) {
        throwInvalidLayerNorm("inputDataType must be fp16, bf16, or fp32; got " + dtypeName(inputDataType));
    }
    if (!isSupportedLayerNormIoDtype(outputDataType)) {
        throwInvalidLayerNorm("outputDataType must be fp16, bf16, or fp32; got " + dtypeName(outputDataType));
    }
    if (parameterDataType != DataType::FP32) {
        throwInvalidLayerNorm("scale/bias parameters are currently required to be fp32; got " + dtypeName(parameterDataType));
    }
    if (computeDataType != DataType::FP32) {
        throwInvalidLayerNorm("computeDataType is currently required to be fp32; got " + dtypeName(computeDataType));
    }
    if (!(epsilon > 0.0f)) {
        throwInvalidLayerNorm("epsilon must be > 0");
    }
}

void CudnnLayerNormDescriptor::validateBackward() const { validateForward(); }

string CudnnLayerNormDescriptor::cacheKey(string_view passName, int gpuNum) const {
    ostringstream out;
    out << "layernorm:" << passName << ":gpu=" << gpuNum << ":outer=" << outerSize << ":hidden=" << normalizedFeatureCount
        << ":in=" << static_cast<int>(inputDataType) << ":out=" << static_cast<int>(outputDataType)
        << ":param=" << static_cast<int>(parameterDataType) << ":compute=" << static_cast<int>(computeDataType)
        << ":eps=" << epsilon << ":training=" << training;
    return out.str();
}

CudnnLayerNorm& CudnnLayerNorm::instance() {
    static CudnnLayerNorm singleton;
    return singleton;
}

void CudnnLayerNorm::forward(const CudnnLayerNormDescriptor& descriptor, const CudnnLayerNormForwardArgs& args, Stream stream) {
    descriptor.validateForward();
    const int gpuNum = stream.getGpuNum();
    requireIoTensor(args.x, descriptor, descriptor.inputDataType, gpuNum, "x");
    requireIoTensor(args.y, descriptor, descriptor.outputDataType, gpuNum, "y");
    requireParameterTensor(args.scale, descriptor, gpuNum, "scale");
    requireParameterTensor(args.bias, descriptor, gpuNum, "bias");
    if (descriptor.training) {
        if (!args.mean.has_value() || !args.invVariance.has_value()) {
            throw invalid_argument("cuDNN LayerNorm forward training requires mean and invVariance output tensors.");
        }
        requireStatsTensor(args.mean.value(), descriptor, gpuNum, "mean");
        requireStatsTensor(args.invVariance.value(), descriptor, gpuNum, "invVariance");
    }

    ScopedGpu scopedGpu(gpuNum);
    BuiltGraph& built = cache().getOrBuildForward(descriptor, gpuNum);
    unordered_map<int64_t, void*> variantPack;
    insertTensor(variantPack, UID_X, args.x);
    insertTensor(variantPack, UID_SCALE, args.scale);
    insertTensor(variantPack, UID_BIAS, args.bias);
    insertTensor(variantPack, UID_Y, args.y);
    if (descriptor.training) {
        insertTensor(variantPack, UID_MEAN, args.mean.value());
        insertTensor(variantPack, UID_INV_VARIANCE, args.invVariance.value());
    }

    void* workspace = built.workspaceBytes > 0 ? built.workspace.getMemPtr<void>() : nullptr;
    auto status = built.graph->execute(stream.getCudnnHandle(), variantPack, workspace);
    if (!status.is_good()) {
        throw runtime_error("Failed to execute cuDNN Frontend LayerNorm graph: " + status.get_message());
    }
}

void CudnnLayerNorm::backward(const CudnnLayerNormDescriptor& descriptor, const CudnnLayerNormBackwardArgs& args, Stream stream) {
    descriptor.validateBackward();
    const int gpuNum = stream.getGpuNum();
    requireIoTensor(args.dy, descriptor, descriptor.outputDataType, gpuNum, "dy");
    requireIoTensor(args.x, descriptor, descriptor.inputDataType, gpuNum, "x");
    requireIoTensor(args.dx, descriptor, descriptor.inputDataType, gpuNum, "dx");
    requireParameterTensor(args.scale, descriptor, gpuNum, "scale");
    requireStatsTensor(args.mean, descriptor, gpuNum, "mean");
    requireStatsTensor(args.invVariance, descriptor, gpuNum, "invVariance");
    requireParameterTensor(args.dscale, descriptor, gpuNum, "dscale");
    requireParameterTensor(args.dbias, descriptor, gpuNum, "dbias");

    ScopedGpu scopedGpu(gpuNum);
    BuiltGraph& built = cache().getOrBuildBackward(descriptor, gpuNum);
    unordered_map<int64_t, void*> variantPack;
    insertTensor(variantPack, UID_DY, args.dy);
    insertTensor(variantPack, UID_X, args.x);
    insertTensor(variantPack, UID_SCALE, args.scale);
    insertTensor(variantPack, UID_MEAN, args.mean);
    insertTensor(variantPack, UID_INV_VARIANCE, args.invVariance);
    insertTensor(variantPack, UID_DX, args.dx);
    insertTensor(variantPack, UID_DSCALE, args.dscale);
    insertTensor(variantPack, UID_DBIAS, args.dbias);

    void* workspace = built.workspaceBytes > 0 ? built.workspace.getMemPtr<void>() : nullptr;
    auto status = built.graph->execute(stream.getCudnnHandle(), variantPack, workspace);
    if (!status.is_good()) {
        throw runtime_error("Failed to execute cuDNN Frontend LayerNorm graph: " + status.get_message());
    }
}

void CudnnLayerNorm::warmForward(const CudnnLayerNormDescriptor& descriptor, int gpuNum) { (void)cache().getOrBuildForward(descriptor, gpuNum); }

void CudnnLayerNorm::warmBackward(const CudnnLayerNormDescriptor& descriptor, int gpuNum) { (void)cache().getOrBuildBackward(descriptor, gpuNum); }

void CudnnLayerNorm::clearCache() { cache().clear(); }

size_t CudnnLayerNorm::cachedGraphCount() const { return cache().size(); }

bool CudnnLayerNorm::frontendAvailable() { return true; }
