#include "Utilities/TensorOperations/DeepLearning/CudnnInstanceNorm.h"

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

[[noreturn]] void throwInvalidInstanceNorm(const string& message) { throw invalid_argument("Invalid cuDNN InstanceNorm descriptor: " + message); }

bool isSupportedInstanceNormIoDtype(TensorDescriptor::DataType dtype) {
    switch (dtype) {
        case TensorDescriptor::DataType::FP16:
        case TensorDescriptor::DataType::BF16:
        case TensorDescriptor::DataType::FP32:
            return true;
        default:
            return false;
    }
}

bool isReducedPrecisionInstanceNormIoDtype(TensorDescriptor::DataType dtype) {
    return dtype == TensorDescriptor::DataType::FP16 || dtype == TensorDescriptor::DataType::BF16;
}

void validateCudnnFrontendPrimaryEngineContract(const CudnnInstanceNormDescriptor& descriptor) {
    if ((isReducedPrecisionInstanceNormIoDtype(descriptor.inputDataType) ||
         isReducedPrecisionInstanceNormIoDtype(descriptor.outputDataType)) &&
        descriptor.channelCount % 8 != 0) {
        throwInvalidInstanceNorm(
            "cuDNN Frontend primary InstanceNorm engines require fp16/bf16 channelCount to be a multiple of 8; got " +
            to_string(descriptor.channelCount));
    }
}

fe::DataType_t toFrontendDataType(TensorDescriptor::DataType dtype) {
    switch (dtype) {
        case TensorDescriptor::DataType::FP16:
            return fe::DataType_t::HALF;
        case TensorDescriptor::DataType::BF16:
            return fe::DataType_t::BFLOAT16;
        case TensorDescriptor::DataType::FP32:
            return fe::DataType_t::FLOAT;
        default:
            throw invalid_argument("Unsupported cuDNN Frontend InstanceNorm dtype: " + TensorDescriptor::getElementTypeName(dtype));
    }
}

int64_t checkedI64(uint64_t value, string_view what) {
    if (value == 0) {
        throwInvalidInstanceNorm(string(what) + " must be non-zero");
    }
    if (value > static_cast<uint64_t>(numeric_limits<int64_t>::max())) {
        throwInvalidInstanceNorm(string(what) + " is too large for cuDNN Frontend int64 dimensions");
    }
    return static_cast<int64_t>(value);
}

string dtypeName(TensorDescriptor::DataType dtype) { return TensorDescriptor::getElementTypeName(dtype); }

uint64_t checkedMul(uint64_t a, uint64_t b, string_view what) {
    if (a != 0 && b > numeric_limits<uint64_t>::max() / a) {
        throw invalid_argument(string("cuDNN InstanceNorm ") + string(what) + " element count overflows uint64_t");
    }
    return a * b;
}

uint64_t ioElementCount(const CudnnInstanceNormDescriptor& descriptor) {
    return checkedMul(checkedMul(descriptor.batchSize, descriptor.channelCount, "IO"), descriptor.spatialElementCount, "IO");
}

uint64_t statsElementCount(const CudnnInstanceNormDescriptor& descriptor) {
    return checkedMul(descriptor.batchSize, descriptor.channelCount, "stats");
}

void requireInitialized(const Tensor& tensor, string_view name) {
    if (!tensor.isInitialized()) {
        throw invalid_argument(string("cuDNN InstanceNorm tensor '") + string(name) + "' is not initialized.");
    }
}

void requireGpuTensor(const Tensor& tensor, string_view name) {
    requireInitialized(tensor, name);
    if (tensor.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw invalid_argument(string("cuDNN InstanceNorm tensor '") + string(name) + "' must be a GPU tensor.");
    }
}

void requireSameGpu(const Tensor& tensor, int gpuNum, string_view name) {
    requireGpuTensor(tensor, name);
    if (tensor.getPlacement().getDeviceNum() != gpuNum) {
        throw invalid_argument(string("cuDNN InstanceNorm tensor '") + string(name) + "' is on GPU " +
                               to_string(tensor.getPlacement().getDeviceNum()) + ", expected GPU " + to_string(gpuNum) + ".");
    }
}

void requireDtype(const Tensor& tensor, TensorDescriptor::DataType expected, string_view name) {
    if (tensor.getDataType() != expected) {
        throw invalid_argument(string("cuDNN InstanceNorm tensor '") + string(name) + "' dtype mismatch. Expected " + dtypeName(expected) +
                               ", got " + dtypeName(tensor.getDataType()) + ".");
    }
}

void requireNumElements(const Tensor& tensor, uint64_t expected, string_view name) {
    const uint64_t actual = tensor.getTotalNumElements();
    if (actual != expected) {
        throw invalid_argument(string("cuDNN InstanceNorm tensor '") + string(name) + "' element-count mismatch. Expected " +
                               to_string(expected) + ", got " + to_string(actual) + ".");
    }
}

void requireIoTensor(const Tensor& tensor,
                     const CudnnInstanceNormDescriptor& descriptor,
                     TensorDescriptor::DataType expectedDtype,
                     int gpuNum,
                     string_view name) {
    requireSameGpu(tensor, gpuNum, name);
    requireDtype(tensor, expectedDtype, name);
    requireNumElements(tensor, ioElementCount(descriptor), name);
}

void requireParameterTensor(const Tensor& tensor, const CudnnInstanceNormDescriptor& descriptor, int gpuNum, string_view name) {
    requireSameGpu(tensor, gpuNum, name);
    requireDtype(tensor, descriptor.parameterDataType, name);
    requireNumElements(tensor, descriptor.channelCount, name);
}

void requireStatsTensor(const Tensor& tensor, const CudnnInstanceNormDescriptor& descriptor, int gpuNum, string_view name) {
    requireSameGpu(tensor, gpuNum, name);
    requireDtype(tensor, TensorDescriptor::DataType::FP32, name);
    requireNumElements(tensor, statsElementCount(descriptor), name);
}

void insertTensor(unordered_map<int64_t, void*>& pack, int64_t uid, const Tensor& tensor) {
    pack[uid] = const_cast<void*>(static_cast<const void*>(tensor.getMemPtr<void>()));
}

vector<int64_t> ioDims(const CudnnInstanceNormDescriptor& descriptor) {
    return {checkedI64(descriptor.batchSize, "batchSize"),
            checkedI64(descriptor.channelCount, "channelCount"),
            checkedI64(descriptor.spatialElementCount, "spatialElementCount"),
            1};
}

vector<int64_t> ioStrides(const CudnnInstanceNormDescriptor& descriptor) {
    const int64_t channels = checkedI64(descriptor.channelCount, "channelCount");
    const int64_t spatial = checkedI64(descriptor.spatialElementCount, "spatialElementCount");
    return {channels * spatial, spatial, 1, 1};
}

vector<int64_t> parameterDims(const CudnnInstanceNormDescriptor& descriptor) {
    return {1, checkedI64(descriptor.channelCount, "channelCount"), 1, 1};
}

vector<int64_t> parameterStrides(const CudnnInstanceNormDescriptor& descriptor) {
    return {checkedI64(descriptor.channelCount, "channelCount"), 1, 1, 1};
}

vector<int64_t> statsDims(const CudnnInstanceNormDescriptor& descriptor) {
    return {checkedI64(descriptor.batchSize, "batchSize"), checkedI64(descriptor.channelCount, "channelCount"), 1, 1};
}

vector<int64_t> statsStrides(const CudnnInstanceNormDescriptor& descriptor) {
    return {checkedI64(descriptor.channelCount, "channelCount"), 1, 1, 1};
}

struct BuiltGraph {
    shared_ptr<fe::graph::Graph> graph;
    Tensor workspace;
    int64_t workspaceBytes = 0;
};

class InstanceNormGraphCache {
   public:
    BuiltGraph& getOrBuildForward(const CudnnInstanceNormDescriptor& descriptor, int gpuNum) {
        const string key = descriptor.cacheKey("forward", gpuNum);
        unique_lock<mutex> lock(mtx);
        auto iter = graphs.find(key);
        if (iter != graphs.end())
            return iter->second;
        BuiltGraph graph = buildForwardGraph(descriptor, gpuNum);
        auto [inserted, _] = graphs.emplace(key, std::move(graph));
        return inserted->second;
    }

    BuiltGraph& getOrBuildBackward(const CudnnInstanceNormDescriptor& descriptor, int gpuNum) {
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
                                                    TensorDescriptor::DataType dtype) {
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
                "Failed to build cuDNN Frontend InstanceNorm graph with primary heuristics only "
                "(Thor InstanceNorm does not permit cuDNN fallback engines): " +
                status.get_message());
        }

        int64_t workspaceBytes = 0;
        status = built.graph->get_workspace_size(workspaceBytes);
        if (!status.is_good()) {
            throw runtime_error("Failed to query cuDNN Frontend InstanceNorm workspace: " + status.get_message());
        }

        built.workspaceBytes = workspaceBytes;
        if (workspaceBytes > 0) {
            built.workspace = Tensor(TensorPlacement(TensorPlacement::MemDevices::GPU, gpuNum),
                                     TensorDescriptor(TensorDescriptor::DataType::UINT8, {static_cast<uint64_t>(workspaceBytes)}),
                                     256);
        }
    }

    BuiltGraph buildForwardGraph(const CudnnInstanceNormDescriptor& descriptor, int gpuNum) {
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
        auto scale = tensor(built.graph,
                            descriptor.debugName + "_scale",
                            UID_SCALE,
                            parameterDims(descriptor),
                            parameterStrides(descriptor),
                            descriptor.parameterDataType);
        auto bias = tensor(built.graph,
                           descriptor.debugName + "_bias",
                           UID_BIAS,
                           parameterDims(descriptor),
                           parameterStrides(descriptor),
                           descriptor.parameterDataType);
        auto epsilon = built.graph->tensor(descriptor.epsilon);

        auto attrs = fe::graph::Instancenorm_attributes()
                         .set_name(descriptor.debugName + "_forward")
                         .set_forward_phase(descriptor.training ? fe::NormFwdPhase_t::TRAINING : fe::NormFwdPhase_t::INFERENCE)
                         .set_epsilon(epsilon)
                         .set_compute_data_type(toFrontendDataType(descriptor.computeDataType));

        auto [y, mean, invVariance] = built.graph->instancenorm(x, scale, bias, attrs);
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
                .set_data_type(toFrontendDataType(TensorDescriptor::DataType::FP32));
            invVariance->set_output(true)
                .set_uid(UID_INV_VARIANCE)
                .set_dim(statsDims(descriptor))
                .set_stride(statsStrides(descriptor))
                .set_data_type(toFrontendDataType(TensorDescriptor::DataType::FP32));
        }

        finalize(built, gpuNum);
        return built;
    }

    BuiltGraph buildBackwardGraph(const CudnnInstanceNormDescriptor& descriptor, int gpuNum) {
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
        auto scale = tensor(built.graph,
                            descriptor.debugName + "_scale",
                            UID_SCALE,
                            parameterDims(descriptor),
                            parameterStrides(descriptor),
                            descriptor.parameterDataType);
        auto mean = tensor(built.graph,
                           descriptor.debugName + "_mean",
                           UID_MEAN,
                           statsDims(descriptor),
                           statsStrides(descriptor),
                           TensorDescriptor::DataType::FP32);
        auto invVariance = tensor(built.graph,
                                  descriptor.debugName + "_inv_variance",
                                  UID_INV_VARIANCE,
                                  statsDims(descriptor),
                                  statsStrides(descriptor),
                                  TensorDescriptor::DataType::FP32);

        auto attrs = fe::graph::Instancenorm_backward_attributes()
                         .set_name(descriptor.debugName + "_backward")
                         .set_saved_mean_and_inv_variance(mean, invVariance)
                         .set_compute_data_type(toFrontendDataType(descriptor.computeDataType));

        auto [dx, dscale, dbias] = built.graph->instancenorm_backward(dy, x, scale, attrs);
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

InstanceNormGraphCache& cache() {
    static InstanceNormGraphCache instance;
    return instance;
}

}  // namespace

void CudnnInstanceNormDescriptor::validateForward() const {
    checkedI64(batchSize, "batchSize");
    checkedI64(channelCount, "channelCount");
    checkedI64(spatialElementCount, "spatialElementCount");
    (void)ioElementCount(*this);
    (void)statsElementCount(*this);
    if (!isSupportedInstanceNormIoDtype(inputDataType)) {
        throwInvalidInstanceNorm("inputDataType must be fp16, bf16, or fp32; got " + dtypeName(inputDataType));
    }
    if (!isSupportedInstanceNormIoDtype(outputDataType)) {
        throwInvalidInstanceNorm("outputDataType must be fp16, bf16, or fp32; got " + dtypeName(outputDataType));
    }
    if (parameterDataType != TensorDescriptor::DataType::FP32) {
        throwInvalidInstanceNorm("scale/bias parameters are currently required to be fp32; got " + dtypeName(parameterDataType));
    }
    if (computeDataType != TensorDescriptor::DataType::FP32) {
        throwInvalidInstanceNorm("computeDataType is currently required to be fp32; got " + dtypeName(computeDataType));
    }
    if (!(epsilon > 0.0f)) {
        throwInvalidInstanceNorm("epsilon must be > 0");
    }
    validateCudnnFrontendPrimaryEngineContract(*this);
}

void CudnnInstanceNormDescriptor::validateBackward() const { validateForward(); }

string CudnnInstanceNormDescriptor::cacheKey(string_view passName, int gpuNum) const {
    ostringstream out;
    out << "instancenorm:" << passName << ":gpu=" << gpuNum << ":n=" << batchSize << ":c=" << channelCount
        << ":s=" << spatialElementCount << ":in=" << static_cast<int>(inputDataType) << ":out=" << static_cast<int>(outputDataType)
        << ":param=" << static_cast<int>(parameterDataType) << ":compute=" << static_cast<int>(computeDataType)
        << ":eps=" << epsilon << ":training=" << training;
    return out.str();
}

CudnnInstanceNorm& CudnnInstanceNorm::instance() {
    static CudnnInstanceNorm singleton;
    return singleton;
}

void CudnnInstanceNorm::forward(const CudnnInstanceNormDescriptor& descriptor, const CudnnInstanceNormForwardArgs& args, Stream stream) {
    descriptor.validateForward();
    const int gpuNum = stream.getGpuNum();
    requireIoTensor(args.x, descriptor, descriptor.inputDataType, gpuNum, "x");
    requireIoTensor(args.y, descriptor, descriptor.outputDataType, gpuNum, "y");
    requireParameterTensor(args.scale, descriptor, gpuNum, "scale");
    requireParameterTensor(args.bias, descriptor, gpuNum, "bias");
    if (descriptor.training) {
        if (!args.mean.has_value() || !args.invVariance.has_value()) {
            throw invalid_argument("cuDNN InstanceNorm forward training requires mean and invVariance output tensors.");
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
        throw runtime_error("Failed to execute cuDNN Frontend InstanceNorm graph: " + status.get_message());
    }
}

void CudnnInstanceNorm::backward(const CudnnInstanceNormDescriptor& descriptor, const CudnnInstanceNormBackwardArgs& args, Stream stream) {
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
        throw runtime_error("Failed to execute cuDNN Frontend InstanceNorm graph: " + status.get_message());
    }
}

void CudnnInstanceNorm::warmForward(const CudnnInstanceNormDescriptor& descriptor, int gpuNum) { (void)cache().getOrBuildForward(descriptor, gpuNum); }

void CudnnInstanceNorm::warmBackward(const CudnnInstanceNormDescriptor& descriptor, int gpuNum) { (void)cache().getOrBuildBackward(descriptor, gpuNum); }

void CudnnInstanceNorm::clearCache() { cache().clear(); }

size_t CudnnInstanceNorm::cachedGraphCount() const { return cache().size(); }

bool CudnnInstanceNorm::frontendAvailable() { return true; }
