#include "Utilities/TensorOperations/DeepLearning/CudnnLayerNorm.h"

#include "Utilities/Common/CudnnExecutionWorkspace.h"
#include "Utilities/Common/CudnnFrontendPlan.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/ScopedGpu.h"

#include <array>
#include <cstddef>
#include <limits>
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

vector<int64_t> explicitOr(const optional<array<int64_t, 4>>& value, array<int64_t, 4> fallback) {
    const array<int64_t, 4>& chosen = value.has_value() ? value.value() : fallback;
    return vector<int64_t>(chosen.begin(), chosen.end());
}

vector<int64_t> ioDims(const CudnnLayerNormDescriptor& descriptor) {
    return explicitOr(descriptor.ioDimensions,
                      {checkedI64(descriptor.outerSize, "outerSize"), checkedI64(descriptor.normalizedFeatureCount, "normalizedFeatureCount"), 1, 1});
}

vector<int64_t> ioStrides(const CudnnLayerNormDescriptor& descriptor) {
    const int64_t hidden = checkedI64(descriptor.normalizedFeatureCount, "normalizedFeatureCount");
    return explicitOr(descriptor.ioStrides, {hidden, 1, hidden, hidden});
}

vector<int64_t> parameterDims(const CudnnLayerNormDescriptor& descriptor) {
    return explicitOr(descriptor.parameterDimensions, {1, checkedI64(descriptor.normalizedFeatureCount, "normalizedFeatureCount"), 1, 1});
}

vector<int64_t> parameterStrides(const CudnnLayerNormDescriptor& descriptor) {
    const int64_t hidden = checkedI64(descriptor.normalizedFeatureCount, "normalizedFeatureCount");
    return explicitOr(descriptor.parameterStrides, {hidden, 1, hidden, hidden});
}

vector<int64_t> statsDims(const CudnnLayerNormDescriptor& descriptor) {
    return explicitOr(descriptor.statsDimensions, {checkedI64(descriptor.outerSize, "outerSize"), 1, 1, 1});
}

vector<int64_t> statsStrides(const CudnnLayerNormDescriptor& descriptor) {
    return explicitOr(descriptor.statsStrides, {1, 1, 1, 1});
}

class LayerNormPlanRepository {
   public:
    CudnnFrontendExecutablePlan prepareForward(const CudnnLayerNormDescriptor& descriptor, Stream stream) {
        descriptor.validateForward();
        return prepare(descriptor, "forward", std::move(stream), [this, descriptor]() { return makeForwardGraph(descriptor); });
    }

    CudnnFrontendExecutablePlan prepareBackward(const CudnnLayerNormDescriptor& descriptor, Stream stream) {
        descriptor.validateBackward();
        return prepare(descriptor, "backward", std::move(stream), [this, descriptor]() { return makeBackwardGraph(descriptor); });
    }

    void clear() { selections.clear(); }
    size_t size() const { return selections.size(); }
    uint64_t hitCount() const { return selections.hitCount(); }
    uint64_t missCount() const { return selections.missCount(); }

   private:
    static constexpr size_t kSelectionCacheCapacity = 1024;

    shared_ptr<fe::graph::Tensor_attributes> tensor(shared_ptr<fe::graph::Graph>& graph,
                                                    string_view name,
                                                    int64_t uid,
                                                    const vector<int64_t>& dim,
                                                    const vector<int64_t>& stride,
                                                    DataType dtype) const {
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
                                                      const vector<int64_t>& stride) const {
        return graph->tensor(fe::graph::Tensor_attributes().set_name(string(name)).set_uid(uid).set_dim(dim).set_stride(stride));
    }

    shared_ptr<fe::graph::Graph> makeForwardGraph(const CudnnLayerNormDescriptor& descriptor) const {
        auto graph = make_shared<fe::graph::Graph>();
        graph->set_io_data_type(toFrontendDataType(descriptor.inputDataType))
            .set_intermediate_data_type(toFrontendDataType(descriptor.computeDataType))
            .set_compute_data_type(toFrontendDataType(descriptor.computeDataType));

        const vector<int64_t> dims = ioDims(descriptor);
        const vector<int64_t> strides = ioStrides(descriptor);
        auto x = ioTensor(graph, descriptor.debugName + "_x", UID_X, dims, strides);
        auto scale = tensor(graph,
                            descriptor.debugName + "_scale",
                            UID_SCALE,
                            parameterDims(descriptor),
                            parameterStrides(descriptor),
                            descriptor.parameterDataType);
        auto bias = tensor(graph,
                           descriptor.debugName + "_bias",
                           UID_BIAS,
                           parameterDims(descriptor),
                           parameterStrides(descriptor),
                           descriptor.parameterDataType);
        auto epsilon = graph->tensor(descriptor.epsilon);

        auto attrs = fe::graph::Layernorm_attributes()
                         .set_name(descriptor.debugName + "_forward")
                         .set_forward_phase(descriptor.training ? fe::NormFwdPhase_t::TRAINING : fe::NormFwdPhase_t::INFERENCE)
                         .set_epsilon(epsilon)
                         .set_compute_data_type(toFrontendDataType(descriptor.computeDataType));

        auto [y, mean, invVariance] = graph->layernorm(x, scale, bias, attrs);
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

        return graph;
    }

    shared_ptr<fe::graph::Graph> makeBackwardGraph(const CudnnLayerNormDescriptor& descriptor) const {
        auto graph = make_shared<fe::graph::Graph>();
        graph->set_io_data_type(toFrontendDataType(descriptor.inputDataType))
            .set_intermediate_data_type(toFrontendDataType(descriptor.computeDataType))
            .set_compute_data_type(toFrontendDataType(descriptor.computeDataType));

        const vector<int64_t> dims = ioDims(descriptor);
        const vector<int64_t> strides = ioStrides(descriptor);
        auto dy = ioTensor(graph, descriptor.debugName + "_dy", UID_DY, dims, strides);
        auto x = ioTensor(graph, descriptor.debugName + "_x", UID_X, dims, strides);
        auto scale = tensor(graph,
                            descriptor.debugName + "_scale",
                            UID_SCALE,
                            parameterDims(descriptor),
                            parameterStrides(descriptor),
                            descriptor.parameterDataType);
        auto mean = tensor(graph,
                           descriptor.debugName + "_mean",
                           UID_MEAN,
                           statsDims(descriptor),
                           statsStrides(descriptor),
                           DataType::FP32);
        auto invVariance = tensor(graph,
                                  descriptor.debugName + "_inv_variance",
                                  UID_INV_VARIANCE,
                                  statsDims(descriptor),
                                  statsStrides(descriptor),
                                  DataType::FP32);

        auto attrs = fe::graph::Layernorm_backward_attributes()
                         .set_name(descriptor.debugName + "_backward")
                         .set_saved_mean_and_inv_variance(mean, invVariance)
                         .set_compute_data_type(toFrontendDataType(descriptor.computeDataType));

        auto [dx, dscale, dbias] = graph->layernorm_backward(dy, x, scale, attrs);
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

        return graph;
    }

    static void checkStatus(fe::error_t status, const string& message) {
        if (!status.is_good()) {
            throw runtime_error(message + ": " + status.get_message());
        }
    }

    static CudnnFrontendPlanSelection selectPrimaryHeuristic(const CudnnFrontendGraphFactory& graphFactory,
                                                              Stream stream,
                                                              string_view passName) {
        ScopedGpu scopedGpu(stream.getGpuNum());
        shared_ptr<fe::graph::Graph> graph = graphFactory();
        if (!graph || graph.use_count() != 1) {
            throw runtime_error("cuDNN Frontend LayerNorm selection requires a pristine operation-local graph.");
        }

        const string operation = "LayerNorm " + string(passName);
        checkStatus(graph->validate(), "Failed to validate cuDNN Frontend " + operation + " graph");
        checkStatus(graph->build_operation_graph(stream.getCudnnHandle()),
                    "Failed to build cuDNN Frontend " + operation + " operation graph");
        checkStatus(graph->create_execution_plans({fe::HeurMode_t::A}),
                    "Failed to enumerate cuDNN Frontend " + operation + " primary-heuristic execution plans");
        checkStatus(graph->check_support(stream.getCudnnHandle()),
                    "Failed to check support for cuDNN Frontend " + operation + " primary-heuristic execution plans");

        const int64_t planCount = graph->get_execution_plan_count();
        string lastReplayFailure;
        for (int64_t planIndex = 0; planIndex < planCount; ++planIndex) {
            const auto status = graph->build_plan_at_index(stream.getCudnnHandle(), planIndex);
            if (!status.is_good()) {
                continue;
            }

            try {
                CudnnFrontendPlanSelection selection =
                    cudnnFrontendPlanSelectionAtIndex(*graph, planIndex, operation);
                // A plan is eligible for the global selection cache only if Thor
                // can recreate it exactly as independent operation-local state. The
                // common helper prefers engine+knob replay and transparently uses an
                // immutable serialized replay token when Frontend's knob enum is
                // lossy. Probe replay once on the cache miss and continue to the next
                // primary-heuristic plan only if recreation still fails.
                (void)replayCudnnFrontendExecutablePlan(
                    graphFactory, selection, stream.getCudnnHandle(), operation);
                return selection;
            } catch (const exception& e) {
                lastReplayFailure = e.what();
            }
        }

        string message = "cuDNN Frontend " + operation +
                         " produced no exactly replayable primary-heuristic execution plan; Thor LayerNorm does not permit fallback engines.";
        if (!lastReplayFailure.empty()) {
            message += " Last replay failure: " + lastReplayFailure;
        }
        throw runtime_error(message);
    }

    CudnnFrontendExecutablePlan prepare(const CudnnLayerNormDescriptor& descriptor,
                                        string_view passName,
                                        Stream stream,
                                        CudnnFrontendGraphFactory graphFactory) {
        ScopedGpu scopedGpu(stream.getGpuNum());
        const string key = descriptor.cacheKey(passName, stream.getGpuNum());
        const CudnnFrontendPlanSelection selection = selections.getOrSelect(key, [&]() {
            return selectPrimaryHeuristic(graphFactory, stream, passName);
        });
        return replayCudnnFrontendExecutablePlan(graphFactory, selection, stream.getCudnnHandle(), "LayerNorm " + string(passName));
    }

    CudnnFrontendPlanSelectionCache<string> selections{kSelectionCacheCapacity};
};

LayerNormPlanRepository& repository() {
    static LayerNormPlanRepository instance;
    return instance;
}

void validateExplicitPhysicalLayout(const CudnnLayerNormDescriptor& descriptor) {
    const bool any = descriptor.ioDimensions.has_value() || descriptor.ioStrides.has_value() ||
                     descriptor.parameterDimensions.has_value() || descriptor.parameterStrides.has_value() ||
                     descriptor.statsDimensions.has_value() || descriptor.statsStrides.has_value();
    const bool all = descriptor.ioDimensions.has_value() && descriptor.ioStrides.has_value() &&
                     descriptor.parameterDimensions.has_value() && descriptor.parameterStrides.has_value() &&
                     descriptor.statsDimensions.has_value() && descriptor.statsStrides.has_value();
    if (any != all) {
        throwInvalidLayerNorm("explicit physical layout requires io/parameter/stats dimensions and strides together");
    }
    if (!all) return;

    auto checkedProduct = [&](const array<int64_t, 4>& dims, string_view what) -> uint64_t {
        uint64_t product = 1;
        for (int64_t dim : dims) {
            if (dim <= 0) throwInvalidLayerNorm(string(what) + " dimensions must be positive");
            const uint64_t u = static_cast<uint64_t>(dim);
            if (product > numeric_limits<uint64_t>::max() / u) throwInvalidLayerNorm(string(what) + " element count overflows uint64_t");
            product *= u;
        }
        return product;
    };
    auto validateStrides = [&](const array<int64_t, 4>& strides, string_view what) {
        for (int64_t stride : strides) if (stride <= 0) throwInvalidLayerNorm(string(what) + " strides must be positive");
    };

    const uint64_t io_count = checkedProduct(descriptor.ioDimensions.value(), "IO");
    const uint64_t parameter_count = checkedProduct(descriptor.parameterDimensions.value(), "parameter");
    const uint64_t stats_count = checkedProduct(descriptor.statsDimensions.value(), "stats");
    validateStrides(descriptor.ioStrides.value(), "IO");
    validateStrides(descriptor.parameterStrides.value(), "parameter");
    validateStrides(descriptor.statsStrides.value(), "stats");
    if (io_count != checkedMul(descriptor.outerSize, descriptor.normalizedFeatureCount, "IO"))
        throwInvalidLayerNorm("explicit IO dimensions do not match outerSize*normalizedFeatureCount");
    if (parameter_count != descriptor.normalizedFeatureCount)
        throwInvalidLayerNorm("explicit parameter dimensions do not match normalizedFeatureCount");
    if (stats_count != descriptor.outerSize)
        throwInvalidLayerNorm("explicit stats dimensions do not match outerSize");
}

string physicalLayoutCacheSuffix(const CudnnLayerNormDescriptor& descriptor) {
    if (!descriptor.ioDimensions.has_value()) return ":layout=contiguous";
    ostringstream out;
    auto append = [&](string_view label, const array<int64_t, 4>& values) {
        out << ':' << label << '=';
        for (size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << values[i]; }
    };
    append("iod", descriptor.ioDimensions.value());
    append("ios", descriptor.ioStrides.value());
    append("pd", descriptor.parameterDimensions.value());
    append("ps", descriptor.parameterStrides.value());
    append("sd", descriptor.statsDimensions.value());
    append("ss", descriptor.statsStrides.value());
    return out.str();
}

}  // namespace

void CudnnLayerNormDescriptor::validateForward() const {
    checkedI64(outerSize, "outerSize");
    checkedI64(normalizedFeatureCount, "normalizedFeatureCount");
    (void)checkedMul(outerSize, normalizedFeatureCount, "IO");
    validateExplicitPhysicalLayout(*this);
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
    out << physicalLayoutCacheSuffix(*this);
    return out.str();
}

CudnnLayerNorm& CudnnLayerNorm::instance() {
    static CudnnLayerNorm singleton;
    return singleton;
}

CudnnLayerNormExecutablePlan CudnnLayerNorm::prepareForward(const CudnnLayerNormDescriptor& descriptor, Stream stream) {
    descriptor.validateForward();
    const int gpuNum = stream.getGpuNum();
    CudnnFrontendExecutablePlan executable = repository().prepareForward(descriptor, stream);
    return CudnnLayerNormExecutablePlan(descriptor, CudnnLayerNormExecutablePlan::Pass::Forward, gpuNum, std::move(executable));
}

CudnnLayerNormExecutablePlan CudnnLayerNorm::prepareBackward(const CudnnLayerNormDescriptor& descriptor, Stream stream) {
    descriptor.validateBackward();
    const int gpuNum = stream.getGpuNum();
    CudnnFrontendExecutablePlan executable = repository().prepareBackward(descriptor, stream);
    return CudnnLayerNormExecutablePlan(descriptor, CudnnLayerNormExecutablePlan::Pass::Backward, gpuNum, std::move(executable));
}

void CudnnLayerNorm::forward(const CudnnLayerNormExecutablePlan& plan,
                             const CudnnLayerNormForwardArgs& args,
                             optional<Tensor>& workspace,
                             Stream stream) {
    if (!plan.isForward()) {
        throw invalid_argument("cuDNN LayerNorm forward requires a forward executable plan.");
    }
    const CudnnLayerNormDescriptor& descriptor = plan.descriptor();
    descriptor.validateForward();
    const int gpuNum = stream.getGpuNum();
    if (gpuNum != plan.gpuNum()) {
        throw invalid_argument("cuDNN LayerNorm forward executable plan cannot move between GPUs.");
    }
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
    unordered_map<int64_t, void*> variantPack;
    insertTensor(variantPack, UID_X, args.x);
    insertTensor(variantPack, UID_SCALE, args.scale);
    insertTensor(variantPack, UID_BIAS, args.bias);
    insertTensor(variantPack, UID_Y, args.y);
    if (descriptor.training) {
        insertTensor(variantPack, UID_MEAN, args.mean.value());
        insertTensor(variantPack, UID_INV_VARIANCE, args.invVariance.value());
    }

    const uint64_t requiredWorkspaceBytes = checkedCudnnWorkspaceSizeInBytes(plan.workspaceBytes(), "LayerNorm forward");
    void* workspacePtr = cudnnExecutionWorkspacePointer(workspace, requiredWorkspaceBytes, gpuNum, "LayerNorm forward");
    plan.executable_.execute(stream.getCudnnHandle(), variantPack, workspacePtr);
}

void CudnnLayerNorm::backward(const CudnnLayerNormExecutablePlan& plan,
                              const CudnnLayerNormBackwardArgs& args,
                              optional<Tensor>& workspace,
                              Stream stream) {
    if (!plan.isBackward()) {
        throw invalid_argument("cuDNN LayerNorm backward requires a backward executable plan.");
    }
    const CudnnLayerNormDescriptor& descriptor = plan.descriptor();
    descriptor.validateBackward();
    const int gpuNum = stream.getGpuNum();
    if (gpuNum != plan.gpuNum()) {
        throw invalid_argument("cuDNN LayerNorm backward executable plan cannot move between GPUs.");
    }
    requireIoTensor(args.dy, descriptor, descriptor.outputDataType, gpuNum, "dy");
    requireIoTensor(args.x, descriptor, descriptor.inputDataType, gpuNum, "x");
    requireIoTensor(args.dx, descriptor, descriptor.inputDataType, gpuNum, "dx");
    requireParameterTensor(args.scale, descriptor, gpuNum, "scale");
    requireStatsTensor(args.mean, descriptor, gpuNum, "mean");
    requireStatsTensor(args.invVariance, descriptor, gpuNum, "invVariance");
    requireParameterTensor(args.dscale, descriptor, gpuNum, "dscale");
    requireParameterTensor(args.dbias, descriptor, gpuNum, "dbias");

    ScopedGpu scopedGpu(gpuNum);
    unordered_map<int64_t, void*> variantPack;
    insertTensor(variantPack, UID_DY, args.dy);
    insertTensor(variantPack, UID_X, args.x);
    insertTensor(variantPack, UID_SCALE, args.scale);
    insertTensor(variantPack, UID_MEAN, args.mean);
    insertTensor(variantPack, UID_INV_VARIANCE, args.invVariance);
    insertTensor(variantPack, UID_DX, args.dx);
    insertTensor(variantPack, UID_DSCALE, args.dscale);
    insertTensor(variantPack, UID_DBIAS, args.dbias);

    const uint64_t requiredWorkspaceBytes = checkedCudnnWorkspaceSizeInBytes(plan.workspaceBytes(), "LayerNorm backward");
    void* workspacePtr = cudnnExecutionWorkspacePointer(workspace, requiredWorkspaceBytes, gpuNum, "LayerNorm backward");
    plan.executable_.execute(stream.getCudnnHandle(), variantPack, workspacePtr);
}

void CudnnLayerNorm::clearSelectionCache() { repository().clear(); }

size_t CudnnLayerNorm::cachedSelectionCount() const { return repository().size(); }

uint64_t CudnnLayerNorm::selectionCacheHitCount() const { return repository().hitCount(); }

uint64_t CudnnLayerNorm::selectionCacheMissCount() const { return repository().missCount(); }

bool CudnnLayerNorm::frontendAvailable() { return true; }
