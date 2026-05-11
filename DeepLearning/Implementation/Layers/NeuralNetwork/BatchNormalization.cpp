#include "DeepLearning/Implementation/Layers/NeuralNetwork/BatchNormalization.h"
#include <optional>
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/DeepLearning/BatchNormFrontendHelpers.h"

#include <cudnn_frontend.h>

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DeepLearning/Implementation/ThorError.h"
using namespace std;

namespace ThorImplementation {

namespace fe = cudnn_frontend;

struct BatchNormalizationFrontendGraph {
    std::shared_ptr<fe::graph::Graph> graph;
    size_t workspace_bytes = 0;
    std::optional<Tensor> workspace = std::nullopt;
    double momentum = std::numeric_limits<double>::quiet_NaN();
};

namespace {
using DataType = TensorDescriptor::DataType;

constexpr int64_t BN_X_UID = 8'100'001;
constexpr int64_t BN_Y_UID = 8'100'002;
constexpr int64_t BN_SCALE_UID = 8'100'003;
constexpr int64_t BN_BIAS_UID = 8'100'004;
constexpr int64_t BN_RUNNING_MEAN_UID = 8'100'005;
constexpr int64_t BN_RUNNING_VARIANCE_UID = 8'100'006;
constexpr int64_t BN_SAVED_MEAN_UID = 8'100'007;
constexpr int64_t BN_SAVED_INV_VARIANCE_UID = 8'100'008;
constexpr int64_t BN_NEXT_RUNNING_MEAN_UID = 8'100'009;
constexpr int64_t BN_NEXT_RUNNING_VARIANCE_UID = 8'100'010;
constexpr int64_t BN_DY_UID = 8'100'011;
constexpr int64_t BN_DX_UID = 8'100'012;
constexpr int64_t BN_DSCALE_UID = 8'100'013;
constexpr int64_t BN_DBIAS_UID = 8'100'014;
constexpr int64_t BN_RUNNING_INV_VARIANCE_UID = 8'100'015;

class BNParameter final : public PhysicalParameter {
   public:
    BNParameter(const string& name, const std::optional<TensorDescriptor::DataType>& storageDataType, bool trainable)
        : PhysicalParameter(name, trainable), storageDataType(storageDataType) {}

    void createStorage(const StorageContext& context) override {
        const Tensor& inputTensor = context.getFeatureInput();
        THOR_THROW_IF_FALSE(inputTensor.getDimensions().size() == 2 || inputTensor.getDimensions().size() == 4);
        const uint64_t channels = inputTensor.getDimensions()[1];
        TensorDescriptor::DataType resolvedDataType;
        if (storageDataType.has_value())
            resolvedDataType = storageDataType.value();
        else
            resolvedDataType = inputTensor.getDataType();

        storage = Tensor(inputTensor.getPlacement(), TensorDescriptor(resolvedDataType, {channels}));
    }

   private:
    const std::optional<TensorDescriptor::DataType> storageDataType;
};

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
        default:
            throw std::runtime_error("Unsupported dtype for cuDNN Frontend batch normalization: " +
                                     TensorDescriptor::getElementTypeName(dtype));
    }
}

static bool isFrontendTrainingBatchNormDataType(DataType dtype) {
    return dtype == DataType::FP16 || dtype == DataType::BF16 || dtype == DataType::FP32;
}

static bool isFrontendInferenceBatchNormDataType(DataType dtype) {
    return isFrontendTrainingBatchNormDataType(dtype) || dtype == DataType::FP8_E4M3;
}

static std::vector<int64_t> frontendBatchNormInputDims(const Tensor& tensor) {
    const std::vector<uint64_t> dims = tensor.getDimensions();
    if (dims.size() == 2) {
        return {static_cast<int64_t>(dims[0]), static_cast<int64_t>(dims[1]), 1, 1};
    }
    if (dims.size() == 4) {
        return {static_cast<int64_t>(dims[0]), static_cast<int64_t>(dims[1]), static_cast<int64_t>(dims[2]), static_cast<int64_t>(dims[3])};
    }
    throw std::runtime_error("BatchNormalization cuDNN Frontend path requires rank-2 [N,C] or rank-4 [N,C,H,W] input tensors.");
}

static std::vector<int64_t> frontendPackedNchwStrides(const std::vector<int64_t>& dims) {
    THOR_THROW_IF_FALSE(dims.size() == 4);
    return {dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1};
}

static std::vector<int64_t> frontendBatchNormParamDims(uint64_t channels) { return {1, static_cast<int64_t>(channels), 1, 1}; }

static std::vector<int64_t> frontendBatchNormParamStrides(uint64_t channels) {
    return {static_cast<int64_t>(channels), 1, static_cast<int64_t>(channels), static_cast<int64_t>(channels)};
}

static std::shared_ptr<fe::graph::Tensor_attributes> frontendTensor(const std::shared_ptr<fe::graph::Graph>& graph,
                                                                    const std::string& name,
                                                                    int64_t uid,
                                                                    const std::vector<int64_t>& dims,
                                                                    const std::vector<int64_t>& strides,
                                                                    DataType dtype) {
    return graph->tensor(fe::graph::Tensor_attributes().set_name(name).set_uid(uid).set_dim(dims).set_stride(strides).set_data_type(
        toFrontendDataType(dtype)));
}

static void setFrontendOutput(std::shared_ptr<fe::graph::Tensor_attributes>& tensor,
                              const std::string& name,
                              int64_t uid,
                              const std::vector<int64_t>& dims,
                              const std::vector<int64_t>& strides,
                              DataType dtype) {
    tensor->set_output(true).set_name(name).set_uid(uid).set_dim(dims).set_stride(strides).set_data_type(toFrontendDataType(dtype));
}

static void buildFrontendBatchNormGraph(BatchNormalizationFrontendGraph& built,
                                        const TensorPlacement& placement,
                                        const Stream& stream,
                                        const char* op_name) {
    if (!built.graph) {
        throw std::runtime_error(std::string(op_name) + " missing cuDNN Frontend graph.");
    }

    ScopedGpu scopedGpu(stream.getGpuNum());
    auto status = built.graph->build(stream.getCudnnHandle(), {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK});
    if (!status.is_good()) {
        throw std::runtime_error(std::string("Failed to build cuDNN Frontend ") + op_name + " graph: " + status.get_message());
    }

    int64_t workspace_bytes = 0;
    status = built.graph->get_workspace_size(workspace_bytes);
    if (!status.is_good()) {
        throw std::runtime_error(std::string("Failed to query cuDNN Frontend ") + op_name + " workspace size: " + status.get_message());
    }
    if (workspace_bytes < 0) {
        throw std::runtime_error(std::string("cuDNN Frontend ") + op_name + " returned a negative workspace size.");
    }
    built.workspace_bytes = static_cast<size_t>(workspace_bytes);
    if (built.workspace_bytes > 0) {
        built.workspace = Tensor(placement, TensorDescriptor(DataType::UINT8, {static_cast<uint64_t>(built.workspace_bytes)}));
    } else {
        built.workspace.reset();
    }
}

static void executeFrontendBatchNormGraph(const BatchNormalizationFrontendGraph& built,
                                          const Stream& stream,
                                          std::unordered_map<int64_t, void*>& tensor_pack,
                                          const char* op_name) {
    if (!built.graph) {
        throw std::runtime_error(std::string(op_name) + " missing cuDNN Frontend graph.");
    }

    void* workspace_ptr = nullptr;
    if (built.workspace_bytes > 0) {
        if (!built.workspace.has_value()) {
            throw std::runtime_error(std::string(op_name) + " requires cuDNN Frontend workspace, but none was allocated.");
        }
        workspace_ptr = (void*)built.workspace.value().getMemPtr<void>();
    }

    auto status = built.graph->execute(stream.getCudnnHandle(), tensor_pack, workspace_ptr);
    if (!status.is_good()) {
        throw std::runtime_error(std::string("Failed to execute cuDNN Frontend ") + op_name + " graph: " + status.get_message());
    }
}

static void putFrontendTensorPointer(std::unordered_map<int64_t, void*>& tensor_pack, int64_t uid, const Tensor& tensor) {
    tensor_pack[uid] = const_cast<void*>(static_cast<const void*>(tensor.getMemPtr<void>()));
}

static std::shared_ptr<BatchNormalizationFrontendGraph> buildTrainingBatchNormGraph(const Tensor& input,
                                                                                    const Tensor& output,
                                                                                    const Tensor& weights,
                                                                                    const Tensor& biases,
                                                                                    const Tensor& runningMean,
                                                                                    const Tensor& runningVariance,
                                                                                    const Tensor& savedMean,
                                                                                    const Tensor& savedInvVariance,
                                                                                    const Tensor& nextMean,
                                                                                    const Tensor& nextVariance,
                                                                                    double epsilon,
                                                                                    double momentum,
                                                                                    const Stream& stream) {
    const DataType io_dtype = input.getDataType();
    if (!isFrontendTrainingBatchNormDataType(io_dtype)) {
        throw std::runtime_error(
            "cuDNN Frontend batch-normalization training currently supports FP32, FP16, and BF16 inputs. "
            "FP8 batch normalization is only exposed for inference in this Thor layer until the backward path is supported.");
    }

    auto built = std::make_shared<BatchNormalizationFrontendGraph>();
    built->momentum = momentum;
    built->graph = std::make_shared<fe::graph::Graph>();
    built->graph->set_io_data_type(toFrontendDataType(io_dtype))
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    const auto input_dims = frontendBatchNormInputDims(input);
    const auto input_strides = frontendPackedNchwStrides(input_dims);
    const auto param_dims = frontendBatchNormParamDims(input.getDimensions()[1]);
    const auto param_strides = frontendBatchNormParamStrides(input.getDimensions()[1]);

    auto x = frontendTensor(built->graph, "batchnorm_x", BN_X_UID, input_dims, input_strides, io_dtype);
    auto scale = frontendTensor(built->graph, "batchnorm_scale", BN_SCALE_UID, param_dims, param_strides, DataType::FP32);
    auto bias = frontendTensor(built->graph, "batchnorm_bias", BN_BIAS_UID, param_dims, param_strides, DataType::FP32);
    auto running_mean =
        frontendTensor(built->graph, "batchnorm_running_mean", BN_RUNNING_MEAN_UID, param_dims, param_strides, DataType::FP32);
    auto running_variance =
        frontendTensor(built->graph, "batchnorm_running_variance", BN_RUNNING_VARIANCE_UID, param_dims, param_strides, DataType::FP32);
    auto epsilon_tensor = built->graph->tensor(static_cast<float>(epsilon));
    auto momentum_tensor = built->graph->tensor(static_cast<float>(momentum));

    auto attrs = fe::graph::Batchnorm_attributes()
                     .set_name("thor_batchnorm_training")
                     .set_epsilon(epsilon_tensor)
                     .set_compute_data_type(fe::DataType_t::FLOAT)
                     .set_previous_running_stats(running_mean, running_variance, momentum_tensor);

    auto [y, mean, inv_variance, next_running_mean, next_running_variance] = built->graph->batchnorm(x, scale, bias, attrs);
    setFrontendOutput(y,
                      "batchnorm_y",
                      BN_Y_UID,
                      output.getDimensions().size() == 2 ? input_dims : frontendBatchNormInputDims(output),
                      input_strides,
                      io_dtype);
    setFrontendOutput(mean, "batchnorm_saved_mean", BN_SAVED_MEAN_UID, param_dims, param_strides, DataType::FP32);
    setFrontendOutput(inv_variance, "batchnorm_saved_inv_variance", BN_SAVED_INV_VARIANCE_UID, param_dims, param_strides, DataType::FP32);
    setFrontendOutput(
        next_running_mean, "batchnorm_next_running_mean", BN_NEXT_RUNNING_MEAN_UID, param_dims, param_strides, DataType::FP32);
    setFrontendOutput(
        next_running_variance, "batchnorm_next_running_variance", BN_NEXT_RUNNING_VARIANCE_UID, param_dims, param_strides, DataType::FP32);

    (void)weights;
    (void)biases;
    (void)runningMean;
    (void)runningVariance;
    (void)savedMean;
    (void)savedInvVariance;
    (void)nextMean;
    (void)nextVariance;
    buildFrontendBatchNormGraph(*built, input.getPlacement(), stream, "batch-normalization training");
    return built;
}

static std::shared_ptr<BatchNormalizationFrontendGraph> buildInferenceBatchNormGraph(const Tensor& input,
                                                                                     const Tensor& output,
                                                                                     const Tensor& weights,
                                                                                     const Tensor& biases,
                                                                                     const Tensor& runningMean,
                                                                                     const Tensor& runningInvVariance,
                                                                                     const Stream& stream) {
    const DataType io_dtype = input.getDataType();
    if (!isFrontendInferenceBatchNormDataType(io_dtype)) {
        throw std::runtime_error(
            "cuDNN Frontend batch-normalization inference supports FP32, FP16, BF16, and FP8_E4M3 inputs in this Thor layer.");
    }

    auto built = std::make_shared<BatchNormalizationFrontendGraph>();
    built->graph = std::make_shared<fe::graph::Graph>();
    built->graph->set_io_data_type(toFrontendDataType(io_dtype))
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    const auto input_dims = frontendBatchNormInputDims(input);
    const auto input_strides = frontendPackedNchwStrides(input_dims);
    const auto param_dims = frontendBatchNormParamDims(input.getDimensions()[1]);
    const auto param_strides = frontendBatchNormParamStrides(input.getDimensions()[1]);

    auto x = frontendTensor(built->graph, "batchnorm_inference_x", BN_X_UID, input_dims, input_strides, io_dtype);
    auto mean = frontendTensor(built->graph, "batchnorm_inference_mean", BN_RUNNING_MEAN_UID, param_dims, param_strides, DataType::FP32);
    auto inv_variance = frontendTensor(
        built->graph, "batchnorm_inference_inv_variance", BN_RUNNING_INV_VARIANCE_UID, param_dims, param_strides, DataType::FP32);
    auto scale = frontendTensor(built->graph, "batchnorm_inference_scale", BN_SCALE_UID, param_dims, param_strides, DataType::FP32);
    auto bias = frontendTensor(built->graph, "batchnorm_inference_bias", BN_BIAS_UID, param_dims, param_strides, DataType::FP32);

    auto y = built->graph->batchnorm_inference(x, mean, inv_variance, scale, bias, fe::graph::Batchnorm_inference_attributes());
    setFrontendOutput(y,
                      "batchnorm_inference_y",
                      BN_Y_UID,
                      output.getDimensions().size() == 2 ? input_dims : frontendBatchNormInputDims(output),
                      input_strides,
                      io_dtype);

    (void)weights;
    (void)biases;
    (void)runningMean;
    (void)runningInvVariance;
    buildFrontendBatchNormGraph(*built, input.getPlacement(), stream, "batch-normalization inference");
    return built;
}

static std::shared_ptr<BatchNormalizationFrontendGraph> buildBackwardBatchNormGraph(const Tensor& input,
                                                                                    const Tensor& gradOutput,
                                                                                    const Tensor& gradInput,
                                                                                    const Tensor& weights,
                                                                                    const Tensor& savedMean,
                                                                                    const Tensor& savedInvVariance,
                                                                                    const Tensor& dscale,
                                                                                    const Tensor& dbias,
                                                                                    const Stream& stream) {
    const DataType io_dtype = input.getDataType();
    if (!isFrontendTrainingBatchNormDataType(io_dtype)) {
        throw std::runtime_error("cuDNN Frontend batch-normalization backward supports FP32, FP16, and BF16 inputs in this Thor layer.");
    }

    auto built = std::make_shared<BatchNormalizationFrontendGraph>();
    built->graph = std::make_shared<fe::graph::Graph>();
    built->graph->set_io_data_type(toFrontendDataType(io_dtype))
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    const auto input_dims = frontendBatchNormInputDims(input);
    const auto input_strides = frontendPackedNchwStrides(input_dims);
    const auto param_dims = frontendBatchNormParamDims(input.getDimensions()[1]);
    const auto param_strides = frontendBatchNormParamStrides(input.getDimensions()[1]);

    auto dy = frontendTensor(built->graph, "batchnorm_backward_dy", BN_DY_UID, input_dims, input_strides, io_dtype);
    auto x = frontendTensor(built->graph, "batchnorm_backward_x", BN_X_UID, input_dims, input_strides, io_dtype);
    auto scale = frontendTensor(built->graph, "batchnorm_backward_scale", BN_SCALE_UID, param_dims, param_strides, DataType::FP32);
    auto mean = frontendTensor(built->graph, "batchnorm_backward_saved_mean", BN_SAVED_MEAN_UID, param_dims, param_strides, DataType::FP32);
    auto inv_variance = frontendTensor(
        built->graph, "batchnorm_backward_saved_inv_variance", BN_SAVED_INV_VARIANCE_UID, param_dims, param_strides, DataType::FP32);

    auto attrs = fe::graph::Batchnorm_backward_attributes()
                     .set_name("thor_batchnorm_backward")
                     .set_saved_mean_and_inv_variance(mean, inv_variance)
                     .set_compute_data_type(fe::DataType_t::FLOAT);
    auto [dx, scale_grad, bias_grad] = built->graph->batchnorm_backward(dy, x, scale, attrs);
    setFrontendOutput(dx, "batchnorm_backward_dx", BN_DX_UID, input_dims, input_strides, io_dtype);
    setFrontendOutput(scale_grad, "batchnorm_backward_dscale", BN_DSCALE_UID, param_dims, param_strides, DataType::FP32);
    setFrontendOutput(bias_grad, "batchnorm_backward_dbias", BN_DBIAS_UID, param_dims, param_strides, DataType::FP32);

    (void)gradOutput;
    (void)gradInput;
    (void)weights;
    (void)savedMean;
    (void)savedInvVariance;
    (void)dscale;
    (void)dbias;
    buildFrontendBatchNormGraph(*built, input.getPlacement(), stream, "batch-normalization backward");
    return built;
}

}  // namespace

const float BatchNormalization::ALPHA_NO_SCALE = 1.0f;
const float BatchNormalization::BETA_CLEAR = 0.0f;
const float BatchNormalization::BETA_ACCUMULATE = 1.0f;

BatchNormalization::BatchNormalization(const TensorPlacement& placement,
                                       bool inferenceOnly,
                                       uint64_t numItemsObserved,
                                       std::optional<double> exponentialRunningAverageFactor,
                                       std::optional<double> epsilon,
                                       std::optional<TensorDescriptor::DataType> storageDataType,
                                       int64_t stampedId)
    : TrainableLayer(placement, inferenceOnly, stampedId),
      exponentialRunningAverageFactor(exponentialRunningAverageFactor.has_value() ? exponentialRunningAverageFactor.value() : 0.05),
      epsilon(epsilon.has_value() ? epsilon.value() : 0.0001) {
    addParameter(make_shared<BNParameter>("weights", DataType::FP32, true));
    addParameter(make_shared<BNParameter>("biases", DataType::FP32, true));
    addParameter(make_shared<BNParameter>("running_mean", DataType::FP32, false));
    addParameter(make_shared<BNParameter>("running_variance", DataType::FP32, false));

    itemsObserved = numItemsObserved;
}

BatchNormalization::~BatchNormalization() { cleanup(); }

std::optional<Tensor> BatchNormalization::createFeatureOutputTensor() {
    std::optional<Tensor> maybeInput = getFirstPresentTensor(featureInputs);
    THOR_THROW_IF_FALSE(maybeInput.has_value());
    return maybeInput.value().clone();
}

std::optional<Tensor> BatchNormalization::createErrorOutputTensor(bool backPropagateError, uint32_t connectionNumber) {
    if (backPropagateError && !isInferenceOnly()) {
        THOR_THROW_IF_FALSE(featureInputs.size() > connectionNumber);
        THOR_THROW_IF_FALSE(featureInputs[connectionNumber].has_value());
        return featureInputs[connectionNumber].value().clone();
    }
    return std::nullopt;
}

uint64_t BatchNormalization::flopCountForward() {
    std::optional<Tensor> maybeInput = getFirstPresentTensor(featureInputs);
    if (!maybeInput.has_value())
        return 0;
    return maybeInput.value().getTotalNumElements() * 8;
}

uint64_t BatchNormalization::flopCountBackward() {
    std::optional<Tensor> maybeInput = getFirstPresentTensor(featureInputs);
    if (!maybeInput.has_value())
        return 0;
    return maybeInput.value().getTotalNumElements() * 16;
}

void BatchNormalization::compileImpl() {
    TrainableLayer::compileImpl();

    THOR_THROW_IF_FALSE(!featureInputs.empty());
    THOR_THROW_IF_FALSE(!featureOutputs.empty());
    THOR_THROW_IF_FALSE(featureInputs.size() == featureOutputs.size());

    std::optional<Tensor> maybeInput = getFirstPresentTensor(featureInputs);
    THOR_THROW_IF_FALSE(maybeInput.has_value());
    const Tensor& input = maybeInput.value();

    placement = input.getPlacement();
    THOR_THROW_IF_FALSE(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
    ensureNoDeviceCrossing(placement);

    attachGradientUpdateStream();

    for (const auto& parameter : parameters) {
        if (!parameter->isStorageInitialized()) {
            parameter->compileStorage(input);
            parameter->compileInitializer(getFanIn(), getFanOut());
        }
        if (parameter->isTrainable()) {
            parameter->compileOptimizer(gradientUpdateStream, isInferenceOnly());
        }
    }

    weights = getParameter("weights")->getStorage().value();
    biases = getParameter("biases")->getStorage().value();
    resultRunningMean = getParameter("running_mean")->getStorage().value();
    resultRunningVariance = getParameter("running_variance")->getStorage().value();
    THOR_THROW_IF_FALSE(weights.getDataType() == DataType::FP32);
    THOR_THROW_IF_FALSE(biases.getDataType() == DataType::FP32);
    THOR_THROW_IF_FALSE(resultRunningMean.getDataType() == DataType::FP32);
    THOR_THROW_IF_FALSE(resultRunningVariance.getDataType() == DataType::FP32);
    THOR_THROW_IF_FALSE(weights.getDimensions() == biases.getDimensions());
    THOR_THROW_IF_FALSE(weights.getDimensions() == resultRunningMean.getDimensions());
    THOR_THROW_IF_FALSE(weights.getDimensions() == resultRunningVariance.getDimensions());

    const vector<uint64_t> inputDimensions = input.getDescriptor().getDimensions();
    THOR_THROW_IF_FALSE(inputDimensions.size() == 2 || inputDimensions.size() == 4);
    batchSize = inputDimensions[0];
    numChannels = inputDimensions[1];
    if (inputDimensions.size() == 2) {
        height = 1;
        width = 1;
    } else {
        height = inputDimensions[2];
        width = inputDimensions[3];
    }

    if (isInferenceOnly()) {
        if (!isFrontendInferenceBatchNormDataType(input.getDataType())) {
            throw std::runtime_error("BatchNormalization inference supports FP32, FP16, BF16, and FP8_E4M3 inputs through cuDNN Frontend.");
        }
    } else {
        if (!isFrontendTrainingBatchNormDataType(input.getDataType())) {
            throw std::runtime_error("BatchNormalization training/backward supports FP32, FP16, and BF16 inputs through cuDNN Frontend.");
        }
    }

    frontendTrainingGraphs.clear();
    frontendInferenceGraph.reset();
    frontendBackwardGraphs.clear();
    resultSaveMean.clear();
    resultSaveInvVariance.clear();
    nextRunningMean.clear();
    nextRunningVariance.clear();
    runningInvVariance.reset();
    weightsGradientScratch.clear();
    biasesGradientScratch.clear();
    scratchErrorOutput.reset();

    frontendTrainingGraphs.resize(featureInputs.size());
    frontendBackwardGraphs.resize(featureInputs.size());
    resultSaveMean.reserve(featureInputs.size());
    resultSaveInvVariance.reserve(featureInputs.size());
    nextRunningMean.reserve(featureInputs.size());
    nextRunningVariance.reserve(featureInputs.size());
    weightsGradientScratch.reserve(featureInputs.size());
    biasesGradientScratch.reserve(featureInputs.size());
    for (unsigned int i = 0; i < featureInputs.size(); ++i) {
        resultSaveMean.push_back(weights.clone());
        resultSaveInvVariance.push_back(weights.clone());
        nextRunningMean.push_back(resultRunningMean.clone());
        nextRunningVariance.push_back(resultRunningVariance.clone());
        weightsGradientScratch.push_back(weights.clone());
        biasesGradientScratch.push_back(biases.clone());
        if (errorInputs.size() > i && errorInputs[i].has_value() && (errorOutputs.size() <= i || !errorOutputs[i].has_value())) {
            THOR_THROW_IF_FALSE(featureInputs[i].has_value());
            // We may need a single, right sized, chunk of scratch memory for back prop pruned paths.
            if (!scratchErrorOutput.has_value())
                scratchErrorOutput = featureInputs[i].value().clone();
        }
    }
    runningInvVariance = resultRunningVariance.clone();

    THOR_THROW_IF_FALSE(exponentialRunningAverageFactor > 0.0);
    THOR_THROW_IF_FALSE(exponentialRunningAverageFactor <= 1.0);
    itemsObserved = 0;
    currentExponentialRunningAverageFactor.assign(featureInputs.size(), 0.0);
}

void BatchNormalization::cleanup() {
    frontendTrainingGraphs.clear();
    frontendInferenceGraph.reset();
    frontendBackwardGraphs.clear();
    resultSaveMean.clear();
    resultSaveInvVariance.clear();
    nextRunningMean.clear();
    nextRunningVariance.clear();
    runningInvVariance.reset();
    weightsGradientScratch.clear();
    biasesGradientScratch.clear();
    scratchErrorOutput.reset();

    Layer::cleanup();
}

void BatchNormalization::runForward(std::optional<Tensor> inputTensor,
                                    std::optional<Tensor> outputTensor,
                                    Stream stream,
                                    unsigned int connectionNumber,
                                    Tensor weights,
                                    std::optional<Tensor> biases) {}

void BatchNormalization::computeFeatureOut(uint32_t connectionNumber) {
    std::optional<Tensor> inputTensor = featureInputs[connectionNumber];
    std::optional<Tensor> outputTensor = featureOutputs[connectionNumber];
    Stream stream = streams[connectionNumber];
    THOR_THROW_IF_FALSE(inputTensor.has_value());
    THOR_THROW_IF_FALSE(outputTensor.has_value());

    if (itemsObserved != UINT64_MAX)
        itemsObserved += 1;
    if (currentExponentialRunningAverageFactor[connectionNumber] != exponentialRunningAverageFactor) {
        currentExponentialRunningAverageFactor[connectionNumber] = 1.0 / itemsObserved;
        if (currentExponentialRunningAverageFactor[connectionNumber] < exponentialRunningAverageFactor)
            currentExponentialRunningAverageFactor[connectionNumber] = exponentialRunningAverageFactor;
    }

    if (!isInferenceOnly()) {
        const double momentum = currentExponentialRunningAverageFactor[connectionNumber];
        if (!frontendTrainingGraphs[connectionNumber] || std::fabs(frontendTrainingGraphs[connectionNumber]->momentum - momentum) > 1e-12) {
            frontendTrainingGraphs[connectionNumber] = buildTrainingBatchNormGraph(inputTensor.value(),
                                                                                   outputTensor.value(),
                                                                                   weights,
                                                                                   biases,
                                                                                   resultRunningMean,
                                                                                   resultRunningVariance,
                                                                                   resultSaveMean[connectionNumber],
                                                                                   resultSaveInvVariance[connectionNumber],
                                                                                   nextRunningMean[connectionNumber],
                                                                                   nextRunningVariance[connectionNumber],
                                                                                   epsilon,
                                                                                   momentum,
                                                                                   stream);
        }

        std::unordered_map<int64_t, void*> tensor_pack;
        putFrontendTensorPointer(tensor_pack, BN_X_UID, inputTensor.value());
        putFrontendTensorPointer(tensor_pack, BN_Y_UID, outputTensor.value());
        putFrontendTensorPointer(tensor_pack, BN_SCALE_UID, weights);
        putFrontendTensorPointer(tensor_pack, BN_BIAS_UID, biases);
        putFrontendTensorPointer(tensor_pack, BN_RUNNING_MEAN_UID, resultRunningMean);
        putFrontendTensorPointer(tensor_pack, BN_RUNNING_VARIANCE_UID, resultRunningVariance);
        putFrontendTensorPointer(tensor_pack, BN_SAVED_MEAN_UID, resultSaveMean[connectionNumber]);
        putFrontendTensorPointer(tensor_pack, BN_SAVED_INV_VARIANCE_UID, resultSaveInvVariance[connectionNumber]);
        putFrontendTensorPointer(tensor_pack, BN_NEXT_RUNNING_MEAN_UID, nextRunningMean[connectionNumber]);
        putFrontendTensorPointer(tensor_pack, BN_NEXT_RUNNING_VARIANCE_UID, nextRunningVariance[connectionNumber]);
        executeFrontendBatchNormGraph(*frontendTrainingGraphs[connectionNumber], stream, tensor_pack, "batch-normalization training");

        resultRunningMean.copyFromAsync(nextRunningMean[connectionNumber], stream);
        resultRunningVariance.copyFromAsync(nextRunningVariance[connectionNumber], stream);
    } else {
        THOR_THROW_IF_FALSE(runningInvVariance.has_value());
        launchComputeBatchNormInvVarianceFp32(resultRunningVariance.getMemPtr<float>(),
                                              runningInvVariance.value().getMemPtr<float>(),
                                              static_cast<float>(epsilon),
                                              numChannels,
                                              stream);

        if (!frontendInferenceGraph) {
            frontendInferenceGraph = buildInferenceBatchNormGraph(
                inputTensor.value(), outputTensor.value(), weights, biases, resultRunningMean, runningInvVariance.value(), stream);
        }

        std::unordered_map<int64_t, void*> tensor_pack;
        putFrontendTensorPointer(tensor_pack, BN_X_UID, inputTensor.value());
        putFrontendTensorPointer(tensor_pack, BN_Y_UID, outputTensor.value());
        putFrontendTensorPointer(tensor_pack, BN_SCALE_UID, weights);
        putFrontendTensorPointer(tensor_pack, BN_BIAS_UID, biases);
        putFrontendTensorPointer(tensor_pack, BN_RUNNING_MEAN_UID, resultRunningMean);
        putFrontendTensorPointer(tensor_pack, BN_RUNNING_INV_VARIANCE_UID, runningInvVariance.value());
        executeFrontendBatchNormGraph(*frontendInferenceGraph, stream, tensor_pack, "batch-normalization inference");
    }
}

// Error in is up-to-date by the end of the gradient stream.
// Gradient accumulation must be performed on the gradient stream, for serialization.
std::optional<Event> BatchNormalization::computeErrorOutAccumulateWeightsGradienFused(uint32_t connectionNumber,
                                                                                      bool clearWeightsGradientFirstIfFused) {
    if (!errorInputs[connectionNumber].has_value())
        return std::nullopt;
    if (isInferenceOnly())
        return std::nullopt;

    auto weightsParameter = getParameter("weights");
    auto biasesParameter = getParameter("biases");
    THOR_THROW_IF_FALSE(weightsParameter->hasOptimizer());
    THOR_THROW_IF_FALSE(biasesParameter->hasOptimizer());

    shared_ptr<Optimizer> weightsOptimizer = weightsParameter->getOptimizer();
    shared_ptr<Optimizer> biasesOptimizer = biasesParameter->getOptimizer();
    THOR_THROW_IF_FALSE(weightsOptimizer != nullptr);
    THOR_THROW_IF_FALSE(biasesOptimizer != nullptr);
    THOR_THROW_IF_FALSE(weightsOptimizer->getWeightsGradient().has_value());
    THOR_THROW_IF_FALSE(biasesOptimizer->getWeightsGradient().has_value());

    std::optional<Tensor> errorOut = std::nullopt;
    if (errorOutputs.size() > connectionNumber && errorOutputs[connectionNumber].has_value()) {
        errorOut = errorOutputs[connectionNumber];
    } else {
        errorOut = scratchErrorOutput;
    }
    THOR_THROW_IF_FALSE(errorOut.has_value());
    THOR_THROW_IF_FALSE(gradientUpdateStream.has_value());
    THOR_THROW_IF_FALSE(featureInputs[connectionNumber].has_value());

    Tensor dscaleOutput =
        clearWeightsGradientFirstIfFused ? weightsOptimizer->getWeightsGradient().value() : weightsGradientScratch[connectionNumber];
    Tensor dbiasOutput =
        clearWeightsGradientFirstIfFused ? biasesOptimizer->getWeightsGradient().value() : biasesGradientScratch[connectionNumber];

    if (!frontendBackwardGraphs[connectionNumber]) {
        frontendBackwardGraphs[connectionNumber] = buildBackwardBatchNormGraph(featureInputs[connectionNumber].value(),
                                                                               errorInputs[connectionNumber].value(),
                                                                               errorOut.value(),
                                                                               weights,
                                                                               resultSaveMean[connectionNumber],
                                                                               resultSaveInvVariance[connectionNumber],
                                                                               dscaleOutput,
                                                                               dbiasOutput,
                                                                               gradientUpdateStream.value());
    }

    std::unordered_map<int64_t, void*> tensor_pack;
    putFrontendTensorPointer(tensor_pack, BN_DY_UID, errorInputs[connectionNumber].value());
    putFrontendTensorPointer(tensor_pack, BN_X_UID, featureInputs[connectionNumber].value());
    putFrontendTensorPointer(tensor_pack, BN_DX_UID, errorOut.value());
    putFrontendTensorPointer(tensor_pack, BN_SCALE_UID, weights);
    putFrontendTensorPointer(tensor_pack, BN_SAVED_MEAN_UID, resultSaveMean[connectionNumber]);
    putFrontendTensorPointer(tensor_pack, BN_SAVED_INV_VARIANCE_UID, resultSaveInvVariance[connectionNumber]);
    putFrontendTensorPointer(tensor_pack, BN_DSCALE_UID, dscaleOutput);
    putFrontendTensorPointer(tensor_pack, BN_DBIAS_UID, dbiasOutput);
    executeFrontendBatchNormGraph(
        *frontendBackwardGraphs[connectionNumber], gradientUpdateStream.value(), tensor_pack, "batch-normalization backward");

    if (!clearWeightsGradientFirstIfFused) {
        launchAccumulateBatchNormGradientFp32(weightsOptimizer->getWeightsGradient().value().getMemPtr<float>(),
                                              weightsGradientScratch[connectionNumber].getMemPtr<float>(),
                                              numChannels,
                                              gradientUpdateStream.value());
        launchAccumulateBatchNormGradientFp32(biasesOptimizer->getWeightsGradient().value().getMemPtr<float>(),
                                              biasesGradientScratch[connectionNumber].getMemPtr<float>(),
                                              numChannels,
                                              gradientUpdateStream.value());
    }

    return gradientUpdateStream.value().putEvent();
}

void BatchNormalization::accumulateWeightsGradient(uint32_t connectionNumber, bool clearGradientFirst) {
    (void)connectionNumber;
    (void)clearGradientFirst;
    // No-op: the cuDNN Frontend batchnorm backward graph already produced dscale/dbias on the gradient stream.
}

}  // namespace ThorImplementation
