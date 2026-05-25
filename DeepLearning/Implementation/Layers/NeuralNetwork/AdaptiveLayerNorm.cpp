#include "DeepLearning/Implementation/Layers/NeuralNetwork/AdaptiveLayerNorm.h"

#include "Utilities/TensorOperations/DeepLearning/CudnnAdaptiveLayerNorm.h"

#include <limits>
#include <stdexcept>
#include <string>

using namespace std;

namespace ThorImplementation {
namespace {

bool isAdaptiveLayerNormIoDataType(DataType dtype) {
    switch (dtype) {
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
            return true;
        default:
            return false;
    }
}

void validateCudnnFrontendContract(uint64_t normalizedFeatureCount, DataType inputDataType) {
    if (inputDataType == DataType::FP32 && normalizedFeatureCount % 32 != 0) {
        throw runtime_error(
            "AdaptiveLayerNorm cuDNN Frontend primary engines require fp32 normalized feature count to be a multiple of 32; got " +
            to_string(normalizedFeatureCount) + ".");
    }
}

string dtypeName(DataType dtype) { return TensorDescriptor::getElementTypeName(dtype); }

void validateEpsilon(double epsilon) {
    if (!(epsilon > 0.0)) {
        throw runtime_error("AdaptiveLayerNorm epsilon must be > 0.");
    }
}

}  // namespace

AdaptiveLayerNorm::AdaptiveLayerNorm(const TensorPlacement& placement,
                                     bool inferenceOnly,
                                     vector<uint64_t> normalizedShape,
                                     optional<double> epsilon,
                                     optional<DataType> scaleBiasDataType,
                                     int64_t stampedId)
    : placement(placement),
      stampedId(stampedId),
      normalizedShape(std::move(normalizedShape)),
      normalizedFeatureCount(checkedNormalizedFeatureCount(this->normalizedShape)),
      epsilon(epsilon.has_value() ? epsilon.value() : 1.0e-5),
      scaleBiasDataType(scaleBiasDataType.has_value() ? scaleBiasDataType.value() : DataType::FP32) {
    (void)this->stampedId;
    setConstructForInferenceOnly(inferenceOnly);
    validateEpsilon(this->epsilon);
    if (this->scaleBiasDataType != DataType::FP32) {
        throw runtime_error("AdaptiveLayerNorm currently requires fp32 scale/bias tensors for cuDNN Frontend AdaptiveLayerNorm; got " +
                            dtypeName(this->scaleBiasDataType) + ".");
    }
}

AdaptiveLayerNorm::~AdaptiveLayerNorm() { cleanup(); }

void AdaptiveLayerNorm::setEpsilon(double value) {
    validateEpsilon(value);
    epsilon = value;
}

uint64_t AdaptiveLayerNorm::checkedNormalizedFeatureCount(const vector<uint64_t>& normalizedShape) {
    if (normalizedShape.empty()) {
        throw runtime_error("AdaptiveLayerNorm normalizedShape must contain at least one dimension.");
    }
    uint64_t count = 1;
    for (uint64_t dim : normalizedShape) {
        if (dim == 0) {
            throw runtime_error("AdaptiveLayerNorm normalizedShape dimensions must be non-zero.");
        }
        if (count > numeric_limits<uint64_t>::max() / dim) {
            throw runtime_error("AdaptiveLayerNorm normalizedShape feature count overflows uint64_t.");
        }
        count *= dim;
    }
    return count;
}

uint32_t AdaptiveLayerNorm::decodeInputConnectionType(int connectionType) {
    if (connectionType < 0 || connectionType >= static_cast<int>(NUM_INPUT_PORTS)) {
        throw runtime_error("AdaptiveLayerNorm received invalid input connection type: " + to_string(connectionType));
    }
    return static_cast<uint32_t>(connectionType);
}

void AdaptiveLayerNorm::validateConfiguredInput(const Tensor& input) const {
    const vector<uint64_t> dims = input.getDimensions();
    if (dims.size() < normalizedShape.size() + 1) {
        throw runtime_error("AdaptiveLayerNorm input must have at least one leading sample dimension plus the normalized trailing shape.");
    }
    if (!isAdaptiveLayerNormIoDataType(input.getDataType())) {
        throw runtime_error("AdaptiveLayerNorm supports fp16, bf16, and fp32 data inputs with cuDNN Frontend; got " +
                            dtypeName(input.getDataType()) + ".");
    }
    const size_t offset = dims.size() - normalizedShape.size();
    for (size_t i = 0; i < normalizedShape.size(); ++i) {
        if (dims[offset + i] != normalizedShape[i]) {
            throw runtime_error("AdaptiveLayerNorm input trailing dimensions do not match normalizedShape.");
        }
    }
    validateCudnnFrontendContract(normalizedFeatureCount, input.getDataType());
}

uint64_t AdaptiveLayerNorm::computeBatchSize(const Tensor& input) const {
    validateConfiguredInput(input);
    const vector<uint64_t> dims = input.getDimensions();
    THOR_THROW_IF_FALSE(!dims.empty());
    if (dims[0] == 0) {
        throw runtime_error("AdaptiveLayerNorm batch size must be non-zero.");
    }
    return dims[0];
}

uint64_t AdaptiveLayerNorm::computeLeadingFeatureCount(const Tensor& input) const {
    validateConfiguredInput(input);
    const vector<uint64_t> dims = input.getDimensions();
    const size_t normalizedOffset = dims.size() - normalizedShape.size();
    uint64_t count = 1;
    for (size_t i = 1; i < normalizedOffset; ++i) {
        if (dims[i] == 0) {
            throw runtime_error("AdaptiveLayerNorm leading feature dimensions must be non-zero.");
        }
        if (count > numeric_limits<uint64_t>::max() / dims[i]) {
            throw runtime_error("AdaptiveLayerNorm leading feature count overflows uint64_t.");
        }
        count *= dims[i];
    }
    return count;
}

void AdaptiveLayerNorm::validateScaleBiasInput(const Tensor& tensor, const Tensor& data, const char* name) const {
    const vector<uint64_t> dataDims = data.getDimensions();
    const vector<uint64_t> dims = tensor.getDimensions();
    if (dims.size() != normalizedShape.size() + 1) {
        throw runtime_error(string("AdaptiveLayerNorm ") + name + " input must have physical dimensions [batch] + normalizedShape.");
    }
    if (dims[0] != dataDims[0]) {
        throw runtime_error(string("AdaptiveLayerNorm ") + name + " batch dimension must match the data input batch dimension.");
    }
    for (size_t i = 0; i < normalizedShape.size(); ++i) {
        if (dims[i + 1] != normalizedShape[i]) {
            throw runtime_error(string("AdaptiveLayerNorm ") + name + " trailing dimensions must match normalizedShape.");
        }
    }
}

void AdaptiveLayerNorm::validateAllConnectedInputs() const {
    for (uint32_t i = 0; i < NUM_INPUT_PORTS; ++i) {
        if (!adaptiveFeatureInputs[i].has_value()) {
            throw runtime_error("AdaptiveLayerNorm requires data, scale, and bias inputs to be connected before compile.");
        }
        if (adaptiveFeatureInputs[i].value().getPlacement() != placement) {
            throw runtime_error("AdaptiveLayerNorm all inputs must have the same GPU placement.");
        }
    }

    const Tensor& data = adaptiveFeatureInputs[DATA].value();
    const Tensor& scale = adaptiveFeatureInputs[SCALE].value();
    const Tensor& bias = adaptiveFeatureInputs[BIAS].value();
    validateConfiguredInput(data);
    validateScaleBiasInput(scale, data, "scale");
    validateScaleBiasInput(bias, data, "bias");
    if (scale.getDataType() != scaleBiasDataType || bias.getDataType() != scaleBiasDataType) {
        throw runtime_error("AdaptiveLayerNorm scale and bias inputs must be fp32 tensors.");
    }
}

bool AdaptiveLayerNorm::anyErrorOutput() const {
    for (const auto& maybeTensor : adaptiveErrorOutputs) {
        if (maybeTensor.has_value())
            return true;
    }
    return false;
}

Stream& AdaptiveLayerNorm::computeStream() {
    THOR_THROW_IF_FALSE(adaptiveStreams[DATA].has_value());
    return adaptiveStreams[DATA].value();
}

const Stream& AdaptiveLayerNorm::computeStream() const {
    THOR_THROW_IF_FALSE(adaptiveStreams[DATA].has_value());
    return adaptiveStreams[DATA].value();
}

optional<Tensor> AdaptiveLayerNorm::createFeatureOutputTensor() {
    THOR_THROW_IF_FALSE(adaptiveFeatureInputs[DATA].has_value());
    return adaptiveFeatureInputs[DATA].value().clone();
}

optional<Tensor> AdaptiveLayerNorm::createErrorOutputTensor(bool backPropagateError) {
    (void)backPropagateError;
    THOR_UNREACHABLE();
}

optional<Tensor> AdaptiveLayerNorm::connectToPreviousLayer(
    Layer* previousLayer, optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType) {
    THOR_THROW_IF_FALSE(!compiled);
    const uint32_t port = decodeInputConnectionType(connectionType);
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(previousLayer != nullptr);
    THOR_THROW_IF_FALSE(!adaptivePreviousLayers[port].has_value());
    THOR_THROW_IF_FALSE(!adaptiveFeatureInputs[port].has_value());
    THOR_THROW_IF_FALSE(!adaptiveErrorOutputs[port].has_value());

    if (featureInput.value().getPlacement() != placement) {
        throw runtime_error("AdaptiveLayerNorm input placement does not match the layer placement.");
    }

    adaptivePreviousLayers[port] = previousLayer;
    adaptiveFeatureInputs[port] = featureInput;
    adaptiveStreams[port] = stream;
    if (backPropagateError && !isInferenceOnly()) {
        adaptiveErrorOutputs[port] = featureInput.value().clone();
    }

    return adaptiveErrorOutputs[port];
}

void AdaptiveLayerNorm::connectToNextLayer(Layer* nextLayer, int driverConnectionType, int loaderConnectionType) {
    THOR_THROW_IF_FALSE(!compiled);
    THOR_THROW_IF_FALSE(driverConnectionType == 0);
    THOR_THROW_IF_FALSE(nextLayer != nullptr);
    THOR_THROW_IF_FALSE(!this->nextLayer.has_value());

    validateAllConnectedInputs();
    if (!featureOutput.has_value()) {
        featureOutput = createFeatureOutputTensor();
    }

    this->nextLayer = nextLayer;
    errorInput = nextLayer->connectToPreviousLayer(
        this, featureOutput, computeStream(), shouldConnectToBackPropErrorIn() && !isBackPropStub(), loaderConnectionType);

    if (!errorInput.has_value()) {
        pruneUpstreamErrorOutputs();
    }
}

void AdaptiveLayerNorm::replaceErrorInput(optional<Tensor> oldErrorInput, optional<Tensor> newErrorInput) {
    THOR_THROW_IF_FALSE(oldErrorInput.has_value());
    if (errorInput.has_value()) {
        THOR_THROW_IF_FALSE(errorInput.value() == oldErrorInput.value());
    }
    errorInput = newErrorInput;
    if (!errorInput.has_value()) {
        pruneUpstreamErrorOutputs();
    }
}

void AdaptiveLayerNorm::pruneUpstreamErrorOutputs() {
    for (uint32_t i = 0; i < NUM_INPUT_PORTS; ++i) {
        if (adaptiveErrorOutputs[i].has_value() && adaptivePreviousLayers[i].has_value()) {
            adaptivePreviousLayers[i].value()->replaceErrorInput(adaptiveErrorOutputs[i], nullopt);
        }
        adaptiveErrorOutputs[i].reset();
    }
}

void AdaptiveLayerNorm::compileImpl() {
    validateAllConnectedInputs();
    THOR_THROW_IF_FALSE(featureOutput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.value().getDescriptor() == adaptiveFeatureInputs[DATA].value().getDescriptor());
    THOR_THROW_IF_FALSE(featureOutput.value().getPlacement() == adaptiveFeatureInputs[DATA].value().getPlacement());

    batchSize = computeBatchSize(adaptiveFeatureInputs[DATA].value());
    leadingFeatureCount = computeLeadingFeatureCount(adaptiveFeatureInputs[DATA].value());
    const uint64_t statsElementCount = batchSize * leadingFeatureCount;

    saveMean = Tensor(placement, TensorDescriptor(DataType::FP32, {statsElementCount}));
    saveInvVariance = Tensor(placement, TensorDescriptor(DataType::FP32, {statsElementCount}));

    for (auto& scratch : scratchErrorOutputs)
        scratch.reset();

    if (errorInput.has_value() && !isInferenceOnly()) {
        for (uint32_t i = 0; i < NUM_INPUT_PORTS; ++i) {
            if (!adaptiveErrorOutputs[i].has_value()) {
                scratchErrorOutputs[i] = adaptiveFeatureInputs[i].value().clone();
            }
        }
    }

    resetForwardArrivalBookkeeping();
}

void AdaptiveLayerNorm::initialize() {
    Layer::initialize();
    resetForwardArrivalBookkeeping();
}

void AdaptiveLayerNorm::cleanup() {
    saveMean = Tensor();
    saveInvVariance = Tensor();
    for (auto& scratch : scratchErrorOutputs)
        scratch.reset();
    allForwardInputTensorIds.clear();
    stillWaitingForForwardInputTensorIds.clear();
    Layer::cleanup();
}

void AdaptiveLayerNorm::resetForwardArrivalBookkeeping() {
    allForwardInputTensorIds.clear();
    for (const auto& maybeTensor : adaptiveFeatureInputs) {
        if (maybeTensor.has_value()) {
            allForwardInputTensorIds.insert(maybeTensor.value().getTensorId());
        }
    }
    stillWaitingForForwardInputTensorIds = allForwardInputTensorIds;
}

void AdaptiveLayerNorm::forward(optional<Tensor> featureInput, bool validationPass, uint32_t batchSize) {
    (void)batchSize;
    THOR_THROW_IF_FALSE(running);
    THOR_THROW_IF_FALSE(featureInput.has_value());

    if (stillWaitingForForwardInputTensorIds.empty()) {
        resetForwardArrivalBookkeeping();
    }

    const unsigned long tensorId = featureInput.value().getTensorId();
    THOR_THROW_IF_FALSE(stillWaitingForForwardInputTensorIds.count(tensorId) == 1);
    stillWaitingForForwardInputTensorIds.erase(tensorId);

    if (!stillWaitingForForwardInputTensorIds.empty()) {
        return;
    }

    resetForwardArrivalBookkeeping();

    for (uint32_t i = 1; i < NUM_INPUT_PORTS; ++i) {
        THOR_THROW_IF_FALSE(adaptiveStreams[i].has_value());
        computeStream().waitEvent(adaptiveStreams[i].value().putEvent());
    }

    CudnnAdaptiveLayerNormDescriptor descriptor;
    descriptor.batchSize = computeBatchSize(adaptiveFeatureInputs[DATA].value());
    descriptor.leadingFeatureCount = computeLeadingFeatureCount(adaptiveFeatureInputs[DATA].value());
    descriptor.normalizedFeatureCount = normalizedFeatureCount;
    descriptor.inputDataType = adaptiveFeatureInputs[DATA].value().getDataType();
    descriptor.outputDataType = featureOutput.value().getDataType();
    descriptor.scaleBiasDataType = scaleBiasDataType;
    descriptor.computeDataType = DataType::FP32;
    descriptor.epsilon = static_cast<float>(epsilon);
    descriptor.training = !isInferenceOnly();

    CudnnAdaptiveLayerNormForwardArgs args;
    args.x = adaptiveFeatureInputs[DATA].value();
    args.scale = adaptiveFeatureInputs[SCALE].value();
    args.bias = adaptiveFeatureInputs[BIAS].value();
    args.y = featureOutput.value();
    if (!isInferenceOnly()) {
        args.mean = saveMean;
        args.invVariance = saveInvVariance;
    }

    CudnnAdaptiveLayerNorm::instance().forward(descriptor, args, computeStream());

    if (nextLayer.has_value()) {
        nextLayer.value()->forward(featureOutput, validationPass, batchSize);
    }
}

void AdaptiveLayerNorm::backward(optional<Tensor> errorInput, uint32_t batchSize) {
    THOR_THROW_IF_FALSE(running);
    if (!errorInput.has_value())
        return;
    if (isInferenceOnly())
        return;
    THOR_THROW_IF_FALSE(this->errorInput.has_value());
    THOR_THROW_IF_FALSE(errorInput.value() == this->errorInput.value());

    if (!anyErrorOutput())
        return;

    array<Tensor, NUM_INPUT_PORTS> gradientOutputs;
    for (uint32_t i = 0; i < NUM_INPUT_PORTS; ++i) {
        if (adaptiveErrorOutputs[i].has_value()) {
            gradientOutputs[i] = adaptiveErrorOutputs[i].value();
        } else {
            THOR_THROW_IF_FALSE(scratchErrorOutputs[i].has_value());
            gradientOutputs[i] = scratchErrorOutputs[i].value();
        }
    }

    CudnnAdaptiveLayerNormDescriptor descriptor;
    descriptor.batchSize = computeBatchSize(adaptiveFeatureInputs[DATA].value());
    descriptor.leadingFeatureCount = computeLeadingFeatureCount(adaptiveFeatureInputs[DATA].value());
    descriptor.normalizedFeatureCount = normalizedFeatureCount;
    descriptor.inputDataType = adaptiveFeatureInputs[DATA].value().getDataType();
    descriptor.outputDataType = errorInput.value().getDataType();
    descriptor.scaleBiasDataType = scaleBiasDataType;
    descriptor.computeDataType = DataType::FP32;
    descriptor.epsilon = static_cast<float>(epsilon);
    descriptor.training = true;

    CudnnAdaptiveLayerNormBackwardArgs args;
    args.dy = errorInput.value();
    args.x = adaptiveFeatureInputs[DATA].value();
    args.scale = adaptiveFeatureInputs[SCALE].value();
    args.mean = saveMean;
    args.invVariance = saveInvVariance;
    args.dx = gradientOutputs[DATA];
    args.dscale = gradientOutputs[SCALE];
    args.dbias = gradientOutputs[BIAS];

    CudnnAdaptiveLayerNorm::instance().backward(descriptor, args, computeStream());

    Event gradientsReady = computeStream().putEvent();
    for (uint32_t i = 0; i < NUM_INPUT_PORTS; ++i) {
        if (!adaptivePreviousLayers[i].has_value() || !adaptiveErrorOutputs[i].has_value()) {
            continue;
        }
        if (i != DATA) {
            THOR_THROW_IF_FALSE(adaptiveStreams[i].has_value());
            adaptiveStreams[i].value().waitEvent(gradientsReady);
        }
        adaptivePreviousLayers[i].value()->backward(adaptiveErrorOutputs[i], batchSize);
    }
}

TensorPlacement AdaptiveLayerNorm::getPlacement() {
    if (placement.getMemDevice() != TensorPlacement::MemDevices::CPU)
        return placement;
    for (const auto& maybeTensor : adaptiveFeatureInputs) {
        if (maybeTensor.has_value())
            return maybeTensor.value().getPlacement();
    }
    return placement;
}

bool AdaptiveLayerNorm::isBackPropStub() { return !anyErrorOutput(); }

uint64_t AdaptiveLayerNorm::floatingPointOperationsPerExampleForward() {
    if (!adaptiveFeatureInputs[DATA].has_value())
        return 0;
    return adaptiveFeatureInputs[DATA].value().getTotalNumElements() * 8;
}

uint64_t AdaptiveLayerNorm::floatingPointOperationsPerExampleBackward() {
    if (!adaptiveFeatureInputs[DATA].has_value())
        return 0;
    return adaptiveFeatureInputs[DATA].value().getTotalNumElements() * 16;
}

void AdaptiveLayerNorm::infer(optional<Tensor> inputTensor, optional<Tensor> outputTensor, Stream stream) {
    (void)inputTensor;
    (void)outputTensor;
    (void)stream;
    THOR_UNREACHABLE();
}

void AdaptiveLayerNorm::backProp(optional<Tensor> dataIn, optional<Tensor> errorIn, optional<Tensor> errorOut, Stream stream) {
    (void)dataIn;
    (void)errorIn;
    (void)errorOut;
    (void)stream;
    THOR_UNREACHABLE();
}

}  // namespace ThorImplementation
