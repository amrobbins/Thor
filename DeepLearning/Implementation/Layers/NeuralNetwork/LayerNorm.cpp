#include "DeepLearning/Implementation/Layers/NeuralNetwork/LayerNorm.h"

#include "DeepLearning/Implementation/Initializers/ConstantInitializer.h"
#include "Utilities/TensorOperations/DeepLearning/BatchNormFrontendHelpers.h"
#include "Utilities/TensorOperations/DeepLearning/CudnnLayerNorm.h"

#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

using namespace std;

namespace ThorImplementation {
namespace {

bool isLayerNormIoDataType(TensorDescriptor::DataType dtype) {
    switch (dtype) {
        case TensorDescriptor::DataType::FP16:
        case TensorDescriptor::DataType::BF16:
        case TensorDescriptor::DataType::FP32:
            return true;
        default:
            return false;
    }
}

string dtypeName(TensorDescriptor::DataType dtype) { return TensorDescriptor::getElementTypeName(dtype); }

class LayerNormParameter final : public PhysicalParameter {
   public:
    LayerNormParameter(string name, uint64_t normalizedFeatureCount, TensorDescriptor::DataType dtype, bool trainable)
        : PhysicalParameter(std::move(name), trainable), normalizedFeatureCount(normalizedFeatureCount), dtype(dtype) {}

    void createStorage(const StorageContext& context) override {
        const Tensor& input = context.getFeatureInput();
        storage = Tensor(input.getPlacement(), TensorDescriptor(dtype, {normalizedFeatureCount}));
    }

   private:
    uint64_t normalizedFeatureCount;
    TensorDescriptor::DataType dtype;
};

void validateEpsilon(double epsilon) {
    if (!(epsilon > 0.0)) {
        throw runtime_error("LayerNorm epsilon must be > 0.");
    }
}

shared_ptr<PhysicalParameter> makeDefaultParameter(const string& name,
                                                   uint64_t normalizedFeatureCount,
                                                   TensorDescriptor::DataType dtype,
                                                   bool trainable,
                                                   float initialValue) {
    auto parameter = make_shared<LayerNormParameter>(name, normalizedFeatureCount, dtype, trainable);
    shared_ptr<Initializer> initializer = make_shared<ConstantInitializer>(initialValue);
    parameter->setInitializer(optional<shared_ptr<Initializer>>(initializer));
    return parameter;
}

}  // namespace

LayerNorm::LayerNorm(const TensorPlacement& placement,
                     bool inferenceOnly,
                     vector<uint64_t> normalizedShape,
                     optional<double> epsilon,
                     optional<TensorDescriptor::DataType> parameterDataType,
                     vector<shared_ptr<PhysicalParameter>> physicalParameters,
                     int64_t stampedId)
    : TrainableLayer(placement, inferenceOnly, stampedId),
      normalizedShape(std::move(normalizedShape)),
      normalizedFeatureCount(checkedNormalizedFeatureCount(this->normalizedShape)),
      epsilon(epsilon.has_value() ? epsilon.value() : 1.0e-5),
      parameterDataType(parameterDataType.has_value() ? parameterDataType.value() : TensorDescriptor::DataType::FP32) {
    validateEpsilon(this->epsilon);
    if (this->parameterDataType != TensorDescriptor::DataType::FP32) {
        throw runtime_error("LayerNorm currently requires fp32 scale/bias parameters for cuDNN Frontend LayerNorm; got " +
                            dtypeName(this->parameterDataType) + ".");
    }

    if (physicalParameters.empty()) {
        addParameter(makeDefaultParameter("weights", normalizedFeatureCount, this->parameterDataType, true, 1.0f));
        addParameter(makeDefaultParameter("biases", normalizedFeatureCount, this->parameterDataType, true, 0.0f));
    } else {
        for (const auto& parameter : physicalParameters) {
            THOR_THROW_IF_FALSE(parameter != nullptr);
            addParameter(parameter);
        }
        THOR_THROW_IF_FALSE(getParameter("weights") != nullptr);
        THOR_THROW_IF_FALSE(getParameter("biases") != nullptr);
    }
}

LayerNorm::~LayerNorm() { cleanup(); }

void LayerNorm::setEpsilon(double value) {
    validateEpsilon(value);
    epsilon = value;
}

uint64_t LayerNorm::checkedNormalizedFeatureCount(const vector<uint64_t>& normalizedShape) {
    if (normalizedShape.empty()) {
        throw runtime_error("LayerNorm normalizedShape must contain at least one dimension.");
    }
    uint64_t count = 1;
    for (uint64_t dim : normalizedShape) {
        if (dim == 0) {
            throw runtime_error("LayerNorm normalizedShape dimensions must be non-zero.");
        }
        if (count > numeric_limits<uint64_t>::max() / dim) {
            throw runtime_error("LayerNorm normalizedShape feature count overflows uint64_t.");
        }
        count *= dim;
    }
    return count;
}

void LayerNorm::validateConfiguredInput(const Tensor& input) const {
    const vector<uint64_t> dims = input.getDimensions();
    if (dims.size() < normalizedShape.size() + 1) {
        throw runtime_error("LayerNorm input must have at least one leading sample dimension plus the normalized trailing shape.");
    }
    if (!isLayerNormIoDataType(input.getDataType())) {
        throw runtime_error("LayerNorm supports fp16, bf16, and fp32 inputs with cuDNN Frontend; got " + dtypeName(input.getDataType()) + ".");
    }
    const size_t offset = dims.size() - normalizedShape.size();
    for (size_t i = 0; i < normalizedShape.size(); ++i) {
        if (dims[offset + i] != normalizedShape[i]) {
            throw runtime_error("LayerNorm input trailing dimensions do not match normalizedShape.");
        }
    }
}

uint64_t LayerNorm::computeOuterSize(const Tensor& input) const {
    validateConfiguredInput(input);
    const uint64_t total = input.getTotalNumElements();
    THOR_THROW_IF_FALSE(total % normalizedFeatureCount == 0);
    const uint64_t outer = total / normalizedFeatureCount;
    if (outer == 0) {
        throw runtime_error("LayerNorm outer sample count must be non-zero.");
    }
    return outer;
}

optional<Tensor> LayerNorm::createFeatureOutputTensor() {
    optional<Tensor> maybeInput = getFirstPresentTensor(featureInputs);
    THOR_THROW_IF_FALSE(maybeInput.has_value());
    return maybeInput.value().clone();
}

optional<Tensor> LayerNorm::createErrorOutputTensor(bool backPropagateError, uint32_t connectionNumber) {
    if (backPropagateError && !isInferenceOnly()) {
        THOR_THROW_IF_FALSE(featureInputs.size() > connectionNumber);
        THOR_THROW_IF_FALSE(featureInputs[connectionNumber].has_value());
        return featureInputs[connectionNumber].value().clone();
    }
    return nullopt;
}

uint64_t LayerNorm::flopCountForward() {
    optional<Tensor> maybeInput = getFirstPresentTensor(featureInputs);
    if (!maybeInput.has_value())
        return 0;
    return maybeInput.value().getTotalNumElements() * 8;
}

uint64_t LayerNorm::flopCountBackward() {
    optional<Tensor> maybeInput = getFirstPresentTensor(featureInputs);
    if (!maybeInput.has_value())
        return 0;
    return maybeInput.value().getTotalNumElements() * 16;
}

void LayerNorm::compileImpl() {
    TrainableLayer::compileImpl();

    THOR_THROW_IF_FALSE(!featureInputs.empty());
    THOR_THROW_IF_FALSE(!featureOutputs.empty());
    THOR_THROW_IF_FALSE(featureInputs.size() == featureOutputs.size());

    optional<Tensor> maybeInput = getFirstPresentTensor(featureInputs);
    THOR_THROW_IF_FALSE(maybeInput.has_value());
    const Tensor& input = maybeInput.value();

    placement = input.getPlacement();
    THOR_THROW_IF_FALSE(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
    ensureNoDeviceCrossing(placement);

    validateConfiguredInput(input);
    outerSize = computeOuterSize(input);

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
    THOR_THROW_IF_FALSE(weights.getDataType() == parameterDataType);
    THOR_THROW_IF_FALSE(biases.getDataType() == parameterDataType);
    THOR_THROW_IF_FALSE(weights.getTotalNumElements() == normalizedFeatureCount);
    THOR_THROW_IF_FALSE(biases.getTotalNumElements() == normalizedFeatureCount);
    THOR_THROW_IF_FALSE(weights.getPlacement() == placement);
    THOR_THROW_IF_FALSE(biases.getPlacement() == placement);

    saveMean.clear();
    saveInvVariance.clear();
    scratchDScale.clear();
    scratchDBias.clear();
    scratchErrorOutput.reset();

    saveMean.reserve(featureInputs.size());
    saveInvVariance.reserve(featureInputs.size());
    scratchDScale.reserve(featureInputs.size());
    scratchDBias.reserve(featureInputs.size());

    for (uint32_t i = 0; i < featureInputs.size(); ++i) {
        if (featureInputs[i].has_value()) {
            const Tensor& in = featureInputs[i].value();
            validateConfiguredInput(in);
            if (computeOuterSize(in) != outerSize) {
                throw runtime_error("LayerNorm all feature inputs must have the same outer sample count.");
            }
        }
        saveMean.emplace_back(placement, TensorDescriptor(TensorDescriptor::DataType::FP32, {outerSize}));
        saveInvVariance.emplace_back(placement, TensorDescriptor(TensorDescriptor::DataType::FP32, {outerSize}));
        scratchDScale.emplace_back(weights.clone());
        scratchDBias.emplace_back(biases.clone());
        if (errorInputs.size() > i && errorInputs[i].has_value() && (errorOutputs.size() <= i || !errorOutputs[i].has_value())) {
            THOR_THROW_IF_FALSE(featureInputs[i].has_value());
            if (!scratchErrorOutput.has_value()) {
                scratchErrorOutput = featureInputs[i].value().clone();
            }
        }
    }
}

void LayerNorm::cleanup() {
    saveMean.clear();
    saveInvVariance.clear();
    scratchDScale.clear();
    scratchDBias.clear();
    scratchErrorOutput.reset();
    Layer::cleanup();
}

void LayerNorm::computeFeatureOut(uint32_t connectionNumber) {
    THOR_THROW_IF_FALSE(featureInputs[connectionNumber].has_value());
    THOR_THROW_IF_FALSE(featureOutputs[connectionNumber].has_value());
    Tensor input = featureInputs[connectionNumber].value();
    Tensor output = featureOutputs[connectionNumber].value();

    validateConfiguredInput(input);
    THOR_THROW_IF_FALSE(output.getDescriptor() == input.getDescriptor());
    THOR_THROW_IF_FALSE(output.getPlacement() == input.getPlacement());

    CudnnLayerNormDescriptor descriptor;
    descriptor.outerSize = computeOuterSize(input);
    descriptor.normalizedFeatureCount = normalizedFeatureCount;
    descriptor.inputDataType = input.getDataType();
    descriptor.outputDataType = output.getDataType();
    descriptor.parameterDataType = parameterDataType;
    descriptor.computeDataType = TensorDescriptor::DataType::FP32;
    descriptor.epsilon = static_cast<float>(epsilon);
    descriptor.training = !isInferenceOnly();

    CudnnLayerNormForwardArgs args;
    args.x = input;
    args.scale = weights;
    args.bias = biases;
    args.y = output;
    if (!isInferenceOnly()) {
        args.mean = saveMean[connectionNumber];
        args.invVariance = saveInvVariance[connectionNumber];
    }

    CudnnLayerNorm::instance().forward(descriptor, args, streams[connectionNumber]);
}

optional<Event> LayerNorm::computeErrorOutAccumulateWeightsGradienFused(uint32_t connectionNumber,
                                                                        bool clearWeightsGradientFirstIfFused) {
    if (!errorInputs[connectionNumber].has_value())
        return nullopt;
    if (isInferenceOnly())
        return nullopt;

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

    optional<Tensor> errorOut = nullopt;
    if (errorOutputs.size() > connectionNumber && errorOutputs[connectionNumber].has_value()) {
        errorOut = errorOutputs[connectionNumber];
    } else {
        errorOut = scratchErrorOutput;
    }
    THOR_THROW_IF_FALSE(errorOut.has_value());
    THOR_THROW_IF_FALSE(gradientUpdateStream.has_value());
    THOR_THROW_IF_FALSE(featureInputs[connectionNumber].has_value());

    Tensor dscaleOutput = clearWeightsGradientFirstIfFused ? weightsOptimizer->getWeightsGradient().value() : scratchDScale[connectionNumber];
    Tensor dbiasOutput = clearWeightsGradientFirstIfFused ? biasesOptimizer->getWeightsGradient().value() : scratchDBias[connectionNumber];

    const Tensor& input = featureInputs[connectionNumber].value();
    CudnnLayerNormDescriptor descriptor;
    descriptor.outerSize = computeOuterSize(input);
    descriptor.normalizedFeatureCount = normalizedFeatureCount;
    descriptor.inputDataType = input.getDataType();
    descriptor.outputDataType = errorInputs[connectionNumber].value().getDataType();
    descriptor.parameterDataType = parameterDataType;
    descriptor.computeDataType = TensorDescriptor::DataType::FP32;
    descriptor.epsilon = static_cast<float>(epsilon);
    descriptor.training = true;

    CudnnLayerNormBackwardArgs args;
    args.dy = errorInputs[connectionNumber].value();
    args.x = input;
    args.scale = weights;
    args.mean = saveMean[connectionNumber];
    args.invVariance = saveInvVariance[connectionNumber];
    args.dx = errorOut.value();
    args.dscale = dscaleOutput;
    args.dbias = dbiasOutput;

    CudnnLayerNorm::instance().backward(descriptor, args, gradientUpdateStream.value());

    if (!clearWeightsGradientFirstIfFused) {
        launchAccumulateBatchNormGradientFp32(weightsOptimizer->getWeightsGradient().value().getMemPtr<float>(),
                                             scratchDScale[connectionNumber].getMemPtr<float>(),
                                             normalizedFeatureCount,
                                             gradientUpdateStream.value());
        launchAccumulateBatchNormGradientFp32(biasesOptimizer->getWeightsGradient().value().getMemPtr<float>(),
                                             scratchDBias[connectionNumber].getMemPtr<float>(),
                                             normalizedFeatureCount,
                                             gradientUpdateStream.value());
    }

    return gradientUpdateStream.value().putEvent();
}

void LayerNorm::accumulateWeightsGradient(uint32_t connectionNumber, bool clearGradientFirst) {
    (void)connectionNumber;
    (void)clearGradientFirst;
    // No-op: cuDNN Frontend LayerNorm backward produces dx, dscale, and dbias together.
}

}  // namespace ThorImplementation
