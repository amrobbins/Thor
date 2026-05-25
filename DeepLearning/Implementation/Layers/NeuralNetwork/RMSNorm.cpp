#include "DeepLearning/Implementation/Layers/NeuralNetwork/RMSNorm.h"

#include "DeepLearning/Implementation/Initializers/ConstantInitializer.h"
#include "Utilities/TensorOperations/DeepLearning/BatchNormFrontendHelpers.h"
#include "Utilities/TensorOperations/DeepLearning/CudnnRmsNorm.h"

#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

using namespace std;

namespace ThorImplementation {
namespace {

bool isRmsNormIoDataType(DataType dtype) {
    switch (dtype) {
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
            return true;
        default:
            return false;
    }
}

string dtypeName(DataType dtype) { return TensorDescriptor::getElementTypeName(dtype); }

class RMSNormParameter final : public PhysicalParameter {
   public:
    RMSNormParameter(string name, uint64_t normalizedFeatureCount, DataType dtype, bool trainable)
        : PhysicalParameter(std::move(name), trainable), normalizedFeatureCount(normalizedFeatureCount), dtype(dtype) {}

    void createStorage(const StorageContext& context) override {
        const Tensor& input = context.getFeatureInput();
        storage = Tensor(input.getPlacement(), TensorDescriptor(dtype, {normalizedFeatureCount}));
    }

   private:
    uint64_t normalizedFeatureCount;
    DataType dtype;
};

void validateEpsilon(double epsilon) {
    if (!(epsilon > 0.0)) {
        throw runtime_error("RMSNorm epsilon must be > 0.");
    }
}

shared_ptr<PhysicalParameter> makeDefaultParameter(const string& name,
                                                   uint64_t normalizedFeatureCount,
                                                   DataType dtype,
                                                   bool trainable,
                                                   float initialValue) {
    auto parameter = make_shared<RMSNormParameter>(name, normalizedFeatureCount, dtype, trainable);
    shared_ptr<Initializer> initializer = make_shared<ConstantInitializer>(initialValue);
    parameter->setInitializer(optional<shared_ptr<Initializer>>(initializer));
    return parameter;
}

}  // namespace

RMSNorm::RMSNorm(const TensorPlacement& placement,
                 bool inferenceOnly,
                 vector<uint64_t> normalizedShape,
                 optional<double> epsilon,
                 optional<DataType> parameterDataType,
                 CudnnRmsNormFusedActivation fusedActivation,
                 vector<shared_ptr<PhysicalParameter>> physicalParameters,
                 int64_t stampedId)
    : TrainableLayer(placement, inferenceOnly, stampedId),
      normalizedShape(std::move(normalizedShape)),
      normalizedFeatureCount(checkedNormalizedFeatureCount(this->normalizedShape)),
      epsilon(epsilon.has_value() ? epsilon.value() : 1.0e-5),
      parameterDataType(parameterDataType.has_value() ? parameterDataType.value() : DataType::FP32),
      fusedActivation(fusedActivation) {
    validateEpsilon(this->epsilon);
    if (this->fusedActivation == CudnnRmsNormFusedActivation::SWISH) {
        if (this->parameterDataType != DataType::BF16) {
            throw runtime_error("RMSNorm fused SWISH currently requires bf16 scale parameters for cuDNN Frontend RMSNorm + SiLU fusion; got " +
                                dtypeName(this->parameterDataType) + ".");
        }
    } else if (this->parameterDataType != DataType::FP32) {
        throw runtime_error("RMSNorm currently requires fp32 scale parameters for cuDNN Frontend RMSNorm; got " +
                            dtypeName(this->parameterDataType) + ".");
    }

    if (physicalParameters.empty()) {
        addParameter(makeDefaultParameter("weights", normalizedFeatureCount, this->parameterDataType, true, 1.0f));
    } else {
        for (const auto& parameter : physicalParameters) {
            THOR_THROW_IF_FALSE(parameter != nullptr);
            addParameter(parameter);
        }
        THOR_THROW_IF_FALSE(getParameter("weights") != nullptr);
    }
}

RMSNorm::~RMSNorm() { cleanup(); }

void RMSNorm::setEpsilon(double value) {
    validateEpsilon(value);
    epsilon = value;
}

uint64_t RMSNorm::checkedNormalizedFeatureCount(const vector<uint64_t>& normalizedShape) {
    if (normalizedShape.empty()) {
        throw runtime_error("RMSNorm normalizedShape must contain at least one dimension.");
    }
    uint64_t count = 1;
    for (uint64_t dim : normalizedShape) {
        if (dim == 0) {
            throw runtime_error("RMSNorm normalizedShape dimensions must be non-zero.");
        }
        if (count > numeric_limits<uint64_t>::max() / dim) {
            throw runtime_error("RMSNorm normalizedShape feature count overflows uint64_t.");
        }
        count *= dim;
    }
    return count;
}

void RMSNorm::validateConfiguredInput(const Tensor& input) const {
    const vector<uint64_t> dims = input.getDimensions();
    if (dims.size() < normalizedShape.size() + 1) {
        throw runtime_error("RMSNorm input must have at least one leading sample dimension plus the normalized trailing shape.");
    }
    if (!isRmsNormIoDataType(input.getDataType())) {
        throw runtime_error("RMSNorm supports fp16, bf16, and fp32 inputs with cuDNN Frontend; got " + dtypeName(input.getDataType()) + ".");
    }
    const size_t offset = dims.size() - normalizedShape.size();
    for (size_t i = 0; i < normalizedShape.size(); ++i) {
        if (dims[offset + i] != normalizedShape[i]) {
            throw runtime_error("RMSNorm input trailing dimensions do not match normalizedShape.");
        }
    }
}

uint64_t RMSNorm::computeOuterSize(const Tensor& input) const {
    validateConfiguredInput(input);
    const uint64_t total = input.getTotalNumElements();
    THOR_THROW_IF_FALSE(total % normalizedFeatureCount == 0);
    const uint64_t outer = total / normalizedFeatureCount;
    if (outer == 0) {
        throw runtime_error("RMSNorm outer sample count must be non-zero.");
    }
    return outer;
}

optional<Tensor> RMSNorm::createFeatureOutputTensor() {
    optional<Tensor> maybeInput = getFirstPresentTensor(featureInputs);
    THOR_THROW_IF_FALSE(maybeInput.has_value());
    return maybeInput.value().clone();
}

optional<Tensor> RMSNorm::createErrorOutputTensor(bool backPropagateError, uint32_t connectionNumber) {
    if (backPropagateError && !isInferenceOnly()) {
        THOR_THROW_IF_FALSE(featureInputs.size() > connectionNumber);
        THOR_THROW_IF_FALSE(featureInputs[connectionNumber].has_value());
        return featureInputs[connectionNumber].value().clone();
    }
    return nullopt;
}

uint64_t RMSNorm::flopCountForward() {
    optional<Tensor> maybeInput = getFirstPresentTensor(featureInputs);
    if (!maybeInput.has_value())
        return 0;
    uint64_t flops = maybeInput.value().getTotalNumElements() * 6;
    if (fusedActivation == CudnnRmsNormFusedActivation::SWISH)
        flops += maybeInput.value().getTotalNumElements() * 5;
    return flops;
}

uint64_t RMSNorm::flopCountBackward() {
    optional<Tensor> maybeInput = getFirstPresentTensor(featureInputs);
    if (!maybeInput.has_value())
        return 0;
    return maybeInput.value().getTotalNumElements() * 12;
}

void RMSNorm::compileImpl() {
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
    if (fusedActivation == CudnnRmsNormFusedActivation::SWISH) {
        if (!isInferenceOnly()) {
            throw runtime_error("RMSNorm fused activation '" + string(toString(fusedActivation)) +
                                "' is inference-only because cuDNN Frontend RMSNorm + SiLU fusion is only selected from an inference-phase RMSNorm graph.");
        }
        if (input.getDataType() != DataType::BF16) {
            throw runtime_error("RMSNorm fused SWISH currently requires bf16 feature inputs for cuDNN Frontend RMSNorm + SiLU fusion; got " +
                                dtypeName(input.getDataType()) + ".");
        }
    }

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
    THOR_THROW_IF_FALSE(weights.getDataType() == parameterDataType);
    THOR_THROW_IF_FALSE(weights.getTotalNumElements() == normalizedFeatureCount);
    THOR_THROW_IF_FALSE(weights.getPlacement() == placement);

    saveInvVariance.clear();
    scratchDScale.clear();
    scratchErrorOutput.reset();

    saveInvVariance.reserve(featureInputs.size());
    scratchDScale.reserve(featureInputs.size());

    for (uint32_t i = 0; i < featureInputs.size(); ++i) {
        if (featureInputs[i].has_value()) {
            const Tensor& in = featureInputs[i].value();
            validateConfiguredInput(in);
            if (computeOuterSize(in) != outerSize) {
                throw runtime_error("RMSNorm all feature inputs must have the same outer sample count.");
            }
        }
        saveInvVariance.emplace_back(placement, TensorDescriptor(DataType::FP32, {outerSize}));
        scratchDScale.emplace_back(weights.clone());
        if (errorInputs.size() > i && errorInputs[i].has_value() && (errorOutputs.size() <= i || !errorOutputs[i].has_value())) {
            THOR_THROW_IF_FALSE(featureInputs[i].has_value());
            if (!scratchErrorOutput.has_value()) {
                scratchErrorOutput = featureInputs[i].value().clone();
            }
        }
    }
}

void RMSNorm::cleanup() {
    saveInvVariance.clear();
    scratchDScale.clear();
    scratchErrorOutput.reset();
    Layer::cleanup();
}

void RMSNorm::computeFeatureOut(uint32_t connectionNumber) {
    THOR_THROW_IF_FALSE(featureInputs[connectionNumber].has_value());
    THOR_THROW_IF_FALSE(featureOutputs[connectionNumber].has_value());
    Tensor input = featureInputs[connectionNumber].value();
    Tensor output = featureOutputs[connectionNumber].value();

    validateConfiguredInput(input);
    THOR_THROW_IF_FALSE(output.getDescriptor() == input.getDescriptor());
    THOR_THROW_IF_FALSE(output.getPlacement() == input.getPlacement());

    CudnnRmsNormDescriptor descriptor;
    descriptor.outerSize = computeOuterSize(input);
    descriptor.normalizedFeatureCount = normalizedFeatureCount;
    descriptor.inputDataType = input.getDataType();
    descriptor.outputDataType = output.getDataType();
    descriptor.parameterDataType = parameterDataType;
    descriptor.computeDataType = DataType::FP32;
    descriptor.epsilon = static_cast<float>(epsilon);
    descriptor.training = !isInferenceOnly();
    descriptor.fusedActivation = fusedActivation;

    CudnnRmsNormForwardArgs args;
    args.x = input;
    args.scale = weights;
    args.y = output;
    if (!isInferenceOnly()) {
        args.invVariance = saveInvVariance[connectionNumber];
    }

    CudnnRmsNorm::instance().forward(descriptor, args, streams[connectionNumber]);
}

optional<Event> RMSNorm::computeErrorOutAccumulateWeightsGradienFused(uint32_t connectionNumber,
                                                                      bool clearWeightsGradientFirstIfFused) {
    if (!errorInputs[connectionNumber].has_value())
        return nullopt;
    if (isInferenceOnly())
        return nullopt;

    auto weightsParameter = getParameter("weights");
    THOR_THROW_IF_FALSE(weightsParameter->hasOptimizer());

    shared_ptr<Optimizer> weightsOptimizer = weightsParameter->getOptimizer();
    THOR_THROW_IF_FALSE(weightsOptimizer != nullptr);
    THOR_THROW_IF_FALSE(weightsOptimizer->getWeightsGradient().has_value());

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

    const Tensor& input = featureInputs[connectionNumber].value();
    CudnnRmsNormDescriptor descriptor;
    descriptor.outerSize = computeOuterSize(input);
    descriptor.normalizedFeatureCount = normalizedFeatureCount;
    descriptor.inputDataType = input.getDataType();
    descriptor.outputDataType = errorInputs[connectionNumber].value().getDataType();
    descriptor.parameterDataType = parameterDataType;
    descriptor.computeDataType = DataType::FP32;
    descriptor.epsilon = static_cast<float>(epsilon);
    descriptor.training = true;

    CudnnRmsNormBackwardArgs args;
    args.dy = errorInputs[connectionNumber].value();
    args.x = input;
    args.scale = weights;
    args.invVariance = saveInvVariance[connectionNumber];
    args.dx = errorOut.value();
    args.dscale = dscaleOutput;

    CudnnRmsNorm::instance().backward(descriptor, args, gradientUpdateStream.value());

    if (!clearWeightsGradientFirstIfFused) {
        launchAccumulateBatchNormGradientFp32(weightsOptimizer->getWeightsGradient().value().getMemPtr<float>(),
                                             scratchDScale[connectionNumber].getMemPtr<float>(),
                                             normalizedFeatureCount,
                                             gradientUpdateStream.value());
    }

    return gradientUpdateStream.value().putEvent();
}

void RMSNorm::accumulateWeightsGradient(uint32_t connectionNumber, bool clearGradientFirst) {
    (void)connectionNumber;
    (void)clearGradientFirst;
    // No-op: cuDNN Frontend RMSNorm backward produces dx and dscale together.
}

}  // namespace ThorImplementation
