#include "LossShaper.h"

#include "DeepLearning/Implementation/ThorError.h"

#include <limits>
#include <optional>

using namespace ThorImplementation;
using namespace std;

vector<uint32_t> LossShaper::getReductionAxes(const vector<uint64_t>& inputDimensions,
                                               OutputLossType outputLossType) {
    THOR_THROW_IF_FALSE(inputDimensions.size() >= 2);

    if (outputLossType == OutputLossType::BATCH) {
        vector<uint32_t> axes(inputDimensions.size());
        for (uint32_t axis = 0; axis < axes.size(); ++axis)
            axes[axis] = axis;
        return axes;
    }
    if (outputLossType == OutputLossType::PER_OUTPUT)
        return {0};
    if (outputLossType == OutputLossType::PER_EXAMPLE) {
        vector<uint32_t> axes(inputDimensions.size() - 1);
        for (uint32_t axis = 1; axis < inputDimensions.size(); ++axis)
            axes[axis - 1] = axis;
        return axes;
    }
    THOR_UNREACHABLE();
}

LossShaper::LossShaper(OutputLossType outputLossType) {
    this->outputLossType = outputLossType;
    setConstructForInferenceOnly(true);
    uninitialized = true;
    reduction = nullptr;
}

LossShaper::~LossShaper() {}

std::optional<Tensor> LossShaper::createFeatureOutputTensor() {
    THOR_THROW_IF_FALSE(featureInput.has_value());

    vector<unsigned long> inputDimensions = featureInput.value().getDescriptor().getDimensions();
    THOR_THROW_IF_FALSE(inputDimensions.size() >= 2);
    vector<unsigned long> outputDimensions = getOutputDimensions(inputDimensions, outputLossType);

    Tensor outputTensor;
    if (outputDimensions == inputDimensions)
        outputTensor = featureInput.value();
    else
        outputTensor = featureInput.value().clone(outputDimensions);
    return outputTensor;
}

void LossShaper::compileImpl() {
    Layer::compileImpl();
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());

    if (featureOutput.value().getDimensions() == featureInput.value().getDimensions()) {
        // There is no ErrorInput to connect to the previous layer, so this is a nop
    } else {
        const vector<uint64_t> inputDimensions = featureInput.value().getDimensions();
        CubReduction cubReduction(CubReductionOp::Sum,
                                  getReductionAxes(inputDimensions, outputLossType),
                                  featureOutput.value().getDataType(),
                                  1.0f);
        reduction = cubReduction.stamp(featureInput.value(), featureOutput.value(), stream);
    }

    uninitialized = false;
}

void LossShaper::forward(std::optional<Tensor> inputTensor, bool validationPass, uint32_t validExampleCount) {
    THOR_THROW_IF_FALSE(featureInput.has_value());
    const vector<uint64_t> inputDimensions = featureInput.value().getDimensions();
    THOR_THROW_IF_FALSE(!inputDimensions.empty());
    THOR_THROW_IF_FALSE(inputDimensions.front() > 0);
    THOR_THROW_IF_FALSE(inputDimensions.front() <= std::numeric_limits<uint32_t>::max());
    const uint32_t physicalBatchCapacity = static_cast<uint32_t>(inputDimensions.front());
    currentValidExampleCount = validExampleCount == 0 ? physicalBatchCapacity : validExampleCount;
    THOR_THROW_IF_FALSE(currentValidExampleCount >= 1);
    THOR_THROW_IF_FALSE(currentValidExampleCount <= physicalBatchCapacity);
    Layer::forward(inputTensor, validationPass, currentValidExampleCount);
}

void LossShaper::infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream) {
    THOR_THROW_IF_FALSE(inputTensor.has_value());
    THOR_THROW_IF_FALSE(outputTensor.has_value());
    THOR_THROW_IF_FALSE(!uninitialized);

    if (featureOutput.value().getDimensions() == featureInput.value().getDimensions()) {
        // Check that the output is properly the same tensor as the input, by checking their ids
        THOR_THROW_IF_FALSE(featureOutput.value() == featureInput.value());
    } else {
        THOR_THROW_IF_FALSE(reduction != nullptr);
        THOR_THROW_IF_FALSE(inputTensor.value() == featureInput.value());
        THOR_THROW_IF_FALSE(outputTensor.value() == featureOutput.value());
        const float outputScale = outputLossType == OutputLossType::PER_EXAMPLE
                                      ? 1.0f
                                      : 1.0f / static_cast<float>(currentValidExampleCount);
        reduction->runOn(stream, outputScale);
    }
}

void LossShaper::backward(std::optional<Tensor> errorInput) {}

void LossShaper::backProp(std::optional<Tensor> dataIn, std::optional<Tensor> errorIn, std::optional<Tensor> errorOut, Stream stream) {
    // This should never be called.
    THOR_UNREACHABLE();
}

vector<uint64_t> LossShaper::getOutputDimensions(vector<uint64_t> inputDimensions, OutputLossType outputLossType) {
    THOR_THROW_IF_FALSE(inputDimensions.size() >= 2);

    if (outputLossType == OutputLossType::BATCH) {
        // Sum all non-batch losses and average those per-example sums across the batch.
        return {1, 1};
    } else if (outputLossType == OutputLossType::PER_OUTPUT) {
        // Average across the batch while preserving the complete per-example loss layout.
        inputDimensions[0] = 1;
        return inputDimensions;
    } else if (outputLossType == OutputLossType::PER_EXAMPLE) {
        // Sum all non-batch losses independently for each batch item.
        return {inputDimensions[0], 1};
    } else {
        THOR_UNREACHABLE();
    }
}

string LossShaper::getType() { return "LossShaper"; }
