#include "LossShaper.h"

#include "DeepLearning/Implementation/ThorError.h"

#include <optional>

using namespace ThorImplementation;
using namespace std;

namespace {

uint64_t flattenedLossDim(const vector<uint64_t>& inputDimensions) {
    THOR_THROW_IF_FALSE(inputDimensions.size() >= 2);
    uint64_t result = 1;
    for (size_t i = 1; i < inputDimensions.size(); ++i) {
        THOR_THROW_IF_FALSE(inputDimensions[i] > 0);
        result *= inputDimensions[i];
    }
    return result;
}

vector<uint64_t> flattenedReductionInputDimensions(const vector<uint64_t>& inputDimensions) {
    THOR_THROW_IF_FALSE(inputDimensions.size() >= 2);
    return {inputDimensions.front(), flattenedLossDim(inputDimensions)};
}

vector<uint32_t> reductionAxes(LossShaper::OutputLossType outputLossType) {
    if (outputLossType == LossShaper::OutputLossType::BATCH) {
        return {0, 1};
    }
    if (outputLossType == LossShaper::OutputLossType::CLASSWISE) {
        return {0};
    }
    if (outputLossType == LossShaper::OutputLossType::ELEMENTWISE) {
        return {1};
    }
    THOR_UNREACHABLE();
}

float reductionOutputScale(const vector<uint64_t>& inputDimensions,
                           LossShaper::OutputLossType outputLossType) {
    if (outputLossType == LossShaper::OutputLossType::ELEMENTWISE) {
        return 1.0f;
    }
    return 1.0f / static_cast<float>(inputDimensions.front());
}

}  // namespace

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
        Tensor reductionInput = featureInput.value();
        reductionInput.reshape(flattenedReductionInputDimensions(inputDimensions));
        CubReduction cubReduction(CubReductionOp::Sum,
                                  reductionAxes(outputLossType),
                                  featureOutput.value().getDataType(),
                                  reductionOutputScale(inputDimensions, outputLossType));
        reduction = cubReduction.stamp(reductionInput, featureOutput.value(), stream);
    }

    uninitialized = false;
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
        reduction->runOn(stream);
    }
}

void LossShaper::backward(std::optional<Tensor> errorInput) {}

void LossShaper::backProp(std::optional<Tensor> dataIn, std::optional<Tensor> errorIn, std::optional<Tensor> errorOut, Stream stream) {
    // This should never be called.
    THOR_UNREACHABLE();
}

vector<uint64_t> LossShaper::getOutputDimensions(vector<uint64_t> inputDimensions, OutputLossType outputLossType) {
    THOR_THROW_IF_FALSE(inputDimensions.size() >= 2);
    const uint64_t classDimSize = flattenedLossDim(inputDimensions);

    if (outputLossType == OutputLossType::BATCH) {
        // Sum all non-batch losses and average those per-item sums across the batch.
        return {1, 1};
    } else if (outputLossType == OutputLossType::CLASSWISE) {
        // Average each flattened non-batch loss position across the batch.
        return {1, classDimSize};
    } else if (outputLossType == OutputLossType::ELEMENTWISE) {
        // Sum all flattened non-batch losses independently for each batch item.
        return {inputDimensions[0], 1};
    } else {
        THOR_UNREACHABLE();
    }
}

string LossShaper::getType() { return "LossShaper"; }
