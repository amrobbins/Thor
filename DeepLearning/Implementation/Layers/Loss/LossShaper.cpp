#include <optional>
#include <limits>
#include "LossShaper.h"

#include "DeepLearning/Implementation/ThorError.h"
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

}  // namespace

LossShaper::LossShaper(OutputLossType outputLossType) {
    this->outputLossType = outputLossType;
    setConstructForInferenceOnly(true);
    uninitialized = true;
    batchReduce = nullptr;
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
        vector<unsigned long> inputDimensions = featureInput.value().getDescriptor().getDimensions();
        THOR_THROW_IF_FALSE(inputDimensions[0] <= std::numeric_limits<uint32_t>::max());
        uint32_t batchSize = static_cast<uint32_t>(inputDimensions[0]);
        uint64_t flattenedClassDimSize64 = flattenedLossDim(inputDimensions);
        THOR_THROW_IF_FALSE(flattenedClassDimSize64 <= std::numeric_limits<uint32_t>::max());
        uint32_t classDimSize = static_cast<uint32_t>(flattenedClassDimSize64);
        bool reduceBatchDim = false;
        if ((outputLossType == OutputLossType::BATCH || outputLossType == OutputLossType::CLASSWISE) &&
            featureInput.value().getDimensions()[0] != 1)
            reduceBatchDim = true;
        bool reduceClassDim = false;
        if ((outputLossType == OutputLossType::BATCH || outputLossType == OutputLossType::ELEMENTWISE) && classDimSize != 1)
            reduceClassDim = true;

        batchReduce = make_shared<BatchReduce>(batchSize,
                                               batchSize,
                                               classDimSize,
                                               reduceBatchDim,
                                               reduceClassDim,
                                               featureInput.value().getDescriptor().getDataType(),
                                               featureOutput.value().getDescriptor().getDataType(),
                                               stream);
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
        // Ensure that batchReduce stream is synchronized properly
        // It is done this way since the cudnnHandle belongs to the stream that batchReduce uses.
        Stream batchReduceStream = batchReduce->getStream();
        if (batchReduceStream != stream)
            batchReduceStream.waitEvent(stream.putEvent());

        DataType lossDataType = inputTensor.value().getDataType();
        THOR_THROW_IF_FALSE(lossDataType == DataType::FP16 || lossDataType == DataType::FP32 ||
               lossDataType == DataType::FP64);
        batchReduce->reduce(inputTensor.value(), outputTensor.value());

        if (batchReduceStream != stream)
            stream.waitEvent(batchReduceStream.putEvent());
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
        // Sum all losses, return a scalar.
        return {1, 1};
    } else if (outputLossType == OutputLossType::CLASSWISE) {
        // Sum all batch items, return a scalar per flattened non-batch output.
        return {1, classDimSize};
    } else if (outputLossType == OutputLossType::ELEMENTWISE) {
        // Sum all non-batch outputs for each batch item, return a scalar per batch item.
        return {inputDimensions[0], 1};
    } else {
        THOR_UNREACHABLE();
    }
}

string LossShaper::getType() { return "LossShaper"; }
