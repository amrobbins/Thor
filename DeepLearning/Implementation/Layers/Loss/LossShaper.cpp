#include "LossShaper.h"

using namespace ThorImplementation;
using namespace std;

LossShaper::LossShaper(OutputLossType outputLossType) {
    this->outputLossType = outputLossType;
    setConstructForInferenceOnly(true);
    uninitialized = true;
    batchReduce = nullptr;
}

LossShaper::~LossShaper() {}

Optional<Tensor> LossShaper::createFeatureOutputTensor() {
    assert(featureInput.isPresent());

    vector<unsigned long> inputDimensions = featureInput.get().getDescriptor().getDimensions();
    assert(inputDimensions.size() == 2);
    vector<unsigned long> outputDimensions = getOutputDimensions(inputDimensions, outputLossType);

    Tensor outputTensor;
    if (outputDimensions == inputDimensions)
        outputTensor = featureInput;
    else
        outputTensor = featureInput.get().clone(outputDimensions);
    return outputTensor;
}

void LossShaper::compile() {
    assert(featureInput.isPresent());
    assert(featureOutput.isPresent());

    if (featureOutput.get().getDimensions() == featureInput.get().getDimensions()) {
        // There is no ErrorInput to connect to the previous layer, so this is a nop
    } else {
        uint32_t batchSize = featureInput.get().getDescriptor().getDimensions()[0];
        uint32_t classDimSize = featureInput.get().getDescriptor().getDimensions()[1];
        bool reduceBatchDim = false;
        if ((outputLossType == OutputLossType::BATCH || outputLossType == OutputLossType::CLASSWISE) &&
            featureInput.get().getDimensions()[0] != 1)
            reduceBatchDim = true;
        bool reduceClassDim = false;
        if ((outputLossType == OutputLossType::BATCH || outputLossType == OutputLossType::ELEMENTWISE) &&
            featureInput.get().getDimensions()[1] != 1)
            reduceClassDim = true;

        batchReduce = make_shared<BatchReduce>(batchSize,
                                               batchSize,
                                               classDimSize,
                                               reduceBatchDim,
                                               reduceClassDim,
                                               featureInput.get().getDescriptor().getDataType(),
                                               featureOutput.get().getDescriptor().getDataType(),
                                               stream);
    }

    uninitialized = false;
}

void LossShaper::infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) {
    assert(inputTensor.isPresent());
    assert(outputTensor.isPresent());
    assert(!uninitialized);

    if (featureOutput.get().getDimensions() == featureInput.get().getDimensions()) {
        // Check that the output is properly the same tensor as the input, by checking their ids
        assert(featureOutput.get() == featureInput.get());
    } else {
        // Ensure that batchReduce stream is synchronized properly
        // It is done this way since the cudnnHandle belongs to the stream that batchReduce uses.
        Stream batchReduceStream = batchReduce->getStream();
        if (batchReduceStream != stream)
            batchReduceStream.waitEvent(stream.putEvent());

        TensorDescriptor::DataType lossDataType = inputTensor.get().getDataType();
        assert(lossDataType == TensorDescriptor::DataType::FP16 || lossDataType == TensorDescriptor::DataType::FP32 ||
               lossDataType == TensorDescriptor::DataType::FP64);
        batchReduce->reduce(inputTensor, outputTensor);

        if (batchReduceStream != stream)
            stream.waitEvent(batchReduceStream.putEvent());
    }
}

void LossShaper::backward(Optional<Tensor> errorInput) {}

void LossShaper::backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream) {
    // This should never be called.
    assert(false);
}

vector<uint64_t> LossShaper::getOutputDimensions(vector<uint64_t> inputDimensions, OutputLossType outputLossType) {
    assert(inputDimensions.size() == 2);

    if (outputLossType == OutputLossType::BATCH) {
        // Sum all losses, return a scalar
        return {1, 1};
    } else if (outputLossType == OutputLossType::CLASSWISE) {
        // sum all batch items, return a scalar per output
        return {1, inputDimensions[1]};
    } else if (outputLossType == OutputLossType::ELEMENTWISE) {
        // Sum all outputs for each batch item, return a scalar per batch item
        return {inputDimensions[0], 1};
    } else {
        assert(false);
    }
}

string LossShaper::getType() { return "LossShaper"; }