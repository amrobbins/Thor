#include "LossShaper.h"

using namespace ThorImplementation;
using namespace std;

LossShaper::LossShaper(InputLossType inputLossType, OutputLossType outputLossType) {
    this->inputLossType = inputLossType;
    this->outputLossType = outputLossType;
    setConstructForInferenceOnly(true);
    uninitialized = true;
    batchReduce = nullptr;
}

LossShaper::~LossShaper() {
    if (batchReduce != nullptr) {
        delete batchReduce;
        batchReduce = nullptr;
    }
}

Optional<Tensor> LossShaper::createFeatureOutputTensor() {
    assert(featureInput.isPresent());

    vector<unsigned long> inputDimensions = featureInput.get().getDescriptor().getDimensions();
    assert(inputDimensions.size() == 2);
    vector<unsigned long> outputDimensions;
    if (outputLossType == OutputLossType::BATCH_LOSS) {
        if (inputLossType == InputLossType::NUMERICAL_LOSS) {
            outputDimensions.push_back(inputDimensions[1]);
        } else if (inputLossType == InputLossType::CATEGORICAL_LOSS) {
            outputDimensions.push_back(1);
        } else {
            assert(false);
        }
    } else if (outputLossType == OutputLossType::CLASSWISE_LOSS) {
        assert(inputLossType == InputLossType::CATEGORICAL_LOSS);
        outputDimensions.push_back(1);
        outputDimensions.push_back(inputDimensions[1]);
    } else {
        assert(false);
    }

    Tensor outputTensor = featureInput.get().clone(outputDimensions);
    return outputTensor;
}

void LossShaper::compile() {
    assert(featureInput.isPresent());
    assert(featureOutput.isPresent());

    bool reduceLossDim = inputLossType == InputLossType::CATEGORICAL_LOSS && outputLossType == OutputLossType::BATCH_LOSS;
    uint32_t batchSize = featureInput.get().getDescriptor().getDimensions()[0];
    uint32_t lossDimSize = featureInput.get().getDescriptor().getDimensions()[1];

    batchReduce = new BatchReduce(batchSize,
                                  batchSize,
                                  lossDimSize,
                                  true,
                                  reduceLossDim,
                                  featureInput.get().getDescriptor().getDataType(),
                                  featureOutput.get().getDescriptor().getDataType(),
                                  stream);

    uninitialized = false;
}

void LossShaper::infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) {
    assert(inputTensor.isPresent());
    assert(outputTensor.isPresent());
    assert(!uninitialized);

    // Ensure that batchReduce stream is synchronized properly
    // It is done this way since the cudnnHandle belongs to the stream that batchReduce uses.
    Stream batchReduceStream = batchReduce->getStream();
    if (batchReduceStream != stream)
        batchReduceStream.waitEvent(stream.putEvent());

    TensorDescriptor::DataType lossDataType = inputTensor.get().getDataType();
    if (lossDataType == TensorDescriptor::DataType::FP16) {
        batchReduce->reduce((half *)inputTensor.get().getMemPtr(), (half *)outputTensor.get().getMemPtr());
    } else if (lossDataType == TensorDescriptor::DataType::FP32) {
        batchReduce->reduce((float *)inputTensor.get().getMemPtr(), (float *)outputTensor.get().getMemPtr());
    } else if (lossDataType == TensorDescriptor::DataType::FP64) {
        batchReduce->reduce((double *)inputTensor.get().getMemPtr(), (double *)outputTensor.get().getMemPtr());
    } else {
        assert(false);
    }

    if (batchReduceStream != stream)
        stream.waitEvent(batchReduceStream.putEvent());
}

void LossShaper::backward(Optional<Tensor> errorInput) {}

void LossShaper::backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream) {
    // This should never be called.
    assert(false);
}
