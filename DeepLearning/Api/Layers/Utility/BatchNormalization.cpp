#include "DeepLearning/Api/Layers/Utility/BatchNormalization.h"

using namespace Thor;
using namespace std;

void BatchNormalization::convertToSingleLayersAndAddToNetwork() {
    assert(isMultiLayer());

    vector<Tensor> currentFeatureInputs;

    for (uint32_t i = 0; i < featureInputs.size(); ++i)
        currentFeatureInputs.push_back(featureInputs[i]);

    // Force the input tensor to this type of layer to be FP16
    if (featureInputs.front().getDataType() != Tensor::DataType::FP16) {
        printf("converting to FP16");
        fflush(stdout);
        for (uint32_t i = 0; i < featureInputs.size(); ++i) {
            TypeConverter typeConverter = TypeConverter::Builder()
                                              .network(*network)
                                              .featureInput(currentFeatureInputs[i])
                                              .newDataType(Tensor::DataType::FP16)
                                              .build();
            currentFeatureInputs[i] = typeConverter.getFeatureOutput();
        }
    } else {
        printf("not converting\n");
        fflush(stdout);
    }

    BatchNormalization::Builder batchNormBuilder;
    for (uint32_t i = 0; i < featureInputs.size(); ++i)
        batchNormBuilder.featureInput(currentFeatureInputs[i]);
    if (exponentialRunningAverageFactor.isPresent())
        batchNormBuilder.exponentialRunningAverageFactor(exponentialRunningAverageFactor);
    if (epsilon.isPresent())
        batchNormBuilder.epsilon(epsilon);
    BatchNormalization standaloneBatchNormalization = batchNormBuilder.network(*network).build();
    currentFeatureInputs = standaloneBatchNormalization.getFeatureOutputs();

    vector<uint64_t> dimensions = currentFeatureInputs[0].getDimensions();

    // Replace the outputs on the compound layer to be the outputs of the last stage
    // i.e. tunnel the actual inputs to actual outputs of the compound layer,
    // these are not necessarily the outputs of the stand-alone fully connected layer.
    // Network uses single layers, user uses compound layer.
    outputTensorFromInputTensor.clear();
    inputTensorFromOutputTensor.clear();
    featureOutputs = currentFeatureInputs;
    for (uint32_t i = 0; i < featureInputs.size(); ++i) {
        outputTensorFromInputTensor[featureInputs[i]] = featureOutputs[i];
        inputTensorFromOutputTensor[featureOutputs[i]] = featureInputs[i];
    }
}
