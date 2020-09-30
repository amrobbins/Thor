#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"

using namespace Thor;

void FullyConnected::convertToSingleLayersAndAddToNetwork() {
    assert(isMultiLayer());

    BatchNormalization::Builder batchNormBuilder;
    if (useBatchNormalization) {
        batchNormBuilder.network(*network);
        if (batchNormExponentialRunningAverageFactor.isPresent())
            batchNormBuilder.exponentialRunningAverageFactor(batchNormExponentialRunningAverageFactor.get());
        if (batchNormEpsilon.isPresent())
            batchNormBuilder.epsilon(batchNormEpsilon.get());
    }

    FullyConnected::Builder fullyConnectedBuilder;
    fullyConnectedBuilder.network(*network)
        .numOutputFeatures(numOutputFeatures)
        .hasBias(hasBias)
        .weightsInitializer(weightsInitializer)
        .biasInitializer(biasInitializer)
        .noActivation();

    vector<Tensor> currentFeatureInputs;

    for (uint32_t i = 0; i < featureInputs.size(); ++i)
        currentFeatureInputs.push_back(featureInputs[i]);

    // Flatten to 2 dimensions {batchSize, numInputFeatures} if not already a 2d tensor.
    vector<uint64_t> featureInputDimensions = featureInputs.front().getDimensions();
    assert(featureInputDimensions.size() >= 1);
    if (featureInputDimensions.size() > 1) {
        for (uint32_t i = 0; i < featureInputs.size(); ++i) {
            Flatten flatten = Flatten::Builder().network(*network).featureInput(currentFeatureInputs[i]).numOutputDimensions(1).build();
            currentFeatureInputs[i] = flatten.getFeatureOutput();
        }
    }

    if (useBatchNormalization) {
        for (uint32_t i = 0; i < featureInputs.size(); ++i) {
            batchNormBuilder.featureInput(currentFeatureInputs[i]);
        }
        BatchNormalization batchNormalization = batchNormBuilder.build();
        currentFeatureInputs = batchNormalization.getFeatureOutputs();
    }

    if (dropProportion > 0.0f) {
        for (uint32_t i = 0; i < featureInputs.size(); ++i) {
            DropOut dropOut =
                DropOut::Builder().network(*network).dropProportion(dropProportion).featureInput(currentFeatureInputs[i]).build();
            currentFeatureInputs[i] = dropOut.getFeatureOutput();
        }
    }

    for (uint32_t i = 0; i < featureInputs.size(); ++i)
        fullyConnectedBuilder.featureInput(currentFeatureInputs[i]);
    FullyConnected fullyConnected = fullyConnectedBuilder.build();
    currentFeatureInputs = fullyConnected.getFeatureOutputs();

    if (activationBuilder) {
        for (uint32_t i = 0; i < featureInputs.size(); ++i) {
            activationBuilder->network(*network);
            activationBuilder->featureInput(currentFeatureInputs[i]);
            shared_ptr<Layer> activation = activationBuilder->build();
            currentFeatureInputs[i] = activation->getFeatureOutput();
        }
    }

    // Replace the outputs on the compound layer to be the outputs of the last stage
    for (uint32_t i = 0; i < featureInputs.size(); ++i) {
        fullyConnected.featureOutputs[i] = currentFeatureInputs[i];
        outputTensorFromInputTensor[featureInputs[i]] = featureOutputs[i];
        inputTensorFromOutputTensor[featureOutputs[i]] = featureInputs[i];
    }
}
