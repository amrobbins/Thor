#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"

using namespace Thor;
using namespace std;

using json = nlohmann::json;

void FullyConnected::buildSupportLayersAndAddToNetwork() {
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

    // Force the input tensor to this type of layer to be FP16
    if (featureInputs.front().getDataType() != Tensor::DataType::FP16) {
        for (uint32_t i = 0; i < featureInputs.size(); ++i) {
            TypeConverter typeConverter = TypeConverter::Builder()
                                              .network(*network)
                                              .featureInput(currentFeatureInputs[i])
                                              .newDataType(Tensor::DataType::FP16)
                                              .build();
            currentFeatureInputs[i] = typeConverter.getFeatureOutput();
        }
    }

    if (useBatchNormalization) {
        BatchNormalization::Builder batchNormBuilder;
        for (uint32_t i = 0; i < featureInputs.size(); ++i) {
            batchNormBuilder.featureInput(currentFeatureInputs[i]);
        }
        if (batchNormExponentialRunningAverageFactor.isPresent())
            batchNormBuilder.exponentialRunningAverageFactor(batchNormExponentialRunningAverageFactor);
        if (batchNormEpsilon.isPresent())
            batchNormBuilder.epsilon(batchNormEpsilon);
        batchNormalization = batchNormBuilder.network(*network).build();
        currentFeatureInputs = batchNormalization.getFeatureOutputs();
    }

    if (dropProportion > 0.0f) {
        for (uint32_t i = 0; i < featureInputs.size(); ++i) {
            dropOut = DropOut::Builder().network(*network).dropProportion(dropProportion).featureInput(currentFeatureInputs[i]).build();
            currentFeatureInputs[i] = dropOut.getFeatureOutput();
        }
    }

    vector<uint64_t> dimensions = currentFeatureInputs[0].getDimensions();

    FullyConnected::Builder fullyConnectedBuilder;
    fullyConnectedBuilder.network(*network)
        .numOutputFeatures(numOutputFeatures)
        .hasBias(hasBias)
        .weightsInitializerBuilder(*weightsInitializerBuilder)
        .biasInitializerBuilder(*biasInitializerBuilder)
        .noActivation();
    for (uint32_t i = 0; i < featureInputs.size(); ++i)
        fullyConnectedBuilder.featureInput(currentFeatureInputs[i]);
    FullyConnected standAloneFullyConnected = fullyConnectedBuilder.build();
    this->id = standAloneFullyConnected.getId();
    currentFeatureInputs = standAloneFullyConnected.getFeatureOutputs();

    if (activationBuilder) {
        for (uint32_t i = 0; i < featureInputs.size(); ++i) {
            shared_ptr<Activation::Builder> activationBuilderClone = activationBuilder->clone();
            activationBuilderClone->network(*network);
            activationBuilderClone->featureInput(currentFeatureInputs[i]);
            // Since activation may be one of many classes, the base class is built and its virtual build function is used.
            activation = activationBuilderClone->build();
            currentFeatureInputs[i] = activation->getFeatureOutput();
        }
    }

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

// Note: get the physical layer from the network via the function that takes the api layer's id and returns it.
json FullyConnected::serialize(Stream stream) {
    // Multi-layers will only serialize the single layer, itself.
    // The other layers will each serialize themselves when walking the api level layer graph that has been added to the network

    json j;
    j["num_output_features"] = numOutputFeatures;
    j["has_bias"] = hasBias;

    // Input connections
    json inputs = json::array();
    for (uint32_t i = 0; i < featureInputs.size(); ++i) {
        inputs.push_back({{"id", featureInputs[i].getId()},
                          {"dimensions", featureInputs[i].getDimensions()},
                          {"data_type", json(featureInputs[i].getDataType())}});
    }
    j["inputs"] = inputs;

    // Output connections
    json outputs = json::array();
    for (uint32_t i = 0; i < featureOutputs.size(); ++i) {
        outputs.push_back({{"id", featureOutputs[i].getId()},
                           {"dimensions", featureOutputs[i].getDimensions()},
                           {"data_type", json(featureOutputs[i].getDataType())}});
    }
    j["outputs"] = outputs;

    // Dump the weights to a file and record its name
    vector<ThorImplementation::StampedNetwork> stampedNetworks = network->getStampedNetworks();
    assert(!stampedNetworks.empty());
    shared_ptr<ThorImplementation::Layer> physicalLayer = stampedNetworks[0].getPhysicalLayerFromApiLayer(getId());
    shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> twbLayer =
        dynamic_pointer_cast<ThorImplementation::TrainableWeightsBiasesLayer>(physicalLayer);
    assert(twbLayer != nullptr);

    string layerName = string("layer") + to_string(getId());
    string weightsFilename = string("/tmp/") + layerName + "_weights.gds";
    j["weights_tensor"] = weightsFilename;
    twbLayer->dumpWeightsToFile(weightsFilename, stream);
    if (hasBias) {
        string biasesFilename = string("/tmp/") + layerName + "_biases.gds";
        j["biases_tensor"] = biasesFilename;
        twbLayer->dumpBiasesToFile(biasesFilename, stream);
    }

    return j;
}
