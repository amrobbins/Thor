#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/TypeConverter.h"

using namespace Thor;

using namespace std;

void NetworkOutput::buildSupportLayersAndAddToNetwork() {
    Tensor currentFeatureInput = featureInput.get();

    // Force the input tensor to this type of layer to be FP16
    if (featureInput.get().getDataType() != dataType) {
        currentFeatureInput =
            TypeConverter::Builder().network(*network).featureInput(currentFeatureInput).newDataType(dataType).build().getFeatureOutput();
    }

    currentFeatureInput = NetworkOutput::Builder()
                              .network(*network)
                              .name(name)
                              .inputTensor(currentFeatureInput)
                              .dataType(dataType)
                              .build()
                              .getFeatureOutput();

    // Replace the output on the compound layer to be the outputs of the last stage
    // i.e. tunnel the actual inputs to actual output of the compound layer,
    // Network uses single layers, user uses compound layer.
    featureOutput = currentFeatureInput;
}

nlohmann::json NetworkOutput::serialize(const std::string &storageDir, Stream stream) const {
    return nlohmann::json{{"version", "1.0.0"},
                          {"layer_type", "network_output"},
                          {"name", name},
                          {"data_type", nlohmann::json(getDataType())},
                          {"feature_input", featureInput.get().serialize()},
                          {"feature_output", featureOutput.get().serialize()}};
}

void NetworkOutput::deserialize(const nlohmann::json &j, Network *network) {
    std::string name = j.at("name").get<std::string>();
    Tensor::DataType dataType = j.at("data_type").get<Tensor::DataType>();

    NetworkOutput networkOutput;
    networkOutput.name = name;
    networkOutput.dataType = dataType;
    uint64_t originalTensorId = j["feature_input"].at("id").get<uint64_t>();
    networkOutput.featureInput = network->getApiTensorByOriginalId(originalTensorId);
    networkOutput.featureOutput = Tensor::deserialize(j["feature_output"]);
    networkOutput.initialized = true;
    networkOutput.addToNetwork(network);
}
