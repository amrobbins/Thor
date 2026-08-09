#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Layers/Utility/BatchNormalization.h"

#include <stdexcept>

using namespace std;
using json = nlohmann::json;

namespace Thor {

json BatchNormalization::architectureJson() const {
    json j;
    j["factory"] = Layer::Factory::Learning.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = "batch_normalization";
    j["layer_name"] = string("layer") + to_string(getId());
    j["exponential_running_average_factor"] = exponentialRunningAverageFactor;
    j["epsilon"] = epsilon;
    j["num_items_observed"] = numItemsObserved;

    json inputs = json::array();
    for (const Tensor& featureInput : featureInputs) {
        inputs.push_back(featureInput.architectureJson());
    }
    j["inputs"] = inputs;

    json outputs = json::array();
    for (const Tensor& featureOutput : featureOutputs) {
        outputs.push_back(featureOutput.architectureJson());
    }
    j["outputs"] = outputs;

    j["parameters"] = getParametersArchitectureJson()["parameters"];
    return j;
}

json BatchNormalization::serialize(thor_file::TarWriter& archiveWriter,
                                   Stream stream,
                                   bool saveOptimizerState,
                                   ThorImplementation::StampedNetwork& stampedNetwork) const {
    json j = architectureJson();

    shared_ptr<ThorImplementation::Layer> physicalLayer = stampedNetwork.getPhysicalLayerFromApiLayer(getId());
    shared_ptr<ThorImplementation::BatchNormalization> batchNorm =
        dynamic_pointer_cast<ThorImplementation::BatchNormalization>(physicalLayer);
    THOR_THROW_IF_FALSE(batchNorm != nullptr);

    j["num_items_observed"] = batchNorm->getNumItemsObserved();
    Parameterizable::serializeParameters(
        j["parameters"], archiveWriter, stream, saveOptimizerState, stampedNetwork, string("layer") + to_string(getId()));
    return j;
}

void BatchNormalization::deserialize(shared_ptr<thor_file::TarReader>& archiveReader, const json& j, Network* network) {
    if (j.at("version").get<string>() != "1.0.0") {
        throw runtime_error("Unsupported version in BatchNormalization::deserialize: " + j.at("version").get<string>());
    }
    if (j.at("layer_type").get<string>() != "batch_normalization") {
        throw runtime_error("Layer type mismatch in BatchNormalization::deserialize: " + j.at("layer_type").get<string>());
    }

    BatchNormalization layer;
    layer.exponentialRunningAverageFactor = j.at("exponential_running_average_factor").get<double>();
    layer.epsilon = j.at("epsilon").get<double>();
    layer.numItemsObserved = j.at("num_items_observed").get<uint64_t>();

    for (const json& inputJson : j.at("inputs")) {
        const uint64_t originalTensorId = inputJson.at("id").get<uint64_t>();
        layer.featureInputs.push_back(network->getApiTensorByOriginalId(originalTensorId));
    }
    for (const json& outputJson : j.at("outputs")) {
        layer.featureOutputs.push_back(Tensor::deserialize(outputJson, archiveReader.get()));
    }
    if (layer.featureInputs.size() != layer.featureOutputs.size()) {
        throw runtime_error("BatchNormalization deserialize expected equal numbers of inputs and outputs.");
    }
    for (uint32_t i = 0; i < layer.featureInputs.size(); ++i) {
        layer.outputTensorFromInputTensor[layer.featureInputs[i]] = layer.featureOutputs[i];
        layer.inputTensorFromOutputTensor[layer.featureOutputs[i]] = layer.featureInputs[i];
    }

    if (!j.contains("parameters") || !j.at("parameters").is_object()) {
        throw runtime_error("BatchNormalization parameters must be an object keyed by parameter name.");
    }
    const json& parametersJson = j.at("parameters");
    static const vector<string> requiredParameters = {"weights", "biases", "running_mean", "running_variance"};
    if (parametersJson.size() != requiredParameters.size()) {
        throw runtime_error("BatchNormalization deserialize expected exactly weights, biases, running_mean, and running_variance parameters.");
    }
    for (const string& parameterName : requiredParameters) {
        if (!parametersJson.contains(parameterName)) {
            throw runtime_error("BatchNormalization deserialize did not find required parameter '" + parameterName + "'.");
        }
        ParameterSpecification parameter = ParameterSpecification::deserialize(parametersJson.at(parameterName), archiveReader);
        if (parameter.getName() != parameterName) {
            throw runtime_error("BatchNormalization parameter key/name mismatch for '" + parameterName + "'.");
        }
        layer.addParameter(make_shared<ParameterSpecification>(std::move(parameter)));
    }

    layer.initialized = true;
    layer.addToNetwork(network);
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::TrainableLayer::register_layer("batch_normalization", &Thor::BatchNormalization::deserialize);
    return true;
}();
}  // namespace
