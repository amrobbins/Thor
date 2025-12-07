#include "DeepLearning/Api/Layers/Metrics/BinaryAccuracy.h"
#include "DeepLearning/Api/Network/Network.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

void BinaryAccuracy::deserialize(const json &j, Network *network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in BinaryAccuracy::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "binary_accuracy")
        throw runtime_error("Layer type mismatch in BinaryAccuracy::deserialize: " + j.at("layer_type").get<std::string>());

    nlohmann::json input = j["predictions"].get<nlohmann::json>();
    uint64_t originalTensorId = input.at("id").get<uint64_t>();
    Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);

    nlohmann::json labels = j["labels"].get<nlohmann::json>();
    originalTensorId = labels.at("id").get<uint64_t>();
    Tensor labelsTensor = network->getApiTensorByOriginalId(originalTensorId);

    Tensor metricTensor = Tensor::deserialize(j.at("metric").get<nlohmann::json>());

    BinaryAccuracy binaryAccuracy;
    binaryAccuracy.featureInput = featureInput;
    binaryAccuracy.labelsTensor = labelsTensor;
    binaryAccuracy.metricTensor = metricTensor;
    binaryAccuracy.initialized = true;
    binaryAccuracy.addToNetwork(network);
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::Metric::register_layer("binary_accuracy", &Thor::BinaryAccuracy::deserialize);
    return true;
}();
}  // namespace
