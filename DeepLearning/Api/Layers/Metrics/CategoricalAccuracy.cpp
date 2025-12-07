#include "DeepLearning/Api/Layers/Metrics/CategoricalAccuracy.h"
#include "DeepLearning/Api/Network/Network.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

json CategoricalAccuracy::serialize(const string &storageDir, Stream stream) const {
    json j = Metric::serialize(storageDir, stream);
    j["label_type"] = labelType;
    return j;
}

void CategoricalAccuracy::deserialize(const json &j, Network *network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in CategoricalAccuracy::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "categorical_accuracy")
        throw runtime_error("Layer type mismatch in CategoricalAccuracy::deserialize: " + j.at("layer_type").get<std::string>());

    nlohmann::json input = j["predictions"].get<nlohmann::json>();
    uint64_t originalTensorId = input.at("id").get<uint64_t>();
    Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);

    nlohmann::json labels = j["labels"].get<nlohmann::json>();
    originalTensorId = labels.at("id").get<uint64_t>();
    Tensor labelsTensor = network->getApiTensorByOriginalId(originalTensorId);

    Tensor metricTensor = Tensor::deserialize(j.at("metric").get<nlohmann::json>());

    LabelType labelType = j.at("label_type").get<LabelType>();

    CategoricalAccuracy categoricalAccuracy;
    categoricalAccuracy.featureInput = featureInput;
    categoricalAccuracy.labelsTensor = labelsTensor;
    categoricalAccuracy.metricTensor = metricTensor;
    categoricalAccuracy.labelType = labelType;
    categoricalAccuracy.initialized = true;
    categoricalAccuracy.addToNetwork(network);
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::Metric::register_layer("categorical_accuracy", &Thor::CategoricalAccuracy::deserialize);
    return true;
}();
}  // namespace
