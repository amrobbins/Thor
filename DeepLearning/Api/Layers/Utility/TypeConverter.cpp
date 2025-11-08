#include "DeepLearning/Api/Layers/Utility/TypeConverter.h"
#include "DeepLearning/Api/Network/Network.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

TypeConverter::TypeConverter() = default;
TypeConverter::~TypeConverter() = default;

json TypeConverter::serialize(const string &storageDir, Stream stream) const {
    assert(initialized);
    assert(featureInput.isPresent());
    assert(featureOutput.isPresent());

    json j;
    j["factory"] = Layer::Factory::Layer.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = to_snake_case(getLayerType());

    j["feature_input"] = featureInput.get().serialize();
    j["feature_output"] = featureOutput.get().serialize();

    return j;
}

void TypeConverter::deserialize(const json &j, Network *network) {
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in TypeConverter::deserialize: " + j["version"].get<string>());
    if (j.at("layer_type").get<string>() != "type_converter")
        throw runtime_error("Layer type mismatch in TypeConverter::deserialize: " + j.at("layer_type").get<string>());

    nlohmann::json input = j["feature_input"].get<nlohmann::json>();
    uint64_t originalTensorId = input.at("id").get<uint64_t>();
    Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);

    Tensor featureOutput = Tensor::deserialize(j.at("feature_output").get<nlohmann::json>());

    TypeConverter typeConverter;
    typeConverter.featureInput = featureInput;
    typeConverter.featureOutput = featureOutput;
    typeConverter.initialized = true;
    typeConverter.addToNetwork(network);
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Layer::registry["type_converter"] = &Thor::TypeConverter::deserialize;
    return true;
}();
}
