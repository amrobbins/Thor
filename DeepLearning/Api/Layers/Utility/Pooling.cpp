#include "DeepLearning/Api/Layers/Utility/Pooling.h"
#include "DeepLearning/Api/Network/Network.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

json Pooling::serialize(thor_file::TarWriter &archiveWriter, Stream stream) const {
    assert(initialized);
    assert(featureInput.isPresent());
    assert(featureOutput.isPresent());

    json j;
    j["factory"] = Layer::Factory::Layer.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = to_snake_case(getLayerType());

    j["feature_input"] = featureInput.get().serialize();
    j["feature_output"] = featureOutput.get().serialize();

    j["type"] = type;
    j["window_height"] = windowHeight;
    j["window_width"] = windowWidth;
    j["vertical_stride"] = verticalStride;
    j["horizontal_stride"] = horizontalStride;
    j["vertical_padding"] = verticalPadding;
    j["horizontal_padding"] = horizontalPadding;

    return j;
}

void Pooling::deserialize(const json &j, Network *network) {
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in Pooling::deserialize: " + j["version"].get<string>());
    if (j.at("layer_type").get<string>() != "pooling")
        throw runtime_error("Layer type mismatch in Pooling::deserialize: " + j.at("layer_type").get<string>());

    nlohmann::json input = j["feature_input"].get<nlohmann::json>();
    uint64_t originalTensorId = input.at("id").get<uint64_t>();
    Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);

    Tensor featureOutput = Tensor::deserialize(j.at("feature_output").get<nlohmann::json>());

    Pooling pooling;
    pooling.featureInput = featureInput;
    pooling.featureOutput = featureOutput;
    pooling.type = j.at("type").get<Pooling::Type>();
    pooling.windowHeight = j.at("window_height").get<uint32_t>();
    pooling.windowWidth = j.at("window_width").get<uint32_t>();
    pooling.verticalStride = j.at("vertical_stride").get<uint32_t>();
    pooling.horizontalStride = j.at("horizontal_stride").get<uint32_t>();
    pooling.verticalPadding = j.at("vertical_padding").get<uint32_t>();
    pooling.horizontalPadding = j.at("horizontal_padding").get<uint32_t>();

    pooling.initialized = true;
    pooling.addToNetwork(network);
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Layer::register_layer("pooling", &Thor::Pooling::deserialize);
    return true;
}();
}  // namespace
