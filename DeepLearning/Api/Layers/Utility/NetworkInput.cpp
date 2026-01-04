#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

json NetworkInput::serialize(thor_file::TarWriter &archiveWriter, Stream stream) const {
    return json{{"factory", Layer::Factory::Layer.value()},
                {"version", "1.0.0"},
                {"layer_type", "network_input"},
                {"name", name},
                {"dimensions", getDimensions()},
                {"data_type", json(getDataType())},
                {"feature_input", featureInput.get().serialize()},
                {"feature_output", featureOutput.get().serialize()}};
}

void NetworkInput::deserialize(const json &j, Network *network) {
    string name = j.at("name").get<string>();
    vector<uint64_t> dimensions = j.at("dimensions").get<vector<uint64_t>>();
    Tensor::DataType dataType = j.at("data_type").get<Tensor::DataType>();

    NetworkInput networkInput;
    networkInput.name = name;
    networkInput.dimensions = dimensions;
    networkInput.dataType = dataType;
    networkInput.featureInput = Tensor::deserialize(j["feature_input"]);
    networkInput.featureOutput = Tensor::deserialize(j["feature_output"]);
    networkInput.initialized = true;
    networkInput.addToNetwork(network);
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Layer::register_layer("network_input", &Thor::NetworkInput::deserialize);
    return true;
}();
}  // namespace
