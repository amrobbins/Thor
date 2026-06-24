#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Network/Network.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

NetworkInput NetworkInput::Builder::build() {
    THOR_THROW_IF_FALSE(_network.has_value());
    THOR_THROW_IF_FALSE(_dimensions.has_value() || _passThroughSource.has_value());
    THOR_THROW_IF_FALSE(_dataType.has_value() || _passThroughSource.has_value());

    NetworkInput networkInput;
    if (_name.has_value())
        networkInput.name = _name.value();
    else
        networkInput.name = std::string("NetworkInput") + std::to_string(networkInput.getId());

    if (_passThroughSource.has_value()) {
        const Tensor& requestedSourceTensor = _passThroughSource.value();
        Tensor sourceTensor = _network.value()->resolveApiTensorByOriginalId(requestedSourceTensor.getOriginalId());
        if (_dimensions.has_value()) {
            THOR_THROW_IF_FALSE(_dimensions.value() == sourceTensor.getDimensions());
        }
        if (_dataType.has_value()) {
            THOR_THROW_IF_FALSE(_dataType.value() == sourceTensor.getDataType());
        }
        networkInput.dimensions = sourceTensor.getDimensions();
        networkInput.dataType = sourceTensor.getDataType();
        networkInput.external_ = false;
        networkInput.passThroughSource_ = sourceTensor;
        networkInput.featureInput = sourceTensor;
        networkInput.featureOutput = sourceTensor;
    } else {
        networkInput.dimensions = _dimensions.value();
        networkInput.dataType = _dataType.value();
        networkInput.external_ = _external;
        networkInput.featureInput = Tensor(_dataType.value(), _dimensions.value());
        networkInput.featureOutput = Tensor(_dataType.value(), _dimensions.value());
    }
    networkInput.dimensionsIncludeBatch_ = _dimensionsIncludeBatch;
    networkInput.initialized = true;
    networkInput.addToNetwork(_network.value());
    return networkInput;
}

json NetworkInput::architectureJson() const {
    json result{{"factory", Layer::Factory::Layer.value()},
                {"version", "1.0.0"},
                {"layer_type", "network_input"},
                {"name", name},
                {"dimensions", getDimensions()},
                {"dimensions_include_batch", dimensionsIncludeBatch()},
                {"external", isExternal()},
                {"data_type", json(getDataType())},
                {"feature_input", featureInput.value().architectureJson()},
                {"feature_output", featureOutput.value().architectureJson()}};
    if (passThroughSource_.has_value()) {
        result["pass_through_source"] = passThroughSource_.value().architectureJson();
    }
    return result;
}

void NetworkInput::deserialize(const json &j, Network *network) {
    string name = j.at("name").get<string>();
    vector<uint64_t> dimensions = j.at("dimensions").get<vector<uint64_t>>();
    DataType dataType = j.at("data_type").get<DataType>();
    bool dimensionsIncludeBatch = j.value("dimensions_include_batch", false);
    bool external = j.value("external", true);

    NetworkInput networkInput;
    networkInput.name = name;
    networkInput.dimensions = dimensions;
    networkInput.dataType = dataType;
    networkInput.dimensionsIncludeBatch_ = dimensionsIncludeBatch;
    if (j.contains("pass_through_source")) {
        // API pass-through NetworkInputs are aliases.  Restore the alias to the
        // canonical source tensor that has already been imported into this
        // network, rather than deserializing a second live Tensor handle for the
        // same serialized tensor id.  Keeping only the canonical tensor ensures
        // this layer is not a graph source and cannot stamp as an external input.
        const uint64_t sourceOriginalId = j.at("pass_through_source").at("id").get<uint64_t>();
        Tensor sourceTensor = network->resolveApiTensorByOriginalId(sourceOriginalId);
        THOR_THROW_IF_FALSE(sourceTensor.getDimensions() == dimensions);
        THOR_THROW_IF_FALSE(sourceTensor.getDataType() == dataType);
        networkInput.external_ = false;
        networkInput.passThroughSource_ = sourceTensor;
        networkInput.featureInput = sourceTensor;
        networkInput.featureOutput = sourceTensor;
    } else {
        networkInput.external_ = external;
        networkInput.featureInput = Tensor::deserialize(j["feature_input"]);
        networkInput.featureOutput = Tensor::deserialize(j["feature_output"]);
    }
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
