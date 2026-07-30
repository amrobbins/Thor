#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Layers/Utility/Stub.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

void Loss::finalizeLossReporting() {
    THOR_THROW_IF_FALSE(network != nullptr);
    THOR_THROW_IF_FALSE(lossShaperInput.isInitialized());

    if (lossShape == LossShape::NONE) {
        // The raw loss remains the training root, but NONE intentionally exposes no
        // user-facing report tensor. Stub the forward output so graph validation does
        // not mistake the backward root for an accidental dangling output.
        lossTensor = lossShaperInput;
        Stub::Builder().network(*network).inputTensor(lossShaperInput).build();
    } else if (lossShape == LossShape::BATCH) {
        LossShaper lossShaper =
            LossShaper::Builder().network(*network).lossInput(lossShaperInput).reportsBatchLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else if (lossShape == LossShape::PER_EXAMPLE) {
        LossShaper lossShaper =
            LossShaper::Builder().network(*network).lossInput(lossShaperInput).reportsPerExampleLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else if (lossShape == LossShape::PER_OUTPUT) {
        LossShaper lossShaper =
            LossShaper::Builder().network(*network).lossInput(lossShaperInput).reportsPerOutputLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else {
        THOR_THROW_IF_FALSE(lossShape == LossShape::RAW);
        lossTensor = lossShaperInput;
    }
}

json Loss::architectureJson() const {
    json j;
    j["factory"] = Factory::Loss.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = to_snake_case(getLayerType());
    string layerName = string("layer") + to_string(getId());
    j["layer_name"] = layerName;
    j["loss_shape"] = LossShape::RAW;
    j["loss_data_type"] = lossDataType;
    ThorImplementation::addLossWeightToJson(j, lossWeight);
    j["labels_tensor"] = labelsTensor.architectureJson();
    j["predictions_tensor"] = predictionsTensor.architectureJson();
    if (exampleWeightsTensor.has_value())
        j["example_weights_tensor"] = exampleWeightsTensor.value().architectureJson();
    j["loss_shaper_input_tensor"] = lossShaperInput.architectureJson();
    j["loss_tensor"] = lossTensor.architectureJson();

    return j;
}

unordered_map<string, Loss::Deserializer> &Loss::get_registry() {
    static unordered_map<string, Deserializer> registry;
    return registry;
}

void Loss::register_layer(string name, Deserializer fn) { get_registry().emplace(move(name), move(fn)); }

void Loss::deserialize(const nlohmann::json &j, Network *network) {
    THOR_THROW_IF_FALSE(j.at("factory").get<std::string>() == Layer::Factory::Loss.value());
    std::string type = j.at("layer_type").get<std::string>();

    unordered_map<string, Loss::Deserializer> &registry = get_registry();
    auto it = registry.find(type);
    if (it == registry.end())
        throw std::runtime_error("Unknown activation type: " + type);

    auto deserializer = it->second;
    deserializer(j, network);
}

}  // namespace Thor
