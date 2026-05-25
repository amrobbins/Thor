#include "DeepLearning/Api/Layers/Utility/Transpose.h"

#include "DeepLearning/Api/Network/Network.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

using namespace std;
using json = nlohmann::json;

namespace Thor {

Transpose::Transpose() = default;
Transpose::Transpose(std::optional<ThorImplementation::Expression> epilogue) : epilogue(std::move(epilogue)) {}
Transpose::~Transpose() = default;

json Transpose::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());

    json j;
    j["factory"] = Layer::Factory::Layer.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = to_snake_case(getLayerType());

    j["output_data_type"] = outputDataType;
    if (epilogue.has_value()) {
        if (!serializableEpilogue.has_value())
            serializableEpilogue = makeEpilogueDefinition(epilogue.value());
        j["epilogue"] = serializableEpilogue.value().architectureJson();
    } else {
        j["epilogue"] = nullptr;
    }

    j["feature_input"] = featureInput.value().architectureJson();
    j["feature_output"] = featureOutput.value().architectureJson();

    return j;
}

void Transpose::deserialize(const json &j, Network *network) {
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in Transpose::deserialize: " + j["version"].get<string>());
    if (j.at("layer_type").get<string>() != "transpose")
        throw runtime_error("Layer type mismatch in Transpose::deserialize: " + j.at("layer_type").get<string>());

    std::optional<ThorImplementation::Expression> epilogue = std::nullopt;
    if (j.contains("epilogue") && !j.at("epilogue").is_null()) {
        ThorImplementation::ExpressionDefinition epilogueDefinition =
            ThorImplementation::ExpressionDefinition::deserialize(j.at("epilogue"));
        epilogue = epilogueExpressionFromDefinition(epilogueDefinition);
    }

    nlohmann::json input = j["feature_input"].get<nlohmann::json>();
    uint64_t originalTensorId = input.at("id").get<uint64_t>();
    Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);

    Tensor featureOutput = Tensor::deserialize(j.at("feature_output").get<nlohmann::json>());
    DataType outputDataType = j.contains("output_data_type") ? j.at("output_data_type").get<DataType>()
                                                                     : featureOutput.getDataType();

    std::vector<uint64_t> expectedOutputDimensions = featureInput.getDimensions();
    if (expectedOutputDimensions.size() < 2) {
        throw runtime_error("Transpose::deserialize requires feature input rank >= 2.");
    }
    std::swap(expectedOutputDimensions[expectedOutputDimensions.size() - 2], expectedOutputDimensions[expectedOutputDimensions.size() - 1]);
    if (featureOutput.getDimensions() != expectedOutputDimensions) {
        throw runtime_error("Transpose::deserialize feature_output dimensions do not match feature_input trailing-dimension transpose.");
    }
    if (featureOutput.getDataType() != outputDataType) {
        throw runtime_error("Transpose::deserialize feature_output dtype must match output_data_type.");
    }

    Transpose transpose(epilogue);
    transpose.outputDataType = outputDataType;
    transpose.featureInput = featureInput;
    transpose.featureOutput = featureOutput;
    transpose.initialized = true;
    transpose.addToNetwork(network);
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Layer::register_layer("transpose", &Thor::Transpose::deserialize);
    return true;
}();
}  // namespace
