#include "DeepLearning/Implementation/ThorError.h"
#include "MeanAbsolutePercentageError.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {
void MAPE::buildSupportLayersAndAddToNetwork() {
    MAPE meanAbsolutePercentageError = MAPE::Builder()
                                                                  .network(*network)
                                                                  .predictions(predictionsTensor)
                                                                  .labels(labelsTensor)
                                                                  .reportsRawLoss()
                                                                  .lossDataType(lossDataType)
                                       .lossWeight(lossWeight.value_or(1.0f))
                                                                  .build();

    lossShaperInput = meanAbsolutePercentageError.getLoss();

    finalizeLossReporting();
}

json MAPE::architectureJson() const {
    json j = Loss::architectureJson();
    j["layer_type"] = "mape";
    return j;
}

void MAPE::deserialize(const json &j, Network *network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in MAPE::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "mape")
        throw runtime_error("Layer type mismatch in MAPE::deserialize: " + j.at("layer_type").get<std::string>());

    MAPE meanAbsolutePercentageError;
    meanAbsolutePercentageError.lossShape = j.at("loss_shape").get<LossShape>();
    meanAbsolutePercentageError.lossDataType = j.at("loss_data_type").get<DataType>();

    meanAbsolutePercentageError.lossWeight = ThorImplementation::lossWeightFromJson(j);

    uint64_t originalTensorId;
    originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    meanAbsolutePercentageError.predictionsTensor = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    meanAbsolutePercentageError.labelsTensor = network->getApiTensorByOriginalId(originalTensorId);

    meanAbsolutePercentageError.lossTensor = Tensor::deserialize(j["loss_shaper_input_tensor"]);

    meanAbsolutePercentageError.initialized = true;
    meanAbsolutePercentageError.addToNetwork(network);
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("mape", &Thor::MAPE::deserialize);
    return true;
}();
}  // namespace
