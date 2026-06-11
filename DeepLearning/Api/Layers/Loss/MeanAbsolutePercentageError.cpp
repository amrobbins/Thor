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

    if (lossShape == LossShape::BATCH) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(lossShaperInput).reportsBatchLoss().build();
        // Replace the output on the compound layer to be the output of the last stage
        // i.e. tunnel the actual input to actual output of the compound layer,
        // Network uses single layers, user uses compound layer.
        lossTensor = lossShaper.getLossOutput();
    } else if (lossShape == LossShape::ELEMENTWISE) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(lossShaperInput).reportsElementwiseLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else if (lossShape == LossShape::CLASSWISE) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(lossShaperInput).reportsClasswiseLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else {
        // No loss shaper needed
        THOR_THROW_IF_FALSE(lossShape == LossShape::RAW);
        lossTensor = lossShaperInput;
    }
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
