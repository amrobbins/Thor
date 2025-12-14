#include "MeanSquaredError.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

void MeanSquaredError::buildSupportLayersAndAddToNetwork() {
    Tensor currentFeatureInput = predictionsTensor;

    MeanSquaredError meanSquaredError = MeanSquaredError::Builder()
                                            .network(*network)
                                            .predictions(predictionsTensor)
                                            .labels(labelsTensor)
                                            .reportsRawLoss()
                                            .lossDataType(lossDataType)
                                            .build();

    if (lossShape == LossShape::BATCH) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(meanSquaredError.getLoss()).reportsBatchLoss().build();
        // Replace the output on the compound layer to be the output of the last stage
        // i.e. tunnel the actual input to actual output of the compound layer,
        // Network uses single layers, user uses compound layer.
        lossTensor = lossShaper.getLossOutput();
    } else if (lossShape == LossShape::ELEMENTWISE) {
        LossShaper lossShaper =
            LossShaper::Builder().network(*network).lossInput(meanSquaredError.getLoss()).reportsElementwiseLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else if (lossShape == LossShape::CLASSWISE) {
        LossShaper lossShaper =
            LossShaper::Builder().network(*network).lossInput(meanSquaredError.getLoss()).reportsClasswiseLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else {
        // No loss shaper needed
        assert(lossShape == LossShape::RAW);
    }
}

void MeanSquaredError::deserialize(const json &j, Network *network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in MeanSquaredError::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "mean_squared_error")
        throw runtime_error("Layer type mismatch in MeanSquaredError::deserialize: " + j.at("layer_type").get<std::string>());

    MeanSquaredError meanSquaredError;
    meanSquaredError.lossShape = j.at("loss_shape").get<LossShape>();
    meanSquaredError.lossDataType = j.at("loss_data_type").get<Tensor::DataType>();

    uint64_t originalTensorId;
    originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    meanSquaredError.predictionsTensor = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    meanSquaredError.labelsTensor = network->getApiTensorByOriginalId(originalTensorId);

    meanSquaredError.lossTensor = Tensor::deserialize(j["loss_tensor"]);

    meanSquaredError.initialized = true;
    meanSquaredError.addToNetwork(network);
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("mean_squared_error", &Thor::MeanSquaredError::deserialize);
    return true;
}();
}  // namespace
