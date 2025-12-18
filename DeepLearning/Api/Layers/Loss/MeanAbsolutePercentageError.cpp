#include "MeanAbsolutePercentageError.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {
void MeanAbsolutePercentageError::buildSupportLayersAndAddToNetwork() {
    MeanAbsolutePercentageError meanAbsolutePercentageError = MeanAbsolutePercentageError::Builder()
                                                                  .network(*network)
                                                                  .predictions(predictionsTensor)
                                                                  .labels(labelsTensor)
                                                                  .reportsRawLoss()
                                                                  .lossDataType(lossDataType)
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
        assert(lossShape == LossShape::RAW);
        lossTensor = lossShaperInput;
    }
}

void MeanAbsolutePercentageError::deserialize(const json &j, Network *network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in MeanAbsolutePercentageError::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "mean_absolute_percentage_error")
        throw runtime_error("Layer type mismatch in MeanAbsolutePercentageError::deserialize: " + j.at("layer_type").get<std::string>());

    MeanAbsolutePercentageError meanAbsolutePercentageError;
    meanAbsolutePercentageError.lossShape = j.at("loss_shape").get<LossShape>();
    meanAbsolutePercentageError.lossDataType = j.at("loss_data_type").get<Tensor::DataType>();

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
    Thor::Loss::register_layer("mean_absolute_percentage_error", &Thor::MeanAbsolutePercentageError::deserialize);
    return true;
}();
}  // namespace
