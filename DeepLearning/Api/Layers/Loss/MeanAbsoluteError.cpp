#include "MeanAbsoluteError.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

void MeanAbsoluteError::buildSupportLayersAndAddToNetwork() {
    Tensor currentFeatureInput = predictionsTensor;

    MeanAbsoluteError meanAbsoluteError =
        MeanAbsoluteError::Builder().network(*network).predictions(predictionsTensor).labels(labelsTensor).reportsRawLoss().build();

    if (lossType == LossType::BATCH) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(meanAbsoluteError.getLoss()).reportsBatchLoss().build();
        // Replace the output on the compound layer to be the output of the last stage
        // i.e. tunnel the actual input to actual output of the compound layer,
        // Network uses single layers, user uses compound layer.
        lossTensor = lossShaper.getLossOutput();
    } else if (lossType == LossType::ELEMENTWISE) {
        LossShaper lossShaper =
            LossShaper::Builder().network(*network).lossInput(meanAbsoluteError.getLoss()).reportsElementwiseLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else if (lossType == LossType::CLASSWISE) {
        LossShaper lossShaper =
            LossShaper::Builder().network(*network).lossInput(meanAbsoluteError.getLoss()).reportsClasswiseLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else {
        // No loss shaper needed
        assert(lossType == LossType::RAW);
    }
}

json MeanAbsoluteError::serialize(const string &storageDir, Stream stream) const {
    json j;
    j["factory"] = Layer::Factory::Loss.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = "mean_absolute_error";
    string layerName = string("layer") + to_string(getId());
    j["layer_name"] = layerName;
    j["loss_type"] = lossType;
    j["loss_data_type"] = lossDataType;
    j["labels_tensor"] = labelsTensor.serialize();
    j["predictions_tensor"] = predictionsTensor.serialize();
    j["loss_tensor"] = lossTensor.serialize();

    return j;
}

void MeanAbsoluteError::deserialize(const json &j, Network *network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in MeanAbsoluteError::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "mean_absolute_error")
        throw runtime_error("Layer type mismatch in MeanAbsoluteError::deserialize: " + j.at("layer_type").get<std::string>());

    MeanAbsoluteError meanAbsoluteError;
    meanAbsoluteError.lossType = j.at("loss_type").get<LossType>();
    meanAbsoluteError.lossDataType = j.at("loss_data_type").get<Tensor::DataType>();

    uint64_t originalTensorId;
    originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    meanAbsoluteError.predictionsTensor = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    meanAbsoluteError.labelsTensor = network->getApiTensorByOriginalId(originalTensorId);

    meanAbsoluteError.lossTensor = Tensor::deserialize(j["loss_tensor"]);

    meanAbsoluteError.initialized = true;
    meanAbsoluteError.addToNetwork(network);
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("mean_absolute_error", &Thor::MeanAbsoluteError::deserialize);
    return true;
}();
}  // namespace
