#include "DeepLearning/Api/Layers/Loss/CategoricalCrossEntropy.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {
void CategoricalCrossEntropy::buildSupportLayersAndAddToNetwork() {
    Tensor currentFeatureInput = predictionsTensor;

    assert(!softmaxStamped);
    Softmax::Builder softmaxBuilder = Softmax::Builder();
    softmaxBuilder.network(*network);
    softmaxBuilder.featureInput(currentFeatureInput);
    softmaxBuilder.backwardComputedExternally();
    shared_ptr<Layer> softmax = softmaxBuilder.build();
    softmaxOutput = softmax->getFeatureOutput();
    currentFeatureInput = softmaxOutput;

    CategoricalCrossEntropy::Builder categoricalCrossEntropyBuilder = CategoricalCrossEntropy::Builder()
                                                                          .network(*network)
                                                                          .predictions(currentFeatureInput)
                                                                          .labels(labelsTensor)
                                                                          .softmaxStamped()
                                                                          .reportsRawLoss()
                                                                          .lossDataType(lossDataType);
    if (labelType == LabelType::INDEX) {
        categoricalCrossEntropyBuilder.receivesClassIndexLabels(numClasses);
    } else {
        assert(labelType == LabelType::ONE_HOT);
        categoricalCrossEntropyBuilder.receivesOneHotLabels();
    }
    CategoricalCrossEntropy crossEntropy = categoricalCrossEntropyBuilder.build();
    currentFeatureInput = crossEntropy.getLoss();

    if (lossType == LossType::BATCH) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(currentFeatureInput).reportsBatchLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else if (lossType == LossType::CLASSWISE) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(currentFeatureInput).reportsClasswiseLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else if (lossType == LossType::ELEMENTWISE) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(currentFeatureInput).reportsElementwiseLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else {
        // No loss shaper needed in this case
        assert(lossType == LossType::RAW);
        lossTensor = currentFeatureInput;
    }
}

json CategoricalCrossEntropy::serialize(const string &storageDir, Stream stream) const {
    json j;
    j["factory"] = Layer::Factory::Loss.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = "categorical_cross_entropy";
    string layerName = string("layer") + to_string(getId());
    j["layer_name"] = layerName;
    j["label_type"] = labelType;
    j["loss_type"] = lossType;
    j["loss_data_type"] = lossDataType;
    j["labels_tensor"] = labelsTensor.serialize();
    j["predictions_tensor"] = predictionsTensor.serialize();
    j["softmax_output_tensor"] = softmaxOutput.serialize();
    j["loss_tensor"] = lossTensor.serialize();

    return j;
}

void CategoricalCrossEntropy::deserialize(const json &j, Network *network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in CategoricalCrossEntropy::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "categorical_cross_entropy")
        throw runtime_error("Layer type mismatch in CategoricalCrossEntropy::deserialize: " + j.at("layer_type").get<std::string>());

    CategoricalCrossEntropy categoricalCrossEntropy;
    categoricalCrossEntropy.labelType = j.at("label_type").get<LabelType>();
    categoricalCrossEntropy.lossType = j.at("loss_type").get<LossType>();

    uint64_t originalTensorId;
    originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    categoricalCrossEntropy.predictionsTensor = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["softmax_output_tensor"].at("id").get<uint64_t>();
    categoricalCrossEntropy.softmaxOutput = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    categoricalCrossEntropy.labelsTensor = network->getApiTensorByOriginalId(originalTensorId);

    categoricalCrossEntropy.lossTensor = Tensor::deserialize(j["loss_tensor"]);

    categoricalCrossEntropy.initialized = true;
    categoricalCrossEntropy.addToNetwork(network);
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("categorical_cross_entropy", &Thor::CategoricalCrossEntropy::deserialize);
    return true;
}();
}  // namespace
