#include "DeepLearning/Api/Layers/Loss/CategoricalCrossEntropy.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {
void CategoricalCrossEntropy::buildSupportLayersAndAddToNetwork() {
    assert(!softmaxAddedToNetwork);
    Softmax::Builder softmaxBuilder = Softmax::Builder();
    softmaxBuilder.network(*network);
    softmaxBuilder.featureInput(predictionsTensor);
    softmaxBuilder.backwardComputedExternally();
    shared_ptr<Layer> softmax = softmaxBuilder.build();
    softmaxOutput = softmax->getFeatureOutput();

    CategoricalCrossEntropy::Builder categoricalCrossEntropyBuilder = CategoricalCrossEntropy::Builder()
                                                                          .network(*network)
                                                                          .predictions(softmaxOutput)
                                                                          .labels(labelsTensor)
                                                                          .softmaxAddedToNetwork()
                                                                          .reportsRawLoss()
                                                                          .lossDataType(lossDataType);
    if (labelType == LabelType::INDEX) {
        categoricalCrossEntropyBuilder.receivesClassIndexLabels(numClasses);
    } else {
        assert(labelType == LabelType::ONE_HOT);
        categoricalCrossEntropyBuilder.receivesOneHotLabels();
    }
    CategoricalCrossEntropy crossEntropy = categoricalCrossEntropyBuilder.build();
    lossShaperInput = crossEntropy.getLoss();

    if (lossShape == LossShape::BATCH) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(lossShaperInput).reportsBatchLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else if (lossShape == LossShape::CLASSWISE) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(lossShaperInput).reportsClasswiseLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else if (lossShape == LossShape::ELEMENTWISE) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(lossShaperInput).reportsElementwiseLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else {
        // No loss shaper needed in this case
        assert(lossShape == LossShape::RAW);
        lossTensor = lossShaperInput;
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
    j["loss_shape"] = lossShape;
    j["loss_data_type"] = lossDataType;
    j["labels_tensor"] = labelsTensor.serialize();
    j["predictions_tensor"] = predictionsTensor.serialize();
    j["softmax_output_tensor"] = softmaxOutput.serialize();
    j["loss_shaper_input_tensor"] = lossShaperInput.serialize();
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
    categoricalCrossEntropy.lossShape = j.at("loss_shape").get<LossShape>();

    uint64_t originalTensorId;
    originalTensorId = j["softmax_output_tensor"].at("id").get<uint64_t>();
    categoricalCrossEntropy.predictionsTensor = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    categoricalCrossEntropy.labelsTensor = network->getApiTensorByOriginalId(originalTensorId);

    categoricalCrossEntropy.lossTensor = Tensor::deserialize(j["loss_shaper_input_tensor"]);

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
