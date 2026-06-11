#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Layers/Loss/CategoricalCrossEntropy.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {
void CategoricalCrossEntropy::buildSupportLayersAndAddToNetwork() {
    THOR_THROW_IF_FALSE(!softmaxAddedToNetwork);
    shared_ptr<Activation> softmax = Softmax::Builder().backwardComputedExternally().build();
    softmaxOutput = softmax->addToNetwork(predictionsTensor, network);

    CategoricalCrossEntropy::Builder categoricalCrossEntropyBuilder = CategoricalCrossEntropy::Builder()
                                                                          .network(*network)
                                                                          .predictions(softmaxOutput)
                                                                          .labels(labelsTensor)
                                                                          .softmaxAddedToNetwork()
                                                                          .reportsRawLoss()
                                                                          .lossDataType(lossDataType);

    CategoricalCrossEntropy crossEntropy;
    categoricalCrossEntropyBuilder.populateAndAdd(
        crossEntropy, labelType, labelType == LabelType::SPARSE ? std::optional<uint32_t>(numClasses) : std::nullopt);
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
        // No loss shaper needed in this case.
        THOR_THROW_IF_FALSE(lossShape == LossShape::RAW);
        lossTensor = lossShaperInput;
    }
}

json CategoricalCrossEntropy::architectureJson() const {
    json j;
    j["factory"] = Layer::Factory::Loss.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = labelType == LabelType::SPARSE ? "sparse_categorical_cross_entropy" : "categorical_cross_entropy";
    string layerName = string("layer") + to_string(getId());
    j["layer_name"] = layerName;
    j["loss_shape"] = LossShape::RAW;
    j["loss_data_type"] = lossDataType;
    j["labels_tensor"] = labelsTensor.architectureJson();
    j["predictions_tensor"] = predictionsTensor.architectureJson();
    j["softmax_output_tensor"] = softmaxOutput.architectureJson();
    j["loss_shaper_input_tensor"] = lossShaperInput.architectureJson();
    j["loss_tensor"] = lossTensor.architectureJson();

    ThorImplementation::addLossWeightToJson(j, lossWeight);
    return j;
}

void CategoricalCrossEntropy::deserializeInto(const json &j,
                                             Network *network,
                                             CategoricalCrossEntropy &categoricalCrossEntropy,
                                             LabelType labelType,
                                             const string &expectedLayerType) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in CategoricalCrossEntropy::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != expectedLayerType)
        throw runtime_error("Layer type mismatch in CategoricalCrossEntropy::deserialize: " + j.at("layer_type").get<std::string>());

    categoricalCrossEntropy.labelType = labelType;
    categoricalCrossEntropy.lossShape = j.at("loss_shape").get<Loss::LossShape>();
    categoricalCrossEntropy.lossDataType = j.at("loss_data_type").get<DataType>();

    categoricalCrossEntropy.lossWeight = ThorImplementation::lossWeightFromJson(j);

    uint64_t originalTensorId;
    originalTensorId = j["softmax_output_tensor"].at("id").get<uint64_t>();
    categoricalCrossEntropy.predictionsTensor = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    categoricalCrossEntropy.labelsTensor = network->getApiTensorByOriginalId(originalTensorId);

    categoricalCrossEntropy.softmaxAddedToNetwork = true;
    categoricalCrossEntropy.softmaxOutput = categoricalCrossEntropy.predictionsTensor;
    categoricalCrossEntropy.lossTensor = Tensor::deserialize(j["loss_shaper_input_tensor"]);
    categoricalCrossEntropy.lossShaperInput = categoricalCrossEntropy.lossTensor;

    categoricalCrossEntropy.initialized = true;
    categoricalCrossEntropy.addToNetwork(network);
}

void CategoricalCrossEntropy::deserialize(const json &j, Network *network) {
    CategoricalCrossEntropy categoricalCrossEntropy;
    deserializeInto(j, network, categoricalCrossEntropy, LabelType::DENSE, "categorical_cross_entropy");
}

void SparseCategoricalCrossEntropy::deserialize(const json &j, Network *network) {
    SparseCategoricalCrossEntropy sparseCategoricalCrossEntropy;
    deserializeInto(j, network, sparseCategoricalCrossEntropy, LabelType::SPARSE, "sparse_categorical_cross_entropy");
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("categorical_cross_entropy", &Thor::CategoricalCrossEntropy::deserialize);
    Thor::Loss::register_layer("sparse_categorical_cross_entropy", &Thor::SparseCategoricalCrossEntropy::deserialize);
    return true;
}();
}  // namespace
