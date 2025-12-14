#include "DeepLearning/Api/Layers/Loss/BinaryCrossEntropy.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

void BinaryCrossEntropy::buildSupportLayersAndAddToNetwork() {
    Tensor currentFeatureInput = predictionsTensor;

    assert(!sigmoidStamped);
    Sigmoid::Builder sigmoidBuilder = Sigmoid::Builder();
    sigmoidBuilder.network(*network);
    sigmoidBuilder.featureInput(currentFeatureInput);
    sigmoidBuilder.backwardComputedExternally();
    shared_ptr<Layer> sigmoid = sigmoidBuilder.build();
    sigmoidOutput = sigmoid->getFeatureOutput();
    currentFeatureInput = sigmoidOutput;

    BinaryCrossEntropy::Builder binaryCrossEntropyBuilder = BinaryCrossEntropy::Builder()
                                                                .network(*network)
                                                                .predictions(currentFeatureInput)
                                                                .labels(labelsTensor)
                                                                .sigmoidStamped()
                                                                .reportsElementwiseLoss()
                                                                .lossDataType(lossDataType);
    BinaryCrossEntropy crossEntropy = binaryCrossEntropyBuilder.build();
    currentFeatureInput = crossEntropy.getLoss();

    if (lossShape == LossShape::BATCH) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(currentFeatureInput).reportsBatchLoss().build();
        lossTensor = lossShaper.getFeatureOutput();
    } else {
        // No loss shaper needed in this case
        assert(lossShape == LossShape::ELEMENTWISE);
        lossTensor = currentFeatureInput;
    }
}

json BinaryCrossEntropy::serialize(const string &storageDir, Stream stream) const {
    json j;
    j["factory"] = Layer::Factory::Loss.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = "binary_cross_entropy";
    string layerName = string("layer") + to_string(getId());
    j["layer_name"] = layerName;
    j["loss_shape"] = lossShape;
    j["loss_data_type"] = lossDataType;
    j["labels_tensor"] = labelsTensor.serialize();
    j["predictions_tensor"] = predictionsTensor.serialize();
    j["softmax_output_tensor"] = sigmoidOutput.serialize();
    j["loss_tensor"] = lossTensor.serialize();

    return j;
}

void BinaryCrossEntropy::deserialize(const json &j, Network *network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in BinaryCrossEntropy::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "binary_cross_entropy")
        throw runtime_error("Layer type mismatch in BinaryCrossEntropy::deserialize: " + j.at("layer_type").get<std::string>());

    BinaryCrossEntropy binaryCrossEntropy;
    binaryCrossEntropy.lossShape = j.at("loss_shape").get<LossShape>();

    uint64_t originalTensorId;
    originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    binaryCrossEntropy.predictionsTensor = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["softmax_output_tensor"].at("id").get<uint64_t>();
    binaryCrossEntropy.sigmoidOutput = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    binaryCrossEntropy.labelsTensor = network->getApiTensorByOriginalId(originalTensorId);

    binaryCrossEntropy.lossTensor = Tensor::deserialize(j["loss_tensor"]);

    binaryCrossEntropy.initialized = true;
    binaryCrossEntropy.addToNetwork(network);
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("binary_cross_entropy", &Thor::BinaryCrossEntropy::deserialize);
    return true;
}();
}  // namespace
