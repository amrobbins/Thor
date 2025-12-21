#include "DeepLearning/Api/Layers/Loss/BinaryCrossEntropy.h"

#include "Utilities/TensorOperations/GpuMatrixMultiply/gpuMatrixMultiply.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

void BinaryCrossEntropy::buildSupportLayersAndAddToNetwork() {
    assert(!sigmoidAddedToNetwork);
    shared_ptr<Layer> sigmoid = Sigmoid::Builder().network(*network).featureInput(predictionsTensor).backwardComputedExternally().build();
    sigmoidOutput = sigmoid->getFeatureOutput();

    BinaryCrossEntropy::Builder binaryCrossEntropyBuilder = BinaryCrossEntropy::Builder()
                                                                .network(*network)
                                                                .predictions(sigmoidOutput)
                                                                .labels(labelsTensor)
                                                                .sigmoidAddedToNetwork()
                                                                .reportsElementwiseLoss()
                                                                .lossDataType(lossDataType);
    BinaryCrossEntropy crossEntropy = binaryCrossEntropyBuilder.build();
    lossShaperInput = crossEntropy.getLoss();

    if (lossShape == LossShape::BATCH) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(lossShaperInput).reportsBatchLoss().build();
        lossTensor = lossShaper.getFeatureOutput();
    } else {
        // No loss shaper needed in this case
        assert(lossShape == LossShape::ELEMENTWISE);
        lossTensor = lossShaperInput;
    }
}

json BinaryCrossEntropy::serialize(const string &storageDir, Stream stream) const {
    // The thing that is deserialized must be just the base layer, any helper layers
    // are themselves deserialized. So loss_shape set to LossShape::ELEMENTWISE

    json j;  // dfgf
    j["factory"] = Layer::Factory::Loss.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = "binary_cross_entropy";
    string layerName = string("layer") + to_string(getId());
    j["layer_name"] = layerName;
    j["loss_shape"] = LossShape::RAW;
    j["loss_data_type"] = lossDataType;
    j["labels_tensor"] = labelsTensor.serialize();
    j["predictions_tensor"] = predictionsTensor.serialize();
    j["sigmoid_output_tensor"] = sigmoidOutput.serialize();
    j["loss_shaper_input_tensor"] = lossShaperInput.serialize();
    j["loss_tensor"] = lossTensor.serialize();

    return j;
}

void BinaryCrossEntropy::deserialize(const json &j, Network *network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in BinaryCrossEntropy::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "binary_cross_entropy")
        throw runtime_error("Layer type mismatch in BinaryCrossEntropy::deserialize: " + j.at("layer_type").get<std::string>());

    // Only connect the single layer and add it to the network, like when it was built.
    // Helper layers deserialize themselves.
    BinaryCrossEntropy binaryCrossEntropy;
    assert(j.at("loss_shape").get<LossShape>() == LossShape::RAW);
    binaryCrossEntropy.lossShape = LossShape::RAW;
    binaryCrossEntropy.lossDataType = j.at("loss_data_type").get<Tensor::DataType>();

    uint64_t originalTensorId;
    originalTensorId = j["sigmoid_output_tensor"].at("id").get<uint64_t>();
    binaryCrossEntropy.predictionsTensor = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    binaryCrossEntropy.labelsTensor = network->getApiTensorByOriginalId(originalTensorId);
    // The sigmoid layer itself, at this point, has been deserialized externally to here.
    binaryCrossEntropy.sigmoidAddedToNetwork = true;

    binaryCrossEntropy.lossTensor = Tensor::deserialize(j["loss_shaper_input_tensor"]);

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
