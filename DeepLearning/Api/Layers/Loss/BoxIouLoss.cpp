#include "DeepLearning/Api/Layers/Loss/BoxIouLoss.h"

#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"

#include <stdexcept>
#include <string>

using namespace std;
using json = nlohmann::json;

namespace Thor {
namespace {

void validateBoxDType(const char* tensorName, DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error(string("Unsupported BoxIouLoss ") + tensorName + " dtype: " +
                            ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validateBoxDimensions(const vector<uint64_t>& dims, const string& tensorName) {
    if (!((dims.size() == 1 && dims[0] == 4) || (dims.size() == 2 && dims[1] == 4))) {
        throw runtime_error("BoxIouLoss " + tensorName + " must have API dimensions [4] or [boxes, 4].");
    }
}

}  // namespace

vector<uint64_t> BoxIouLoss::rawLossDimensionsForBoxes(const vector<uint64_t>& boxDims) {
    validateBoxDimensions(boxDims, "boxes");
    if (boxDims.size() == 1)
        return {1};
    return {boxDims[0]};
}

void BoxIouLoss::populateAndAdd(BoxIouLoss& loss,
                                Network* network,
                                Tensor predictions,
                                Tensor labels,
                                LossShape lossShape,
                                DataType lossDataType,
                                float eps,
                                std::optional<float> lossWeight) {
    THOR_THROW_IF_FALSE(network != nullptr);
    THOR_THROW_IF_FALSE(predictions != labels);
    validateBoxDimensions(predictions.getDimensions(), "predictions");
    if (predictions.getDimensions() != labels.getDimensions())
        throw runtime_error("BoxIouLoss labels dimensions must match predictions dimensions.");
    validateBoxDType("predictions", predictions.getDataType());
    validateBoxDType("labels", labels.getDataType());
    THOR_THROW_IF_FALSE(lossDataType == DataType::FP16 || lossDataType == DataType::FP32);
    THOR_THROW_IF_FALSE(lossShape == LossShape::BATCH || lossShape == LossShape::CLASSWISE ||
                        lossShape == LossShape::ELEMENTWISE || lossShape == LossShape::RAW);
    THOR_THROW_IF_FALSE(eps > 0.0f);
    lossWeight = ThorImplementation::normalizeLossWeight(lossWeight);

    loss.predictionsTensor = predictions;
    loss.labelsTensor = labels;
    loss.lossShape = lossShape;
    loss.lossDataType = lossDataType;
    loss.lossWeight = lossWeight;
    loss.eps = eps;
    loss.network = network;
    loss.initialized = true;

    if (loss.isMultiLayer()) {
        loss.buildSupportLayersAndAddToNetwork();
    } else {
        loss.lossTensor = Tensor(lossDataType, rawLossDimensionsForBoxes(predictions.getDimensions()));
        loss.lossShaperInput = loss.lossTensor;
        loss.addToNetwork(network);
    }
}

void BoxIouLoss::buildSupportLayersAndAddToNetwork() {
    shared_ptr<BoxIouLoss> rawLoss = makeRawSupportLoss();
    rawLoss->predictionsTensor = predictionsTensor;
    rawLoss->labelsTensor = labelsTensor;
    rawLoss->lossShape = LossShape::RAW;
    rawLoss->lossDataType = lossDataType;
    rawLoss->lossWeight = lossWeight;
    rawLoss->eps = eps;
    rawLoss->network = network;
    rawLoss->initialized = true;
    rawLoss->lossTensor = Tensor(lossDataType, rawLossDimensionsForBoxes(predictionsTensor.getDimensions()));
    rawLoss->lossShaperInput = rawLoss->lossTensor;
    rawLoss->addToNetwork(network);

    lossShaperInput = rawLoss->getLoss();

    if (lossShape == LossShape::BATCH) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(lossShaperInput).reportsBatchLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else if (lossShape == LossShape::ELEMENTWISE) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(lossShaperInput).reportsElementwiseLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else if (lossShape == LossShape::CLASSWISE) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(lossShaperInput).reportsClasswiseLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else {
        THOR_THROW_IF_FALSE(lossShape == LossShape::RAW);
        lossTensor = lossShaperInput;
    }
}

json BoxIouLoss::architectureJson() const {
    json j;
    j["factory"] = Factory::Loss.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = getSerializedLayerType();
    j["layer_name"] = string("layer") + to_string(getId());
    j["loss_shape"] = lossShape;
    j["loss_data_type"] = lossDataType;
    j["predictions_tensor"] = predictionsTensor.architectureJson();
    j["labels_tensor"] = labelsTensor.architectureJson();
    j["loss_shaper_input_tensor"] = lossShaperInput.architectureJson();
    j["loss_tensor"] = lossTensor.architectureJson();
    j["box_format"] = "xyxy";
    j["eps"] = eps;
    ThorImplementation::addLossWeightToJson(j, lossWeight);
    return j;
}

void BoxIouLoss::deserializeInto(const json& j, Network* network, BoxIouLoss& loss, const string& expectedLayerType) {
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in BoxIouLoss::deserialize: " + j.at("version").get<string>());
    if (j.at("layer_type").get<string>() != expectedLayerType)
        throw runtime_error("Layer type mismatch in BoxIouLoss::deserialize: " + j.at("layer_type").get<string>());
    if (j.value("box_format", string("xyxy")) != "xyxy")
        throw runtime_error("Unsupported BoxIouLoss box_format: " + j.value("box_format", string("")));

    uint64_t originalTensorId = j.at("predictions_tensor").at("id").get<uint64_t>();
    Tensor predictions = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j.at("labels_tensor").at("id").get<uint64_t>();
    Tensor labels = network->getApiTensorByOriginalId(originalTensorId);

    loss.lossShape = j.at("loss_shape").get<LossShape>();
    loss.lossDataType = j.at("loss_data_type").get<DataType>();

    loss.lossWeight = ThorImplementation::lossWeightFromJson(j);
    loss.eps = j.value("eps", 1.0e-7f);
    loss.predictionsTensor = predictions;
    loss.labelsTensor = labels;
    loss.network = network;
    loss.initialized = true;

    if (loss.isMultiLayer()) {
        loss.buildSupportLayersAndAddToNetwork();
    } else {
        loss.lossTensor = Tensor::deserialize(j.at("loss_tensor"));
        loss.lossShaperInput = loss.lossTensor;
        loss.addToNetwork(network);
    }
}

void IoULoss::deserialize(const json& j, Network* network) {
    IoULoss loss;
    deserializeInto(j, network, loss, "iou_loss");
}

void GIoULoss::deserialize(const json& j, Network* network) {
    GIoULoss loss;
    deserializeInto(j, network, loss, "giou_loss");
}

void DIoULoss::deserialize(const json& j, Network* network) {
    DIoULoss loss;
    deserializeInto(j, network, loss, "diou_loss");
}

void CIoULoss::deserialize(const json& j, Network* network) {
    CIoULoss loss;
    deserializeInto(j, network, loss, "ciou_loss");
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("iou_loss", &Thor::IoULoss::deserialize);
    Thor::Loss::register_layer("giou_loss", &Thor::GIoULoss::deserialize);
    Thor::Loss::register_layer("diou_loss", &Thor::DIoULoss::deserialize);
    Thor::Loss::register_layer("ciou_loss", &Thor::CIoULoss::deserialize);
    return true;
}();
}  // namespace
