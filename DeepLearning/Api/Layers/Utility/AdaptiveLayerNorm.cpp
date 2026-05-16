#include "DeepLearning/Api/Layers/Utility/AdaptiveLayerNorm.h"

#include <limits>
#include <stdexcept>
#include <string>

using namespace std;
using json = nlohmann::json;

namespace Thor {

bool AdaptiveLayerNorm::isAdaptiveLayerNormInputDataType(Tensor::DataType dataType) {
    switch (dataType) {
        case Tensor::DataType::FP16:
        case Tensor::DataType::BF16:
        case Tensor::DataType::FP32:
            return true;
        default:
            return false;
    }
}

void AdaptiveLayerNorm::validateCudnnFrontendContract(uint64_t normalizedFeatureCount, Tensor::DataType inputDataType) {
    if (inputDataType == Tensor::DataType::FP32 && normalizedFeatureCount % 32 != 0) {
        throw invalid_argument(
            "AdaptiveLayerNorm cuDNN Frontend primary engines require fp32 normalized feature count to be a multiple of 32; got " +
            to_string(normalizedFeatureCount) + ".");
    }
}

uint64_t AdaptiveLayerNorm::checkedFeatureCount(const vector<uint64_t>& shape, const string& what) {
    if (shape.empty()) {
        throw invalid_argument("AdaptiveLayerNorm " + what + " must contain at least one dimension.");
    }
    uint64_t count = 1;
    for (uint64_t dim : shape) {
        if (dim == 0) {
            throw invalid_argument("AdaptiveLayerNorm " + what + " dimensions must be non-zero.");
        }
        if (count > numeric_limits<uint64_t>::max() / dim) {
            throw invalid_argument("AdaptiveLayerNorm " + what + " feature count overflows uint64_t.");
        }
        count *= dim;
    }
    return count;
}

void AdaptiveLayerNorm::validateNormalizedShapeForInput(const vector<uint64_t>& inputDims, const vector<uint64_t>& normalizedShape) {
    if (inputDims.empty()) {
        throw invalid_argument("AdaptiveLayerNorm feature input must have at least one feature dimension.");
    }
    if (inputDims.size() < normalizedShape.size()) {
        throw invalid_argument("AdaptiveLayerNorm normalizedShape rank cannot exceed feature input rank.");
    }
    const size_t offset = inputDims.size() - normalizedShape.size();
    for (size_t i = 0; i < normalizedShape.size(); ++i) {
        if (inputDims[offset + i] != normalizedShape[i]) {
            throw invalid_argument("AdaptiveLayerNorm normalizedShape must match trailing feature input dimensions.");
        }
    }
}

const char* AdaptiveLayerNorm::portName(uint32_t port) {
    switch (port) {
        case DATA:
            return "feature_input";
        case SCALE:
            return "scale_input";
        case BIAS:
            return "bias_input";
        default:
            return "unknown";
    }
}

void AdaptiveLayerNorm::resetInputConnectionTracking() { connectedInputOriginalIds.clear(); }

AdaptiveLayerNorm AdaptiveLayerNorm::Builder::build() {
    if (!_featureInput.has_value()) {
        throw invalid_argument("AdaptiveLayerNorm::Builder requires featureInput().");
    }
    if (!_scaleInput.has_value()) {
        throw invalid_argument("AdaptiveLayerNorm::Builder requires scaleInput().");
    }
    if (!_biasInput.has_value()) {
        throw invalid_argument("AdaptiveLayerNorm::Builder requires biasInput().");
    }
    if (_normalizedShape.empty()) {
        const vector<uint64_t> dims = _featureInput.value().getDimensions();
        _normalizedShape = {dims.back()};
    }
    if (!_epsilon.has_value())
        _epsilon = 1.0e-5;
    if (!_scaleBiasDataType.has_value())
        _scaleBiasDataType = Tensor::DataType::FP32;

    verifyConfig();

    AdaptiveLayerNorm layer;
    layer.featureInputs = {_featureInput.value(), _scaleInput.value(), _biasInput.value()};
    layer.normalizedShape = _normalizedShape;
    layer.epsilon = _epsilon.value();
    layer.scaleBiasDataType = _scaleBiasDataType.value();

    Tensor out = layer.featureInputs[DATA].clone();
    layer.featureOutputs.push_back(out);
    for (uint32_t i = 0; i < layer.featureInputs.size(); ++i) {
        layer.outputTensorFromInputTensor[layer.featureInputs[i]] = out;
    }
    layer.inputTensorFromOutputTensor[out] = layer.featureInputs[DATA];
    layer.resetInputConnectionTracking();

    layer.initialized = true;
    layer.addToNetwork(_network.value());
    return layer;
}

void AdaptiveLayerNorm::Builder::verifyConfig() const {
    if (!_network.has_value()) {
        throw invalid_argument("AdaptiveLayerNorm::Builder requires network().");
    }
    if (!_featureInput.has_value() || !_scaleInput.has_value() || !_biasInput.has_value()) {
        throw invalid_argument("AdaptiveLayerNorm::Builder requires featureInput(), scaleInput(), and biasInput().");
    }
    const uint64_t normalizedFeatureCount = checkedFeatureCount(_normalizedShape, "normalizedShape");
    if (!_epsilon.has_value() || !(_epsilon.value() > 0.0)) {
        throw invalid_argument("AdaptiveLayerNorm epsilon must be > 0.");
    }
    if (_scaleBiasDataType.value() != Tensor::DataType::FP32) {
        throw invalid_argument("AdaptiveLayerNorm currently requires fp32 scale/bias tensors for cuDNN Frontend AdaptiveLayerNorm.");
    }

    const Tensor& featureInput = _featureInput.value();
    const Tensor& scaleInput = _scaleInput.value();
    const Tensor& biasInput = _biasInput.value();
    if (!AdaptiveLayerNorm::isAdaptiveLayerNormInputDataType(featureInput.getDataType())) {
        throw invalid_argument("AdaptiveLayerNorm feature input dtype must be fp16, bf16, or fp32.");
    }
    if (scaleInput.getDataType() != _scaleBiasDataType.value() || biasInput.getDataType() != _scaleBiasDataType.value()) {
        throw invalid_argument("AdaptiveLayerNorm scale_input and bias_input must be fp32 tensors.");
    }
    const vector<uint64_t> inputDims = featureInput.getDimensions();
    AdaptiveLayerNorm::validateNormalizedShapeForInput(inputDims, _normalizedShape);
    AdaptiveLayerNorm::validateCudnnFrontendContract(normalizedFeatureCount, featureInput.getDataType());
    if (scaleInput.getDimensions() != _normalizedShape || biasInput.getDimensions() != _normalizedShape) {
        throw invalid_argument("AdaptiveLayerNorm scale_input and bias_input dimensions must match normalizedShape.");
    }

    set<uint64_t> originalIds = {featureInput.getOriginalId(), scaleInput.getOriginalId(), biasInput.getOriginalId()};
    if (originalIds.size() != NUM_INPUT_PORTS) {
        throw invalid_argument("AdaptiveLayerNorm feature_input, scale_input, and bias_input must be distinct tensors.");
    }
}

int AdaptiveLayerNorm::getConnectionType(Tensor connectingTensor) const {
    for (uint32_t i = 0; i < featureInputs.size(); ++i) {
        if (connectingTensor == featureInputs[i])
            return static_cast<int>(i);
    }
    if (featureOutputs.size() == 1 && connectingTensor == featureOutputs[0])
        return 0;
    throw runtime_error("Tensor is not connected to this AdaptiveLayerNorm layer.");
}

void AdaptiveLayerNorm::informThatInputConnectionMade(Tensor inputTensor) {
    bool found = false;
    for (uint32_t i = 0; i < featureInputs.size(); ++i) {
        if (inputTensor == featureInputs[i]) {
            found = true;
            connectedInputOriginalIds.insert(inputTensor.getOriginalId());
            break;
        }
    }
    if (!found) {
        throw runtime_error("AdaptiveLayerNorm informed of connection for unknown input tensor.");
    }
}

vector<Tensor> AdaptiveLayerNorm::getOutputsFromInput(Tensor inputTensor) {
    (void)inputTensor;
    THOR_THROW_IF_FALSE(featureOutputs.size() == 1);
    if (connectedInputOriginalIds.size() == NUM_INPUT_PORTS) {
        return {featureOutputs[0]};
    }
    return {};
}

shared_ptr<ThorImplementation::Layer> AdaptiveLayerNorm::stamp(ThorImplementation::TensorPlacement placement,
                                                               shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                               shared_ptr<Thor::Layer> drivingApiLayer,
                                                               Thor::Tensor connectingApiTensor,
                                                               const bool inferenceOnly) const {
    (void)drivingLayer;
    (void)drivingApiLayer;
    (void)connectingApiTensor;
    THOR_THROW_IF_FALSE(initialized);
    return make_shared<ThorImplementation::AdaptiveLayerNorm>(
        placement, inferenceOnly, normalizedShape, epsilon, scaleBiasDataType, static_cast<int64_t>(getId()));
}

json AdaptiveLayerNorm::architectureJson() const {
    json j;
    j["factory"] = Layer::Factory::Layer.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = "adaptive_layer_norm";
    j["layer_name"] = string("layer") + to_string(getId());
    j["normalized_shape"] = normalizedShape;
    j["epsilon"] = epsilon;
    j["scale_bias_data_type"] = scaleBiasDataType;

    json inputs = json::array();
    for (uint32_t i = 0; i < featureInputs.size(); ++i) {
        json input = featureInputs[i].architectureJson();
        input["port"] = portName(i);
        inputs.push_back(input);
    }
    j["inputs"] = inputs;

    json outputs = json::array();
    for (uint32_t i = 0; i < featureOutputs.size(); ++i)
        outputs.push_back(featureOutputs[i].architectureJson());
    j["outputs"] = outputs;

    return j;
}

void AdaptiveLayerNorm::deserialize(const json& j, Network* network) {
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in AdaptiveLayerNorm::deserialize: " + j.at("version").get<string>());
    if (j.at("layer_type").get<string>() != "adaptive_layer_norm")
        throw runtime_error("Layer type mismatch in AdaptiveLayerNorm::deserialize: " + j.at("layer_type").get<string>());

    AdaptiveLayerNorm layer;
    layer.normalizedShape = j.at("normalized_shape").get<vector<uint64_t>>();
    layer.epsilon = j.at("epsilon").get<double>();
    layer.scaleBiasDataType = j.at("scale_bias_data_type").get<Tensor::DataType>();

    const auto inputsJson = j.at("inputs").get<vector<json>>();
    if (inputsJson.size() != NUM_INPUT_PORTS) {
        throw runtime_error("AdaptiveLayerNorm deserialize expected exactly three inputs.");
    }
    layer.featureInputs.resize(NUM_INPUT_PORTS);
    for (uint32_t i = 0; i < NUM_INPUT_PORTS; ++i) {
        const uint64_t originalTensorId = inputsJson[i].at("id").get<uint64_t>();
        layer.featureInputs[i] = network->getApiTensorByOriginalId(originalTensorId);
    }

    for (const json& outputJson : j.at("outputs")) {
        layer.featureOutputs.push_back(Tensor::deserialize(outputJson));
    }
    if (layer.featureOutputs.size() != 1) {
        throw runtime_error("AdaptiveLayerNorm deserialize expected exactly one output.");
    }

    for (uint32_t i = 0; i < layer.featureInputs.size(); ++i) {
        layer.outputTensorFromInputTensor[layer.featureInputs[i]] = layer.featureOutputs[0];
    }
    layer.inputTensorFromOutputTensor[layer.featureOutputs[0]] = layer.featureInputs[DATA];
    layer.resetInputConnectionTracking();

    layer.initialized = true;
    layer.addToNetwork(network);
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::Layer::register_layer("adaptive_layer_norm", &Thor::AdaptiveLayerNorm::deserialize);
    return true;
}();
}  // namespace
