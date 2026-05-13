#include "DeepLearning/Api/Layers/Utility/LayerNorm.h"

#include <limits>
#include <stdexcept>
#include <string>

using namespace std;
using json = nlohmann::json;

namespace Thor {

bool LayerNorm::isLayerNormInputDataType(Tensor::DataType dataType) {
    switch (dataType) {
        case Tensor::DataType::FP16:
        case Tensor::DataType::BF16:
        case Tensor::DataType::FP32:
            return true;
        default:
            return false;
    }
}

uint64_t LayerNorm::checkedFeatureCount(const vector<uint64_t>& shape, const string& what) {
    if (shape.empty()) {
        throw invalid_argument("LayerNorm " + what + " must contain at least one dimension.");
    }
    uint64_t count = 1;
    for (uint64_t dim : shape) {
        if (dim == 0) {
            throw invalid_argument("LayerNorm " + what + " dimensions must be non-zero.");
        }
        if (count > numeric_limits<uint64_t>::max() / dim) {
            throw invalid_argument("LayerNorm " + what + " feature count overflows uint64_t.");
        }
        count *= dim;
    }
    return count;
}

void LayerNorm::validateNormalizedShapeForInput(const vector<uint64_t>& inputDims, const vector<uint64_t>& normalizedShape) {
    if (inputDims.empty()) {
        throw invalid_argument("LayerNorm feature input must have at least one feature dimension.");
    }
    if (inputDims.size() < normalizedShape.size()) {
        throw invalid_argument("LayerNorm normalizedShape rank cannot exceed feature input rank.");
    }
    const size_t offset = inputDims.size() - normalizedShape.size();
    for (size_t i = 0; i < normalizedShape.size(); ++i) {
        if (inputDims[offset + i] != normalizedShape[i]) {
            throw invalid_argument("LayerNorm normalizedShape must match trailing feature input dimensions.");
        }
    }
}

LayerNorm LayerNorm::Builder::build() {
    if (_featureInputs.empty()) {
        throw invalid_argument("LayerNorm::Builder requires at least one featureInput().");
    }
    if (_normalizedShape.empty()) {
        const vector<uint64_t> dims = _featureInputs.front().getDimensions();
        _normalizedShape = {dims.back()};
    }
    if (!_epsilon.has_value())
        _epsilon = 1.0e-5;
    if (!_parameterDataType.has_value())
        _parameterDataType = Tensor::DataType::FP32;
    if (_weightsInitializer == nullptr)
        _weightsInitializer = UniformRandom::Builder().minValue(1.0f).maxValue(1.0f).build();
    if (_biasesInitializer == nullptr)
        _biasesInitializer = UniformRandom::Builder().minValue(0.0f).maxValue(0.0f).build();

    verifyConfig();

    LayerNorm layer;
    layer.featureInputs = _featureInputs;
    layer.normalizedShape = _normalizedShape;
    layer.epsilon = _epsilon.value();
    layer.parameterDataType = _parameterDataType.value();

    const uint64_t hidden = LayerNorm::checkedFeatureCount(layer.normalizedShape, "normalizedShape");

    ParameterSpecification::Builder weightsBuilder;
    weightsBuilder.name("weights").shape({hidden}).dtype(layer.parameterDataType).initializer(_weightsInitializer).trainable(true);
    if (_weightsOptimizer != nullptr)
        weightsBuilder.optimizer(_weightsOptimizer);
    layer.addParameter(make_shared<ParameterSpecification>(weightsBuilder.build()));

    ParameterSpecification::Builder biasesBuilder;
    biasesBuilder.name("biases").shape({hidden}).dtype(layer.parameterDataType).initializer(_biasesInitializer).trainable(true);
    if (_biasesOptimizer != nullptr)
        biasesBuilder.optimizer(_biasesOptimizer);
    layer.addParameter(make_shared<ParameterSpecification>(biasesBuilder.build()));

    layer.initialized = true;

    for (uint32_t i = 0; i < layer.featureInputs.size(); ++i) {
        Tensor out = layer.featureInputs[i].clone();
        layer.featureOutputs.push_back(out);
        layer.outputTensorFromInputTensor[layer.featureInputs[i]] = out;
        layer.inputTensorFromOutputTensor[out] = layer.featureInputs[i];
    }

    layer.addToNetwork(_network.value());
    return layer;
}

void LayerNorm::Builder::verifyConfig() const {
    if (!_network.has_value()) {
        throw invalid_argument("LayerNorm::Builder requires network().");
    }
    if (_featureInputs.empty()) {
        throw invalid_argument("LayerNorm::Builder requires featureInput().");
    }
    checkedFeatureCount(_normalizedShape, "normalizedShape");
    if (!_epsilon.has_value() || !(_epsilon.value() > 0.0)) {
        throw invalid_argument("LayerNorm epsilon must be > 0.");
    }
    if (_parameterDataType.value() != Tensor::DataType::FP32) {
        throw invalid_argument("LayerNorm currently requires fp32 weights/biases for cuDNN Frontend LayerNorm.");
    }
    const Tensor::DataType inputDataType = _featureInputs.front().getDataType();
    if (!LayerNorm::isLayerNormInputDataType(inputDataType)) {
        throw invalid_argument("LayerNorm feature input dtype must be fp16, bf16, or fp32.");
    }
    const vector<uint64_t> inputDims = _featureInputs.front().getDimensions();
    LayerNorm::validateNormalizedShapeForInput(inputDims, _normalizedShape);
    for (uint32_t i = 0; i < _featureInputs.size(); ++i) {
        if (!_featureInputs[i].isInitialized()) {
            throw invalid_argument("LayerNorm feature input is not initialized.");
        }
        if (_featureInputs[i].getDataType() != inputDataType) {
            throw invalid_argument("LayerNorm all feature inputs must have the same dtype.");
        }
        if (_featureInputs[i].getDimensions() != inputDims) {
            throw invalid_argument("LayerNorm all feature inputs must have the same dimensions.");
        }
    }
}

shared_ptr<ThorImplementation::Layer> LayerNorm::stamp(ThorImplementation::TensorPlacement placement,
                                                       shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                       shared_ptr<Thor::Layer> drivingApiLayer,
                                                       Thor::Tensor connectingApiTensor,
                                                       const bool inferenceOnly) const {
    (void)drivingLayer;
    (void)drivingApiLayer;
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(outputTensorFromInputTensor.find(connectingApiTensor) != outputTensorFromInputTensor.end());

    vector<shared_ptr<ThorImplementation::PhysicalParameter>> physicalParameters;
    for (const auto& parameter : getParameters()) {
        THOR_THROW_IF_FALSE(parameter != nullptr);
        physicalParameters.push_back(parameter->stamp());
    }

    return make_shared<ThorImplementation::LayerNorm>(
        placement, inferenceOnly, normalizedShape, epsilon, parameterDataType, physicalParameters, getId());
}

json LayerNorm::architectureJson() const {
    json j;
    j["factory"] = Layer::Factory::Learning.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = "layer_norm";
    j["layer_name"] = string("layer") + to_string(getId());
    j["normalized_shape"] = normalizedShape;
    j["epsilon"] = epsilon;
    j["parameter_data_type"] = parameterDataType;

    json inputs = json::array();
    for (uint32_t i = 0; i < featureInputs.size(); ++i)
        inputs.push_back(featureInputs[i].architectureJson());
    j["inputs"] = inputs;

    json outputs = json::array();
    for (uint32_t i = 0; i < featureOutputs.size(); ++i)
        outputs.push_back(featureOutputs[i].architectureJson());
    j["outputs"] = outputs;

    j["parameters"] = getParametersArchitectureJson()["parameters"];
    return j;
}

json LayerNorm::serialize(thor_file::TarWriter& archiveWriter,
                          Stream stream,
                          bool saveOptimizerState,
                          ThorImplementation::StampedNetwork& stampedNetwork) const {
    json j = architectureJson();
    Parameterizable::serializeParameters(j["parameters"], archiveWriter, stream, saveOptimizerState, stampedNetwork, "layer" + to_string(getId()));
    return j;
}

void LayerNorm::deserialize(shared_ptr<thor_file::TarReader>& archiveReader, const json& j, Network* network) {
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in LayerNorm::deserialize: " + j.at("version").get<string>());
    if (j.at("layer_type").get<string>() != "layer_norm")
        throw runtime_error("Layer type mismatch in LayerNorm::deserialize: " + j.at("layer_type").get<string>());

    LayerNorm layer;
    layer.normalizedShape = j.at("normalized_shape").get<vector<uint64_t>>();
    layer.epsilon = j.at("epsilon").get<double>();
    layer.parameterDataType = j.at("parameter_data_type").get<Tensor::DataType>();

    for (const json& inputJson : j.at("inputs")) {
        const uint64_t originalTensorId = inputJson.at("id").get<uint64_t>();
        layer.featureInputs.push_back(network->getApiTensorByOriginalId(originalTensorId));
    }
    for (const json& outputJson : j.at("outputs")) {
        layer.featureOutputs.push_back(Tensor::deserialize(outputJson, archiveReader.get()));
    }
    if (layer.featureInputs.size() != layer.featureOutputs.size()) {
        throw runtime_error("LayerNorm deserialize expected equal numbers of inputs and outputs.");
    }
    for (uint32_t i = 0; i < layer.featureInputs.size(); ++i) {
        layer.outputTensorFromInputTensor[layer.featureInputs[i]] = layer.featureOutputs[i];
        layer.inputTensorFromOutputTensor[layer.featureOutputs[i]] = layer.featureInputs[i];
    }

    if (j.contains("parameters")) {
        const json& parametersJson = j.at("parameters");
        if (!parametersJson.is_object()) {
            throw runtime_error("LayerNorm parameters must be an object keyed by parameter name.");
        }
        for (auto it = parametersJson.begin(); it != parametersJson.end(); ++it) {
            ParameterSpecification parameter = ParameterSpecification::deserialize(it.value(), archiveReader);
            layer.addParameter(make_shared<ParameterSpecification>(std::move(parameter)));
        }
    }
    if (!layer.hasParameter("weights") || !layer.hasParameter("biases")) {
        throw runtime_error("LayerNorm deserialize did not find required weights/biases parameters.");
    }

    layer.initialized = true;
    layer.addToNetwork(network);
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::TrainableLayer::register_layer("layer_norm", &Thor::LayerNorm::deserialize);
    return true;
}();
}  // namespace
