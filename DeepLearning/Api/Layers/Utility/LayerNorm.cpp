#include "DeepLearning/Api/Layers/Utility/LayerNorm.h"

#include "DeepLearning/Implementation/Layers/RaggedCustomLayer.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/RaggedExpression.h"

#include <limits>
#include <stdexcept>
#include <string>

using namespace std;
using json = nlohmann::json;

namespace Thor {

std::vector<Tensor> LayerNorm::getFeatureInputs() const {
    if (raggedFeatureInputs.empty()) {
        return featureInputs;
    }

    std::vector<Tensor> inputs;
    inputs.reserve(raggedFeatureInputs.size() * 2);
    for (const RaggedTensor& ragged : raggedFeatureInputs) {
        inputs.push_back(ragged.getValues());
        inputs.push_back(ragged.getOffsets());
    }
    return inputs;
}

std::vector<uint32_t> LayerNorm::inputPortIndicesForTensor(Tensor tensor) const {
    std::vector<uint32_t> ports;
    if (!raggedFeatureInputs.empty()) {
        for (uint32_t i = 0; i < raggedFeatureInputs.size(); ++i) {
            if (tensor.getOriginalId() == raggedFeatureInputs[i].getValues().getOriginalId()) {
                ports.push_back(i * 2);
            }
            if (tensor.getOriginalId() == raggedFeatureInputs[i].getOffsets().getOriginalId()) {
                ports.push_back(i * 2 + 1);
            }
        }
        return ports;
    }

    for (uint32_t i = 0; i < featureInputs.size(); ++i) {
        if (tensor == featureInputs[i]) {
            ports.push_back(i);
        }
    }
    return ports;
}

Tensor LayerNorm::getFeatureOutput(Tensor inputTensor) const {
    auto it = outputTensorFromInputTensor.find(inputTensor);
    if (it == outputTensorFromInputTensor.end()) {
        throw std::runtime_error("Tensor is not connected to this LayerNorm layer.");
    }
    return it->second;
}

std::vector<Tensor> LayerNorm::getOutputsFromInput(Tensor inputTensor) {
    if (raggedFeatureInputs.empty()) {
        return {getFeatureOutput(inputTensor)};
    }
    if (inputPortIndicesForTensor(inputTensor).empty()) {
        throw std::runtime_error("LayerNorm received an unknown ragged input tensor.");
    }

    std::vector<Tensor> readyOutputs;
    for (uint32_t applicationIndex = 0; applicationIndex < raggedFeatureInputs.size(); ++applicationIndex) {
        const uint32_t valuesPort = applicationIndex * 2;
        const uint32_t offsetsPort = valuesPort + 1;
        if (!connectedInputPortIndices.contains(valuesPort) || !connectedInputPortIndices.contains(offsetsPort) ||
            emittedRaggedOutputApplications.contains(applicationIndex)) {
            continue;
        }
        THOR_THROW_IF_FALSE(applicationIndex < featureOutputs.size());
        emittedRaggedOutputApplications.insert(applicationIndex);
        readyOutputs.push_back(featureOutputs[applicationIndex]);
    }
    return readyOutputs;
}

void LayerNorm::informThatInputConnectionMade(Tensor inputTensor) {
    if (raggedFeatureInputs.empty()) {
        return;
    }
    std::vector<uint32_t> ports = inputPortIndicesForTensor(inputTensor);
    if (ports.empty()) {
        throw std::runtime_error("LayerNorm informed of connection for unknown ragged input tensor.");
    }
    uint32_t& cursor = nextTraversalInputCursorByTensorOriginalId[inputTensor.getOriginalId()];
    connectedInputPortIndices.insert(ports[cursor % ports.size()]);
    ++cursor;
}

void LayerNorm::resetGraphTraversalState() {
    connectedInputPortIndices.clear();
    emittedRaggedOutputApplications.clear();
    nextInputConnectionCursorByTensorOriginalId.clear();
    nextTraversalInputCursorByTensorOriginalId.clear();
}

int LayerNorm::getConnectionType(Tensor connectingTensor) const {
    if (!raggedFeatureInputs.empty()) {
        std::vector<uint32_t> ports = inputPortIndicesForTensor(connectingTensor);
        if (!ports.empty()) {
            uint32_t& cursor = nextInputConnectionCursorByTensorOriginalId[connectingTensor.getOriginalId()];
            const uint32_t port = ports[cursor % ports.size()];
            ++cursor;
            return static_cast<int>(port);
        }
    } else {
        for (uint32_t i = 0; i < featureInputs.size(); ++i) {
            if (connectingTensor == featureInputs[i]) {
                return static_cast<int>(i);
            }
        }
    }
    for (uint32_t i = 0; i < featureOutputs.size(); ++i) {
        if (connectingTensor == featureOutputs[i]) {
            return static_cast<int>(i);
        }
    }
    throw std::runtime_error("Tensor is not connected to this LayerNorm layer.");
}

bool LayerNorm::isLayerNormInputDataType(DataType dataType) {
    switch (dataType) {
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
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
        _parameterDataType = DataType::FP32;
    if (_weightsInitializer == nullptr)
        _weightsInitializer = UniformRandom::Builder().minValue(1.0f).maxValue(1.0f).build();
    if (_biasesInitializer == nullptr)
        _biasesInitializer = UniformRandom::Builder().minValue(0.0f).maxValue(0.0f).build();

    verifyConfig();

    LayerNorm layer;
    layer.featureInputs = _featureInputs;
    layer.raggedFeatureInputs = _raggedFeatureInputs;
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
        if (!layer.raggedFeatureInputs.empty()) {
            layer.raggedFeatureOutputs.push_back(layer.raggedFeatureInputs[i].withValues(out));
        }
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
    if (_parameterDataType.value() != DataType::FP32) {
        throw invalid_argument("LayerNorm currently requires fp32 weights/biases for cuDNN Frontend LayerNorm.");
    }
    const DataType inputDataType = _featureInputs.front().getDataType();
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
    if (!_raggedFeatureInputs.empty()) {
        if (_raggedFeatureInputs.size() != _featureInputs.size()) {
            throw invalid_argument("LayerNorm cannot mix dense and ragged feature inputs.");
        }
        for (uint32_t i = 0; i < _raggedFeatureInputs.size(); ++i) {
            const RaggedTensor& ragged = _raggedFeatureInputs[i];
            if (ragged.getValues() != _featureInputs[i]) {
                throw invalid_argument("LayerNorm ragged feature input values do not match the packed feature tensor.");
            }
            const vector<uint64_t> trailingDims = ragged.getTrailingDimensions();
            if (trailingDims.size() != 1 || trailingDims.front() == 0) {
                throw invalid_argument(
                    "LayerNorm(RaggedTensor) currently requires exactly one non-zero trailing channel dimension.");
            }
            if (_normalizedShape != trailingDims) {
                throw invalid_argument(
                    "LayerNorm(RaggedTensor) normalizedShape must be exactly the single trailing channel dimension and may not include the packed ragged row dimension.");
            }
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
    if (!raggedFeatureInputs.empty()) {
        THOR_THROW_IF_FALSE(!inputPortIndicesForTensor(connectingApiTensor).empty());
    } else {
        THOR_THROW_IF_FALSE(outputTensorFromInputTensor.find(connectingApiTensor) != outputTensorFromInputTensor.end());
    }

    vector<shared_ptr<ThorImplementation::PhysicalParameter>> physicalParameters;
    for (const auto& parameter : getParameters()) {
        THOR_THROW_IF_FALSE(parameter != nullptr);
        physicalParameters.push_back(parameter->stamp());
    }

    if (raggedFeatureInputs.empty()) {
        return make_shared<ThorImplementation::LayerNorm>(
            placement, inferenceOnly, normalizedShape, epsilon, parameterDataType, physicalParameters, getId());
    }

    const RaggedTensor& ragged = raggedFeatureInputs.front();
    const vector<uint64_t> trailingDims = ragged.getTrailingDimensions();
    THOR_THROW_IF_FALSE(trailingDims.size() == 1);
    const uint64_t elementsPerValue = trailingDims.front();

    ThorImplementation::RaggedExpression input =
        ThorImplementation::RaggedExpression::input("feature_input", "feature_offsets", ragged.getDescriptor());
    ThorImplementation::Expression weights =
        ThorImplementation::Expression::input("weights", std::nullopt, parameterDataType);
    ThorImplementation::Expression biases =
        ThorImplementation::Expression::input("biases", std::nullopt, parameterDataType);
    ThorImplementation::RaggedExpression output =
        input.layerNorm(weights, biases, epsilon, DataType::FP32, ragged.getValuesDataType());

    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{"feature_output", output.getValues()}}));
    auto physicalLayer = make_shared<ThorImplementation::RaggedCustomLayer>(
        ThorImplementation::DynamicExpression::fromExpressionDefinition(definition),
        vector<string>{"feature_input", "feature_offsets"},
        vector<string>{"feature_output"},
        placement,
        physicalParameters,
        inferenceOnly,
        ragged.getMaxTotalValues(),
        vector<uint64_t>{elementsPerValue},
        vector<uint64_t>{elementsPerValue},
        vector<uint32_t>{0},
        1,
        getId());
    physicalLayer->setLayerName(getLayerType());
    return physicalLayer;
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
    j["use_ragged"] = !raggedFeatureInputs.empty();
    if (!raggedFeatureInputs.empty()) {
        json raggedInputsJson = json::array();
        json raggedOutputsJson = json::array();
        for (const RaggedTensor& input : raggedFeatureInputs) raggedInputsJson.push_back(input.architectureJson());
        for (const RaggedTensor& output : raggedFeatureOutputs) raggedOutputsJson.push_back(output.architectureJson());
        j["ragged_inputs"] = std::move(raggedInputsJson);
        j["ragged_outputs"] = std::move(raggedOutputsJson);
    }

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
    layer.parameterDataType = j.at("parameter_data_type").get<DataType>();

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
    const bool useRagged = j.value("use_ragged", false);
    if (useRagged) {
        if (!j.contains("ragged_inputs") || !j.contains("ragged_outputs") ||
            j.at("ragged_inputs").size() != layer.featureInputs.size() ||
            j.at("ragged_outputs").size() != layer.featureOutputs.size()) {
            throw runtime_error("LayerNorm serialized ragged metadata does not match its input/output arity.");
        }
        for (uint32_t i = 0; i < layer.featureInputs.size(); ++i) {
            const json& raggedInputJson = j.at("ragged_inputs").at(i);
            if (raggedInputJson.at("values").at("id").get<uint64_t>() !=
                j.at("inputs").at(i).at("id").get<uint64_t>()) {
                throw runtime_error("LayerNorm serialized ragged input values must match the corresponding feature input.");
            }
            const uint64_t inputOffsetsId = raggedInputJson.at("offsets").at("id").get<uint64_t>();
            Tensor inputOffsets = network->getApiTensorByOriginalId(inputOffsetsId);
            RaggedTensor raggedInput = raggedInputJson.contains("max_values_per_row")
                ? RaggedTensor(layer.featureInputs[i], inputOffsets, raggedInputJson.at("max_values_per_row").get<uint64_t>())
                : RaggedTensor(layer.featureInputs[i], inputOffsets);
            if (raggedInput.getBatchSize() != raggedInputJson.at("batch_size").get<uint64_t>() ||
                raggedInput.getMaxTotalValues() != raggedInputJson.at("max_total_values").get<uint64_t>()) {
                throw runtime_error("LayerNorm serialized ragged input metadata does not match reconstructed tensors.");
            }
            layer.raggedFeatureInputs.push_back(raggedInput);
            layer.raggedFeatureOutputs.push_back(raggedInput.withValues(layer.featureOutputs[i]));
            const json& raggedOutputJson = j.at("ragged_outputs").at(i);
            if (raggedOutputJson.at("values").at("id").get<uint64_t>() !=
                    j.at("outputs").at(i).at("id").get<uint64_t>() ||
                raggedOutputJson.at("offsets").at("id").get<uint64_t>() != inputOffsetsId ||
                raggedOutputJson.at("batch_size").get<uint64_t>() != raggedInput.getBatchSize() ||
                raggedOutputJson.at("max_total_values").get<uint64_t>() != raggedInput.getMaxTotalValues() ||
                (raggedOutputJson.contains("max_values_per_row") &&
                 (!raggedInput.hasMaxValuesPerRow() ||
                  raggedOutputJson.at("max_values_per_row").get<uint64_t>() != raggedInput.getMaxValuesPerRow()))) {
                throw runtime_error("LayerNorm serialized ragged output must preserve the input row partition and capacity metadata.");
            }
        }
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
