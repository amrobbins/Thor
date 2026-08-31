#include "DeepLearning/Api/Layers/Utility/AdaptiveLayerNorm.h"

#include "DeepLearning/Implementation/Layers/RaggedCustomLayer.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/RaggedExpression.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

using namespace std;
using json = nlohmann::json;

namespace Thor {

bool AdaptiveLayerNorm::isAdaptiveLayerNormInputDataType(DataType dataType) {
    switch (dataType) {
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
            return true;
        default:
            return false;
    }
}

void AdaptiveLayerNorm::validateCudnnFrontendContract(uint64_t normalizedFeatureCount, DataType inputDataType) {
    if (inputDataType == DataType::FP32 && normalizedFeatureCount % 32 != 0) {
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

vector<Tensor> AdaptiveLayerNorm::getFeatureInputs() const {
    if (!raggedFeatureInput.has_value()) {
        return featureInputs;
    }
    THOR_THROW_IF_FALSE(featureInputs.size() == NUM_INPUT_PORTS);
    return {raggedFeatureInput->getValues(), raggedFeatureInput->getOffsets(), featureInputs[SCALE], featureInputs[BIAS]};
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
        _scaleBiasDataType = DataType::FP32;

    verifyConfig();

    AdaptiveLayerNorm layer;
    layer.featureInputs = {_featureInput.value(), _scaleInput.value(), _biasInput.value()};
    layer.raggedFeatureInput = _raggedFeatureInput;
    layer.normalizedShape = _normalizedShape;
    layer.epsilon = _epsilon.value();
    layer.scaleBiasDataType = _scaleBiasDataType.value();

    Tensor out = layer.featureInputs[DATA].clone();
    layer.featureOutputs.push_back(out);
    if (layer.raggedFeatureInput.has_value()) {
        layer.raggedFeatureOutput = layer.raggedFeatureInput->withValues(out);
    }
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
    if (_scaleBiasDataType.value() != DataType::FP32) {
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
    if (_raggedFeatureInput.has_value()) {
        const vector<uint64_t> trailingDims = _raggedFeatureInput->getTrailingDimensions();
        if (trailingDims.size() != 1 || trailingDims.front() == 0) {
            throw invalid_argument(
                "AdaptiveLayerNorm(RaggedTensor) currently requires exactly one non-zero trailing channel dimension.");
        }
        if (_normalizedShape != trailingDims) {
            throw invalid_argument(
                "AdaptiveLayerNorm(RaggedTensor) normalizedShape must equal the single trailing channel dimension.");
        }
    } else {
        AdaptiveLayerNorm::validateCudnnFrontendContract(normalizedFeatureCount, featureInput.getDataType());
    }
    if (scaleInput.getDimensions() != _normalizedShape || biasInput.getDimensions() != _normalizedShape) {
        throw invalid_argument("AdaptiveLayerNorm scale_input and bias_input dimensions must match normalizedShape.");
    }

    set<uint64_t> originalIds = {featureInput.getOriginalId(), scaleInput.getOriginalId(), biasInput.getOriginalId()};
    if (originalIds.size() != NUM_INPUT_PORTS) {
        throw invalid_argument("AdaptiveLayerNorm feature_input, scale_input, and bias_input must be distinct tensors.");
    }
    if (_raggedFeatureInput.has_value()) {
        const uint64_t offsetsOriginalId = _raggedFeatureInput->getOffsets().getOriginalId();
        if (originalIds.contains(offsetsOriginalId)) {
            throw invalid_argument("AdaptiveLayerNorm ragged offsets must be distinct from data, scale, and bias tensors.");
        }
    }
}

int AdaptiveLayerNorm::getConnectionType(Tensor connectingTensor) const {
    const vector<Tensor> physicalInputs = getFeatureInputs();
    for (uint32_t i = 0; i < physicalInputs.size(); ++i) {
        if (connectingTensor == physicalInputs[i])
            return static_cast<int>(i);
    }
    if (featureOutputs.size() == 1 && connectingTensor == featureOutputs[0])
        return 0;
    throw runtime_error("Tensor is not connected to this AdaptiveLayerNorm layer.");
}

void AdaptiveLayerNorm::informThatInputConnectionMade(Tensor inputTensor) {
    const vector<Tensor> physicalInputs = getFeatureInputs();
    auto it = find(physicalInputs.begin(), physicalInputs.end(), inputTensor);
    if (it == physicalInputs.end()) {
        throw runtime_error("AdaptiveLayerNorm informed of connection for unknown input tensor.");
    }
    connectedInputOriginalIds.insert(inputTensor.getOriginalId());
}

void AdaptiveLayerNorm::resetGraphTraversalState() {
    connectedInputOriginalIds.clear();
}

vector<Tensor> AdaptiveLayerNorm::getOutputsFromInput(Tensor inputTensor) {
    (void)getConnectionType(inputTensor);
    THOR_THROW_IF_FALSE(featureOutputs.size() == 1);
    if (connectedInputOriginalIds.size() == getFeatureInputs().size()) {
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
    THOR_THROW_IF_FALSE(initialized);
    (void)getConnectionType(connectingApiTensor);
    if (raggedFeatureInput.has_value()) {
        using ThorImplementation::DataType;
        using ThorImplementation::DynamicExpression;
        using ThorImplementation::Expression;
        using ThorImplementation::ExpressionDefinition;
        using ThorImplementation::RaggedCustomLayer;
        using ThorImplementation::RaggedExpression;
        using ThorImplementation::RaggedTensorDescriptor;
        using ThorImplementation::TensorDescriptor;

        const RaggedTensor& ragged = raggedFeatureInput.value();
        const vector<uint64_t> trailingDims = ragged.getTrailingDimensions();
        THOR_THROW_IF_FALSE(trailingDims.size() == 1);
        const uint64_t channels = trailingDims.front();

        RaggedExpression data = RaggedExpression::input("feature_input", "feature_offsets", ragged.getDescriptor());
        Expression perRowScale = Expression::input("scale_input", nullopt, scaleBiasDataType);
        Expression perRowBias = Expression::input("bias_input", nullopt, scaleBiasDataType);

        const TensorDescriptor conditioningValuesDescriptor(
            scaleBiasDataType, vector<uint64_t>{ragged.getMaxTotalValues(), channels});
        const RaggedTensorDescriptor conditioningDescriptor(
            conditioningValuesDescriptor, ragged.getDescriptor().getRowPartition());
        RaggedExpression tokenScale = data.segment_broadcast(perRowScale, conditioningDescriptor);
        RaggedExpression tokenBias = data.segment_broadcast(perRowBias, conditioningDescriptor);

        // Keep normalization on the optimized packed LayerNorm backend.  The
        // adaptive affine is applied afterward using the row-conditioned
        // broadcasts, so the scale/bias semantics are explicit and partition
        // preserving rather than relying on packed-capacity coincidence.
        Expression unitScale = Expression::fill(1.0, {channels}, DataType::FP32);
        Expression zeroBias = Expression::fill(0.0, {channels}, DataType::FP32);
        RaggedExpression normalized = data.layerNorm(unitScale, zeroBias, epsilon, DataType::FP32, DataType::FP32);
        RaggedExpression output = ((normalized * tokenScale) + tokenBias).cast(ragged.getValuesDataType());

        ExpressionDefinition definition =
            ExpressionDefinition::fromOutputs(Expression::outputs({{"feature_output", output.getValues()}}));
        auto physicalLayer = make_shared<RaggedCustomLayer>(
            DynamicExpression::fromExpressionDefinition(definition),
            vector<string>{"feature_input", "feature_offsets", "scale_input", "bias_input"},
            vector<string>{"feature_output"},
            placement,
            vector<shared_ptr<ThorImplementation::PhysicalParameter>>{},
            inferenceOnly,
            ragged.getMaxTotalValues(),
            vector<uint64_t>{channels},
            vector<uint64_t>{channels},
            vector<uint32_t>{0},
            1,
            getId(),
            vector<ThorImplementation::CustomLayer::DeclaredOutputDescriptor>{
                {ragged.getValuesDataType(), ragged.getValuesDimensions(), true}});
        physicalLayer->setLayerName(getLayerType());
        return physicalLayer;
    }
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
    j["use_ragged"] = raggedFeatureInput.has_value();
    if (raggedFeatureInput.has_value()) {
        THOR_THROW_IF_FALSE(raggedFeatureOutput.has_value());
        j["ragged_feature_input"] = raggedFeatureInput->architectureJson();
        j["ragged_feature_output"] = raggedFeatureOutput->architectureJson();
    }

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
    layer.scaleBiasDataType = j.at("scale_bias_data_type").get<DataType>();

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
    if (j.value("use_ragged", false)) {
        const json& raggedInputJson = j.at("ragged_feature_input");
        const uint64_t offsetsId = raggedInputJson.at("offsets").at("id").get<uint64_t>();
        Tensor offsets = network->getApiTensorByOriginalId(offsetsId);
        RaggedTensor raggedInput = raggedInputJson.contains("max_values_per_row")
            ? RaggedTensor(layer.featureInputs[DATA], offsets, raggedInputJson.at("max_values_per_row").get<uint64_t>())
            : RaggedTensor(layer.featureInputs[DATA], offsets);
        if (raggedInput.getBatchSize() != raggedInputJson.at("batch_size").get<uint64_t>() ||
            raggedInput.getMaxTotalValues() != raggedInputJson.at("max_total_values").get<uint64_t>()) {
            throw runtime_error("AdaptiveLayerNorm serialized ragged input metadata does not match reconstructed tensors.");
        }
        const json& raggedOutputJson = j.at("ragged_feature_output");
        if (raggedOutputJson.at("values").at("id").get<uint64_t>() != layer.featureOutputs[0].getOriginalId() ||
            raggedOutputJson.at("offsets").at("id").get<uint64_t>() != offsetsId ||
            raggedOutputJson.at("batch_size").get<uint64_t>() != raggedInput.getBatchSize() ||
            raggedOutputJson.at("max_total_values").get<uint64_t>() != raggedInput.getMaxTotalValues() ||
            (raggedOutputJson.contains("max_values_per_row") &&
             (!raggedInput.hasMaxValuesPerRow() ||
              raggedOutputJson.at("max_values_per_row").get<uint64_t>() != raggedInput.getMaxValuesPerRow()))) {
            throw runtime_error(
                "AdaptiveLayerNorm serialized ragged output must preserve the input row partition and capacity metadata.");
        }
        layer.raggedFeatureInput = raggedInput;
        layer.raggedFeatureOutput = raggedInput.withValues(layer.featureOutputs[0]);
    }
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
