#include "DeepLearning/Api/Layers/Utility/LayerNorm.h"

#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitivePlacement.h"
#include "DeepLearning/Implementation/Layers/RaggedCustomLayer.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/RaggedExpression.h"

#include <limits>
#include <stdexcept>
#include <string>

using namespace std;
using json = nlohmann::json;

namespace Thor {
namespace {

void validateCanonicalTensorReference(const json& record, const Tensor& tensor) {
    if (!record.is_object() || record.size() != 4 ||
        record.at("version").get<string>() != "1.0.0" ||
        record.at("id").get<uint64_t>() != tensor.getOriginalId() ||
        record.at("data_type").get<DataType>() != tensor.getDataType() ||
        record.at("dimensions").get<vector<uint64_t>>() != tensor.getDimensions()) {
        throw runtime_error("LayerNorm canonical tensor reference disagrees with the live graph tensor.");
    }
}

void validateCanonicalRaggedReference(const json& record, const RaggedTensor& tensor) {
    if (!SegmentedPrimitiveDetail::hasCanonicalLayerRaggedReferenceFields(record) ||
        SegmentedPrimitiveDetail::layerRaggedReferenceGrammar(record, "LayerNorm") !=
            SegmentedPrimitiveDetail::LayerRaggedReferenceGrammar::CANONICAL_LOGICAL ||
        record.at("ragged_rank").get<uint32_t>() != 1 ||
        record.at("max_values_per_row").get<uint64_t>() != tensor.getMaxValuesPerRow() ||
        record.contains("batch_size") || record.contains("max_total_values")) {
        throw runtime_error("LayerNorm canonical ragged reference has invalid semantic geometry.");
    }
    validateCanonicalTensorReference(record.at("values"), tensor.getValues());
    validateCanonicalTensorReference(record.at("offsets"), tensor.getRowPartitionToken());
}

uint64_t checkedRaggedOutputBytes(const vector<Tensor>& outputs, uint32_t batchSize) {
    if (outputs.empty()) throw runtime_error("LayerNorm requires a feature output.");
    if (batchSize == 0) throw overflow_error("LayerNorm placement B must be positive.");
    uint64_t result = 0;
    for (const Tensor& output : outputs) {
        const uint64_t bytes = output.getTotalSizeInBytes();
        const uint64_t multiplier = batchSize;
        if (bytes > numeric_limits<uint64_t>::max() / multiplier ||
            result > numeric_limits<uint64_t>::max() - bytes * multiplier) {
            throw overflow_error("LayerNorm placement output bytes overflow uint64_t.");
        }
        result += bytes * multiplier;
    }
    return result;
}

}  // namespace

uint64_t LayerNorm::raggedOutputBytesForPlacement(uint32_t batchSize) const {
    return checkedRaggedOutputBytes(featureOutputs, batchSize);
}

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
    const ThorImplementation::RaggedTensorDescriptor physicalRagged =
        SegmentedPrimitivePlacementDetail::physicalRaggedDescriptorForStamp(ragged, drivingLayer, "LayerNorm");
    const uint64_t runtimeBatch = physicalRagged.getBatchSize();
    const uint64_t runtimeCapacity = physicalRagged.getMaxTotalValues();
    if (runtimeBatch == 0 || runtimeBatch > numeric_limits<uint32_t>::max() || runtimeCapacity == 0 ||
        ragged.getMaxValuesPerRow() > numeric_limits<uint64_t>::max() / runtimeBatch ||
        runtimeCapacity != runtimeBatch * ragged.getMaxValuesPerRow()) {
        throw runtime_error("LayerNorm invalid placed ragged geometry.");
    }
    {
        // HOST_EXTENT is routed through the owner values fanout, not a device
        // [1] offsets/count tensor. Either logical port can stamp this layer
        // first (notably after saving/reloading a graph with a GradientRivet),
        // so the partition-token edge may also arrive with a packed values
        // descriptor. The placement helper above has already verified B * M,
        // the values dtype and the complete physical descriptor.
        const auto driver = SegmentedPrimitivePlacementDetail::physicalDescriptorForStamp(drivingLayer, "LayerNorm");
        if (driver != physicalRagged.getValuesDescriptor()) {
            throw runtime_error("LayerNorm stamp driver disagrees with packed values/HOST_EXTENT carrier.");
        }
    }

    // LayerNorm is a HOST_EXTENT consumer: its cuDNN stage chooses the packed
    // execution bucket from authoritative host partition metadata and never reads
    // the structural carrier payload on device. Build the marker explicitly with
    // HOST_EXTENT rather than using RaggedExpression::input()'s conservative
    // DEVICE_OFFSETS default; placement may legitimately route this port through
    // the FP16/BF16/FP32 values carrier.
    ThorImplementation::Expression featureInput = ThorImplementation::Expression::input(
        "feature_input", std::nullopt, ragged.getValuesDataType());
    ThorImplementation::Expression hostPartitionCarrier = ThorImplementation::Expression::input(
        "feature_offsets", std::nullopt, ragged.getOffsetsDataType());
    featureInput = featureInput.withRaggedRuntimeExtent(hostPartitionCarrier,
                                                       runtimeBatch,
                                                       runtimeCapacity,
                                                       elementsPerValue,
                                                       ThorImplementation::RaggedRuntimeExtentSource::HOST_EXTENT);
    ThorImplementation::Expression weights =
        ThorImplementation::Expression::input("weights", std::nullopt, parameterDataType);
    ThorImplementation::Expression biases =
        ThorImplementation::Expression::input("biases", std::nullopt, parameterDataType);
    ThorImplementation::Expression output = ThorImplementation::Expression::layerNorm(featureInput,
                                                                                      weights,
                                                                                      biases,
                                                                                      elementsPerValue,
                                                                                      epsilon,
                                                                                      DataType::FP32,
                                                                                      ragged.getValuesDataType(),
                                                                                      runtimeCapacity);

    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{"feature_output", output}}));
    auto physicalLayer = make_shared<ThorImplementation::RaggedCustomLayer>(
        ThorImplementation::DynamicExpression::fromExpressionDefinition(definition),
        vector<string>{"feature_input", "feature_offsets"},
        vector<string>{"feature_output"},
        placement,
        physicalParameters,
        inferenceOnly,
        runtimeCapacity,
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
    j["version"] = raggedFeatureInputs.empty() ? getLayerVersion() : "1.1.0";
    j["layer_type"] = "layer_norm";
    j["layer_name"] = string("layer") + to_string(getId());
    j["normalized_shape"] = normalizedShape;
    j["epsilon"] = epsilon;
    j["parameter_data_type"] = parameterDataType;
    j["use_ragged"] = !raggedFeatureInputs.empty();
    if (!raggedFeatureInputs.empty()) {
        json raggedInputsJson = json::array();
        json raggedOutputsJson = json::array();
        for (const RaggedTensor& input : raggedFeatureInputs) raggedInputsJson.push_back(SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(input));
        for (const RaggedTensor& output : raggedFeatureOutputs) raggedOutputsJson.push_back(SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(output));
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
    const string version = j.at("version").get<string>();
    if (version != "1.0.0" && version != "1.1.0")
        throw runtime_error("Unsupported version in LayerNorm::deserialize: " + version);
    const bool neutral = version == "1.1.0";
    if (neutral && (!j.is_object() || j.size() != 13 ||
                    j.at("factory") != Layer::Factory::Learning.value() ||
                    !j.at("use_ragged").get<bool>())) {
        throw runtime_error("LayerNorm 1.1.0 requires the explicit neutral ragged schema.");
    }
    if (j.at("layer_type").get<string>() != "layer_norm")
        throw runtime_error("Layer type mismatch in LayerNorm::deserialize: " + j.at("layer_type").get<string>());

    LayerNorm layer;
    layer.normalizedShape = j.at("normalized_shape").get<vector<uint64_t>>();
    layer.epsilon = j.at("epsilon").get<double>();
    layer.parameterDataType = j.at("parameter_data_type").get<DataType>();
    if (neutral && (layer.normalizedShape.size() != 1 ||
                    checkedFeatureCount(layer.normalizedShape, "normalizedShape") == 0 ||
                    !(layer.epsilon > 0.0) || layer.parameterDataType != DataType::FP32 ||
                    !j.at("inputs").is_array() || j.at("inputs").empty() ||
                    !j.at("outputs").is_array() || !j.at("ragged_inputs").is_array() ||
                    !j.at("ragged_outputs").is_array())) {
        throw runtime_error("LayerNorm neutral configuration is malformed.");
    }

    for (const json& inputJson : j.at("inputs")) {
        const uint64_t originalTensorId = inputJson.at("id").get<uint64_t>();
        Tensor input = network->getApiTensorByOriginalId(originalTensorId);
        if (neutral) validateCanonicalTensorReference(inputJson, input);
        layer.featureInputs.push_back(input);
    }
    const bool useRagged = j.value("use_ragged", false);
    std::vector<RaggedTensor> logicalInputs;
    if (useRagged) {
        const auto& records = j.at("ragged_inputs");
        if (!records.is_array() || records.size() != layer.featureInputs.size())
            throw runtime_error("LayerNorm serialized ragged input arity mismatch.");
        for (uint32_t i = 0; i < layer.featureInputs.size(); ++i) {
            if ((SegmentedPrimitiveDetail::layerRaggedReferenceGrammar(records.at(i), "LayerNorm") ==
                 SegmentedPrimitiveDetail::LayerRaggedReferenceGrammar::CANONICAL_LOGICAL) != neutral)
                throw runtime_error("LayerNorm layer version disagrees with ragged reference grammar.");
            if ((SegmentedPrimitiveDetail::layerRaggedReferenceGrammar(j.at("ragged_outputs").at(i), "LayerNorm") ==
                 SegmentedPrimitiveDetail::LayerRaggedReferenceGrammar::CANONICAL_LOGICAL) != neutral)
                throw runtime_error("LayerNorm layer version disagrees with ragged reference grammar.");
            RaggedTensor logical = SegmentedPrimitiveDetail::reconstructInput(records.at(i), network, "LayerNorm");
            if (logical.getValues().getOriginalId() != layer.featureInputs[i].getOriginalId())
                throw runtime_error("LayerNorm ragged feature input identity mismatch.");
            logicalInputs.push_back(std::move(logical));
        }
    }
    for (uint32_t i = 0; i < j.at("outputs").size(); ++i) {
        const json& outputJson = j.at("outputs").at(i);
        layer.featureOutputs.push_back(useRagged
            ? SegmentedPrimitiveDetail::reconstructPreservedOutputValues(
                  outputJson, j.at("ragged_outputs").at(i), logicalInputs.at(i), "LayerNorm")
            : Tensor::deserialize(outputJson, archiveReader.get()));
    }
    if (layer.featureInputs.size() != layer.featureOutputs.size()) {
        throw runtime_error("LayerNorm deserialize expected equal numbers of inputs and outputs.");
    }
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
            const RaggedTensor raggedInput = logicalInputs.at(i);
            const RaggedTensor raggedOutput = raggedInput.withValues(layer.featureOutputs[i]);
            const json& raggedOutputJson = j.at("ragged_outputs").at(i);
            SegmentedPrimitiveDetail::validateSerializedPreservedPartition(
                raggedOutputJson, raggedInputJson, raggedInput, "LayerNorm");
            if (neutral) {
                validateCanonicalRaggedReference(raggedInputJson, raggedInput);
                validateCanonicalRaggedReference(raggedOutputJson, raggedOutput);
            }
            {
                if (raggedInput.getValues().getOriginalId() != layer.featureInputs[i].getOriginalId() ||
                    raggedOutput.getValues().getOriginalId() != layer.featureOutputs[i].getOriginalId() ||
                    raggedOutput.getValues().getOriginalId() == raggedInput.getValues().getOriginalId() ||
                    raggedOutput.getValues().getOriginalId() == raggedInput.getRowPartitionToken().getOriginalId() ||
                    raggedInput.getTrailingDimensions() != layer.normalizedShape ||
                    raggedInput.getValuesDataType() != raggedOutput.getValuesDataType() ||
                    !isLayerNormInputDataType(raggedInput.getValuesDataType()) ||
                    raggedInput.getValuesDimensions() != raggedOutput.getValuesDimensions()) {
                    throw runtime_error("LayerNorm neutral input/output identity, type or shape mismatch.");
                }
            }
            layer.raggedFeatureInputs.push_back(raggedInput);
            layer.raggedFeatureOutputs.push_back(raggedOutput);
        }
    }
    if (neutral) {
        const RaggedTensor& first = layer.raggedFeatureInputs.front();
        for (const RaggedTensor& current : layer.raggedFeatureInputs) {
            if (current.getValuesDataType() != first.getValuesDataType() ||
                current.getValuesDimensions() != first.getValuesDimensions() ||
                current.getOffsetsDataType() != first.getOffsetsDataType() ||
                current.getMaxValuesPerRow() != first.getMaxValuesPerRow()) {
                throw runtime_error("LayerNorm neutral multi-input applications require the same semantic geometry.");
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
    if (neutral) {
        const auto& specs = layer.getParameters();
        if (specs.size() != 2) throw runtime_error("LayerNorm neutral parameters require weights and biases only.");
        for (const char* name : {"weights", "biases"}) {
            const auto spec = layer.getParameterSpecification(name);
            const json parameterJson = spec->architectureJson();
            const bool direct = parameterJson.contains("shape") && parameterJson.contains("dtype");
            const auto& shape = direct ? parameterJson.at("shape") : parameterJson.at("storage").at("dimensions");
            const auto& dtype = direct ? parameterJson.at("dtype") : parameterJson.at("storage").at("data_type");
            if (shape.get<vector<uint64_t>>() != layer.normalizedShape ||
                dtype.get<DataType>() != layer.parameterDataType || !spec->isTrainable()) {
                throw runtime_error("LayerNorm neutral affine parameter shape/dtype is invalid.");
            }
        }
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
