#include "DeepLearning/Api/Layers/Learning/Embedding.h"

#include "DeepLearning/Api/Initializers/Glorot.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitivePlacement.h"
#include "DeepLearning/Api/Parameter/ParameterSpecification.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/Embedding.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <nlohmann/json.hpp>

#include <limits>
#include <stdexcept>
#include <string>

using json = nlohmann::json;

namespace Thor {
namespace {

void validateCanonicalTensorReference(const json& record, const Tensor& tensor) {
    if (!record.is_object() || record.size() != 4 ||
        record.at("version").get<std::string>() != "1.0.0" ||
        record.at("id").get<uint64_t>() != tensor.getOriginalId() ||
        record.at("data_type").get<DataType>() != tensor.getDataType() ||
        record.at("dimensions").get<std::vector<uint64_t>>() != tensor.getDimensions()) {
        throw std::runtime_error("Embedding canonical tensor reference does not match the live graph tensor.");
    }
}

void validateCanonicalRaggedReference(const json& record, const RaggedTensor& tensor) {
    if (!SegmentedPrimitiveDetail::hasCanonicalLayerRaggedReferenceFields(record) ||
        SegmentedPrimitiveDetail::layerRaggedReferenceGrammar(record, "Embedding") !=
            SegmentedPrimitiveDetail::LayerRaggedReferenceGrammar::CANONICAL_LOGICAL ||
        record.at("ragged_rank").get<uint32_t>() != 1 ||
        record.at("max_values_per_row").get<uint64_t>() != tensor.getMaxValuesPerRow() ||
        record.contains("batch_size") || record.contains("max_total_values")) {
        throw std::runtime_error("Embedding canonical ragged geometry is malformed.");
    }
    validateCanonicalTensorReference(record.at("values"), tensor.getValues());
    validateCanonicalTensorReference(record.at("offsets"), tensor.getRowPartitionToken());
}

uint64_t checkedOutputBytes(const std::vector<Tensor>& outputs, uint32_t batchSize) {
    if (outputs.empty()) throw std::runtime_error("Embedding has no feature outputs.");
    if (batchSize == 0) throw std::overflow_error("Embedding placement batch must be positive.");
    uint64_t bytes = 0;
    for (const Tensor& output : outputs) {
        const uint64_t n = output.getTotalSizeInBytes();
        if (n > std::numeric_limits<uint64_t>::max() / batchSize ||
            bytes > std::numeric_limits<uint64_t>::max() - n * batchSize) {
            throw std::overflow_error("Embedding output memory requirement overflows uint64_t.");
        }
        bytes += n * batchSize;
    }
    return bytes;
}

}  // namespace

uint64_t Embedding::outputBytesForRaggedPlacement(uint32_t batchSize) const {
    return checkedOutputBytes(featureOutputs, batchSize);
}

bool Embedding::isSupportedIndexDataType(DataType dataType) {
    switch (dataType) {
        case DataType::UINT8:
        case DataType::UINT16:
        case DataType::UINT32:
        case DataType::UINT64:
            return true;
        default:
            return false;
    }
}

bool Embedding::isSupportedWeightsDataType(DataType dataType) {
    switch (dataType) {
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
            return true;
        default:
            return false;
    }
}

std::string Embedding::dataTypeName(DataType dataType) { return ThorImplementation::TensorDescriptor::getElementTypeName(dataType); }

void Embedding::validateIndexTensor(const Tensor& tensor, const std::string& what) {
    if (!tensor.isInitialized()) {
        throw std::invalid_argument("Embedding " + what + " tensor is not initialized.");
    }
    if (tensor.getDimensions().empty()) {
        throw std::invalid_argument("Embedding " + what + " tensor must have at least one dimension.");
    }
    if (!isSupportedIndexDataType(tensor.getDataType())) {
        throw std::invalid_argument("Embedding " + what + " dtype must be uint8, uint16, uint32, or uint64. Got " +
                                    dataTypeName(tensor.getDataType()) + ".");
    }
}

std::vector<Tensor> Embedding::getFeatureInputs() const {
    if (raggedFeatureInputs.empty()) return featureInputs;
    std::vector<Tensor> inputs;
    inputs.reserve(raggedFeatureInputs.size() * 2);
    for (const RaggedTensor& input : raggedFeatureInputs) {
        inputs.push_back(input.getValues());
        inputs.push_back(input.getOffsets());
    }
    return inputs;
}

std::vector<uint32_t> Embedding::inputPortIndicesForTensor(Tensor tensor) const {
    std::vector<uint32_t> ports;
    if (raggedFeatureInputs.empty()) return ports;
    for (uint32_t i = 0; i < raggedFeatureInputs.size(); ++i) {
        if (tensor.getOriginalId() == raggedFeatureInputs[i].getValues().getOriginalId()) ports.push_back(i * 2);
        if (tensor.getOriginalId() == raggedFeatureInputs[i].getOffsets().getOriginalId()) ports.push_back(i * 2 + 1);
    }
    return ports;
}

std::vector<Tensor> Embedding::getOutputsFromInput(Tensor inputTensor) {
    if (raggedFeatureInputs.empty()) return {getFeatureOutput(inputTensor)};
    if (inputPortIndicesForTensor(inputTensor).empty()) {
        throw std::runtime_error("Embedding received an unknown ragged input tensor.");
    }
    std::vector<Tensor> readyOutputs;
    for (uint32_t app = 0; app < raggedFeatureInputs.size(); ++app) {
        const uint32_t valuesPort = app * 2;
        const uint32_t offsetsPort = valuesPort + 1;
        if (!connectedInputPortIndices.contains(valuesPort) || !connectedInputPortIndices.contains(offsetsPort) ||
            emittedRaggedOutputApplications.contains(app)) {
            continue;
        }
        emittedRaggedOutputApplications.insert(app);
        readyOutputs.push_back(featureOutputs.at(app));
    }
    return readyOutputs;
}

void Embedding::informThatInputConnectionMade(Tensor inputTensor) {
    if (raggedFeatureInputs.empty()) return;
    const std::vector<uint32_t> ports = inputPortIndicesForTensor(inputTensor);
    if (ports.empty()) throw std::runtime_error("Embedding informed of a connection for an unknown ragged input tensor.");
    uint32_t& cursor = nextTraversalInputCursorByTensorOriginalId[inputTensor.getOriginalId()];
    connectedInputPortIndices.insert(ports[cursor % ports.size()]);
    ++cursor;
}

void Embedding::resetGraphTraversalState() {
    connectedInputPortIndices.clear();
    emittedRaggedOutputApplications.clear();
    nextInputConnectionCursorByTensorOriginalId.clear();
    nextTraversalInputCursorByTensorOriginalId.clear();
}

int Embedding::getConnectionType(Tensor connectingTensor) const {
    if (!raggedFeatureInputs.empty()) {
        const std::vector<uint32_t> inputPorts = inputPortIndicesForTensor(connectingTensor);
        if (!inputPorts.empty()) {
            uint32_t& cursor = nextInputConnectionCursorByTensorOriginalId[connectingTensor.getOriginalId()];
            const uint32_t port = inputPorts[cursor % inputPorts.size()];
            ++cursor;
            return static_cast<int>(port);
        }
    } else {
        for (uint32_t i = 0; i < featureInputs.size(); ++i) {
            if (connectingTensor == featureInputs[i]) return static_cast<int>(i);
        }
    }
    for (uint32_t i = 0; i < featureOutputs.size(); ++i) {
        if (connectingTensor == featureOutputs[i]) return static_cast<int>(i);
    }
    throw std::runtime_error("Tensor is not connected to this Embedding layer.");
}

Embedding Embedding::Builder::build() {
    if (!_sparseGradients.has_value())
        _sparseGradients = true;
    if (!_weightsDataType.has_value())
        _weightsDataType = DataType::FP32;
    if (_weightsInitializer == nullptr)
        _weightsInitializer = Glorot::Builder().build();

    verifyConfig();

    Embedding embedding;
    embedding.featureInputs = _featureInputs;
    embedding.raggedFeatureInputs = _raggedFeatureInputs;
    embedding.vocabularySize = _vocabularySize.value();
    embedding.embeddingDim = _embeddingDim.value();
    embedding.weightsDataType = _weightsDataType.value();
    embedding.paddingIndex = _paddingIndex;
    embedding.sparseGradients = _sparseGradients.value();

    ParameterSpecification::Builder weightsParameterBuilder;
    weightsParameterBuilder.name("weights")
        .shape({embedding.vocabularySize, embedding.embeddingDim})
        .dtype(embedding.weightsDataType)
        .initializer(_weightsInitializer)
        .trainable(true);
    if (_weightsOptimizer != nullptr)
        weightsParameterBuilder.optimizer(_weightsOptimizer);
    embedding.addParameter(std::make_shared<ParameterSpecification>(weightsParameterBuilder.build()));

    embedding.initialized = true;

    for (uint32_t i = 0; i < embedding.featureInputs.size(); ++i) {
        std::vector<uint64_t> outDims = embedding.featureInputs[i].getDimensions();
        outDims.push_back(embedding.embeddingDim);
        Tensor out(embedding.weightsDataType, outDims);
        embedding.featureOutputs.push_back(out);
        if (!embedding.raggedFeatureInputs.empty()) {
            embedding.raggedFeatureOutputs.push_back(embedding.raggedFeatureInputs[i].withValues(out));
        }
        embedding.outputTensorFromInputTensor[embedding.featureInputs[i]] = out;
        embedding.inputTensorFromOutputTensor[out] = embedding.featureInputs[i];
    }

    embedding.addToNetwork(_network.value());
    return embedding;
}

void Embedding::Builder::verifyConfig() const {
    if (!_network.has_value()) {
        throw std::invalid_argument("Embedding::Builder requires network().");
    }
    if (_featureInputs.empty()) {
        throw std::invalid_argument("Embedding::Builder requires at least one featureInput().");
    }
    if (!_vocabularySize.has_value() || _vocabularySize.value() == 0) {
        throw std::invalid_argument("Embedding vocabularySize must be non-zero.");
    }
    if (!_embeddingDim.has_value() || _embeddingDim.value() == 0) {
        throw std::invalid_argument("Embedding embeddingDim must be non-zero.");
    }
    if (!_sparseGradients.value()) {
        throw std::invalid_argument("Embedding only supports sparseGradients(true); dense gradients are intentionally not implemented.");
    }
    if (!Embedding::isSupportedWeightsDataType(_weightsDataType.value())) {
        throw std::invalid_argument("Embedding weightsDataType must be fp16, bf16, or fp32. Got " +
                                    Embedding::dataTypeName(_weightsDataType.value()) + ".");
    }
    if (_paddingIndex.has_value() && _paddingIndex.value() >= _vocabularySize.value()) {
        throw std::invalid_argument("Embedding paddingIndex must be less than vocabularySize.");
    }

    const DataType inputDataType = _featureInputs.front().getDataType();
    const std::vector<uint64_t> inputDimensions = _featureInputs.front().getDimensions();
    for (uint32_t i = 0; i < _featureInputs.size(); ++i) {
        validateIndexTensor(_featureInputs[i], "featureInput " + std::to_string(i));
        if (_featureInputs[i].getDataType() != inputDataType) {
            throw std::invalid_argument("Embedding all feature inputs must have the same data type.");
        }
        if (_featureInputs[i].getDimensions() != inputDimensions) {
            throw std::invalid_argument("Embedding all feature inputs must have the same dimensions.");
        }
    }
}

std::shared_ptr<ThorImplementation::Layer> Embedding::stamp(ThorImplementation::TensorPlacement placement,
                                                            std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                            std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                            Thor::Tensor connectingApiTensor,
                                                            const bool inferenceOnly) const {
    (void)drivingApiLayer;

    THOR_THROW_IF_FALSE(initialized);
    if (!raggedFeatureInputs.empty()) {
        THOR_THROW_IF_FALSE(!inputPortIndicesForTensor(connectingApiTensor).empty());
    } else {
        THOR_THROW_IF_FALSE(outputTensorFromInputTensor.find(connectingApiTensor) != outputTensorFromInputTensor.end());
    }

    std::vector<std::shared_ptr<ThorImplementation::PhysicalParameter>> physicalParameters;
    for (const auto& parameter : getParameters()) {
        THOR_THROW_IF_FALSE(parameter != nullptr);
        physicalParameters.push_back(parameter->stamp());
    }

    std::optional<ThorImplementation::RaggedEmbeddingConfig> raggedConfig = std::nullopt;
    if (!raggedFeatureInputs.empty()) {
        const RaggedTensor& input = raggedFeatureInputs.front();
        const Tensor values = input.getValues();
        ThorImplementation::RaggedTensorDescriptor physicalInput =
            SegmentedPrimitivePlacementDetail::physicalRaggedDescriptorForStamp(
                input, drivingLayer, "Embedding");
        const uint64_t batchSize = physicalInput.getBatchSize();
        const uint64_t maxTotalValues = physicalInput.getMaxTotalValues();
        if (batchSize == 0 || batchSize > std::numeric_limits<uint32_t>::max() || maxTotalValues == 0 ||
            input.getMaxValuesPerRow() > std::numeric_limits<uint64_t>::max() / batchSize ||
            maxTotalValues != batchSize * input.getMaxValuesPerRow()) {
            throw std::runtime_error("Embedding physical placement geometry is invalid.");
        }
        const auto driver = SegmentedPrimitivePlacementDetail::physicalDescriptorForStamp(drivingLayer, "Embedding");
        const auto ports = inputPortIndicesForTensor(connectingApiTensor);
        THOR_THROW_IF_FALSE(!ports.empty());
        const bool valuesPort = ports.front() % 2 == 0;
        const auto expected = valuesPort ? physicalInput.getValuesDescriptor()
                                         : ThorImplementation::TensorDescriptor(input.getOffsetsDataType(), {1});
        if (driver != expected) {
            throw std::runtime_error("Embedding physical stamp driver disagrees with the values/active-count contract.");
        }
        const uint64_t elementsPerValue = SegmentedPrimitiveDetail::elementsPerValue(
            input.getTrailingDimensions(), "Embedding");
        if (values.getTotalNumElements() % elementsPerValue != 0) {
            throw std::runtime_error("Embedding logical indices dimensions are not a whole number of values.");
        }
        raggedConfig = ThorImplementation::RaggedEmbeddingConfig{
            .batchSize = batchSize,
            .maxTotalValues = maxTotalValues,
            .elementsPerValue = elementsPerValue,
            .offsetsDataType = input.getOffsetsDataType(),
        };
    }

    std::shared_ptr<ThorImplementation::Embedding> physicalEmbedding = std::make_shared<ThorImplementation::Embedding>(placement,
                                                                                                                        physicalParameters,
                                                                                                                        vocabularySize,
                                                                                                                        embeddingDim,
                                                                                                                        weightsDataType,
                                                                                                                        paddingIndex,
                                                                                                                        sparseGradients,
                                                                                                                        inferenceOnly,
                                                                                                                        getId(),
                                                                                                                        raggedConfig);
    physicalEmbedding->setName(getLayerType());
    return physicalEmbedding;
}

json Embedding::architectureJson() const {
    json j;
    j["factory"] = Layer::Factory::Learning.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = "embedding";
    j["layer_name"] = std::string("layer") + std::to_string(getId());
    j["vocabulary_size"] = vocabularySize;
    j["embedding_dim"] = embeddingDim;
    j["weights_data_type"] = weightsDataType;
    j["sparse_gradients"] = sparseGradients;
    j["use_ragged"] = !raggedFeatureInputs.empty();
    if (paddingIndex.has_value()) {
        j["padding_index"] = paddingIndex.value();
    } else {
        j["padding_index"] = nullptr;
    }

    json inputs = json::array();
    for (const Tensor& input : featureInputs)
        inputs.push_back(input.architectureJson());
    j["inputs"] = inputs;

    json outputs = json::array();
    for (const Tensor& output : featureOutputs)
        outputs.push_back(output.architectureJson());
    j["outputs"] = outputs;

    if (!raggedFeatureInputs.empty()) {
        json raggedInputs = json::array();
        json raggedOutputs = json::array();
        for (const RaggedTensor& input : raggedFeatureInputs)
            raggedInputs.push_back(SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(input));
        for (const RaggedTensor& output : raggedFeatureOutputs)
            raggedOutputs.push_back(SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(output));
        j["ragged_inputs"] = std::move(raggedInputs);
        j["ragged_outputs"] = std::move(raggedOutputs);
    }

    j["parameters"] = getParametersArchitectureJson()["parameters"];
    return j;
}

json Embedding::serialize(thor_file::TarWriter& archiveWriter,
                          Stream stream,
                          bool saveOptimizerState,
                          ThorImplementation::StampedNetwork& stampedNetwork) const {
    json j = architectureJson();
    Parameterizable::serializeParameters(j["parameters"], archiveWriter, stream, saveOptimizerState, stampedNetwork,
                                         "layer" + std::to_string(getId()));
    return j;
}

void Embedding::deserialize(std::shared_ptr<thor_file::TarReader>& archiveReader, const json& j, Network* network) {
    const std::string serializedVersion = j.at("version").get<std::string>();
    if (serializedVersion != "1.0.0" && serializedVersion != "1.1.0" && serializedVersion != "1.2.0")
        throw std::runtime_error("Unsupported version in Embedding::deserialize: " + serializedVersion);
    if (j.at("layer_type").get<std::string>() != "embedding")
        throw std::runtime_error("Layer type mismatch in Embedding::deserialize: " + j.at("layer_type").get<std::string>());

    const bool useRagged = j.value("use_ragged", false);
    // The layer version is an input grammar, not a switch between fixed and
    // placement-independent model geometries. Both historical packed and
    // canonical logical references reconstruct the same logical tensors.
    const bool canonicalRagged = useRagged && j.contains("ragged_inputs") &&
        j.at("ragged_inputs").is_array() && !j.at("ragged_inputs").empty() &&
        SegmentedPrimitiveDetail::layerRaggedReferenceGrammar(j.at("ragged_inputs").at(0), "Embedding") ==
            SegmentedPrimitiveDetail::LayerRaggedReferenceGrammar::CANONICAL_LOGICAL;
    // Archives actually written as 1.2.0 used the canonical reference grammar.
    // Continue to recognize those archived records; new writers use 1.1.0.
    if (serializedVersion == "1.2.0" && !canonicalRagged) {
        throw std::runtime_error("Embedding 1.2.0 archive requires canonical ragged references.");
    }
    if (canonicalRagged && (!j.is_object() || j.size() != (j.contains("padding_index") ? 15 : 14) ||
                            j.at("factory") != Layer::Factory::Learning.value() || !useRagged)) {
        throw std::runtime_error("Embedding canonical ragged layer JSON is malformed.");
    }

    Embedding embedding;
    embedding.vocabularySize = j.at("vocabulary_size").get<uint64_t>();
    embedding.embeddingDim = j.at("embedding_dim").get<uint64_t>();
    embedding.weightsDataType = j.at("weights_data_type").get<DataType>();
    embedding.sparseGradients = j.value("sparse_gradients", true);
    if (j.contains("padding_index") && !j.at("padding_index").is_null()) {
        embedding.paddingIndex = j.at("padding_index").get<uint64_t>();
    }

    for (const json& inputJson : j.at("inputs")) {
        uint64_t originalTensorId = inputJson.at("id").get<uint64_t>();
        Tensor input = network->getApiTensorByOriginalId(originalTensorId);
        if (canonicalRagged) validateCanonicalTensorReference(inputJson, input);
        embedding.featureInputs.push_back(input);
    }
    std::vector<RaggedTensor> logicalInputs;
    if (useRagged) {
        const auto& records = j.at("ragged_inputs");
        if (!records.is_array() || records.size() != embedding.featureInputs.size())
            throw std::runtime_error("Embedding ragged input arity mismatch.");
        for (uint32_t i = 0; i < embedding.featureInputs.size(); ++i) {
            if ((SegmentedPrimitiveDetail::layerRaggedReferenceGrammar(records.at(i), "Embedding") ==
                 SegmentedPrimitiveDetail::LayerRaggedReferenceGrammar::CANONICAL_LOGICAL) != canonicalRagged ||
                (SegmentedPrimitiveDetail::layerRaggedReferenceGrammar(j.at("ragged_outputs").at(i), "Embedding") ==
                 SegmentedPrimitiveDetail::LayerRaggedReferenceGrammar::CANONICAL_LOGICAL) != canonicalRagged) {
                throw std::runtime_error("Embedding ragged references use inconsistent historical grammars.");
            }
            RaggedTensor logical = SegmentedPrimitiveDetail::reconstructInput(records.at(i), network, "Embedding");
            if (logical.getValues().getOriginalId() != embedding.featureInputs[i].getOriginalId())
                throw std::runtime_error("Embedding ragged feature input identity mismatch.");
            logicalInputs.push_back(std::move(logical));
        }
    }
    for (uint32_t i = 0; i < j.at("outputs").size(); ++i) {
        const json& outputJson = j.at("outputs").at(i);
        embedding.featureOutputs.push_back(useRagged
            ? SegmentedPrimitiveDetail::reconstructPreservedOutputValues(
                  outputJson, j.at("ragged_outputs").at(i), logicalInputs.at(i), "Embedding")
            : Tensor::deserialize(outputJson, archiveReader.get()));
    }
    if (embedding.featureInputs.size() != embedding.featureOutputs.size()) {
        throw std::runtime_error("Embedding deserialize expected equal numbers of inputs and outputs.");
    }
    if (canonicalRagged && (embedding.featureInputs.empty() || !embedding.sparseGradients || embedding.vocabularySize == 0 || embedding.embeddingDim == 0 ||
                    !isSupportedIndexDataType(embedding.featureInputs.front().getDataType()) ||
                    !isSupportedWeightsDataType(embedding.weightsDataType) ||
                    (embedding.paddingIndex.has_value() && embedding.paddingIndex.value() >= embedding.vocabularySize))) {
        throw std::runtime_error("Embedding canonical layer configuration is invalid.");
    }
    if (useRagged) {
        if (serializedVersion == "1.0.0") {
            throw std::runtime_error("Embedding 1.0.0 archives cannot contain ragged metadata.");
        }
        if (!j.contains("ragged_inputs") || !j.contains("ragged_outputs") ||
            j.at("ragged_inputs").size() != embedding.featureInputs.size() ||
            j.at("ragged_outputs").size() != embedding.featureOutputs.size()) {
            throw std::runtime_error("Embedding serialized ragged metadata does not match its input/output arity.");
        }
        for (uint32_t i = 0; i < embedding.featureInputs.size(); ++i) {
            const json& raggedInputJson = j.at("ragged_inputs").at(i);
            const RaggedTensor raggedInput = logicalInputs.at(i);
            if (canonicalRagged) validateCanonicalRaggedReference(raggedInputJson, raggedInput);
            const json& raggedOutputJson = j.at("ragged_outputs").at(i);
            const RaggedTensor raggedOutput = raggedInput.withValues(embedding.featureOutputs[i]);
            SegmentedPrimitiveDetail::validateSerializedPreservedPartition(
                raggedOutputJson, raggedInputJson, raggedInput, "Embedding");
            if (canonicalRagged) validateCanonicalRaggedReference(raggedOutputJson, raggedOutput);
            {
                const auto expectedDimensions = raggedInput.getValuesDimensions();
                auto actualDimensions = expectedDimensions;
                actualDimensions.push_back(embedding.embeddingDim);
                if (raggedOutput.getValuesDimensions() != actualDimensions ||
                    raggedOutput.getValuesDataType() != embedding.weightsDataType ||
                    raggedOutput.getValues().getOriginalId() == raggedInput.getValues().getOriginalId() ||
                    raggedOutput.getValues().getOriginalId() == raggedInput.getOffsets().getOriginalId()) {
                    throw std::runtime_error("Embedding canonical output identity or shape is not an embedding of the input.");
                }
            }
            embedding.raggedFeatureInputs.push_back(raggedInput);
            embedding.raggedFeatureOutputs.push_back(raggedOutput);
        }
    }
    if (canonicalRagged) {
        for (uint32_t i = 0; i < embedding.raggedFeatureInputs.size(); ++i) {
            const RaggedTensor& input = embedding.raggedFeatureInputs[i];
            if (input.getValuesDataType() != embedding.featureInputs.front().getDataType() ||
                input.getValuesDimensions() != embedding.featureInputs.front().getDimensions() ||
                input.getOffsetsDataType() != embedding.raggedFeatureInputs.front().getOffsetsDataType() ||
                input.getMaxValuesPerRow() != embedding.raggedFeatureInputs.front().getMaxValuesPerRow()) {
                throw std::runtime_error("Embedding canonical applications require identical semantic geometry and dtypes.");
            }
        }
    }
    for (uint32_t i = 0; i < embedding.featureInputs.size(); ++i) {
        embedding.outputTensorFromInputTensor[embedding.featureInputs[i]] = embedding.featureOutputs[i];
        embedding.inputTensorFromOutputTensor[embedding.featureOutputs[i]] = embedding.featureInputs[i];
    }

    if (j.contains("parameters")) {
        const json& parametersJson = j.at("parameters");
        if (!parametersJson.is_object()) {
            throw std::runtime_error("Embedding parameters must be an object keyed by parameter name.");
        }
        for (auto it = parametersJson.begin(); it != parametersJson.end(); ++it) {
            ParameterSpecification parameter = ParameterSpecification::deserialize(it.value(), archiveReader);
            embedding.addParameter(std::make_shared<ParameterSpecification>(std::move(parameter)));
        }
    }
    if (!embedding.hasParameter("weights")) {
        throw std::runtime_error("Embedding deserialize did not find required weights parameter.");
    }
    if (canonicalRagged) {
        const auto& params = embedding.getParameters();
        if (params.size() != 1 || params.front() == nullptr || params.front()->getName() != "weights") {
            throw std::runtime_error("Embedding canonical model must have one named weights parameter.");
        }
        const json spec = params.front()->architectureJson();
        const bool staticShape = spec.contains("shape") && spec.contains("dtype");
        const auto& shapeJson = staticShape ? spec.at("shape") : spec.at("storage").at("dimensions");
        const auto& dtypeJson = staticShape ? spec.at("dtype") : spec.at("storage").at("data_type");
        if (shapeJson.get<std::vector<uint64_t>>() !=
                std::vector<uint64_t>{embedding.vocabularySize, embedding.embeddingDim} ||
            dtypeJson.get<DataType>() != embedding.weightsDataType || !params.front()->isTrainable()) {
            throw std::runtime_error("Embedding canonical weights specification must match vocabulary and embedding dimensions.");
        }
    }

    embedding.initialized = true;
    embedding.addToNetwork(network);
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::TrainableLayer::register_layer("embedding", &Thor::Embedding::deserialize);
    return true;
}();
}  // namespace
