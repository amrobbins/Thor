#include "DeepLearning/Api/Layers/Learning/Embedding.h"

#include "DeepLearning/Api/Initializers/Glorot.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Parameter/ParameterSpecification.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/Embedding.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <nlohmann/json.hpp>

#include <limits>
#include <stdexcept>
#include <string>

using json = nlohmann::json;

namespace Thor {

bool Embedding::isSupportedIndexDataType(Tensor::DataType dataType) {
    switch (dataType) {
        case Tensor::DataType::UINT32:
        case Tensor::DataType::UINT64:
            return true;
        default:
            return false;
    }
}

bool Embedding::isSupportedWeightsDataType(Tensor::DataType dataType) {
    switch (dataType) {
        case Tensor::DataType::FP16:
        case Tensor::DataType::BF16:
        case Tensor::DataType::FP32:
            return true;
        default:
            return false;
    }
}

std::string Embedding::dataTypeName(Tensor::DataType dataType) { return ThorImplementation::TensorDescriptor::getElementTypeName(dataType); }

void Embedding::validateIndexTensor(const Tensor& tensor, const std::string& what) {
    if (!tensor.isInitialized()) {
        throw std::invalid_argument("Embedding " + what + " tensor is not initialized.");
    }
    if (tensor.getDimensions().empty()) {
        throw std::invalid_argument("Embedding " + what + " tensor must have at least one dimension.");
    }
    if (!isSupportedIndexDataType(tensor.getDataType())) {
        throw std::invalid_argument("Embedding " + what + " dtype must be uint32 or uint64. Got " +
                                    dataTypeName(tensor.getDataType()) + ".");
    }
}

Embedding Embedding::Builder::build() {
    if (!_sparseGradients.has_value())
        _sparseGradients = true;
    if (!_weightsDataType.has_value())
        _weightsDataType = Tensor::DataType::FP32;
    if (_weightsInitializer == nullptr)
        _weightsInitializer = Glorot::Builder().build();

    verifyConfig();

    Embedding embedding;
    embedding.featureInputs = _featureInputs;
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

    const Tensor::DataType inputDataType = _featureInputs.front().getDataType();
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
    (void)drivingLayer;
    (void)drivingApiLayer;

    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(outputTensorFromInputTensor.find(connectingApiTensor) != outputTensorFromInputTensor.end());

    std::vector<std::shared_ptr<ThorImplementation::PhysicalParameter>> physicalParameters;
    for (const auto& parameter : getParameters()) {
        THOR_THROW_IF_FALSE(parameter != nullptr);
        physicalParameters.push_back(parameter->stamp());
    }

    std::shared_ptr<ThorImplementation::Embedding> physicalEmbedding = std::make_shared<ThorImplementation::Embedding>(
        placement, physicalParameters, vocabularySize, embeddingDim, weightsDataType, paddingIndex, sparseGradients, inferenceOnly, getId());
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
    if (j.at("version").get<std::string>() != "1.0.0")
        throw std::runtime_error("Unsupported version in Embedding::deserialize: " + j.at("version").get<std::string>());
    if (j.at("layer_type").get<std::string>() != "embedding")
        throw std::runtime_error("Layer type mismatch in Embedding::deserialize: " + j.at("layer_type").get<std::string>());

    Embedding embedding;
    embedding.vocabularySize = j.at("vocabulary_size").get<uint64_t>();
    embedding.embeddingDim = j.at("embedding_dim").get<uint64_t>();
    embedding.weightsDataType = j.at("weights_data_type").get<Tensor::DataType>();
    embedding.sparseGradients = j.value("sparse_gradients", true);
    if (j.contains("padding_index") && !j.at("padding_index").is_null()) {
        embedding.paddingIndex = j.at("padding_index").get<uint64_t>();
    }

    for (const json& inputJson : j.at("inputs")) {
        uint64_t originalTensorId = inputJson.at("id").get<uint64_t>();
        embedding.featureInputs.push_back(network->getApiTensorByOriginalId(originalTensorId));
    }
    for (const json& outputJson : j.at("outputs")) {
        embedding.featureOutputs.push_back(Tensor::deserialize(outputJson, archiveReader.get()));
    }
    if (embedding.featureInputs.size() != embedding.featureOutputs.size()) {
        throw std::runtime_error("Embedding deserialize expected equal numbers of inputs and outputs.");
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
