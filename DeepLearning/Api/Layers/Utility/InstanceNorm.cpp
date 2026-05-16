#include "DeepLearning/Api/Layers/Utility/InstanceNorm.h"

#include <limits>
#include <stdexcept>
#include <string>

using namespace std;
using json = nlohmann::json;

namespace Thor {

bool InstanceNorm::isInstanceNormInputDataType(Tensor::DataType dataType) {
    switch (dataType) {
        case Tensor::DataType::FP16:
        case Tensor::DataType::BF16:
        case Tensor::DataType::FP32:
            return true;
        default:
            return false;
    }
}

namespace {

bool isReducedPrecisionInstanceNormInputDataType(Tensor::DataType dataType) {
    return dataType == Tensor::DataType::FP16 || dataType == Tensor::DataType::BF16;
}

}  // namespace

uint64_t InstanceNorm::checkedChannelCount(uint64_t channelCount) {
    if (channelCount == 0) {
        throw invalid_argument("InstanceNorm channel count must be non-zero.");
    }
    return channelCount;
}

uint64_t InstanceNorm::channelCountFromInputDims(const vector<uint64_t>& inputDims) {
    validateInputShape(inputDims);
    return checkedChannelCount(inputDims[0]);
}

void InstanceNorm::validateInputShape(const vector<uint64_t>& inputDims) {
    if (inputDims.size() < 2) {
        throw invalid_argument("InstanceNorm feature input dimensions must be [C, spatial...] with at least one spatial dimension.");
    }
    if (inputDims[0] == 0) {
        throw invalid_argument("InstanceNorm channel dimension must be non-zero.");
    }
    uint64_t spatial = 1;
    for (size_t i = 1; i < inputDims.size(); ++i) {
        if (inputDims[i] == 0) {
            throw invalid_argument("InstanceNorm spatial dimensions must be non-zero.");
        }
        if (spatial > numeric_limits<uint64_t>::max() / inputDims[i]) {
            throw invalid_argument("InstanceNorm spatial element count overflows uint64_t.");
        }
        spatial *= inputDims[i];
    }
}

void InstanceNorm::validateCudnnFrontendContract(uint64_t channelCount, Tensor::DataType inputDataType) {
    if (isReducedPrecisionInstanceNormInputDataType(inputDataType) && channelCount % 8 != 0) {
        throw invalid_argument(
            "InstanceNorm cuDNN Frontend primary engines require fp16/bf16 channel count to be a multiple of 8; got " +
            to_string(channelCount) + ".");
    }
}

InstanceNorm InstanceNorm::Builder::build() {
    if (_featureInputs.empty()) {
        throw invalid_argument("InstanceNorm::Builder requires at least one featureInput().");
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

    InstanceNorm layer;
    layer.featureInputs = _featureInputs;
    layer.channelCount = InstanceNorm::channelCountFromInputDims(_featureInputs.front().getDimensions());
    layer.epsilon = _epsilon.value();
    layer.parameterDataType = _parameterDataType.value();

    ParameterSpecification::Builder weightsBuilder;
    weightsBuilder.name("weights").shape({layer.channelCount}).dtype(layer.parameterDataType).initializer(_weightsInitializer).trainable(true);
    if (_weightsOptimizer != nullptr)
        weightsBuilder.optimizer(_weightsOptimizer);
    layer.addParameter(make_shared<ParameterSpecification>(weightsBuilder.build()));

    ParameterSpecification::Builder biasesBuilder;
    biasesBuilder.name("biases").shape({layer.channelCount}).dtype(layer.parameterDataType).initializer(_biasesInitializer).trainable(true);
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

void InstanceNorm::Builder::verifyConfig() const {
    if (!_network.has_value()) {
        throw invalid_argument("InstanceNorm::Builder requires network().");
    }
    if (_featureInputs.empty()) {
        throw invalid_argument("InstanceNorm::Builder requires featureInput().");
    }
    if (!_epsilon.has_value() || !(_epsilon.value() > 0.0)) {
        throw invalid_argument("InstanceNorm epsilon must be > 0.");
    }
    if (_parameterDataType.value() != Tensor::DataType::FP32) {
        throw invalid_argument("InstanceNorm currently requires fp32 weights/biases for cuDNN Frontend InstanceNorm.");
    }
    const Tensor::DataType inputDataType = _featureInputs.front().getDataType();
    if (!InstanceNorm::isInstanceNormInputDataType(inputDataType)) {
        throw invalid_argument("InstanceNorm feature input dtype must be fp16, bf16, or fp32.");
    }
    const vector<uint64_t> inputDims = _featureInputs.front().getDimensions();
    InstanceNorm::validateInputShape(inputDims);
    InstanceNorm::validateCudnnFrontendContract(inputDims.front(), inputDataType);
    for (uint32_t i = 0; i < _featureInputs.size(); ++i) {
        if (!_featureInputs[i].isInitialized()) {
            throw invalid_argument("InstanceNorm feature input is not initialized.");
        }
        if (_featureInputs[i].getDataType() != inputDataType) {
            throw invalid_argument("InstanceNorm all feature inputs must have the same dtype.");
        }
        if (_featureInputs[i].getDimensions() != inputDims) {
            throw invalid_argument("InstanceNorm all feature inputs must have the same dimensions.");
        }
    }
}

shared_ptr<ThorImplementation::Layer> InstanceNorm::stamp(ThorImplementation::TensorPlacement placement,
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

    return make_shared<ThorImplementation::InstanceNorm>(
        placement, inferenceOnly, channelCount, epsilon, parameterDataType, physicalParameters, getId());
}

json InstanceNorm::architectureJson() const {
    json j;
    j["factory"] = Layer::Factory::Learning.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = "instance_norm";
    j["layer_name"] = string("layer") + to_string(getId());
    j["channel_count"] = channelCount;
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

json InstanceNorm::serialize(thor_file::TarWriter& archiveWriter,
                             Stream stream,
                             bool saveOptimizerState,
                             ThorImplementation::StampedNetwork& stampedNetwork) const {
    json j = architectureJson();
    Parameterizable::serializeParameters(j["parameters"], archiveWriter, stream, saveOptimizerState, stampedNetwork, "layer" + to_string(getId()));
    return j;
}

void InstanceNorm::deserialize(shared_ptr<thor_file::TarReader>& archiveReader, const json& j, Network* network) {
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in InstanceNorm::deserialize: " + j.at("version").get<string>());
    if (j.at("layer_type").get<string>() != "instance_norm")
        throw runtime_error("Layer type mismatch in InstanceNorm::deserialize: " + j.at("layer_type").get<string>());

    InstanceNorm layer;
    layer.channelCount = j.at("channel_count").get<uint64_t>();
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
        throw runtime_error("InstanceNorm deserialize expected equal numbers of inputs and outputs.");
    }
    for (uint32_t i = 0; i < layer.featureInputs.size(); ++i) {
        layer.outputTensorFromInputTensor[layer.featureInputs[i]] = layer.featureOutputs[i];
        layer.inputTensorFromOutputTensor[layer.featureOutputs[i]] = layer.featureInputs[i];
    }

    if (j.contains("parameters")) {
        const json& parametersJson = j.at("parameters");
        if (!parametersJson.is_object()) {
            throw runtime_error("InstanceNorm parameters must be an object keyed by parameter name.");
        }
        for (auto it = parametersJson.begin(); it != parametersJson.end(); ++it) {
            ParameterSpecification parameter = ParameterSpecification::deserialize(it.value(), archiveReader);
            layer.addParameter(make_shared<ParameterSpecification>(std::move(parameter)));
        }
    }
    if (!layer.hasParameter("weights") || !layer.hasParameter("biases")) {
        throw runtime_error("InstanceNorm deserialize did not find required weights/biases parameters.");
    }

    layer.initialized = true;
    layer.addToNetwork(network);
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::TrainableLayer::register_layer("instance_norm", &Thor::InstanceNorm::deserialize);
    return true;
}();
}  // namespace
