#include "DeepLearning/Api/Layers/Utility/FiniteCheck.h"

#include "DeepLearning/Implementation/ThorError.h"

#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

using json = nlohmann::json;
using namespace std;

namespace Thor {
namespace {

std::mutex finiteCheckWarningMutex;
std::unordered_set<uint64_t> warnedFiniteCheckLayerIds;

void warnIfEnabled(const FiniteCheck &finiteCheck) {
    if (!finiteCheck.getEnabled())
        return;

    std::lock_guard<std::mutex> lock(finiteCheckWarningMutex);
    if (!warnedFiniteCheckLayerIds.insert(finiteCheck.getId()).second)
        return;

    std::cerr << "Thor warning: FiniteCheck layer is enabled"
              << " label=\"" << (finiteCheck.getTensorLabel().empty() ? "<unnamed>" : finiteCheck.getTensorLabel()) << '"'
              << " api_layer_id=" << finiteCheck.getId()
              << ". FiniteCheck is intended for diagnostic runs and will hurt performance because it synchronizes execution. "
                 "Disable it for performance runs."
              << std::endl;
}

RaggedTensor reconstructRaggedInput(const json& inputJson, Network* network, const char* context) {
    const uint64_t valuesId = inputJson.at("values").at("id").get<uint64_t>();
    const uint64_t offsetsId = inputJson.at("offsets").at("id").get<uint64_t>();
    Tensor values = network->getApiTensorByOriginalId(valuesId);
    Tensor offsets = network->getApiTensorByOriginalId(offsetsId);
    RaggedTensor ragged = inputJson.contains("max_values_per_row")
        ? RaggedTensor(values, offsets, inputJson.at("max_values_per_row").get<uint64_t>())
        : RaggedTensor(values, offsets);
    if (ragged.getBatchSize() != inputJson.at("batch_size").get<uint64_t>() ||
        ragged.getMaxTotalValues() != inputJson.at("max_total_values").get<uint64_t>()) {
        throw runtime_error(string(context) + " serialized ragged input metadata does not match reconstructed tensors.");
    }
    return ragged;
}

void validateSerializedRaggedOutput(const json& outputJson,
                                    const json& denseOutputJson,
                                    const RaggedTensor& input,
                                    const Tensor& outputValues,
                                    const char* context) {
    if (outputJson.at("values").at("id").get<uint64_t>() != denseOutputJson.at("id").get<uint64_t>() ||
        outputJson.at("offsets").at("id").get<uint64_t>() != input.getOffsets().getOriginalId() ||
        outputJson.at("batch_size").get<uint64_t>() != input.getBatchSize() ||
        outputJson.at("max_total_values").get<uint64_t>() != input.getMaxTotalValues() ||
        outputValues.getDimensions() != input.getValues().getDimensions() ||
        outputValues.getDataType() != input.getValues().getDataType()) {
        throw runtime_error(string(context) + " serialized ragged output must preserve the input values descriptor and row partition.");
    }
    if (outputJson.contains("max_values_per_row")) {
        if (!input.hasMaxValuesPerRow() ||
            outputJson.at("max_values_per_row").get<uint64_t>() != input.getMaxValuesPerRow()) {
            throw runtime_error(string(context) + " serialized ragged output max_values_per_row does not match its input.");
        }
    }
}

}  // namespace

FiniteCheck::FiniteCheck() = default;
FiniteCheck::~FiniteCheck() = default;

vector<Tensor> FiniteCheck::getAllInputTensors() const {
    if (!raggedFeatureInput.has_value()) return Layer::getAllInputTensors();
    return {raggedFeatureInput->getValues(), raggedFeatureInput->getOffsets()};
}

vector<Tensor> FiniteCheck::getOutputsFromInput(Tensor inputTensor) {
    (void)getConnectionType(inputTensor);
    if (!raggedFeatureInput.has_value()) {
        THOR_THROW_IF_FALSE(featureOutput.has_value());
        return {featureOutput.value()};
    }
    if (emittedFeatureOutputAfterAllInputsConnected || connectedInputPortIndices.size() != 2) return {};
    emittedFeatureOutputAfterAllInputsConnected = true;
    THOR_THROW_IF_FALSE(featureOutput.has_value());
    return {featureOutput.value()};
}

void FiniteCheck::informThatInputConnectionMade(Tensor inputTensor) {
    if (!raggedFeatureInput.has_value()) return;
    connectedInputPortIndices.insert(static_cast<uint32_t>(getConnectionType(inputTensor)));
}

void FiniteCheck::resetGraphTraversalState() {
    connectedInputPortIndices.clear();
    emittedFeatureOutputAfterAllInputsConnected = false;
}

int FiniteCheck::getConnectionType(Tensor connectingTensor) const {
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());
    if (connectingTensor == featureInput.value()) return 0;
    if (raggedFeatureInput.has_value() && connectingTensor == raggedFeatureInput->getOffsets()) return 1;
    if (connectingTensor == featureOutput.value()) return 0;
    throw runtime_error("Tensor is not connected to this FiniteCheck layer.");
}

shared_ptr<ThorImplementation::Layer> FiniteCheck::stamp(ThorImplementation::TensorPlacement placement,
                                                         shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                         shared_ptr<Thor::Layer> drivingApiLayer,
                                                         Thor::Tensor connectingApiTensor,
                                                         const bool inferenceOnly) const {
    (void)placement;
    (void)drivingLayer;
    (void)drivingApiLayer;
    (void)inferenceOnly;
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(featureInput.has_value());
    bool knownInput = connectingApiTensor == featureInput.value();
    if (raggedFeatureInput.has_value() && connectingApiTensor == raggedFeatureInput->getOffsets()) knownInput = true;
    THOR_THROW_IF_FALSE(knownInput);

    warnIfEnabled(*this);

    optional<ThorImplementation::FiniteCheck::RaggedConfiguration> raggedConfiguration;
    if (raggedFeatureInput.has_value()) {
        uint64_t elementsPerValue = 1;
        for (uint64_t dim : raggedFeatureInput->getTrailingDimensions()) {
            if (dim == 0 || elementsPerValue > numeric_limits<uint64_t>::max() / dim)
                throw overflow_error("Ragged FiniteCheck elements-per-value overflow.");
            elementsPerValue *= dim;
        }
        raggedConfiguration = ThorImplementation::FiniteCheck::RaggedConfiguration{
            .batchSize = raggedFeatureInput->getBatchSize(),
            .maxTotalValues = raggedFeatureInput->getMaxTotalValues(),
            .elementsPerValue = elementsPerValue,
            .offsetsDataType = raggedFeatureInput->getOffsetsDataType(),
        };
    }

    return make_shared<ThorImplementation::FiniteCheck>(tensorLabel,
                                                         featureInput.value().getId(),
                                                         featureInput.value().getOriginalId(),
                                                         checkForward,
                                                         checkBackward,
                                                         failOnNonFinite,
                                                         maxReportedIndices,
                                                         enabled,
                                                         raggedConfiguration);
}

uint64_t FiniteCheck::getFirstInstanceMemRequirementInBytes(uint32_t batchSize,
                                                            ThorImplementation::TensorPlacement tensorPlacement) const {
    (void)batchSize;
    if (!enabled)
        return 0;
    if (!featureInput.has_value() || ThorImplementation::TensorDescriptor::isIntegralType(featureInput.value().getDataType()) ||
        tensorPlacement.getMemDevice() != ThorImplementation::TensorPlacement::MemDevices::GPU) {
        return 0;
    }
    return sizeof(ThorImplementation::FiniteCheckResult);
}

json FiniteCheck::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());

    json j;
    j["factory"] = Layer::Factory::Layer.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = to_snake_case(getLayerType());
    j["feature_input"] = featureInput.value().architectureJson();
    j["feature_output"] = featureOutput.value().architectureJson();
    j["use_ragged"] = raggedFeatureInput.has_value();
    if (raggedFeatureInput.has_value()) {
        THOR_THROW_IF_FALSE(raggedFeatureOutput.has_value());
        j["ragged_feature_input"] = raggedFeatureInput->architectureJson();
        j["ragged_feature_output"] = raggedFeatureOutput->architectureJson();
    }
    j["tensor_label"] = tensorLabel;
    j["enabled"] = enabled;
    j["check_forward"] = checkForward;
    j["check_backward"] = checkBackward;
    j["fail_on_non_finite"] = failOnNonFinite;
    j["max_reported_indices"] = maxReportedIndices;
    return j;
}

void FiniteCheck::deserialize(const json &j, Network *network) {
    const string version = j.at("version").get<string>();
    if (version != "1.0.0" && version != "1.1.0")
        throw runtime_error("Unsupported version in FiniteCheck::deserialize: " + version);
    if (j.at("layer_type").get<string>() != "finite_check")
        throw runtime_error("Layer type mismatch in FiniteCheck::deserialize: " + j.at("layer_type").get<string>());

    const bool useRagged = version == "1.1.0" && j.value("use_ragged", false);
    Tensor featureInput;
    optional<RaggedTensor> raggedInput;
    if (useRagged) {
        const json& inputJson = j.at("ragged_feature_input");
        if (j.at("feature_input").at("id").get<uint64_t>() != inputJson.at("values").at("id").get<uint64_t>()) {
            throw runtime_error("FiniteCheck serialized ragged feature_input must reference the ragged values tensor.");
        }
        raggedInput = reconstructRaggedInput(inputJson, network, "FiniteCheck");
        featureInput = raggedInput->getValues();
    } else {
        const json input = j.at("feature_input").get<json>();
        featureInput = network->getApiTensorByOriginalId(input.at("id").get<uint64_t>());
    }
    Tensor featureOutput = Tensor::deserialize(j.at("feature_output").get<json>());

    FiniteCheck finiteCheck;
    finiteCheck.featureInput = featureInput;
    finiteCheck.featureOutput = featureOutput;
    if (raggedInput.has_value()) {
        validateSerializedRaggedOutput(j.at("ragged_feature_output"), j.at("feature_output"), raggedInput.value(), featureOutput, "FiniteCheck");
        finiteCheck.raggedFeatureInput = raggedInput;
        finiteCheck.raggedFeatureOutput = raggedInput->withValues(featureOutput);
    }
    finiteCheck.tensorLabel = j.value("tensor_label", string{});
    finiteCheck.enabled = j.value("enabled", true);
    finiteCheck.checkForward = j.value("check_forward", true);
    finiteCheck.checkBackward = j.value("check_backward", true);
    finiteCheck.failOnNonFinite = j.value("fail_on_non_finite", true);
    finiteCheck.maxReportedIndices = j.value("max_reported_indices", 8U);
    if (!finiteCheck.checkForward && !finiteCheck.checkBackward)
        throw runtime_error("Deserialized FiniteCheck must check forward, backward, or both.");
    if (finiteCheck.maxReportedIndices > ThorImplementation::FINITE_CHECK_MAX_REPORTED_INDICES)
        throw runtime_error("Deserialized FiniteCheck max_reported_indices exceeds the supported maximum.");

    finiteCheck.initialized = true;
    finiteCheck.addToNetwork(network);
}

FiniteCheck FiniteCheck::Builder::build() {
    THOR_THROW_IF_FALSE(_network.has_value());
    THOR_THROW_IF_FALSE(_featureInput.has_value());
    if (!_checkForward && !_checkBackward)
        throw invalid_argument("FiniteCheck must check forward, backward, or both.");
    if (_maxReportedIndices > ThorImplementation::FINITE_CHECK_MAX_REPORTED_INDICES) {
        throw invalid_argument("FiniteCheck maxReportedIndices exceeds the supported maximum of " +
                               to_string(ThorImplementation::FINITE_CHECK_MAX_REPORTED_INDICES) + ".");
    }

    FiniteCheck finiteCheck;
    finiteCheck.featureInput = _featureInput.value();
    finiteCheck.featureOutput = _featureInput.value().clone();
    if (_raggedFeatureInput.has_value()) {
        finiteCheck.raggedFeatureInput = _raggedFeatureInput;
        finiteCheck.raggedFeatureOutput = _raggedFeatureInput->withValues(finiteCheck.featureOutput.value());
    }
    finiteCheck.tensorLabel = std::move(_tensorLabel);
    finiteCheck.enabled = _enabled;
    finiteCheck.checkForward = _checkForward;
    finiteCheck.checkBackward = _checkBackward;
    finiteCheck.failOnNonFinite = _failOnNonFinite;
    finiteCheck.maxReportedIndices = _maxReportedIndices;
    finiteCheck.initialized = true;
    finiteCheck.addToNetwork(_network.value());
    return finiteCheck;
}

FiniteCheck::Builder &FiniteCheck::Builder::network(Network &network) {
    THOR_THROW_IF_FALSE(!_network.has_value());
    _network = &network;
    return *this;
}

FiniteCheck::Builder &FiniteCheck::Builder::featureInput(Tensor featureInput) {
    THOR_THROW_IF_FALSE(!_featureInput.has_value());
    THOR_THROW_IF_FALSE(!_raggedFeatureInput.has_value());
    THOR_THROW_IF_FALSE(featureInput.isInitialized());
    _featureInput = featureInput;
    return *this;
}

FiniteCheck::Builder &FiniteCheck::Builder::featureInput(RaggedTensor featureInput) {
    THOR_THROW_IF_FALSE(!_featureInput.has_value());
    THOR_THROW_IF_FALSE(!_raggedFeatureInput.has_value());
    THOR_THROW_IF_FALSE(featureInput.isInitialized());
    _raggedFeatureInput = featureInput;
    _featureInput = featureInput.getValues();
    return *this;
}

FiniteCheck::Builder &FiniteCheck::Builder::tensorLabel(string tensorLabel) {
    _tensorLabel = std::move(tensorLabel);
    return *this;
}

FiniteCheck::Builder &FiniteCheck::Builder::enabled(bool enabled) {
    _enabled = enabled;
    return *this;
}

FiniteCheck::Builder &FiniteCheck::Builder::checkForward(bool checkForward) {
    _checkForward = checkForward;
    return *this;
}

FiniteCheck::Builder &FiniteCheck::Builder::checkBackward(bool checkBackward) {
    _checkBackward = checkBackward;
    return *this;
}

FiniteCheck::Builder &FiniteCheck::Builder::failOnNonFinite(bool failOnNonFinite) {
    _failOnNonFinite = failOnNonFinite;
    return *this;
}

FiniteCheck::Builder &FiniteCheck::Builder::maxReportedIndices(uint32_t maxReportedIndices) {
    _maxReportedIndices = maxReportedIndices;
    return *this;
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Layer::register_layer("finite_check", &Thor::FiniteCheck::deserialize);
    return true;
}();
}  // namespace
