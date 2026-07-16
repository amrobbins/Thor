#include "DeepLearning/Api/Layers/Utility/FiniteCheck.h"

#include "DeepLearning/Implementation/ThorError.h"

#include <stdexcept>
#include <string>
#include <utility>

using json = nlohmann::json;
using namespace std;

namespace Thor {

FiniteCheck::FiniteCheck() = default;
FiniteCheck::~FiniteCheck() = default;

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
    THOR_THROW_IF_FALSE(connectingApiTensor == featureInput.value());

    return make_shared<ThorImplementation::FiniteCheck>(tensorLabel,
                                                         featureInput.value().getId(),
                                                         featureInput.value().getOriginalId(),
                                                         checkForward,
                                                         checkBackward,
                                                         failOnNonFinite,
                                                         maxReportedIndices);
}

uint64_t FiniteCheck::getFirstInstanceMemRequirementInBytes(uint32_t batchSize,
                                                            ThorImplementation::TensorPlacement tensorPlacement) const {
    (void)batchSize;
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
    j["tensor_label"] = tensorLabel;
    j["check_forward"] = checkForward;
    j["check_backward"] = checkBackward;
    j["fail_on_non_finite"] = failOnNonFinite;
    j["max_reported_indices"] = maxReportedIndices;
    return j;
}

void FiniteCheck::deserialize(const json &j, Network *network) {
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in FiniteCheck::deserialize: " + j.at("version").get<string>());
    if (j.at("layer_type").get<string>() != "finite_check")
        throw runtime_error("Layer type mismatch in FiniteCheck::deserialize: " + j.at("layer_type").get<string>());

    const json input = j.at("feature_input").get<json>();
    const uint64_t originalTensorId = input.at("id").get<uint64_t>();
    Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);
    Tensor featureOutput = Tensor::deserialize(j.at("feature_output").get<json>());

    FiniteCheck finiteCheck;
    finiteCheck.featureInput = featureInput;
    finiteCheck.featureOutput = featureOutput;
    finiteCheck.tensorLabel = j.value("tensor_label", string{});
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
    finiteCheck.tensorLabel = std::move(_tensorLabel);
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
    THOR_THROW_IF_FALSE(featureInput.isInitialized());
    _featureInput = featureInput;
    return *this;
}

FiniteCheck::Builder &FiniteCheck::Builder::tensorLabel(string tensorLabel) {
    _tensorLabel = std::move(tensorLabel);
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
