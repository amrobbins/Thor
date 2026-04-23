#include "DeepLearning/Api/Layers/Learning/CustomLayer.h"

#include <algorithm>
#include <stdexcept>

using namespace std;
using json = nlohmann::json;

using DynamicExpression = ThorImplementation::DynamicExpression;

namespace Thor {

CustomLayer::CustomLayer(DynamicExpression expr,
                         const std::vector<NamedTensor>& namedInputs,
                         const std::vector<NamedTensor>& namedOutputs,
                         bool inferenceOnly,
                         bool useFastMath)
    : expr(std::move(expr)), inferenceOnly(inferenceOnly), useFastMath(useFastMath) {
    validateNamedTensorList(namedInputs, "input");
    validateNamedTensorList(namedOutputs, "output");
    assignNamedInputs(namedInputs);
    assignNamedOutputs(namedOutputs);
    initialized = true;
}

void CustomLayer::validateNamedTensorList(const std::vector<NamedTensor>& namedTensors, const std::string& what) {
    if (namedTensors.empty()) {
        throw runtime_error("CustomLayer requires at least one named " + what + ".");
    }

    std::set<std::string> seenNames;
    std::set<uint64_t> seenTensorIds;
    for (const auto& [name, tensor] : namedTensors) {
        if (name.empty()) {
            throw runtime_error("CustomLayer " + what + " name cannot be empty.");
        }
        if (!tensor.isInitialized()) {
            throw runtime_error("CustomLayer " + what + " tensor for port '" + name + "' is not initialized.");
        }
        if (!seenNames.insert(name).second) {
            throw runtime_error("Duplicate CustomLayer " + what + " name: " + name);
        }
        if (!seenTensorIds.insert(tensor.getId()).second) {
            throw runtime_error("Duplicate CustomLayer " + what + " tensor used for multiple named ports.");
        }
    }
}

void CustomLayer::assignNamedInputs(const std::vector<NamedTensor>& namedInputs) {
    featureInputs.clear();
    inputNames.clear();
    inputPortByName.clear();
    connectedInputTensorOriginalIds.clear();

    for (uint32_t i = 0; i < namedInputs.size(); ++i) {
        const auto& [name, tensor] = namedInputs[i];
        inputNames.push_back(name);
        featureInputs.push_back(tensor);
        inputPortByName.emplace(name, i);
    }
}

void CustomLayer::assignNamedOutputs(const std::vector<NamedTensor>& namedOutputs) {
    featureOutputs.clear();
    outputNames.clear();
    outputPortByName.clear();

    for (uint32_t i = 0; i < namedOutputs.size(); ++i) {
        const auto& [name, tensor] = namedOutputs[i];
        outputNames.push_back(name);
        featureOutputs.push_back(tensor);
        outputPortByName.emplace(name, i);
    }
}

int CustomLayer::getConnectionType(Tensor connectingTensor) const {
    for (uint32_t i = 0; i < featureInputs.size(); ++i) {
        if (featureInputs[i] == connectingTensor)
            return static_cast<int>(i);
    }
    for (uint32_t i = 0; i < featureOutputs.size(); ++i) {
        if (featureOutputs[i] == connectingTensor)
            return static_cast<int>(i);
    }
    throw runtime_error("Tensor is not connected to this CustomLayer.");
}

void CustomLayer::informThatInputConnectionMade(Tensor inputTensor) {
    bool found = false;
    for (const Tensor& tensor : featureInputs) {
        if (tensor == inputTensor) {
            found = true;
            connectedInputTensorOriginalIds.insert(inputTensor.getOriginalId());
            break;
        }
    }
    if (!found) {
        throw runtime_error("CustomLayer informed of connection for unknown input tensor.");
    }
}

std::vector<Tensor> CustomLayer::getOutputsFromInput(Tensor inputTensor) {
    bool found = false;
    for (const Tensor& tensor : featureInputs) {
        if (tensor == inputTensor) {
            found = true;
            break;
        }
    }
    if (!found) {
        throw runtime_error("CustomLayer asked for outputs from unknown input tensor.");
    }

    if (connectedInputTensorOriginalIds.size() != featureInputs.size()) {
        return {};
    }
    return featureOutputs;
}

uint64_t CustomLayer::getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
    uint64_t totalBytes = 0;
    for (const Tensor& tensor : featureInputs)
        totalBytes += tensor.getTotalSizeInBytes();
    for (const Tensor& tensor : featureOutputs)
        totalBytes += tensor.getTotalSizeInBytes();
    return totalBytes * std::max<uint32_t>(1, batchSize);
}

std::shared_ptr<ThorImplementation::Layer> CustomLayer::stamp(ThorImplementation::TensorPlacement placement,
                                                              std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                              std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                              Thor::Tensor connectingApiTensor) const {
    (void)drivingLayer;
    (void)drivingApiLayer;

    bool connectingTensorKnown = false;
    for (const Tensor& tensor : featureInputs) {
        if (tensor == connectingApiTensor) {
            connectingTensorKnown = true;
            break;
        }
    }
    if (!connectingTensorKnown) {
        throw runtime_error("CustomLayer::stamp called with a tensor that is not one of its declared inputs.");
    }

    auto physicalLayer = std::make_shared<ThorImplementation::CustomLayer>(expr,
                                                                           inputNames,
                                                                           outputNames,
                                                                           placement,
                                                                           std::vector<std::shared_ptr<ThorImplementation::Parameter>>{},
                                                                           inferenceOnly,
                                                                           Layer::getId(),
                                                                           useFastMath);
    physicalLayer->setLayerName(getLayerType());
    return physicalLayer;
}

json CustomLayer::architectureJson() const {
    json j;
    j["factory"] = Layer::Factory::Learning.value();
    j["version"] = "1.0.0";
    j["layer_type"] = "custom_layer";
    j["inference_only"] = inferenceOnly;
    j["input_names"] = inputNames;
    j["output_names"] = outputNames;
    j["inputs"] = json::array();
    j["outputs"] = json::array();
    for (uint32_t i = 0; i < inputNames.size(); ++i) {
        j["inputs"].push_back(json{{"name", inputNames[i]}, {"tensor", featureInputs[i].architectureJson()}});
    }
    for (uint32_t i = 0; i < outputNames.size(); ++i) {
        j["outputs"].push_back(json{{"name", outputNames[i]}, {"tensor", featureOutputs[i].architectureJson()}});
    }
    return j;
}

json CustomLayer::serialize(thor_file::TarWriter& archiveWriter,
                            Stream stream,
                            bool saveOptimizerState,
                            ThorImplementation::StampedNetwork& stampedNetwork) const {
    (void)archiveWriter;
    (void)stream;
    (void)saveOptimizerState;
    (void)stampedNetwork;
    return architectureJson();
}

}  // namespace Thor
