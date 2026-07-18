#include "DeepLearning/Api/Network/Network.h"
#include "Utilities/Expression/CudaKernelSecurity.h"
#include <optional>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <set>
#include <string_view>
#include "DeepLearning/Api/Layers/Learning/CustomLayer.h"
#include "DeepLearning/Api/Layers/Loss/CustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/MultiInputCustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/QuantileLoss.h"
#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Parameter/ParameterSpecification.h"
#include "DeepLearning/Api/Parameter/Parameterizable.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Training/DeviceStartupCoordinator.h"
#include <cstdio>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <unordered_set>

using namespace std;
using json = nlohmann::json;

using ThorImplementation::TensorPlacement;

namespace {

json cudaKernelOutOfBandKeysFileJson(const std::string& networkName,
                                      const std::vector<ThorImplementation::CudaKernelOutOfBandKeys>& cudaKernelKeys,
                                      const std::string& status) {
    json j;
    j["type"] = "thor.cuda_kernel_expression_out_of_band_keys";
    j["schema_version"] = 1;
    j["network_name"] = networkName;
    j["status"] = status;
    j["warning"] =
        "Keep this file outside the saved model archive. A Thor model containing CudaKernelExpression CUDA source cannot be loaded "
        "or compiled from the saved artifact unless the Ed25519 public key and AES-256-GCM source decryption key are preserved out of band.";
    j["loaded_model_safety_disclaimer"] = ThorImplementation::cudaKernelLoadedModelSafetyDisclaimer();
    j["keys"] = json::array();
    for (const ThorImplementation::CudaKernelOutOfBandKeys& keys : cudaKernelKeys) {
        j["keys"].push_back(json{{"signing_public_key", keys.signing_public_key},
                                  {"source_decryption_key", keys.source_decryption_key}});
    }
    return j;
}

void writeJsonFile(const std::filesystem::path& path, const json& j) {
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("Unable to open CudaKernelExpression save-key capture file for writing: " + path.string());
    }
    out << j.dump(4) << "\n";
    if (!out) {
        throw std::runtime_error("Failed while writing CudaKernelExpression save-key capture file: " + path.string());
    }
}


bool hasArchiveShard0(const std::filesystem::path& directory, const std::string& archiveName) {
    return std::filesystem::exists(directory / (archiveName + ".thor.tar")) ||
           std::filesystem::exists(directory / (archiveName + ".000000.thor.tar"));
}


struct NetworkArchiveSelection {
    std::string archiveName;
    std::string modelJsonFileName;
};

NetworkArchiveSelection selectNetworkArchiveForLoad(const std::filesystem::path& directory, const std::string& archiveName) {
    if (archiveName.empty()) {
        throw std::runtime_error("Network::load requires a non-empty network/archive name.");
    }
    std::error_code errorCode;
    if (!std::filesystem::exists(directory, errorCode)) {
        throw std::runtime_error("Network::load: directory does not exist: " + directory.string());
    }
    if (errorCode) {
        throw std::runtime_error("Network::load: failed to inspect directory '" + directory.string() + "': " + errorCode.message());
    }
    if (!std::filesystem::is_directory(directory, errorCode)) {
        throw std::runtime_error("Network::load: expected a directory containing archive '" + archiveName +
                                 ".thor.tar', got: " + directory.string());
    }
    if (errorCode) {
        throw std::runtime_error("Network::load: failed to inspect directory '" + directory.string() + "': " + errorCode.message());
    }
    if (!hasArchiveShard0(directory, archiveName)) {
        throw std::runtime_error("Network::load: expected archive '" + archiveName + ".thor.tar' or sharded archive '" +
                                 archiveName + ".000000.thor.tar' in directory " + directory.string() + ".");
    }

    auto reader = std::make_shared<thor_file::TarReader>(archiveName, directory);
    const std::string modelJsonFileName = archiveName + ".thor.json";
    if (!reader->containsFile(modelJsonFileName)) {
        throw std::runtime_error("Network::load: archive '" + archiveName + "' in " + directory.string() +
                                 " is missing expected model file '" + modelJsonFileName + "'.");
    }
    return {archiveName, modelJsonFileName};
}

void printCudaKernelOutOfBandKeys(const std::vector<ThorImplementation::CudaKernelOutOfBandKeys>& cudaKernelKeys) {
    if (cudaKernelKeys.empty()) {
        return;
    }
    std::cout << "\nThor saved a model containing CudaKernelExpression custom CUDA source.\n";
    std::cout << ThorImplementation::cudaKernelLoadedModelSafetyDisclaimer() << "\n";
    std::cout << "The serialized model stores encrypted CUDA source and records only key fingerprints, not keys. "
              << "Persist the out-of-band keys below separately if this saved model should be loadable.\n";
    std::cout << "The Ed25519 public key verifies the signed encrypted CUDA manifest. The AES-256-GCM source decryption key "
              << "decrypts the CUDA source after signature verification. Provide both keys to the load API; enable compilation only "
              << "after inspecting the decrypted CUDA source and applying your own security policy.\n";
    for (const ThorImplementation::CudaKernelOutOfBandKeys& keys : cudaKernelKeys) {
        std::cout << "CudaKernelExpression Ed25519 public key: " << keys.signing_public_key << "\n";
        std::cout << "CudaKernelExpression AES-256-GCM source decryption key: " << keys.source_decryption_key << "\n";
    }
    std::cout << std::endl;
}


bool isSerializedApiTensorJson(const json& value) {
    return value.is_object() && value.contains("version") && value.contains("id") && value.contains("dimensions") &&
           value.contains("data_type");
}

uint64_t allocateApiTensorOriginalIdForClone(const json& tensorJson) {
    std::vector<uint64_t> dimensions = tensorJson.at("dimensions").get<std::vector<uint64_t>>();
    Thor::DataType dataType = tensorJson.at("data_type").get<Thor::DataType>();
    Thor::Tensor placeholder(dataType, dimensions);
    return placeholder.getOriginalId();
}

void rewriteApiTensorIdsForClone(json& value,
                                 const std::set<uint64_t>& cloneGraphTensorOriginalIds,
                                 const std::map<uint64_t, uint64_t>& sourceOriginalIdBySerializedTensorId,
                                 const std::set<uint64_t>& clonedLayerOutputOriginalIds,
                                 std::map<uint64_t, uint64_t>& destinationOriginalIdBySourceOriginalId,
                                 const std::string& layerContext) {
    if (isSerializedApiTensorJson(value)) {
        const uint64_t serializedTensorId = value.at("id").get<uint64_t>();

        // Tensor::architectureJson() serializes the live Tensor id.  For a freshly
        // constructed network that is the same value as originalId, but after
        // loading a saved artifact the live id is intentionally different from
        // the original id used by Network's API tensor indexes.  Cloning must
        // canonicalize the serialized id back to the source tensor's originalId
        // before consulting destinationOriginalIdBySourceOriginalId.
        auto sourceOriginalIt = sourceOriginalIdBySerializedTensorId.find(serializedTensorId);
        uint64_t sourceOriginalId = 0;
        if (sourceOriginalIt != sourceOriginalIdBySerializedTensorId.end()) {
            sourceOriginalId = sourceOriginalIt->second;
        } else if (cloneGraphTensorOriginalIds.count(serializedTensorId) != 0) {
            // Be tolerant of older/manual layer JSON that stored the API tensor
            // originalId directly instead of Tensor::architectureJson()'s live id.
            sourceOriginalId = serializedTensorId;
        } else {
            // Some layer JSON contains non-API tensor-shaped metadata, such as
            // saved parameter storage descriptors.  Only rewrite tensors that
            // are part of the source API subgraph being cloned.
            return;
        }

        auto mappedIt = destinationOriginalIdBySourceOriginalId.find(sourceOriginalId);
        if (mappedIt == destinationOriginalIdBySourceOriginalId.end()) {
            if (clonedLayerOutputOriginalIds.count(sourceOriginalId) == 0) {
                throw std::runtime_error("cloneInferenceSubgraphInto: layer '" + layerContext +
                                         "' references source API tensor " + std::to_string(sourceOriginalId) +
                                         " (serialized id " + std::to_string(serializedTensorId) +
                                         ") before that tensor has been remapped or cloned.");
            }
            mappedIt = destinationOriginalIdBySourceOriginalId.emplace(sourceOriginalId, allocateApiTensorOriginalIdForClone(value)).first;
        }
        value["id"] = mappedIt->second;
        return;
    }

    if (value.is_object()) {
        for (auto it = value.begin(); it != value.end(); ++it) {
            rewriteApiTensorIdsForClone(it.value(),
                                        cloneGraphTensorOriginalIds,
                                        sourceOriginalIdBySerializedTensorId,
                                        clonedLayerOutputOriginalIds,
                                        destinationOriginalIdBySourceOriginalId,
                                        layerContext);
        }
        return;
    }

    if (value.is_array()) {
        for (json& element : value) {
            rewriteApiTensorIdsForClone(element,
                                        cloneGraphTensorOriginalIds,
                                        sourceOriginalIdBySerializedTensorId,
                                        clonedLayerOutputOriginalIds,
                                        destinationOriginalIdBySourceOriginalId,
                                        layerContext);
        }
    }
}


void rewriteAlreadyMappedApiTensorIdsForClone(json& value,
                                             const std::map<uint64_t, uint64_t>& destinationOriginalIdBySourceOriginalId) {
    if (value.is_object()) {
        auto idIt = value.find("id");
        if (idIt != value.end() && idIt->is_number_unsigned()) {
            const uint64_t sourceOriginalId = idIt->get<uint64_t>();
            auto mappedIt = destinationOriginalIdBySourceOriginalId.find(sourceOriginalId);
            if (mappedIt != destinationOriginalIdBySourceOriginalId.end()) {
                *idIt = mappedIt->second;
            }
        }
        for (auto it = value.begin(); it != value.end(); ++it) {
            rewriteAlreadyMappedApiTensorIdsForClone(it.value(), destinationOriginalIdBySourceOriginalId);
        }
        return;
    }

    if (value.is_array()) {
        for (json& element : value) {
            rewriteAlreadyMappedApiTensorIdsForClone(element, destinationOriginalIdBySourceOriginalId);
        }
    }
}

std::string sourceLayerContext(const std::shared_ptr<Thor::Layer>& layer) {
    if (layer == nullptr) {
        return "<null>";
    }
    return layer->getLayerType() + "#" + std::to_string(layer->getId());
}

void prefixLayerNamesForClone(json& layerJson, const std::string& namePrefix) {
    if (namePrefix.empty()) {
        return;
    }
    if (layerJson.contains("layer_name") && layerJson.at("layer_name").is_string()) {
        layerJson["layer_name"] = namePrefix + layerJson.at("layer_name").get<std::string>();
    }
}

}  // namespace

namespace Thor {
string Network::statusCodeToString(StatusCode statusCode) {
    if (statusCode == StatusCode::SUCCESS)
        return "SUCCESS";
    else if (statusCode == StatusCode::FLOATING_INPUT)
        return "FLOATING INPUT";
    else if (statusCode == StatusCode::DANGLING_OUTPUT)
        return "DANGLING OUTPUT";
    else if (statusCode == StatusCode::GPU_OUT_OF_MEMORY)
        return "GPU OUT OF MEMORY";
    else if (statusCode == StatusCode::DUPLICATE_NAMED_NETWORK_INPUT)
        return "DUPLICATE NAMED NETWORK INPUT";
    else if (statusCode == StatusCode::DUPLICATE_NAMED_NETWORK_OUTPUT)
        return "DUPLICATE NAMED NETWORK OUTPUT";
    else if (statusCode == StatusCode::DEADLOCK_CYCLE)
        return "DEADLOCK CYCLE";
    THOR_UNREACHABLE();
}


void ApiTensorRemap::map(const Tensor& sourceTensor, const Tensor& destinationTensor) {
    THOR_THROW_IF_FALSE(sourceTensor.isInitialized());
    THOR_THROW_IF_FALSE(destinationTensor.isInitialized());
    if (sourceTensor.getDimensions() != destinationTensor.getDimensions()) {
        throw std::runtime_error("ApiTensorRemap shape mismatch: source " + sourceTensor.getDescriptorString() +
                                 " destination " + destinationTensor.getDescriptorString() + ".");
    }
    if (sourceTensor.getDataType() != destinationTensor.getDataType()) {
        throw std::runtime_error("ApiTensorRemap dtype mismatch: source " + sourceTensor.getDescriptorString() +
                                 " destination " + destinationTensor.getDescriptorString() + ".");
    }
    const uint64_t sourceOriginalId = sourceTensor.getOriginalId();
    auto [it, inserted] = destinationTensorBySourceOriginalId.emplace(sourceOriginalId, destinationTensor);
    if (!inserted && it->second != destinationTensor) {
        throw std::runtime_error("ApiTensorRemap source tensor " + std::to_string(sourceOriginalId) +
                                 " was mapped to two different destination tensors.");
    }
}

bool ApiTensorRemap::contains(const Tensor& sourceTensor) const {
    THOR_THROW_IF_FALSE(sourceTensor.isInitialized());
    return destinationTensorBySourceOriginalId.count(sourceTensor.getOriginalId()) != 0;
}

Tensor ApiTensorRemap::get(const Tensor& sourceTensor) const {
    THOR_THROW_IF_FALSE(sourceTensor.isInitialized());
    auto it = destinationTensorBySourceOriginalId.find(sourceTensor.getOriginalId());
    if (it == destinationTensorBySourceOriginalId.end()) {
        throw std::runtime_error("ApiTensorRemap has no destination tensor for source tensor " +
                                 std::to_string(sourceTensor.getOriginalId()) + ".");
    }
    return it->second;
}

ApiSubgraphCloneResult Network::cloneInferenceSubgraphInto(const Network& sourceNetworkConst,
                                                           const std::vector<std::string>& outputNames,
                                                           const ApiTensorRemap& initialRemap,
                                                           const ApiSubgraphCloneOptions& options) {
    if (outputNames.empty()) {
        throw std::runtime_error("cloneInferenceSubgraphInto requires at least one output name.");
    }
    if (!options.cloneTrainableParameters) {
        throw std::runtime_error("cloneInferenceSubgraphInto currently requires cloneTrainableParameters=true.");
    }

    Network& sourceNetwork = const_cast<Network&>(sourceNetworkConst);
    sourceNetwork.rebuildApiGraphIndexes(options.inferenceOnly);
    this->rebuildApiGraphIndexes(options.inferenceOnly);

    std::map<uint64_t, uint64_t> destinationOriginalIdBySourceOriginalId;
    std::map<uint64_t, uint64_t> sourceOriginalIdBySerializedTensorId;
    auto rememberSourceCloneGraphTensor = [&](const Tensor& tensor) {
        // Tensor JSON currently stores the live Tensor id.  Network indexes,
        // initial remaps, and clone results are keyed by originalId.
        sourceOriginalIdBySerializedTensorId[tensor.getId()] = tensor.getOriginalId();
    };

    ApiSubgraphCloneResult result;
    for (const auto& [sourceOriginalId, destinationTensor] : initialRemap.entriesBySourceOriginalId()) {
        if (!sourceNetwork.hasApiTensorByOriginalId(sourceOriginalId)) {
            throw std::runtime_error("cloneInferenceSubgraphInto initial remap source tensor " + std::to_string(sourceOriginalId) +
                                     " does not belong to source network '" + sourceNetwork.getNetworkName() + "'.");
        }
        Tensor sourceTensor = sourceNetwork.resolveApiTensorByOriginalId(sourceOriginalId);
        rememberSourceCloneGraphTensor(sourceTensor);
        if (sourceTensor.getDimensions() != destinationTensor.getDimensions() || sourceTensor.getDataType() != destinationTensor.getDataType()) {
            throw std::runtime_error("cloneInferenceSubgraphInto initial remap tensor descriptor mismatch for source tensor " +
                                     std::to_string(sourceOriginalId) + ".");
        }
        Tensor destinationCanonicalTensor = this->resolveApiTensorByOriginalId(destinationTensor.getOriginalId());
        destinationOriginalIdBySourceOriginalId[sourceOriginalId] = destinationCanonicalTensor.getOriginalId();
        result.clonedTensorBySourceOriginalId[sourceOriginalId] = destinationCanonicalTensor;
    }

    std::map<std::string, Tensor> sourceOutputTensorByName;
    for (const std::shared_ptr<Layer>& layer : sourceNetwork.allLayersInNetworkList) {
        std::shared_ptr<NetworkOutput> networkOutput = std::dynamic_pointer_cast<NetworkOutput>(layer);
        if (networkOutput == nullptr) {
            continue;
        }
        const std::string outputName = networkOutput->getName();
        sourceOutputTensorByName[outputName] = networkOutput->getFeatureInput().value();
    }

    std::vector<Tensor> requestedSourceOutputTensors;
    requestedSourceOutputTensors.reserve(outputNames.size());
    for (const std::string& outputName : outputNames) {
        auto outputIt = sourceOutputTensorByName.find(outputName);
        if (outputIt == sourceOutputTensorByName.end()) {
            throw std::runtime_error("cloneInferenceSubgraphInto could not find source NetworkOutput named '" + outputName +
                                     "' in network '" + sourceNetwork.getNetworkName() + "'.");
        }
        requestedSourceOutputTensors.push_back(outputIt->second);
    }

    std::set<uint64_t> requiredLayerIds;
    std::set<uint64_t> visitingLayerIds;
    std::vector<std::shared_ptr<Layer>> layersToClone;
    std::function<void(const Tensor&)> collectUpstream = [&](const Tensor& tensor) {
        if (destinationOriginalIdBySourceOriginalId.count(tensor.getOriginalId()) != 0) {
            return;
        }
        auto driverIt = sourceNetwork.apiTensorToApiDrivingLayer.find(tensor);
        if (driverIt == sourceNetwork.apiTensorToApiDrivingLayer.end()) {
            throw std::runtime_error("cloneInferenceSubgraphInto could not find a driving source layer for tensor " +
                                     std::to_string(tensor.getOriginalId()) + ".");
        }
        std::shared_ptr<Layer> driver = driverIt->second;
        if (std::dynamic_pointer_cast<NetworkInput>(driver) != nullptr) {
            throw std::runtime_error("cloneInferenceSubgraphInto output depends on source NetworkInput '" +
                                     std::dynamic_pointer_cast<NetworkInput>(driver)->getName() +
                                     "' that was not provided in the initial ApiTensorRemap.");
        }
        if (std::dynamic_pointer_cast<NetworkOutput>(driver) != nullptr) {
            throw std::runtime_error("cloneInferenceSubgraphInto encountered a NetworkOutput as a tensor driver; request the output's input tensor instead.");
        }

        const uint64_t driverId = driver->getId();
        if (requiredLayerIds.count(driverId) != 0) {
            return;
        }
        if (!visitingLayerIds.insert(driverId).second) {
            throw std::runtime_error("cloneInferenceSubgraphInto encountered a cycle while cloning " + sourceLayerContext(driver) + ".");
        }

        auto inputsIt = sourceNetwork.apiLayerToApiInputTensors.find(driver);
        if (inputsIt != sourceNetwork.apiLayerToApiInputTensors.end()) {
            for (const Tensor& inputTensor : inputsIt->second) {
                collectUpstream(inputTensor);
            }
        }

        visitingLayerIds.erase(driverId);
        requiredLayerIds.insert(driverId);
        layersToClone.push_back(driver);
    };

    for (const Tensor& outputTensor : requestedSourceOutputTensors) {
        collectUpstream(outputTensor);
    }

    std::set<uint64_t> cloneGraphTensorOriginalIds;
    auto rememberCloneGraphTensor = [&](const Tensor& tensor) {
        cloneGraphTensorOriginalIds.insert(tensor.getOriginalId());
        rememberSourceCloneGraphTensor(tensor);
    };
    for (const auto& [sourceOriginalId, _] : initialRemap.entriesBySourceOriginalId()) {
        Tensor sourceTensor = sourceNetwork.resolveApiTensorByOriginalId(sourceOriginalId);
        rememberCloneGraphTensor(sourceTensor);
    }
    for (const Tensor& outputTensor : requestedSourceOutputTensors) {
        rememberCloneGraphTensor(outputTensor);
    }
    for (const std::shared_ptr<Layer>& layer : layersToClone) {
        auto inputIt = sourceNetwork.apiLayerToApiInputTensors.find(layer);
        if (inputIt != sourceNetwork.apiLayerToApiInputTensors.end()) {
            for (const Tensor& inputTensor : inputIt->second) {
                rememberCloneGraphTensor(inputTensor);
            }
        }
        auto outputIt = sourceNetwork.apiLayerToApiOutputTensors.find(layer);
        if (outputIt != sourceNetwork.apiLayerToApiOutputTensors.end()) {
            for (const Tensor& outputTensor : outputIt->second) {
                rememberCloneGraphTensor(outputTensor);
            }
        }
    }

    std::shared_ptr<thor_file::TarReader> sourceArchiveReader = sourceNetwork.archiveReader;
    if (sourceArchiveReader != nullptr) {
        cloneInitializationArchiveReaders.push_back(sourceArchiveReader);
    }

    for (const std::shared_ptr<Layer>& layer : layersToClone) {
        std::set<uint64_t> clonedLayerOutputOriginalIds;
        auto outputIt = sourceNetwork.apiLayerToApiOutputTensors.find(layer);
        if (outputIt != sourceNetwork.apiLayerToApiOutputTensors.end()) {
            for (const Tensor& outputTensor : outputIt->second) {
                clonedLayerOutputOriginalIds.insert(outputTensor.getOriginalId());
            }
        }

        json layerJson = layer->architectureJson();
        if (sourceArchiveReader != nullptr && layerJson.contains("parameters")) {
            std::shared_ptr<Parameterizable> parameterizableLayer = std::dynamic_pointer_cast<Parameterizable>(layer);
            if (parameterizableLayer != nullptr) {
                layerJson["parameters"] = parameterizableLayer->getParametersArchitectureJson(
                    /*includeArchiveStorageFiles=*/true)["parameters"];
            }
        }
        prefixLayerNamesForClone(layerJson, options.namePrefix);
        rewriteApiTensorIdsForClone(layerJson,
                                    cloneGraphTensorOriginalIds,
                                    sourceOriginalIdBySerializedTensorId,
                                    clonedLayerOutputOriginalIds,
                                    destinationOriginalIdBySourceOriginalId,
                                    sourceLayerContext(layer));
        rewriteAlreadyMappedApiTensorIdsForClone(layerJson, destinationOriginalIdBySourceOriginalId);

        const size_t previousLayerCount = this->allLayersInNetworkList.size();
        try {
            Layer::deserialize(sourceArchiveReader, layerJson, this);
        } catch (const std::exception& e) {
            throw std::runtime_error("cloneInferenceSubgraphInto failed to deserialize cloned " + sourceLayerContext(layer) +
                                     " into network '" + this->getNetworkName() + "': " + e.what());
        }
        if (this->allLayersInNetworkList.size() <= previousLayerCount) {
            throw std::runtime_error("cloneInferenceSubgraphInto did not add a cloned layer for " + sourceLayerContext(layer) + ".");
        }

        const std::string cloneSourceKey = options.namePrefix + sourceLayerContext(layer);
        for (size_t clonedLayerIndex = previousLayerCount; clonedLayerIndex < this->allLayersInNetworkList.size(); ++clonedLayerIndex) {
            const std::shared_ptr<Layer>& clonedLayer = this->allLayersInNetworkList[clonedLayerIndex];
            if (clonedLayer != nullptr) {
                cloneSourceKeyByLayerId[clonedLayer->getId()] = cloneSourceKey;
            }
        }

        for (uint64_t sourceOutputOriginalId : clonedLayerOutputOriginalIds) {
            auto mappedIt = destinationOriginalIdBySourceOriginalId.find(sourceOutputOriginalId);
            if (mappedIt == destinationOriginalIdBySourceOriginalId.end()) {
                throw std::runtime_error("cloneInferenceSubgraphInto did not allocate a destination tensor for output tensor " +
                                         std::to_string(sourceOutputOriginalId) + " of " + sourceLayerContext(layer) + ".");
            }
            Tensor destinationTensor = this->resolveApiTensorByOriginalId(mappedIt->second);
            result.clonedTensorBySourceOriginalId[sourceOutputOriginalId] = destinationTensor;
        }
    }

    for (size_t i = 0; i < outputNames.size(); ++i) {
        const Tensor& sourceOutputTensor = requestedSourceOutputTensors[i];
        auto mappedIt = destinationOriginalIdBySourceOriginalId.find(sourceOutputTensor.getOriginalId());
        if (mappedIt == destinationOriginalIdBySourceOriginalId.end()) {
            throw std::runtime_error("cloneInferenceSubgraphInto did not produce a destination tensor for output '" + outputNames[i] + "'.");
        }
        result.outputTensorsByName[outputNames[i]] = this->resolveApiTensorByOriginalId(mappedIt->second);
    }

    this->rebuildApiGraphIndexes(options.inferenceOnly);

    return result;
}

std::optional<std::string> Network::getCloneSourceKeyForLayerId(uint64_t layerId) const {
    auto it = cloneSourceKeyByLayerId.find(layerId);
    if (it == cloneSourceKeyByLayerId.end()) {
        return std::nullopt;
    }
    return it->second;
}

bool Network::hasCudaKernelExpressions() const {
    json modelJson = architectureJson();
    std::vector<ThorImplementation::CudaKernelSourceInspection> cudaKernelSources =
        ThorImplementation::collectCudaKernelSourceInfo(modelJson);
    return !cudaKernelSources.empty();
}

void Network::captureCudaKernelSaveKeysToFile(const std::string& path, bool overwrite) {
    if (path.empty()) {
        throw std::invalid_argument("CudaKernelExpression save-key capture file path cannot be empty.");
    }

    const std::filesystem::path capturePath = std::filesystem::absolute(std::filesystem::path(path));
    if (std::filesystem::exists(capturePath) && !overwrite) {
        throw std::runtime_error("CudaKernelExpression save-key capture file already exists: " + capturePath.string() +
                                 ". Pass overwrite=true or choose a different file before training.");
    }

    cudaKernelSaveKeyCaptureFile_ = capturePath.string();
    writeJsonFile(capturePath, cudaKernelOutOfBandKeysFileJson(networkName, {}, "pending"));
}

void Network::clearCudaKernelSaveKeyCapture() {
    cudaKernelSaveKeyCaptureFile_.reset();
}

void Network::enforceCudaKernelSaveKeyCaptureForTraining() const {
    if (!hasCudaKernelExpressions()) {
        return;
    }
    if (cudaKernelSaveKeyCaptureFile_.has_value()) {
        return;
    }

    throw std::runtime_error(
        "Refusing to place a training network that contains CudaKernelExpression CUDA source without configured save-key capture. "
        "CudaKernelExpression source is encrypted and signed when the model is saved; the saved model stores only key fingerprints. "
        "Configure captureCudaKernelSaveKeysToFile(...) before training so the out-of-band Ed25519 public key and AES-256-GCM "
        "source decryption key produced by save() are persisted. Otherwise a long training run could produce a model that cannot "
        "be loaded or compiled.");
}

void Network::requireCudaKernelSaveKeyCaptureForKeys(
    const std::vector<ThorImplementation::CudaKernelOutOfBandKeys>& cudaKernelKeys) const {
    if (cudaKernelKeys.empty()) {
        return;
    }
    if (cudaKernelSaveKeyCaptureFile_.has_value()) {
        return;
    }

    throw std::runtime_error(
        "Refusing to save a model containing CudaKernelExpression CUDA source without configured save-key capture. The saved model "
        "would contain encrypted CUDA source and key fingerprints but not the out-of-band keys needed to load it. Configure "
        "captureCudaKernelSaveKeysToFile(...) before training or before save(), then save again.");
}

void Network::writeCudaKernelSaveKeysToCaptureFile(
    const std::vector<ThorImplementation::CudaKernelOutOfBandKeys>& cudaKernelKeys) const {
    if (cudaKernelKeys.empty()) {
        return;
    }
    requireCudaKernelSaveKeyCaptureForKeys(cudaKernelKeys);
    writeJsonFile(std::filesystem::path(cudaKernelSaveKeyCaptureFile_.value()),
                  cudaKernelOutOfBandKeysFileJson(networkName, cudaKernelKeys, "complete"));
}

// Records the layers in sorted DAG order.
Network::StatusCode Network::createDagAndFreeze(bool inferenceOnly) {
    if (!frozen) {
        StatusCode status = evaluateGraph(inferenceOnly);
        if (status != StatusCode::SUCCESS)
            return status;
        topologicalSort();
        frozen = true;
    }

    return StatusCode::SUCCESS;
}

// Calls preomptimize on each layer, one at a time, in DAG order.
void Network::preOptimize(uint32_t gpuNum, uint32_t batchSize) {
    for (auto it = orderedNetwork.begin(); it != orderedNetwork.end(); ++it) {
        std::optional<Tensor> inputTensor = it->first;
        shared_ptr<Layer> layer = it->second;

        if (inputTensor.has_value()) {
            layer->preOptimize(inputTensor.value(), batchSize, Stream::getNextUploadStream(gpuNum));
        }
    }
}

// Returns 0 on success, returns an error code (i.e. out of memory) on failure
Network::StatusCode Network::stampNetwork(uint32_t gpuNum,
                                          vector<Event> &initDoneEvents,
                                          uint32_t batchSize,
                                          vector<ThorImplementation::StampedNetwork> &stampedNetworks,
                                          const shared_ptr<GradientUpdateStreamPool>& gradientUpdateStreamPool,
                                          const bool inferenceOnly,
                                          bool networkOutputsOnGpu) {
    ThorImplementation::StampedNetwork stampedNetwork;
    stampedNetwork.gpuNum = gpuNum;
    stampedNetwork.floatingPointOperationsPerExampleForward = 0;
    stampedNetwork.floatingPointOperationsPerExampleBackward = 0;

    // FIXME: check for non-first instance to use shared weights
    // FIXME: support other gpus
    firstInstanceBytes = computeFirstInstanceMemRequirements(batchSize, TensorPlacement(TensorPlacement::MemDevices::GPU, gpuNum));
    nonFirstInstanceBytes = computeNonFirstInstanceMemRequirements(batchSize, TensorPlacement(TensorPlacement::MemDevices::GPU, gpuNum));
    stampedNetwork.bytesRequired = firstInstanceBytes;
    stampedNetwork.batchSize = batchSize;

    // Preserve the same 1 GiB safety reserve used by serialized model
    // startup. This is only the early model-placement check; the complete
    // startup transaction checks again after output, dataset-session, and input
    // staging allocations have also been created.
    // FIXME: need to determine if this is the not the first instance and use shared weights and shared weights mem requirements
    const uint64_t freeMemBytes = MachineEvaluator::instance().getFreeMemBytes(gpuNum);
    if (freeMemBytes <= ThorImplementation::DEVICE_STARTUP_SAFETY_RESERVE_BYTES ||
        firstInstanceBytes >
            freeMemBytes - ThorImplementation::DEVICE_STARTUP_SAFETY_RESERVE_BYTES) {
        return StatusCode::GPU_OUT_OF_MEMORY;
    }

    // 1. Stamp (i.e. construct) all layers
    // 2. At the moment, I connect the layers upon stamping them, I think I should change that and stamp everything first,
    //    then for the next phase, connect them
    stampedNetwork.clear();
    THOR_THROW_IF_FALSE(gradientUpdateStreamPool != nullptr);
    THOR_THROW_IF_FALSE(gradientUpdateStreamPool->getDeviceNum() == gpuNum);
    stampedNetwork.gradientUpdateStreamPool = gradientUpdateStreamPool;
    try {
        // FIXME: need to throw GPU_OUT_OF_MEMORY when stamping and run out of memory

        for (const shared_ptr<Layer>& layer : network) {
            if (layer != nullptr) {
                layer->resetGraphTraversalState();
            }
        }

        uint64_t orderedIndex = 0;
        for (auto it = orderedNetwork.begin(); it != orderedNetwork.end(); ++it, ++orderedIndex) {
            std::optional<Tensor> inputTensor = it->first;
            shared_ptr<Layer> layer = it->second;

            const shared_ptr<NetworkInput> networkInput = dynamic_pointer_cast<NetworkInput>(layer);
            if (networkInput) {
                stampNetworkInput(networkInput, gpuNum, batchSize, stampedNetwork);
                continue;
            }

            THOR_THROW_IF_FALSE(inputTensor.has_value());
            const shared_ptr<NetworkOutput> networkOutput = dynamic_pointer_cast<NetworkOutput>(layer);
            if (networkOutput) {
                stampNetworkOutput(inputTensor.value(), networkOutput, gpuNum, batchSize, stampedNetwork, inferenceOnly, networkOutputsOnGpu);
                continue;
            }

            const shared_ptr<Stub> stub = dynamic_pointer_cast<Stub>(layer);
            if (stub) {
                // FIXME: Stub should cause all dangling tensors to be optimized away.
                //        currently when forward is called for a layer that is a stub, output tensor will not have been allocated
                //        and can cause memory out of bounds. Since stub is a future feature it is not being fixed yet.
                continue;
            }

            stampLayer(inputTensor.value(), layer, gpuNum, batchSize, stampedNetwork, inferenceOnly);
        }

        for (const auto& [name, record] : raggedNetworkInputs) {
            THOR_THROW_IF_FALSE(record.raggedTensor.getBatchSize() == batchSize);
            ThorImplementation::StampedNetwork::RaggedInputBinding binding{
                record.valuesInputName, record.offsetsInputName, record.raggedTensor.getDescriptor()};
            stampedNetwork.raggedInputNamedShared[name] = binding;
            stampedNetwork.raggedInputNamed[name] = binding;
        }

        // 3. Now that all layers have been constructed and connected, compile all layers.
        // Trainable layers already have this model's lazy three-stream pool;
        // training-enabled layers request their stream during stamping because
        // CustomLayer may prepare optimizer expressions during connection.
        //
        // Some physical layers are inserted by stamping and do not have a corresponding API layer.
        // TensorFanout is the important case: it can fuse or prune a single live backward edge during
        // compile by calling replaceErrorInput() on the previous physical layer.  That must happen
        // before API-layer compile, otherwise a trainable layer can compile its backward application
        // against the unfused fanout scratch tensor and never receive the downstream loss gradient.
        std::unordered_set<ThorImplementation::Layer *> apiImplementationLayers;
        for (const auto &apiPhysicalLayer : stampedNetwork.apiLayerToPhysicalLayerShared) {
            if (apiPhysicalLayer.second != nullptr)
                apiImplementationLayers.insert(apiPhysicalLayer.second.get());
        }

        for (const shared_ptr<ThorImplementation::Layer> &implementationLayer : stampedNetwork.otherLayersShared) {
            if (implementationLayer == nullptr)
                continue;
            if (apiImplementationLayers.find(implementationLayer.get()) != apiImplementationLayers.end())
                continue;
            implementationLayer->compile();
            stampedNetwork.floatingPointOperationsPerExampleForward += implementationLayer->floatingPointOperationsPerExampleForward();
            stampedNetwork.floatingPointOperationsPerExampleBackward += implementationLayer->floatingPointOperationsPerExampleBackward();
        }

        for (const shared_ptr<Layer> &apiLayer : allLayersInNetworkList) {
            // It is possible for an implementation layer to have no physical layer, for example a stub layer
            if (stampedNetwork.apiLayerToPhysicalLayerShared.count(apiLayer->getId()) == 0)
                continue;

            shared_ptr<ThorImplementation::Layer> implementationLayer = stampedNetwork.apiLayerToPhysicalLayerShared[apiLayer->getId()];
            apiLayer->compile(implementationLayer);
            stampedNetwork.floatingPointOperationsPerExampleForward += implementationLayer->floatingPointOperationsPerExampleForward();
            stampedNetwork.floatingPointOperationsPerExampleBackward += implementationLayer->floatingPointOperationsPerExampleBackward();
        }

        // Now that all layers are constructed, connected and compiled, initialize all layers
        // (that in turn initialize their optimizers)
        for (shared_ptr<Layer> layer : allLayersInNetworkList) {
            auto physicalLayerIt = stampedNetwork.apiLayerToPhysicalLayerShared.find(layer->getId());
            if (physicalLayerIt == stampedNetwork.apiLayerToPhysicalLayerShared.end() || physicalLayerIt->second == nullptr)
                continue;

            shared_ptr<ThorImplementation::Layer> implementationLayer = physicalLayerIt->second;
            shared_ptr<TrainableLayer> trainableLayer = dynamic_pointer_cast<TrainableLayer>(layer);
            if (trainableLayer != nullptr) {
                shared_ptr<ThorImplementation::TrainableLayer> implementationTrainableLayer =
                    dynamic_pointer_cast<ThorImplementation::TrainableLayer>(implementationLayer);
                THOR_THROW_IF_FALSE(implementationTrainableLayer != nullptr);
                vector<Event> layerEvents = trainableLayer->initialize(implementationTrainableLayer, true, nullptr, std::nullopt);
                stampedNetwork.initializationDoneEvents.insert(stampedNetwork.initializationDoneEvents.end(),
                                                               layerEvents.begin(),
                                                               layerEvents.end());
                initDoneEvents.insert(initDoneEvents.end(), make_move_iterator(layerEvents.begin()), make_move_iterator(layerEvents.end()));
            } else {
                vector<Event> layerEvents = layer->initialize(implementationLayer);
                stampedNetwork.initializationDoneEvents.insert(stampedNetwork.initializationDoneEvents.end(),
                                                               layerEvents.begin(),
                                                               layerEvents.end());
                initDoneEvents.insert(initDoneEvents.end(), make_move_iterator(layerEvents.begin()), make_move_iterator(layerEvents.end()));
            }
        }

        for (const shared_ptr<ThorImplementation::Layer> &implementationLayer : stampedNetwork.otherLayersShared) {
            if (implementationLayer == nullptr)
                continue;
            if (apiImplementationLayers.find(implementationLayer.get()) != apiImplementationLayers.end())
                continue;
            implementationLayer->initialize();
        }

    } catch (const GpuOutOfMemoryError&) {
        // A failed stamp may already own tensors, CUDA handles, and auxiliary
        // allocations whose release lives in Layer::cleanup(), not solely in
        // C++ member destructors. Always tear the partial physical graph down
        // before reporting admission failure.
        stampedNetwork.clearNoThrow();
        return StatusCode::GPU_OUT_OF_MEMORY;
    } catch (...) {
        // CUDA_CHECK-based allocation failures and kernel-launch failures are
        // std::runtime_error, not GpuOutOfMemoryError. Without this catch, a
        // partial stamp can escape without Layer::cleanup(), retaining GPU
        // resources and causing later otherwise-valid models to OOM.
        const std::exception_ptr originalFailure = std::current_exception();
        stampedNetwork.clearNoThrow();
        std::rethrow_exception(originalFailure);
    }

    stampedNetworks.push_back(stampedNetwork);

    std::set<thor_file::TarReader*> executedArchiveReaders;
    if (archiveReader != nullptr) {
        archiveReader->executeReadRequests();
        executedArchiveReaders.insert(archiveReader.get());
    }
    for (const std::shared_ptr<thor_file::TarReader>& cloneArchiveReader : cloneInitializationArchiveReaders) {
        if (cloneArchiveReader != nullptr && executedArchiveReaders.insert(cloneArchiveReader.get()).second) {
            cloneArchiveReader->executeReadRequests();
        }
    }

    return StatusCode::SUCCESS;
}

Network::StatusCode Network::connect(bool inferenceOnly) {
    if (!inferenceOnly) {
        if (defaultOptimizer != nullptr)
            attachOptimizerToLayers(false);

        for (auto trainableLayer : allTrainableLayersInNetwork) {
            if (!trainableLayer->hasOptimizer()) {
                string message = "A layer of type ";
                message += trainableLayer->getLayerType();
                message += " does not have an optimizer assigned, but the network is being placed for training.";
                throw runtime_error(message);
            }
        }
    }

    StatusCode dagStatus = createDagAndFreeze(inferenceOnly);
    if (dagStatus != StatusCode::SUCCESS) {
        printf("ERROR: evaluateGraph() returned %s\n", statusCodeToString(dagStatus).c_str());
        fflush(stdout);
    }

    return dagStatus;
}

shared_ptr<PlacedNetwork> Network::place(uint32_t batchSize,
                                         vector<Event> &initDoneEvents,
                                         bool inferenceOnly,
                                         vector<int32_t> forcedDevices,
                                         uint32_t forcedNumStampsPerGpu,
                                         bool networkOutputsOnGpu) {
    if (networkOutputsOnGpu && !inferenceOnly) {
        throw invalid_argument("networkOutputsOnGpu is only supported for inference-only placement.");
    }
    if (!inferenceOnly) {
        enforceCudaKernelSaveKeyCaptureForTraining();
    }

    if (!frozen) {
        StatusCode dagStatus = connect(inferenceOnly);
        if (dagStatus != StatusCode::SUCCESS)
            throw logic_error("Network graph is invalid, error: " + statusCodeToString(dagStatus));
    }
    THOR_THROW_IF_FALSE(frozen);

    // FIXME: multiple stamps, multiple gpus
    // FIXME: smart placement and stamping
    THOR_THROW_IF_FALSE(forcedNumStampsPerGpu == 0 || forcedNumStampsPerGpu == 1);
    vector<int32_t> gpu0 = {0};
    THOR_THROW_IF_FALSE(forcedDevices == gpu0 || forcedDevices.empty());

    vector<int32_t> devices;
    vector<uint32_t> numStampsPerDevice;
    if (forcedDevices.empty())
        devices = {0};
    else
        devices = forcedDevices;
    for (uint32_t i = 0; i < devices.size(); ++i) {
        if (forcedNumStampsPerGpu > 0) {
            numStampsPerDevice.push_back(forcedNumStampsPerGpu);
        } else {
            numStampsPerDevice.push_back(1);
        }
    }

    vector<ThorImplementation::StampedNetwork> stampedNetworks;
    vector<shared_ptr<GradientUpdateStreamPool>> gradientUpdateStreamPools;
    gradientUpdateStreamPools.reserve(devices.size());
    for (int32_t device : devices) {
        THOR_THROW_IF_FALSE(device >= 0);
        gradientUpdateStreamPools.push_back(std::make_shared<GradientUpdateStreamPool>(static_cast<uint32_t>(device)));
    }

    // FIXME: pull preOptimize into initialize
    for (uint32_t i = 0; i < devices.size(); ++i) {
        preOptimize(devices[i], batchSize);
    }
    for (uint32_t i = 0; i < devices.size(); ++i) {
        for (uint32_t j = 0; j < numStampsPerDevice[i]; ++j) {
            // FIXME: need to propagate inferenceOnly from here through to the API layer to the implementation layer
            StatusCode statusCode = stampNetwork(devices[i],
                                                 initDoneEvents,
                                                 batchSize,
                                                 stampedNetworks,
                                                 gradientUpdateStreamPools[i],
                                                 inferenceOnly,
                                                 networkOutputsOnGpu);
            if (statusCode != StatusCode::SUCCESS)
                throw logic_error("Error when stamping network, error: " + statusCodeToString(statusCode));
        }
    }

    auto placedNetwork = make_shared<PlacedNetwork>(networkName, *this, stampedNetworks);
    return placedNetwork;
}

// Save the architecture only, does not use a stamped network so no state
void Network::save(const string &directory, const bool overwrite) {
    if (defaultOptimizer != nullptr)
        attachOptimizerToLayers(false);

    thor_file::TarWriter archiveWriter(networkName);

    // string modelJsonDump = architectureJsonString();
    json modelJson = architectureJson();
    const std::vector<ThorImplementation::CudaKernelOutOfBandKeys> cudaKernelKeys =
        ThorImplementation::cudaKernelGenerateAndAttachManifestSignatures(modelJson);
    requireCudaKernelSaveKeyCaptureForKeys(cudaKernelKeys);
    writeCudaKernelSaveKeysToCaptureFile(cudaKernelKeys);
    printCudaKernelOutOfBandKeys(cudaKernelKeys);
    string modelJsonDump = modelJson.dump(4);

    string qualifiedModelName = networkName + ".thor.json";
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    ThorImplementation::TensorDescriptor jsonTensorDescriptor(ThorImplementation::DataType::UINT8,
                                                              {modelJsonDump.size()});
    ThorImplementation::Tensor jsonDumpTensor(cpuPlacement, jsonTensorDescriptor);
    memcpy(jsonDumpTensor.getMemPtr<void>(), modelJsonDump.data(), modelJsonDump.size());
    archiveWriter.addArchiveFile(qualifiedModelName, jsonDumpTensor);

    archiveWriter.createArchive(directory, overwrite);
}

// Save the architecture and state - requires a stamped network.
void Network::save(vector<ThorImplementation::StampedNetwork> &stampedNetworks,
                   const string &directory,
                   const bool overwrite,
                   const bool saveOptimizerState) const {
    thor_file::TarWriter archiveWriter(networkName);

    // Snapshot only the streams owned by these stamps.  Training snapshot callers
    // stop batch submission before entering save(), so synchronizing these events
    // is sufficient to make parameter and optimizer state stable without draining
    // unrelated models that happen to share the same CUDA device.
    std::vector<Event> synchronizeEvents;
    for (ThorImplementation::StampedNetwork& stampedNetwork : stampedNetworks) {
        std::vector<Event> stampEvents = stampedNetwork.getSynchronizeEvents();
        synchronizeEvents.insert(synchronizeEvents.end(),
                                 std::make_move_iterator(stampEvents.begin()),
                                 std::make_move_iterator(stampEvents.end()));
    }
    for (Event& event : synchronizeEvents) {
        event.synchronize();
    }

    // Serialization-specific work must not queue behind unrelated users of the
    // process-wide download-stream pool. For the initial implementation, state
    // serialization is still forced through GPU 0.
    Stream stream(0);
    json modelJson;
    modelJson["layers"] = json::array();
    uint32_t stampIndex = 0;
    for (const shared_ptr<Layer> &layer : allLayersInNetworkList) {
        modelJson["layers"].push_back(layer->serialize(archiveWriter, stream, saveOptimizerState, stampedNetworks[stampIndex]));
        stampIndex += 1;
        if (stampIndex >= stampedNetworks.size())
            stampIndex = 0;
    }
    if (!cloneSourceKeyByLayerId.empty()) {
        modelJson["clone_source_keys"] = json::array();
        for (size_t layerIndex = 0; layerIndex < allLayersInNetworkList.size(); ++layerIndex) {
            const shared_ptr<Layer>& layer = allLayersInNetworkList[layerIndex];
            if (layer == nullptr) {
                continue;
            }
            auto keyIt = cloneSourceKeyByLayerId.find(layer->getId());
            if (keyIt != cloneSourceKeyByLayerId.end()) {
                modelJson["clone_source_keys"].push_back(json{{"layer_index", layerIndex}, {"key", keyIt->second}});
            }
        }
    }
    if (!raggedNetworkInputs.empty()) {
        modelJson["ragged_network_inputs"] = json::array();
        for (const auto& [name, record] : raggedNetworkInputs) {
            (void)name;
            modelJson["ragged_network_inputs"].push_back(json{{"version", "1.0.0"},
                                                             {"name", record.name},
                                                             {"values_input_name", record.valuesInputName},
                                                             {"offsets_input_name", record.offsetsInputName},
                                                             {"ragged_tensor", record.raggedTensor.serialize(archiveWriter)}});
        }
    }
    if (defaultOptimizer != nullptr)
        modelJson["default_optimizer"] = defaultOptimizer->architectureJson();

    string qualifiedModelName = networkName + ".thor.json";
    const std::vector<ThorImplementation::CudaKernelOutOfBandKeys> cudaKernelKeys =
        ThorImplementation::cudaKernelGenerateAndAttachManifestSignatures(modelJson);
    requireCudaKernelSaveKeyCaptureForKeys(cudaKernelKeys);
    writeCudaKernelSaveKeysToCaptureFile(cudaKernelKeys);
    printCudaKernelOutOfBandKeys(cudaKernelKeys);

    string jsonDump = modelJson.dump(4);
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    ThorImplementation::TensorDescriptor jsonTensorDescriptor(ThorImplementation::DataType::UINT8, {jsonDump.size()});
    ThorImplementation::Tensor jsonDumpTensor(cpuPlacement, jsonTensorDescriptor);
    memcpy(jsonDumpTensor.getMemPtr<void>(), jsonDump.data(), jsonDump.size());
    archiveWriter.addArchiveFile(qualifiedModelName, jsonDumpTensor);

    archiveWriter.createArchive(directory, overwrite);
}


void Network::registerRaggedNetworkInput(const std::string& name,
                                         const RaggedTensor& raggedTensor,
                                         const std::string& valuesInputName,
                                         const std::string& offsetsInputName) {
    THOR_THROW_IF_FALSE(!name.empty());
    THOR_THROW_IF_FALSE(raggedTensor.isInitialized());
    THOR_THROW_IF_FALSE(!valuesInputName.empty());
    THOR_THROW_IF_FALSE(!offsetsInputName.empty());
    THOR_THROW_IF_FALSE(valuesInputName != offsetsInputName);
    THOR_THROW_IF_FALSE(raggedNetworkInputs.count(name) == 0);

    RaggedNetworkInputRecord record;
    record.name = name;
    record.valuesInputName = valuesInputName;
    record.offsetsInputName = offsetsInputName;
    record.raggedTensor = raggedTensor;
    raggedNetworkInputs[name] = record;
}

bool Network::hasRaggedNetworkInput(const std::string& name) const { return raggedNetworkInputs.count(name) != 0; }

json Network::architectureJson() const {
    json modelJson;
    modelJson["layers"] = json::array();
    uint32_t layerIndex = 0;
    for (const shared_ptr<Layer> &layer : allLayersInNetworkList) {
        modelJson["layers"].push_back(layer->architectureJson());
        ++layerIndex;
    }
    if (!raggedNetworkInputs.empty()) {
        modelJson["ragged_network_inputs"] = json::array();
        for (const auto& [name, record] : raggedNetworkInputs) {
            (void)name;
            modelJson["ragged_network_inputs"].push_back(json{{"version", "1.0.0"},
                                                             {"name", record.name},
                                                             {"values_input_name", record.valuesInputName},
                                                             {"offsets_input_name", record.offsetsInputName},
                                                             {"ragged_tensor", record.raggedTensor.architectureJson()}});
        }
    }
    if (!cloneSourceKeyByLayerId.empty()) {
        modelJson["clone_source_keys"] = json::array();
        for (size_t layerIndex = 0; layerIndex < allLayersInNetworkList.size(); ++layerIndex) {
            const std::shared_ptr<Layer>& layer = allLayersInNetworkList[layerIndex];
            if (layer == nullptr) {
                continue;
            }
            auto keyIt = cloneSourceKeyByLayerId.find(layer->getId());
            if (keyIt != cloneSourceKeyByLayerId.end()) {
                modelJson["clone_source_keys"].push_back(json{{"layer_index", layerIndex}, {"key", keyIt->second}});
            }
        }
    }
    if (defaultOptimizer != nullptr) {
        modelJson["default_optimizer"] = defaultOptimizer->architectureJson();
    }
    return modelJson;
}

string Network::architectureJsonString() const {
    json modelJson = architectureJson();
    (void)ThorImplementation::cudaKernelGenerateAndAttachManifestSignatures(modelJson);
    return modelJson.dump(4);
}

std::vector<std::string> Network::cudaKernelSigningPublicKeys() const {
    json modelJson = architectureJson();
    (void)ThorImplementation::cudaKernelGenerateAndAttachManifestSignatures(modelJson);
    return ThorImplementation::collectCudaKernelSigningPublicKeys(modelJson);
}

std::vector<ThorImplementation::CudaKernelOutOfBandKeys> Network::cudaKernelOutOfBandKeys() const {
    json modelJson = architectureJson();
    return ThorImplementation::cudaKernelGenerateAndAttachManifestSignatures(modelJson);
}

std::vector<ThorImplementation::CudaKernelSourceInspection> Network::cudaKernelSourceInfo() const {
    return ThorImplementation::collectCudaKernelSourceInfo(architectureJson());
}

std::vector<std::string> Network::cudaKernelSources() const {
    std::vector<std::string> sources;
    for (const ThorImplementation::CudaKernelSourceInspection& info : cudaKernelSourceInfo()) {
        sources.push_back(info.source);
    }
    return sources;
}

std::string Network::cudaKernelSourceInfoJsonString() const {
    return ThorImplementation::cudaKernelSourceInspectionListToJson(cudaKernelSourceInfo()).dump(4);
}

void Network::load(const string &directory) { load(directory, false, "", ""); }

void Network::load(const string &directory, bool allowUnsafeLoadedCudaKernelSource) { load(directory, allowUnsafeLoadedCudaKernelSource, "", ""); }

void Network::load(const string &directory, bool allowUnsafeLoadedCudaKernelSource, const string &trustedCudaKernelPublicKey) {
    load(directory, allowUnsafeLoadedCudaKernelSource, trustedCudaKernelPublicKey, "");
}

void Network::load(const string &directory,
                   bool allowUnsafeLoadedCudaKernelSource,
                   const string &trustedCudaKernelPublicKey,
                   const string &trustedCudaKernelSourceDecryptionKey) {
    allowUnsafeLoadedCudaKernelSourceCompilation_ = allowUnsafeLoadedCudaKernelSource;
    trustedLoadedCudaKernelPublicKey_ = trustedCudaKernelPublicKey;
    trustedLoadedCudaKernelSourceDecryptionKey_ = trustedCudaKernelSourceDecryptionKey;
    // Read the model JSON from the exact archive matching this Network name.
    // Thor intentionally does not fall back to "whatever unique model" is in
    // the directory; callers must name the artifact they intend to load.
    const NetworkArchiveSelection archiveSelection = selectNetworkArchiveForLoad(directory, networkName);
    archiveReader = make_shared<thor_file::TarReader>(archiveSelection.archiveName, directory);
    const string& modelJsonFileName = archiveSelection.modelJsonFileName;
    uint32_t modelJsonNumBytes = archiveReader->getFileSize(modelJsonFileName);
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    ThorImplementation::TensorDescriptor modelJsonTensorDescriptor(ThorImplementation::DataType::UINT8,
                                                                   {modelJsonNumBytes});
    ThorImplementation::Tensor modelJsonStrTensor(cpuPlacement, modelJsonTensorDescriptor);
    archiveReader->registerReadRequest(modelJsonFileName, modelJsonStrTensor);
    archiveReader->executeReadRequests();

    char *jsonStr = modelJsonStrTensor.getMemPtr<char>();  // Not null terminated.
    json modelJson = json::parse(jsonStr, jsonStr + modelJsonNumBytes);
    const json layers = modelJson["layers"];
    if (!layers.is_array()) {
        throw runtime_error("\"layers\" is not a JSON array");
    }
    if (modelJson.contains("default_optimizer"))
        defaultOptimizer = Optimizer::deserialize(archiveReader, modelJson["default_optimizer"], this);

    const size_t firstLoadedLayerIndex = allLayersInNetworkList.size();
    for (const json &layerJson : layers) {
        // printf("%s\n", layerJson.dump(4).c_str());
        Layer::deserialize(archiveReader, layerJson, this);
    }
    cloneSourceKeyByLayerId.clear();
    if (modelJson.contains("clone_source_keys")) {
        const json cloneSourceKeys = modelJson.at("clone_source_keys");
        if (!cloneSourceKeys.is_array()) {
            throw runtime_error("\"clone_source_keys\" is not a JSON array");
        }
        for (const json& entry : cloneSourceKeys) {
            const size_t layerIndex = entry.at("layer_index").get<size_t>();
            const string key = entry.at("key").get<string>();
            const size_t absoluteLayerIndex = firstLoadedLayerIndex + layerIndex;
            if (absoluteLayerIndex >= allLayersInNetworkList.size() || allLayersInNetworkList[absoluteLayerIndex] == nullptr) {
                throw runtime_error("clone_source_keys entry references invalid layer_index " + to_string(layerIndex));
            }
            cloneSourceKeyByLayerId[allLayersInNetworkList[absoluteLayerIndex]->getId()] = key;
        }
    }
    loadedFromArchive = true;

    raggedNetworkInputs.clear();
    if (modelJson.contains("ragged_network_inputs")) {
        const json raggedInputs = modelJson["ragged_network_inputs"];
        if (!raggedInputs.is_array()) {
            throw runtime_error("\"ragged_network_inputs\" is not a JSON array");
        }
        for (const json& raggedInputJson : raggedInputs) {
            if (raggedInputJson.at("version").get<string>() != "1.0.0") {
                throw runtime_error("Unsupported ragged_network_inputs version: " + raggedInputJson.at("version").get<string>());
            }
            const string name = raggedInputJson.at("name").get<string>();
            const string valuesInputName = raggedInputJson.at("values_input_name").get<string>();
            const string offsetsInputName = raggedInputJson.at("offsets_input_name").get<string>();
            RaggedTensor raggedTensor = RaggedTensor::deserialize(raggedInputJson.at("ragged_tensor"), archiveReader.get());
            registerRaggedNetworkInput(name, raggedTensor, valuesInputName, offsetsInputName);
        }
    }
}



std::vector<std::shared_ptr<NetworkInput>> Network::getExternalNetworkInputs() const {
    std::vector<std::shared_ptr<NetworkInput>> inputs;
    inputs.reserve(allLayersInNetworkList.size());
    for (const std::shared_ptr<Layer>& layer : allLayersInNetworkList) {
        std::shared_ptr<NetworkInput> input = std::dynamic_pointer_cast<NetworkInput>(layer);
        if (input == nullptr || !input->isExternal() || input->hasPassThroughSource()) {
            continue;
        }
        inputs.push_back(std::move(input));
    }
    return inputs;
}

std::vector<std::string> Network::getInferenceNetworkInputNames() {
    // Build the same inference-only graph view used by place(..., inferenceOnly=true).
    // For saved training artifacts this prunes loss roots and label-only inputs, so
    // deployable ensemble manifests do not advertise labels as inference inputs.
    const StatusCode status = evaluateGraph(/*inferenceOnly=*/true);
    if (status != StatusCode::SUCCESS) {
        throw std::runtime_error("Unable to evaluate inference graph while collecting network input names: " + statusCodeToString(status));
    }

    std::vector<std::string> names;
    const uint32_t numLayers = getNumLayers();
    names.reserve(numLayers);
    for (uint32_t i = 0; i < numLayers; ++i) {
        std::shared_ptr<NetworkInput> input = std::dynamic_pointer_cast<NetworkInput>(getLayer(i));
        if (input == nullptr || !input->getFeatureOutput().has_value()) {
            continue;
        }
        const Tensor outputTensor = input->getFeatureOutput().value();
        if (allTensors.count(outputTensor) == 0) {
            continue;
        }
        auto driverIt = apiTensorToApiDrivingLayer.find(outputTensor);
        if (driverIt == apiTensorToApiDrivingLayer.end() || driverIt->second != input) {
            continue;
        }
        auto loadingIt = apiTensorToApiLoadingLayers.find(outputTensor);
        if (loadingIt == apiTensorToApiLoadingLayers.end() || loadingIt->second.empty()) {
            continue;
        }
        names.push_back(input->getName());
    }
    return names;
}

std::vector<std::string> Network::getInferenceNetworkInputNamesForOutputs(const std::vector<std::string>& outputNames) {
    if (outputNames.empty()) {
        return {};
    }

    // Build the same inference-only graph view used by place(..., inferenceOnly=true),
    // then walk backward only from the requested deployable outputs.  A saved
    // training artifact can legitimately have report-only NetworkOutputs such as
    // Mean(labels).  Those reports make labels visible to getInferenceNetworkInputNames(),
    // but labels are not required to compute prediction-only deployable outputs.
    const StatusCode status = evaluateGraph(/*inferenceOnly=*/true);
    if (status != StatusCode::SUCCESS) {
        throw std::runtime_error("Unable to evaluate inference graph while collecting network input names for outputs: " +
                                 statusCodeToString(status));
    }

    std::map<std::string, Tensor> outputTensorByName;
    const uint32_t numLayers = getNumLayers();
    for (uint32_t i = 0; i < numLayers; ++i) {
        std::shared_ptr<NetworkOutput> output = std::dynamic_pointer_cast<NetworkOutput>(getLayer(i));
        if (output == nullptr || !output->getFeatureInput().has_value()) {
            continue;
        }
        outputTensorByName[output->getName()] = output->getFeatureInput().value();
    }

    std::set<std::string> requiredInputNames;
    std::set<uint64_t> visitedLayerIds;
    std::set<uint64_t> visitingLayerIds;
    std::function<void(const Tensor&)> collectUpstreamInputs = [&](const Tensor& tensor) {
        auto driverIt = apiTensorToApiDrivingLayer.find(tensor);
        if (driverIt == apiTensorToApiDrivingLayer.end() || driverIt->second == nullptr) {
            throw std::runtime_error("Unable to find a driving layer while collecting inference inputs for tensor " +
                                     std::to_string(tensor.getOriginalId()) + ".");
        }

        std::shared_ptr<Layer> driver = driverIt->second;
        std::shared_ptr<NetworkInput> input = std::dynamic_pointer_cast<NetworkInput>(driver);
        if (input != nullptr) {
            requiredInputNames.insert(input->getName());
            return;
        }
        if (std::dynamic_pointer_cast<NetworkOutput>(driver) != nullptr) {
            throw std::runtime_error("Encountered a NetworkOutput as a tensor driver while collecting inference inputs.");
        }

        const uint64_t driverId = driver->getId();
        if (visitedLayerIds.count(driverId) != 0) {
            return;
        }
        if (!visitingLayerIds.insert(driverId).second) {
            throw std::runtime_error("Encountered a cycle while collecting inference inputs for output-bounded subgraph.");
        }

        auto inputsIt = apiLayerToApiInputTensors.find(driver);
        if (inputsIt != apiLayerToApiInputTensors.end()) {
            for (const Tensor& inputTensor : inputsIt->second) {
                collectUpstreamInputs(inputTensor);
            }
        }

        visitingLayerIds.erase(driverId);
        visitedLayerIds.insert(driverId);
    };

    for (const std::string& outputName : outputNames) {
        auto outputIt = outputTensorByName.find(outputName);
        if (outputIt == outputTensorByName.end()) {
            throw std::runtime_error("Unable to find NetworkOutput '" + outputName +
                                     "' while collecting inference input names for network '" + getNetworkName() + "'.");
        }
        collectUpstreamInputs(outputIt->second);
    }

    std::vector<std::string> names;
    names.reserve(requiredInputNames.size());
    for (uint32_t i = 0; i < numLayers; ++i) {
        std::shared_ptr<NetworkInput> input = std::dynamic_pointer_cast<NetworkInput>(getLayer(i));
        if (input == nullptr || requiredInputNames.count(input->getName()) == 0) {
            continue;
        }
        names.push_back(input->getName());
    }
    return names;
}

std::vector<std::string> Network::getTrainingOnlyNetworkInputNames() {
    // Identify training-only inputs structurally instead of relying on archive-time
    // inference pruning.  The original in-memory training graph is not
    // loadedFromArchive, so evaluateGraph(inferenceOnly=true) intentionally leaves
    // loss roots in place there.  A training-only input is one whose value is used
    // only by loss/metric label-style inputs, possibly through pure transform layers
    // such as type conversions or reshapes.
    rebuildApiGraphIndexes(/*inferenceOnly=*/false);

    std::map<Tensor, bool> memo;
    std::set<Tensor> visiting;
    std::function<bool(const Tensor&)> tensorFeedsOnlyTrainingLabelConsumers = [&](const Tensor& tensor) -> bool {
        auto memoIt = memo.find(tensor);
        if (memoIt != memo.end()) {
            return memoIt->second;
        }
        if (visiting.count(tensor) != 0) {
            memo[tensor] = false;
            return false;
        }
        visiting.insert(tensor);

        auto loadingIt = apiTensorToApiLoadingLayers.find(tensor);
        if (loadingIt == apiTensorToApiLoadingLayers.end() || loadingIt->second.empty()) {
            visiting.erase(tensor);
            memo[tensor] = false;
            return false;
        }

        bool result = true;
        for (const std::shared_ptr<Layer>& consumer : loadingIt->second) {
            if (std::dynamic_pointer_cast<NetworkOutput>(consumer) != nullptr) {
                result = false;
                break;
            }

            std::shared_ptr<MultiInputCustomLoss> multiInputCustomLoss = std::dynamic_pointer_cast<MultiInputCustomLoss>(consumer);
            if (multiInputCustomLoss != nullptr) {
                bool isAuxiliaryLossInput = false;
                for (const MultiInputCustomLoss::InputSpec& inputSpec : multiInputCustomLoss->getInputs()) {
                    if (inputSpec.tensor == tensor) {
                        isAuxiliaryLossInput = !inputSpec.isDifferentiable();
                        break;
                    }
                }
                if (!isAuxiliaryLossInput) {
                    result = false;
                    break;
                }
                continue;
            }

            std::shared_ptr<Loss> loss = std::dynamic_pointer_cast<Loss>(consumer);
            if (loss != nullptr) {
                const int connectionType = loss->getConnectionType(tensor);
                if (connectionType != static_cast<int>(ThorImplementation::Loss::ConnectionType::LABELS)) {
                    result = false;
                    break;
                }
                continue;
            }

            std::shared_ptr<Metric> metric = std::dynamic_pointer_cast<Metric>(consumer);
            if (metric != nullptr) {
                const int connectionType = metric->getConnectionType(tensor);
                if (connectionType != static_cast<int>(ThorImplementation::Metric::ConnectionType::LABELS)) {
                    result = false;
                    break;
                }
                continue;
            }

            auto outputsIt = apiLayerToApiOutputTensors.find(consumer);
            if (outputsIt == apiLayerToApiOutputTensors.end() || outputsIt->second.empty()) {
                result = false;
                break;
            }
            for (const Tensor& outputTensor : outputsIt->second) {
                if (!tensorFeedsOnlyTrainingLabelConsumers(outputTensor)) {
                    result = false;
                    break;
                }
            }
            if (!result) {
                break;
            }
        }

        visiting.erase(tensor);
        memo[tensor] = result;
        return result;
    };

    std::vector<std::string> names;
    const uint32_t numLayers = getNumLayers();
    names.reserve(numLayers);
    for (uint32_t i = 0; i < numLayers; ++i) {
        std::shared_ptr<NetworkInput> input = std::dynamic_pointer_cast<NetworkInput>(getLayer(i));
        if (input == nullptr || !input->getFeatureOutput().has_value()) {
            continue;
        }
        if (tensorFeedsOnlyTrainingLabelConsumers(input->getFeatureOutput().value())) {
            names.push_back(input->getName());
        }
    }
    std::sort(names.begin(), names.end());
    names.erase(std::unique(names.begin(), names.end()), names.end());
    return names;
}

std::vector<NetworkLossReference> Network::getReportableLosses() {
    // Reportable loss discovery is based on actual Loss layers that the user
    // explicitly exposes through NetworkOutput.  A loss can remain reportable
    // for the source network even when its prediction tensor is an internal
    // hidden/intermediate tensor that cannot be remapped in a composed ensemble
    // evaluator.  Composition-time reporting decides whether that loss exists
    // for that composition.
    rebuildApiGraphIndexes(/*inferenceOnly=*/false);

    auto sourceNetworkInputName = [&](const Tensor& tensor) -> std::optional<std::string> {
        std::set<Tensor> visiting;
        std::function<std::optional<std::string>(const Tensor&)> visit = [&](const Tensor& current) -> std::optional<std::string> {
            if (visiting.count(current) != 0) {
                return std::nullopt;
            }
            visiting.insert(current);

            auto driverIt = apiTensorToApiDrivingLayer.find(current);
            if (driverIt == apiTensorToApiDrivingLayer.end() || driverIt->second == nullptr) {
                visiting.erase(current);
                return std::nullopt;
            }
            std::shared_ptr<Layer> driver = driverIt->second;
            std::shared_ptr<NetworkInput> input = std::dynamic_pointer_cast<NetworkInput>(driver);
            if (input != nullptr) {
                std::string name = input->getName();
                visiting.erase(current);
                return name;
            }

            auto inputsIt = apiLayerToApiInputTensors.find(driver);
            if (inputsIt == apiLayerToApiInputTensors.end() || inputsIt->second.empty()) {
                visiting.erase(current);
                return std::nullopt;
            }
            std::optional<std::string> found;
            for (const Tensor& upstream : inputsIt->second) {
                std::optional<std::string> candidate = visit(upstream);
                if (!candidate.has_value()) {
                    continue;
                }
                if (found.has_value() && found.value() != candidate.value()) {
                    visiting.erase(current);
                    return std::nullopt;
                }
                found = candidate;
            }
            visiting.erase(current);
            return found;
        };
        return visit(tensor);
    };

    std::map<Tensor, std::vector<std::string>> outputNamesByTensor;
    const uint32_t numLayers = getNumLayers();
    for (uint32_t i = 0; i < numLayers; ++i) {
        std::shared_ptr<NetworkOutput> output = std::dynamic_pointer_cast<NetworkOutput>(getLayer(i));
        if (output == nullptr) {
            continue;
        }
        std::optional<Tensor> outputTensor = output->getFeatureInput();
        if (!outputTensor.has_value()) {
            outputTensor = output->getFeatureOutput();
        }
        if (!outputTensor.has_value()) {
            continue;
        }
        outputNamesByTensor[outputTensor.value()].push_back(output->getName());
    }

    auto addUniqueName = [](std::vector<std::string>& names, const std::string& name) {
        if (std::find(names.begin(), names.end(), name) == names.end()) {
            names.push_back(name);
        }
    };

    std::map<uint64_t, std::vector<std::string>> lossNamesByLayerId;
    auto addLossOutputNameForTensor = [&](const Tensor& tensor, const std::string& outputName) {
        std::set<Tensor> visiting;
        std::function<void(const Tensor&)> visit = [&](const Tensor& current) {
            if (visiting.count(current) != 0) {
                return;
            }
            visiting.insert(current);
            auto driverIt = apiTensorToApiDrivingLayer.find(current);
            if (driverIt == apiTensorToApiDrivingLayer.end() || driverIt->second == nullptr) {
                visiting.erase(current);
                return;
            }
            std::shared_ptr<Layer> driver = driverIt->second;
            std::shared_ptr<Loss> lossDriver = std::dynamic_pointer_cast<Loss>(driver);
            if (lossDriver != nullptr) {
                addUniqueName(lossNamesByLayerId[lossDriver->getId()], outputName);
                visiting.erase(current);
                return;
            }
            auto inputsIt = apiLayerToApiInputTensors.find(driver);
            if (inputsIt != apiLayerToApiInputTensors.end()) {
                for (const Tensor& upstream : inputsIt->second) {
                    visit(upstream);
                }
            }
            visiting.erase(current);
        };
        visit(tensor);
    };

    for (uint32_t i = 0; i < numLayers; ++i) {
        std::shared_ptr<NetworkOutput> output = std::dynamic_pointer_cast<NetworkOutput>(getLayer(i));
        if (output == nullptr) {
            continue;
        }
        std::optional<Tensor> outputTensor = output->getFeatureInput();
        if (!outputTensor.has_value()) {
            outputTensor = output->getFeatureOutput();
        }
        if (!outputTensor.has_value()) {
            continue;
        }
        addLossOutputNameForTensor(outputTensor.value(), output->getName());
    }
    auto predictionOutputNamesForTensor = [&](const Tensor& tensor) -> std::vector<std::string> {
        std::vector<std::string> names;
        std::set<Tensor> visiting;
        std::function<void(const Tensor&)> visit = [&](const Tensor& current) {
            if (visiting.count(current) != 0) {
                return;
            }
            visiting.insert(current);

            auto outputIt = outputNamesByTensor.find(current);
            if (outputIt != outputNamesByTensor.end()) {
                for (const std::string& outputName : outputIt->second) {
                    addUniqueName(names, outputName);
                }
                visiting.erase(current);
                return;
            }

            auto driverIt = apiTensorToApiDrivingLayer.find(current);
            if (driverIt == apiTensorToApiDrivingLayer.end() || driverIt->second == nullptr) {
                visiting.erase(current);
                return;
            }
            std::shared_ptr<Layer> driver = driverIt->second;
            if (std::dynamic_pointer_cast<NetworkInput>(driver) != nullptr || std::dynamic_pointer_cast<Loss>(driver) != nullptr ||
                std::dynamic_pointer_cast<Metric>(driver) != nullptr) {
                visiting.erase(current);
                return;
            }

            auto inputsIt = apiLayerToApiInputTensors.find(driver);
            if (inputsIt != apiLayerToApiInputTensors.end()) {
                for (const Tensor& upstream : inputsIt->second) {
                    visit(upstream);
                }
            }
            visiting.erase(current);
        };
        visit(tensor);
        return names;
    };

    std::vector<NetworkLossReference> references;
    for (uint32_t i = 0; i < numLayers; ++i) {
        std::shared_ptr<Loss> loss = std::dynamic_pointer_cast<Loss>(getLayer(i));
        if (loss == nullptr) {
            continue;
        }

        Tensor predictions;
        std::optional<Tensor> labels;
        std::optional<Tensor> exampleWeights;
        std::shared_ptr<MultiInputCustomLoss> multiInputCustomLoss = std::dynamic_pointer_cast<MultiInputCustomLoss>(loss);
        if (multiInputCustomLoss != nullptr) {
            try {
                predictions = multiInputCustomLoss->getPredictions();
            } catch (const std::exception&) {
                continue;
            }
            for (const MultiInputCustomLoss::InputSpec& input : multiInputCustomLoss->getInputs()) {
                if (input.isDifferentiable()) {
                    continue;
                }
                if (input.name == "labels") {
                    labels = input.tensor;
                } else if (input.name == "example_weights") {
                    exampleWeights = input.tensor;
                }
            }
            if (!labels.has_value()) {
                continue;
            }
        } else {
            try {
                predictions = loss->getPredictions();
                labels = loss->getLabels();
            } catch (const std::exception&) {
                continue;
            }
            exampleWeights = loss->getExampleWeights();
        }

        std::vector<std::string> lossNames;
        auto lossNamesIt = lossNamesByLayerId.find(loss->getId());
        if (lossNamesIt != lossNamesByLayerId.end()) {
            lossNames = lossNamesIt->second;
        }
        if (lossNames.empty()) {
            // Only user-exposed graph losses are reportable.  Do not synthesize
            // report names for unexposed Loss layers.
            continue;
        }

        const std::vector<std::string> predictionOutputNames = predictionOutputNamesForTensor(predictions);
        std::optional<std::string> labelInputName = sourceNetworkInputName(labels.value());
        if (!labelInputName.has_value()) {
            continue;
        }

        std::optional<std::string> weightInputName;
        if (exampleWeights.has_value()) {
            weightInputName = sourceNetworkInputName(exampleWeights.value());
        }

        const double lossWeight = static_cast<double>(loss->getLossWeight().value_or(1.0f));

        std::optional<double> quantile;
        std::shared_ptr<QuantileLoss> quantileLoss = std::dynamic_pointer_cast<QuantileLoss>(loss);
        if (quantileLoss != nullptr) {
            quantile = static_cast<double>(quantileLoss->getQuantile());
        }

        std::vector<std::string> predictionReportSources = predictionOutputNames;
        if (predictionReportSources.empty()) {
            // The loss itself is explicitly exposed, but its prediction side is
            // not an exposed NetworkOutput.  Keep it reportable for the source
            // network; composed ensemble evaluators will skip it because there is
            // no averaged prediction tensor to remap.
            predictionReportSources.push_back("");
        }

        for (const std::string& outputName : predictionReportSources) {
            for (const std::string& lossName : lossNames) {
                NetworkLossReference reference;
                reference.lossName = lossName;
                reference.predictionOutputName = outputName;
                reference.targetInputName = labelInputName.value();
                reference.weightInputName = weightInputName;
                reference.lossLayerType = loss->getLayerType();
                reference.lossWeight = lossWeight;
                reference.quantile = quantile;
                references.push_back(std::move(reference));
            }
        }
    }

    std::sort(references.begin(), references.end(), [](const NetworkLossReference& lhs, const NetworkLossReference& rhs) {
        if (lhs.lossName != rhs.lossName) {
            return lhs.lossName < rhs.lossName;
        }
        if (lhs.predictionOutputName != rhs.predictionOutputName) {
            return lhs.predictionOutputName < rhs.predictionOutputName;
        }
        if (lhs.targetInputName != rhs.targetInputName) {
            return lhs.targetInputName < rhs.targetInputName;
        }
        if (lhs.lossLayerType != rhs.lossLayerType) {
            return lhs.lossLayerType < rhs.lossLayerType;
        }
        if (lhs.weightInputName != rhs.weightInputName) {
            return lhs.weightInputName < rhs.weightInputName;
        }
        return lhs.quantile < rhs.quantile;
    });
    return references;
}

std::vector<NetworkMetricReference> Network::getReportableMetrics() {
    rebuildApiGraphIndexes(/*inferenceOnly=*/false);

    auto sourceNetworkInputName = [&](const Tensor& tensor) -> std::optional<std::string> {
        std::set<Tensor> visiting;
        std::function<std::optional<std::string>(const Tensor&)> visit = [&](const Tensor& current) -> std::optional<std::string> {
            if (visiting.count(current) != 0) {
                return std::nullopt;
            }
            visiting.insert(current);

            auto driverIt = apiTensorToApiDrivingLayer.find(current);
            if (driverIt == apiTensorToApiDrivingLayer.end() || driverIt->second == nullptr) {
                visiting.erase(current);
                return std::nullopt;
            }
            std::shared_ptr<Layer> driver = driverIt->second;
            std::shared_ptr<NetworkInput> input = std::dynamic_pointer_cast<NetworkInput>(driver);
            if (input != nullptr) {
                std::string name = input->getName();
                visiting.erase(current);
                return name;
            }

            auto inputsIt = apiLayerToApiInputTensors.find(driver);
            if (inputsIt == apiLayerToApiInputTensors.end() || inputsIt->second.empty()) {
                visiting.erase(current);
                return std::nullopt;
            }
            std::optional<std::string> found;
            for (const Tensor& upstream : inputsIt->second) {
                std::optional<std::string> candidate = visit(upstream);
                if (!candidate.has_value()) {
                    continue;
                }
                if (found.has_value() && found.value() != candidate.value()) {
                    visiting.erase(current);
                    return std::nullopt;
                }
                found = candidate;
            }
            visiting.erase(current);
            return found;
        };
        return visit(tensor);
    };

    std::map<Tensor, std::vector<std::string>> outputNamesByTensor;
    const uint32_t numLayers = getNumLayers();
    for (uint32_t i = 0; i < numLayers; ++i) {
        std::shared_ptr<NetworkOutput> output = std::dynamic_pointer_cast<NetworkOutput>(getLayer(i));
        if (output == nullptr) {
            continue;
        }
        std::optional<Tensor> outputTensor = output->getFeatureInput();
        if (!outputTensor.has_value()) {
            outputTensor = output->getFeatureOutput();
        }
        if (!outputTensor.has_value()) {
            continue;
        }
        outputNamesByTensor[outputTensor.value()].push_back(output->getName());
    }

    auto addUniqueName = [](std::vector<std::string>& names, const std::string& name) {
        if (std::find(names.begin(), names.end(), name) == names.end()) {
            names.push_back(name);
        }
    };

    std::map<uint64_t, std::vector<std::string>> metricNamesByLayerId;
    auto addMetricOutputNameForTensor = [&](const Tensor& tensor, const std::string& outputName) {
        std::set<Tensor> visiting;
        std::function<void(const Tensor&)> visit = [&](const Tensor& current) {
            if (visiting.count(current) != 0) {
                return;
            }
            visiting.insert(current);
            auto driverIt = apiTensorToApiDrivingLayer.find(current);
            if (driverIt == apiTensorToApiDrivingLayer.end() || driverIt->second == nullptr) {
                visiting.erase(current);
                return;
            }
            std::shared_ptr<Layer> driver = driverIt->second;
            std::shared_ptr<Metric> metricDriver = std::dynamic_pointer_cast<Metric>(driver);
            if (metricDriver != nullptr) {
                addUniqueName(metricNamesByLayerId[metricDriver->getId()], outputName);
                visiting.erase(current);
                return;
            }
            auto inputsIt = apiLayerToApiInputTensors.find(driver);
            if (inputsIt != apiLayerToApiInputTensors.end()) {
                for (const Tensor& upstream : inputsIt->second) {
                    visit(upstream);
                }
            }
            visiting.erase(current);
        };
        visit(tensor);
    };

    for (uint32_t i = 0; i < numLayers; ++i) {
        std::shared_ptr<NetworkOutput> output = std::dynamic_pointer_cast<NetworkOutput>(getLayer(i));
        if (output == nullptr) {
            continue;
        }
        std::optional<Tensor> outputTensor = output->getFeatureInput();
        if (!outputTensor.has_value()) {
            outputTensor = output->getFeatureOutput();
        }
        if (!outputTensor.has_value()) {
            continue;
        }
        addMetricOutputNameForTensor(outputTensor.value(), output->getName());
    }

    auto predictionOutputNamesForTensor = [&](const Tensor& tensor) -> std::vector<std::string> {
        std::vector<std::string> names;
        std::set<Tensor> visiting;
        std::function<void(const Tensor&)> visit = [&](const Tensor& current) {
            if (visiting.count(current) != 0) {
                return;
            }
            visiting.insert(current);

            auto outputIt = outputNamesByTensor.find(current);
            if (outputIt != outputNamesByTensor.end()) {
                for (const std::string& outputName : outputIt->second) {
                    addUniqueName(names, outputName);
                }
                visiting.erase(current);
                return;
            }

            auto driverIt = apiTensorToApiDrivingLayer.find(current);
            if (driverIt == apiTensorToApiDrivingLayer.end() || driverIt->second == nullptr) {
                visiting.erase(current);
                return;
            }
            std::shared_ptr<Layer> driver = driverIt->second;
            if (std::dynamic_pointer_cast<NetworkInput>(driver) != nullptr || std::dynamic_pointer_cast<Loss>(driver) != nullptr ||
                std::dynamic_pointer_cast<Metric>(driver) != nullptr) {
                visiting.erase(current);
                return;
            }

            auto inputsIt = apiLayerToApiInputTensors.find(driver);
            if (inputsIt != apiLayerToApiInputTensors.end()) {
                for (const Tensor& upstream : inputsIt->second) {
                    visit(upstream);
                }
            }
            visiting.erase(current);
        };
        visit(tensor);
        return names;
    };

    std::vector<NetworkMetricReference> references;
    for (uint32_t i = 0; i < numLayers; ++i) {
        std::shared_ptr<Metric> metric = std::dynamic_pointer_cast<Metric>(getLayer(i));
        if (metric == nullptr) {
            continue;
        }

        std::vector<std::string> metricNames;
        auto metricNamesIt = metricNamesByLayerId.find(metric->getId());
        if (metricNamesIt != metricNamesByLayerId.end()) {
            metricNames = metricNamesIt->second;
        }
        if (metricNames.empty()) {
            // A graph metric is reportable only when the user explicitly exposes it
            // through a NetworkOutput.  The metric's upstream source can be a
            // prediction, label, weight, feature, or any other API tensor.
            continue;
        }

        Tensor predictions;
        std::optional<Tensor> labels;
        try {
            predictions = metric->getPredictions();
            if (metric->requiresLabels()) {
                labels = metric->getLabels();
            }
        } catch (const std::exception&) {
            continue;
        }

        const std::vector<std::string> predictionOutputNames = predictionOutputNamesForTensor(predictions);
        std::optional<std::string> inputSourceName;
        if (predictionOutputNames.empty()) {
            // A metric explicitly exposed through NetworkOutput is reportable even
            // when its prediction/source tensor is hidden inside the graph.  The
            // composed ensemble evaluator later decides whether that metric is
            // valid for a particular composition based on which remapped tensors
            // are actually present.
            inputSourceName = sourceNetworkInputName(predictions);
        }
        std::optional<std::string> labelInputName;
        if (labels.has_value()) {
            labelInputName = sourceNetworkInputName(labels.value());
        }

        for (const std::string& metricName : metricNames) {
            if (!predictionOutputNames.empty()) {
                for (const std::string& outputName : predictionOutputNames) {
                    NetworkMetricReference reference;
                    reference.metricName = metricName;
                    reference.predictionOutputName = outputName;
                    reference.targetInputName = labelInputName;
                    reference.inputSourceName = inputSourceName;
                    reference.metricLayerType = metric->getLayerType();
                    references.push_back(std::move(reference));
                }
            } else {
                NetworkMetricReference reference;
                reference.metricName = metricName;
                reference.targetInputName = labelInputName;
                reference.inputSourceName = inputSourceName;
                reference.metricLayerType = metric->getLayerType();
                references.push_back(std::move(reference));
            }
        }
    }

    std::sort(references.begin(), references.end(), [](const NetworkMetricReference& lhs, const NetworkMetricReference& rhs) {
        if (lhs.metricName != rhs.metricName) {
            return lhs.metricName < rhs.metricName;
        }
        if (lhs.predictionOutputName != rhs.predictionOutputName) {
            return lhs.predictionOutputName < rhs.predictionOutputName;
        }
        if (lhs.targetInputName != rhs.targetInputName) {
            return lhs.targetInputName < rhs.targetInputName;
        }
        if (lhs.inputSourceName != rhs.inputSourceName) {
            return lhs.inputSourceName < rhs.inputSourceName;
        }
        return lhs.metricLayerType < rhs.metricLayerType;
    });
    return references;
}

void Network::pruneLoadedTrainingArtifactsForInference() {
    // Saved training artifacts often contain loss layers and label-only NetworkInputs.
    // When such an artifact is loaded and placed for inference, keep the graph rooted
    // at non-loss NetworkOutputs and prune the training-only loss/label subgraph.
    std::set<Tensor> lossOutputTensors;
    for (const std::shared_ptr<Layer>& layer : network) {
        std::shared_ptr<Loss> loss = std::dynamic_pointer_cast<Loss>(layer);
        if (loss) {
            lossOutputTensors.insert(loss->getLoss());
        }
    }
    if (lossOutputTensors.empty()) {
        return;
    }

    std::set<std::shared_ptr<Layer>, Network::LayerComparator> liveLayers;
    std::set<Tensor> liveTensors;

    std::function<void(const std::shared_ptr<Layer>&)> markLiveLayer = [&](const std::shared_ptr<Layer>& layer) {
        if (liveLayers.count(layer) != 0) {
            return;
        }
        if (std::dynamic_pointer_cast<Loss>(layer)) {
            return;
        }
        liveLayers.insert(layer);

        auto outputsIt = apiLayerToApiOutputTensors.find(layer);
        if (outputsIt != apiLayerToApiOutputTensors.end()) {
            for (const Tensor& outputTensor : outputsIt->second) {
                liveTensors.insert(outputTensor);
            }
        }

        auto inputsIt = apiLayerToApiInputTensors.find(layer);
        if (inputsIt == apiLayerToApiInputTensors.end()) {
            return;
        }
        for (const Tensor& inputTensor : inputsIt->second) {
            liveTensors.insert(inputTensor);
            auto driverIt = apiTensorToApiDrivingLayer.find(inputTensor);
            if (driverIt != apiTensorToApiDrivingLayer.end()) {
                markLiveLayer(driverIt->second);
            }
        }
    };

    std::map<Tensor, bool> tensorDependsOnLossCache;
    std::function<bool(const Tensor&)> tensorDependsOnLoss = [&](const Tensor& tensor) -> bool {
        auto cacheIt = tensorDependsOnLossCache.find(tensor);
        if (cacheIt != tensorDependsOnLossCache.end()) {
            return cacheIt->second;
        }

        auto driverIt = apiTensorToApiDrivingLayer.find(tensor);
        if (driverIt == apiTensorToApiDrivingLayer.end()) {
            tensorDependsOnLossCache[tensor] = false;
            return false;
        }

        const std::shared_ptr<Layer>& driver = driverIt->second;
        if (std::dynamic_pointer_cast<Loss>(driver)) {
            tensorDependsOnLossCache[tensor] = true;
            return true;
        }

        auto inputsIt = apiLayerToApiInputTensors.find(driver);
        if (inputsIt != apiLayerToApiInputTensors.end()) {
            for (const Tensor& inputTensor : inputsIt->second) {
                if (tensorDependsOnLoss(inputTensor)) {
                    tensorDependsOnLossCache[tensor] = true;
                    return true;
                }
            }
        }

        tensorDependsOnLossCache[tensor] = false;
        return false;
    };

    for (const std::shared_ptr<Layer>& layer : network) {
        std::shared_ptr<NetworkOutput> networkOutput = std::dynamic_pointer_cast<NetworkOutput>(layer);
        if (!networkOutput) {
            continue;
        }
        Tensor inputTensor = networkOutput->getFeatureInput().value();
        if (lossOutputTensors.count(inputTensor) != 0 || tensorDependsOnLoss(inputTensor)) {
            continue;
        }
        markLiveLayer(networkOutput);
    }

    // Preserve existing behavior for artifacts that only expose loss outputs. In
    // that case there is no prediction root to prune to, and keeping the loss
    // graph is more useful than producing a graph with no outputs.
    bool hasLiveNetworkOutput = false;
    for (const std::shared_ptr<Layer>& layer : liveLayers) {
        if (std::dynamic_pointer_cast<NetworkOutput>(layer)) {
            hasLiveNetworkOutput = true;
            break;
        }
    }
    if (!hasLiveNetworkOutput) {
        return;
    }

    for (auto it = apiTensorToApiDrivingLayer.begin(); it != apiTensorToApiDrivingLayer.end();) {
        if (liveTensors.count(it->first) == 0 || liveLayers.count(it->second) == 0) {
            it = apiTensorToApiDrivingLayer.erase(it);
        } else {
            ++it;
        }
    }

    for (auto it = apiTensorToApiLoadingLayers.begin(); it != apiTensorToApiLoadingLayers.end();) {
        if (liveTensors.count(it->first) == 0) {
            it = apiTensorToApiLoadingLayers.erase(it);
            continue;
        }
        std::vector<std::shared_ptr<Layer>> filtered;
        for (const std::shared_ptr<Layer>& loadingLayer : it->second) {
            if (liveLayers.count(loadingLayer) != 0) {
                filtered.push_back(loadingLayer);
            }
        }
        if (filtered.empty()) {
            it = apiTensorToApiLoadingLayers.erase(it);
        } else {
            it->second = std::move(filtered);
            ++it;
        }
    }

    for (auto it = apiLayerToApiInputTensors.begin(); it != apiLayerToApiInputTensors.end();) {
        if (liveLayers.count(it->first) == 0) {
            it = apiLayerToApiInputTensors.erase(it);
            continue;
        }
        std::vector<Tensor> filtered;
        for (const Tensor& tensor : it->second) {
            if (liveTensors.count(tensor) != 0) {
                filtered.push_back(tensor);
            }
        }
        it->second = std::move(filtered);
        ++it;
    }

    for (auto it = apiLayerToApiOutputTensors.begin(); it != apiLayerToApiOutputTensors.end();) {
        if (liveLayers.count(it->first) == 0) {
            it = apiLayerToApiOutputTensors.erase(it);
            continue;
        }
        std::vector<Tensor> filtered;
        for (const Tensor& tensor : it->second) {
            if (liveTensors.count(tensor) != 0) {
                filtered.push_back(tensor);
            }
        }
        it->second = std::move(filtered);
        ++it;
    }


    allTensors = std::move(liveTensors);
}

// Determine the graph structure
// Tensors are the edges that connect the Layers which are nodes.
void Network::rebuildApiGraphIndexes(bool inferenceOnly) {
    allTensors.clear();
    apiTensorToApiLoadingLayers.clear();
    apiTensorToApiDrivingLayer.clear();
    apiLayerToApiInputTensors.clear();
    apiLayerToApiOutputTensors.clear();

    for (auto it = network.begin(); it != network.end(); ++it) {
        shared_ptr<Layer> layer = *it;

        // Handle each class of layers
        shared_ptr<NetworkInput> networkInput = dynamic_pointer_cast<NetworkInput>(layer);
        if (networkInput) {
            if (networkInput->hasPassThroughSource()) {
                // API-level pass-through inputs are aliases for another API tensor.
                // They are intentionally invisible to the graph as source layers:
                // downstream layers consume the source tensor directly, so stamping
                // can fan it out without creating another external-load endpoint.
                continue;
            }
            Tensor outputTensor = networkInput->getFeatureOutput().value();
            allTensors.insert(outputTensor);
            THOR_THROW_IF_FALSE(apiTensorToApiDrivingLayer.count(outputTensor) == 0);
            apiTensorToApiDrivingLayer[outputTensor] = networkInput;
            apiLayerToApiOutputTensors[networkInput].push_back(outputTensor);
            continue;
        }

        shared_ptr<NetworkOutput> networkOutput = dynamic_pointer_cast<NetworkOutput>(layer);
        if (networkOutput) {
            Tensor inputTensor = networkOutput->getFeatureInput().value();
            allTensors.insert(inputTensor);
            apiTensorToApiLoadingLayers[inputTensor].push_back(networkOutput);
            apiLayerToApiInputTensors[networkOutput].push_back(inputTensor);
            continue;
        }

        shared_ptr<Stub> stub = dynamic_pointer_cast<Stub>(layer);
        if (stub) {
            Tensor inputTensor = stub->getFeatureInput().value();
            allTensors.insert(inputTensor);
            apiTensorToApiLoadingLayers[inputTensor].push_back(stub);
            apiLayerToApiInputTensors[stub].push_back(inputTensor);
            continue;
        }

        shared_ptr<Loss> loss = dynamic_pointer_cast<Loss>(layer);
        if (loss) {
            // Loss inputs in, Loss out. Most losses expose predictions + labels, while multi-input losses expose
            // multiple differentiable operands.
            vector<Tensor> lossInputTensors = loss->getLossInputTensors();
            Tensor lossTensor = loss->getLoss();
            THOR_THROW_IF_FALSE(!lossInputTensors.empty());
            for (const Tensor& inputTensor : lossInputTensors) {
                allTensors.insert(inputTensor);
                apiTensorToApiLoadingLayers[inputTensor].push_back(loss);
                apiLayerToApiInputTensors[loss].push_back(inputTensor);
            }
            allTensors.insert(lossTensor);
            THOR_THROW_IF_FALSE(apiTensorToApiDrivingLayer.count(lossTensor) == 0);
            apiTensorToApiDrivingLayer[lossTensor] = loss;
            apiLayerToApiOutputTensors[loss].push_back(lossTensor);
            continue;
        }

        shared_ptr<Metric> metric = dynamic_pointer_cast<Metric>(layer);
        if (metric) {
            Tensor inputTensor = metric->getFeatureInput().value();
            Tensor outputTensor = metric->getFeatureOutput().value();
            allTensors.insert(inputTensor);
            allTensors.insert(outputTensor);
            apiTensorToApiLoadingLayers[inputTensor].push_back(metric);
            apiLayerToApiInputTensors[metric].push_back(inputTensor);

            if (metric->requiresLabels()) {
                Tensor labelsTensor = metric->getLabels();
                allTensors.insert(labelsTensor);
                apiTensorToApiLoadingLayers[labelsTensor].push_back(metric);
                apiLayerToApiInputTensors[metric].push_back(labelsTensor);
            }

            THOR_THROW_IF_FALSE(apiTensorToApiDrivingLayer.count(outputTensor) == 0);
            apiTensorToApiDrivingLayer[outputTensor] = metric;
            apiLayerToApiOutputTensors[metric].push_back(outputTensor);
            continue;
        }

        shared_ptr<CustomLayer> customLayer = dynamic_pointer_cast<CustomLayer>(layer);
        if (customLayer) {
            vector<Tensor> inputTensors = customLayer->getFeatureInputs();
            vector<Tensor> outputTensors = customLayer->getFeatureOutputs();
            THOR_THROW_IF_FALSE(!inputTensors.empty());
            THOR_THROW_IF_FALSE(!outputTensors.empty());
            for (uint32_t i = 0; i < inputTensors.size(); ++i) {
                allTensors.insert(inputTensors[i]);
                apiTensorToApiLoadingLayers[inputTensors[i]].push_back(customLayer);
                apiLayerToApiInputTensors[customLayer].push_back(inputTensors[i]);
            }
            for (uint32_t i = 0; i < outputTensors.size(); ++i) {
                allTensors.insert(outputTensors[i]);
                THOR_THROW_IF_FALSE(apiTensorToApiDrivingLayer.count(outputTensors[i]) == 0);
                apiTensorToApiDrivingLayer[outputTensors[i]] = customLayer;
                apiLayerToApiOutputTensors[customLayer].push_back(outputTensors[i]);
            }
            continue;
        }

        shared_ptr<Activation> activationLayer = dynamic_pointer_cast<Activation>(layer);
        if (activationLayer && activationLayer->mustConnectAllInputsToDriveOutput()) {
            vector<Tensor> inputTensors = activationLayer->getFeatureInputs();
            vector<Tensor> outputTensors = activationLayer->getFeatureOutputs();
            THOR_THROW_IF_FALSE(!inputTensors.empty());
            THOR_THROW_IF_FALSE(!outputTensors.empty());
            for (uint32_t i = 0; i < inputTensors.size(); ++i) {
                allTensors.insert(inputTensors[i]);
                apiTensorToApiLoadingLayers[inputTensors[i]].push_back(activationLayer);
                apiLayerToApiInputTensors[activationLayer].push_back(inputTensors[i]);
            }
            for (uint32_t i = 0; i < outputTensors.size(); ++i) {
                allTensors.insert(outputTensors[i]);
                THOR_THROW_IF_FALSE(apiTensorToApiDrivingLayer.count(outputTensors[i]) == 0);
                apiTensorToApiDrivingLayer[outputTensors[i]] = activationLayer;
                apiLayerToApiOutputTensors[activationLayer].push_back(outputTensors[i]);
            }
            continue;
        }

        shared_ptr<MultiConnectionLayer> multiConnectionLayer = dynamic_pointer_cast<MultiConnectionLayer>(layer);
        if (multiConnectionLayer) {
            vector<Tensor> inputTensors = multiConnectionLayer->getFeatureInputs();
            vector<Tensor> outputTensors = multiConnectionLayer->getFeatureOutputs();
            THOR_THROW_IF_FALSE(!inputTensors.empty());
            THOR_THROW_IF_FALSE(!outputTensors.empty());
            for (uint32_t i = 0; i < inputTensors.size(); ++i) {
                allTensors.insert(inputTensors[i]);
                apiTensorToApiLoadingLayers[inputTensors[i]].push_back(multiConnectionLayer);
                apiLayerToApiInputTensors[multiConnectionLayer].push_back(inputTensors[i]);
            }
            for (uint32_t i = 0; i < outputTensors.size(); ++i) {
                allTensors.insert(outputTensors[i]);
                THOR_THROW_IF_FALSE(apiTensorToApiDrivingLayer.count(outputTensors[i]) == 0);
                apiTensorToApiDrivingLayer[outputTensors[i]] = multiConnectionLayer;
                apiLayerToApiOutputTensors[multiConnectionLayer].push_back(outputTensors[i]);
            }
            continue;
        }

        // So it is a base single connection layer
        Tensor inputTensor = layer->getFeatureInput().value();
        Tensor outputTensor = layer->getFeatureOutput().value();
        allTensors.insert(inputTensor);
        allTensors.insert(outputTensor);
        THOR_THROW_IF_FALSE(apiTensorToApiDrivingLayer.count(outputTensor) == 0);
        apiTensorToApiDrivingLayer[outputTensor] = layer;
        apiTensorToApiLoadingLayers[inputTensor].push_back(layer);
        apiLayerToApiInputTensors[layer].push_back(inputTensor);
        apiLayerToApiOutputTensors[layer].push_back(outputTensor);
    }

    if (inferenceOnly && loadedFromArchive) {
        pruneLoadedTrainingArtifactsForInference();
    }
}

Network::StatusCode Network::evaluateGraph(bool inferenceOnly) {
    rebuildApiGraphIndexes(inferenceOnly);

    StatusCode status;
    status = checkForDuplicateInOutPortNames();
    if (status != StatusCode::SUCCESS)
        return status;

    status = checkForFloatingInputs();
    if (status != StatusCode::SUCCESS)
        return status;

    status = checkForDanglingOutputs();
    if (status != StatusCode::SUCCESS)
        return status;

    status = checkForDeadlockCycles();
    if (status != StatusCode::SUCCESS)
        return status;

    return StatusCode::SUCCESS;
}

Network::StatusCode Network::checkForDuplicateInOutPortNames() {
    StatusCode status = StatusCode::SUCCESS;

    set<string> inputNames;
    for (auto it = network.begin(); it != network.end(); ++it) {
        shared_ptr<Layer> layer = *it;
        const shared_ptr<NetworkInput> networkInput = dynamic_pointer_cast<NetworkInput>(layer);
        if (networkInput != nullptr) {
            if (networkInput->hasPassThroughSource()) {
                continue;
            }
            if (inputNames.count(networkInput->getName()) != 0) {
                printf("Duplicate network input name used: %s\n", networkInput->getName().c_str());
                status = StatusCode::DUPLICATE_NAMED_NETWORK_INPUT;
            }
            inputNames.insert(networkInput->getName());
        }
    }

    for (const auto& [name, record] : raggedNetworkInputs) {
        (void)record;
        if (inputNames.count(name) != 0) {
            printf("Duplicate network input name used: %s\n", name.c_str());
            status = StatusCode::DUPLICATE_NAMED_NETWORK_INPUT;
        }
        inputNames.insert(name);
    }

    set<string> outputNames;
    for (auto it = network.begin(); it != network.end(); ++it) {
        shared_ptr<Layer> layer = *it;
        const shared_ptr<NetworkOutput> networkOutput = dynamic_pointer_cast<NetworkOutput>(layer);
        if (networkOutput != nullptr) {
            if (outputNames.count(networkOutput->getName()) != 0) {
                printf("Duplicate network output name used: %s\n", networkOutput->getName().c_str());
                status = StatusCode::DUPLICATE_NAMED_NETWORK_OUTPUT;
            }
            outputNames.insert(networkOutput->getName());
        }
    }

    return status;
}

/**
 * A tensor has a floating input when nothing is connected to write to it. -> No Driver.
 */
Network::StatusCode Network::checkForFloatingInputs() {
    for (auto it = allTensors.begin(); it != allTensors.end(); ++it) {
        Tensor tensor = *it;
        if (apiTensorToApiDrivingLayer.count(tensor) == 0) {
            printf("Tensor with id = %ld (original id %ld) is not driven.\n", tensor.getId(), tensor.getOriginalId());
            fflush(stdout);
            return StatusCode::FLOATING_INPUT;
        }
    }
    return StatusCode::SUCCESS;
}

/**
 * A tensor has a dangling output when nothing is connected to read from it -> No consumer.
 */
Network::StatusCode Network::checkForDanglingOutputs() {
    for (auto it = allTensors.begin(); it != allTensors.end(); ++it) {
        Tensor tensor = *it;
        if (apiTensorToApiLoadingLayers.count(tensor) == 0) {
            printf("tensor with id = %ld (original id %ld) is not loaded.\n", tensor.getId(), tensor.getOriginalId());
            fflush(stdout);
            return StatusCode::DANGLING_OUTPUT;
        }
    }
    return StatusCode::SUCCESS;
}

/**
 * A deadlock cycle occurs when a layer that requires all of its input to arrive before it drives its output
 * is connected in a way where there is a path from its output to its input.
 */
Network::StatusCode Network::checkForDeadlockCycles() {
    for (auto it = network.begin(); it != network.end(); ++it) {
        shared_ptr<Layer> layer = *it;
        if (layer->mustConnectAllInputsToDriveOutput()) {
            vector<Tensor> outputs = apiLayerToApiOutputTensors[layer];
            for (uint32_t i = 0; i < outputs.size(); ++i) {
                if (terminatesWithoutHitting(outputs[i], layer) == false)
                    return StatusCode::DEADLOCK_CYCLE;
            }
        }
    }
    return StatusCode::SUCCESS;
}

bool Network::terminatesWithoutHitting(Tensor tensor, shared_ptr<Layer> layer) {
    vector<shared_ptr<Layer>> tensorLoadingLayers = apiTensorToApiLoadingLayers[tensor];
    for (uint32_t i = 0; i < tensorLoadingLayers.size(); ++i) {
        shared_ptr<Layer> loadingLayer = tensorLoadingLayers[i];
        if (loadingLayer == layer) {
            return false;
        } else {
            vector<Tensor> layerOutputTensors = apiLayerToApiOutputTensors[loadingLayer];
            for (uint32_t j = 0; j < layerOutputTensors.size(); ++j) {
                Tensor outputTensor = layerOutputTensors[j];
                if (terminatesWithoutHitting(outputTensor, layer) == false)
                    return false;
            }
        }
    }
    return true;
}

void Network::topologicalSort() {
    deque<pair<std::optional<Tensor>, shared_ptr<Layer>>> workQueue;

    for (const shared_ptr<Layer>& layer : network) {
        if (layer != nullptr) {
            layer->resetGraphTraversalState();
        }
    }
    orderedNetwork.clear();

    // Put all network inputs into the work queue
    for (auto it = network.begin(); it != network.end(); ++it) {
        shared_ptr<Layer> layer = *it;

        const shared_ptr<NetworkInput> networkInput = dynamic_pointer_cast<NetworkInput>(layer);
        if (networkInput) {
            if (networkInput->hasPassThroughSource()) {
                continue;
            }
            Tensor outputTensor = layer->getFeatureOutput().value();
            if (allTensors.count(outputTensor) == 0)
                continue;
            auto loadingLayerIt = apiTensorToApiLoadingLayers.find(outputTensor);
            if (loadingLayerIt == apiTensorToApiLoadingLayers.end())
                continue;
            vector<shared_ptr<Layer>> loadingLayers = loadingLayerIt->second;
            for (uint32_t i = 0; i < loadingLayers.size(); ++i) {
                workQueue.push_front(make_pair(outputTensor, loadingLayers[i]));
            }

            orderedNetwork.push_back(make_pair(std::nullopt, layer));
        }
    }

    while (!workQueue.empty()) {
        // Visit a node, connect the output tensor that corresponds to this input tensor by adding the loading layer
        // and its input tensor to orderedNetwork
        // After connecting an output tensor to its loading layer, add that loading layer and its input tensor to the work queue.
        pair<std::optional<Tensor>, shared_ptr<Layer>> workNode = workQueue.back();
        workQueue.pop_back();
        std::optional<Tensor> inputTensor = workNode.first;
        shared_ptr<Layer> layer = workNode.second;
        THOR_THROW_IF_FALSE(inputTensor.has_value());

        // printf("connecting tensor %ld into layer id %ld\n", inputTensor.value().getId(), layer->getId());

        // For layers, such as concatenate, that need all inputs to be connected before creating the output
        layer->informThatInputConnectionMade(inputTensor.value());

        vector<Tensor> outputTensors = layer->getOutputsFromInput(inputTensor.value());
        for (uint32_t t = 0; t < outputTensors.size(); ++t) {
            Tensor outputTensor = outputTensors[t];
            vector<shared_ptr<Layer>> loadingLayers = apiTensorToApiLoadingLayers[outputTensor];
            for (uint32_t i = 0; i < loadingLayers.size(); ++i) {
                workQueue.push_front(make_pair(outputTensor, loadingLayers[i]));
            }
        }

        orderedNetwork.push_back(make_pair(inputTensor, layer));
    }
}

// TODO: create a slice of a network that uses at most N bytes, given a specified batch size. return both network slices.
uint64_t Network::computeFirstInstanceMemRequirements(uint32_t batchSize, TensorPlacement tensorPlacement) {
    uint64_t bytes = 0;

    for (auto it = network.begin(); it != network.end(); ++it) {
        const shared_ptr<Layer> layer = *it;
        // It is only valid to get first instance bytes on single layers
        bytes += layer->getFirstInstanceMemRequirementInBytes(batchSize, tensorPlacement);
    }
    return bytes;
}

uint64_t Network::computeNonFirstInstanceMemRequirements(uint32_t batchSize, TensorPlacement tensorPlacement) {
    uint64_t bytes = 0;

    for (auto it = network.begin(); it != network.end(); ++it) {
        const shared_ptr<Layer> layer = *it;
        bytes += layer->getNonFirstInstanceMemRequirementInBytes(batchSize, tensorPlacement);
    }
    return bytes;
}

void Network::createBatchDimensions(vector<uint64_t> &batchDimensions, vector<uint64_t> tensorDimensions, uint32_t batchSize) {
    THOR_THROW_IF_FALSE(!tensorDimensions.empty());

    batchDimensions.clear();
    batchDimensions.push_back(batchSize);
    for (uint32_t i = 0; i < tensorDimensions.size(); ++i)
        batchDimensions.push_back(tensorDimensions[i]);
}

// Note that when stamping, a stamped layer does not connect to
// adjacent layers. That is done later.
// A stamped layer may be implemented by serveral actual layers, in that case
// the intermediate layers are connected to form a single logical layer
// that is ready to connect to its inputs and outputs.
void Network::stampNetworkInput(const shared_ptr<Thor::NetworkInput> networkInput,
                                uint32_t gpuNum,
                                uint32_t batchSize,
                                ThorImplementation::StampedNetwork &stampedNetwork) {
    ThorImplementation::TensorPlacement placement(TensorPlacement::MemDevices::GPU, gpuNum);
    shared_ptr<ThorImplementation::Layer> outputLayer;
    Tensor outputTensor = networkInput->getFeatureOutput().value();

    // Stamp network input
    shared_ptr<ThorImplementation::NetworkInput> implementationNetworkInput = networkInput->stamp(placement, batchSize);
    if (DEBUG_STAMP) {
        printf("stamped network input\n");
        fflush(stdout);
    }
    stampedNetwork.inputsShared.push_back(implementationNetworkInput);
    stampedNetwork.inputs.push_back(implementationNetworkInput.get());
    stampedNetwork.inputNamedShared[implementationNetworkInput->getName()] = implementationNetworkInput;
    stampedNetwork.inputNamed[implementationNetworkInput->getName()] = implementationNetworkInput.get();
    outputLayer = implementationNetworkInput;
    stampedNetwork.apiLayerToPhysicalLayerShared[networkInput->getId()] = implementationNetworkInput;
    stampedNetwork.apiLayerToPhysicalLayer[networkInput->getId()] = implementationNetworkInput.get();
    stampedNetwork.physicalLayerToApiLayerShared[implementationNetworkInput] = networkInput->getId();
    stampedNetwork.physicalLayerToApiLayer[implementationNetworkInput.get()] = networkInput->getId();
    // stampedNetwork.recordIfParameterizable(networkInput, implementationNetworkInput);

    // Map the api tensor to its physical driving layer
    stampedNetwork.apiTensorToPhysicalDrivingLayerShared[outputTensor] = outputLayer;
    stampedNetwork.apiTensorToPhysicalDrivingLayer[outputTensor] = outputLayer.get();
}

void Network::addToNetwork(Layer *layer) {
    frozen = false;

    THOR_THROW_IF_FALSE(layer != nullptr);
    addLayerToNetwork(layer);
}

void Network::addLayerToNetwork(const Layer *layer) {
    // Ensure every layer can be added just once
    shared_ptr<Layer> layerClone = layer->clone();
    if (allLayersInNetwork.count(layerClone) == 1)
        return;
    allLayersInNetwork.insert(layerClone);
    allLayersInNetworkList.push_back(layerClone);
    shared_ptr<TrainableLayer> TrainableLayerInstance = dynamic_pointer_cast<TrainableLayer>(layerClone);
    if (TrainableLayerInstance)
        allTrainableLayersInNetwork.push_back(TrainableLayerInstance);
    network.push_back(layerClone);

    auto networkInput = dynamic_cast<const NetworkInput *>(layer);
    auto networkOutput = dynamic_cast<const NetworkOutput *>(layer);
    auto stub = dynamic_cast<const Stub *>(layer);
    auto loss = dynamic_cast<const Loss *>(layer);
    auto metric = dynamic_cast<const Metric *>(layer);
    auto customLayer = dynamic_cast<const CustomLayer *>(layer);
    auto activationLayer = dynamic_cast<const Activation *>(layer);
    auto multiConnectionLayer = dynamic_cast<const MultiConnectionLayer *>(layer);
    if (networkInput) {
        Tensor outputTensor = networkInput->getFeatureOutput().value();
        apiTensorByOriginalId[outputTensor.getOriginalId()] = outputTensor;
        if (networkInput->hasPassThroughSource()) {
            Tensor sourceTensor = networkInput->getPassThroughSource();
            apiTensorByOriginalId[sourceTensor.getOriginalId()] = sourceTensor;
        }
    } else if (networkOutput) {
        Tensor inputTensor = networkOutput->getFeatureInput().value();
        apiTensorByOriginalId[inputTensor.getOriginalId()] = inputTensor;
    } else if (stub) {
        Tensor inputTensor = stub->getFeatureInput().value();
        apiTensorByOriginalId[inputTensor.getOriginalId()] = inputTensor;
    } else if (loss) {
        vector<Tensor> lossInputTensors = loss->getLossInputTensors();
        Tensor lossTensor = loss->getLoss();
        for (const Tensor& inputTensor : lossInputTensors) {
            apiTensorByOriginalId[inputTensor.getOriginalId()] = inputTensor;
        }
        apiTensorByOriginalId[lossTensor.getOriginalId()] = lossTensor;
    } else if (metric) {
        Tensor inputTensor = metric->getFeatureInput().value();
        Tensor outputTensor = metric->getFeatureOutput().value();
        apiTensorByOriginalId[inputTensor.getOriginalId()] = inputTensor;
        if (metric->requiresLabels()) {
            Tensor labelsTensor = metric->getLabels();
            apiTensorByOriginalId[labelsTensor.getOriginalId()] = labelsTensor;
        }
        apiTensorByOriginalId[outputTensor.getOriginalId()] = outputTensor;
    } else if (customLayer) {
        vector<Tensor> inputTensors = customLayer->getFeatureInputs();
        vector<Tensor> outputTensors = customLayer->getFeatureOutputs();
        THOR_THROW_IF_FALSE(!inputTensors.empty());
        THOR_THROW_IF_FALSE(!outputTensors.empty());
        for (uint32_t i = 0; i < inputTensors.size(); ++i) {
            apiTensorByOriginalId[inputTensors[i].getOriginalId()] = inputTensors[i];
        }
        for (uint32_t i = 0; i < outputTensors.size(); ++i) {
            apiTensorByOriginalId[outputTensors[i].getOriginalId()] = outputTensors[i];
        }
    } else if (activationLayer && activationLayer->mustConnectAllInputsToDriveOutput()) {
        vector<Tensor> inputTensors = activationLayer->getFeatureInputs();
        vector<Tensor> outputTensors = activationLayer->getFeatureOutputs();
        THOR_THROW_IF_FALSE(!inputTensors.empty());
        THOR_THROW_IF_FALSE(!outputTensors.empty());
        for (uint32_t i = 0; i < inputTensors.size(); ++i) {
            apiTensorByOriginalId[inputTensors[i].getOriginalId()] = inputTensors[i];
        }
        for (uint32_t i = 0; i < outputTensors.size(); ++i) {
            apiTensorByOriginalId[outputTensors[i].getOriginalId()] = outputTensors[i];
        }
    } else if (multiConnectionLayer) {
        vector<Tensor> inputTensors = multiConnectionLayer->getFeatureInputs();
        vector<Tensor> outputTensors = multiConnectionLayer->getFeatureOutputs();
        THOR_THROW_IF_FALSE(!inputTensors.empty());
        THOR_THROW_IF_FALSE(!outputTensors.empty());
        for (uint32_t i = 0; i < inputTensors.size(); ++i) {
            apiTensorByOriginalId[inputTensors[i].getOriginalId()] = inputTensors[i];
        }
        for (uint32_t i = 0; i < outputTensors.size(); ++i) {
            apiTensorByOriginalId[outputTensors[i].getOriginalId()] = outputTensors[i];
        }
    } else {
        // base Layer type
        Tensor inputTensor = layer->getFeatureInput().value();
        Tensor outputTensor = layer->getFeatureOutput().value();
        apiTensorByOriginalId[inputTensor.getOriginalId()] = inputTensor;
        apiTensorByOriginalId[outputTensor.getOriginalId()] = outputTensor;
    }
}

// An optimizer is used to optimize all weights and biases in a network
void Network::addToNetwork(Optimizer *optimizer) {
    // If the default optimizer is specified more than once, the user has an error in their code, call it out.
    if (this->defaultOptimizer != nullptr) {
        string errorMessage = "Error: Multiple default optimizers specified on network " + this->getNetworkName() +
                              ". You may specify at most one default optimizer. Had " + this->defaultOptimizer->getType() +
                              " and tried to add " + optimizer->getType();
        throw(runtime_error(errorMessage.c_str()));
    }

    this->defaultOptimizer = optimizer->clone();
}

shared_ptr<Optimizer> Network::getDefaultOptimizer() { return defaultOptimizer; }

void Network::setDefaultOptimizer(std::shared_ptr<Optimizer> optimizer) {
    THOR_THROW_IF_FALSE(optimizer != nullptr);
    addToNetwork(optimizer.get());
}

bool Network::allTrainingEnabledParametersHaveOptimizers() const {
    for (const auto& trainableLayer : allTrainableLayersInNetwork) {
        if (trainableLayer != nullptr && !trainableLayer->hasOptimizer()) {
            return false;
        }
    }
    return true;
}

std::vector<Tensor> Network::getLossRootTensors() const {
    std::vector<Tensor> result;
    for (const std::shared_ptr<Layer>& layer : allLayersInNetwork) {
        std::shared_ptr<Loss> loss = std::dynamic_pointer_cast<Loss>(layer);
        if (loss != nullptr) {
            result.push_back(loss->getLoss());
        }
    }
    return result;
}

std::vector<Tensor> Network::getRawLossTensorsForTrainingRoots(const std::vector<Tensor>& lossRoots) const {
    bool needsGraphEvaluation = apiTensorToApiDrivingLayer.empty() || apiLayerToApiInputTensors.empty();
    if (!needsGraphEvaluation) {
        for (const Tensor& lossRoot : lossRoots) {
            if (!lossRoot.isInitialized()) {
                continue;
            }
            bool foundRootOriginalId = false;
            for (const auto& [apiTensor, drivingLayer] : apiTensorToApiDrivingLayer) {
                (void)drivingLayer;
                if (apiTensor.getOriginalId() == lossRoot.getOriginalId()) {
                    foundRootOriginalId = true;
                    break;
                }
            }
            if (!foundRootOriginalId) {
                needsGraphEvaluation = true;
                break;
            }
        }
    }
    if (needsGraphEvaluation) {
        // Loss root resolution may be needed before a full placement pass.  evaluateGraph() populates the
        // tensor/layer adjacency maps before running validity checks.  We intentionally validate the requested
        // active roots below instead of accepting any invalid graph state silently.
        (void)const_cast<Network*>(this)->evaluateGraph(false);
    }

    auto findDrivenTensorByOriginalId = [&](uint64_t originalId) -> std::optional<std::pair<Tensor, std::shared_ptr<Layer>>> {
        for (const auto& [apiTensor, drivingLayer] : apiTensorToApiDrivingLayer) {
            if (apiTensor.getOriginalId() == originalId) {
                return std::make_pair(apiTensor, drivingLayer);
            }
        }
        return std::nullopt;
    };

    std::vector<Tensor> rawLossRoots;
    std::set<uint64_t> emittedLossOriginalIds;

    std::function<void(const Tensor&, std::set<uint64_t>&, std::set<uint64_t>&)> visitTensor =
        [&](const Tensor& candidateRoot, std::set<uint64_t>& visitedTensorOriginalIds, std::set<uint64_t>& rootResolvedLossOriginalIds) {
            if (!candidateRoot.isInitialized()) {
                throw std::runtime_error("Cannot resolve uninitialized active loss root tensor.");
            }

            const uint64_t candidateOriginalId = candidateRoot.getOriginalId();
            if (!visitedTensorOriginalIds.insert(candidateOriginalId).second) {
                return;
            }

            std::optional<std::pair<Tensor, std::shared_ptr<Layer>>> drivenTensorAndLayer =
                findDrivenTensorByOriginalId(candidateOriginalId);
            if (!drivenTensorAndLayer.has_value()) {
                throw std::runtime_error("Active loss root tensor with original id " + std::to_string(candidateOriginalId) +
                                         " does not have a driving layer in network '" + networkName + "'.");
            }

            const Tensor& tensor = drivenTensorAndLayer->first;
            const std::shared_ptr<Layer>& drivingLayer = drivenTensorAndLayer->second;
            std::shared_ptr<Loss> lossLayer = std::dynamic_pointer_cast<Loss>(drivingLayer);
            if (lossLayer != nullptr) {
                rootResolvedLossOriginalIds.insert(tensor.getOriginalId());
                if (emittedLossOriginalIds.insert(tensor.getOriginalId()).second) {
                    // Return the canonical tensor object used as the key in the evaluated/stamped graph.  Do not
                    // round-trip through apiTensorByOriginalId here: that map is keyed by original id and may contain
                    // an equivalent tensor object from the pre-clone builder side, while downstream stamped-network
                    // lookup is by Tensor identity.
                    rawLossRoots.push_back(tensor);
                }
                return;
            }

            auto inputsIt = apiLayerToApiInputTensors.find(drivingLayer);
            if (inputsIt == apiLayerToApiInputTensors.end()) {
                return;
            }
            for (const Tensor& inputTensor : inputsIt->second) {
                visitTensor(inputTensor, visitedTensorOriginalIds, rootResolvedLossOriginalIds);
            }
        };

    for (const Tensor& lossRoot : lossRoots) {
        std::set<uint64_t> visitedTensorOriginalIds;
        std::set<uint64_t> rootResolvedLossOriginalIds;
        visitTensor(lossRoot, visitedTensorOriginalIds, rootResolvedLossOriginalIds);
        if (rootResolvedLossOriginalIds.empty()) {
            throw std::runtime_error("Active loss root tensor with original id " + std::to_string(lossRoot.getOriginalId()) +
                                     " does not resolve to any physical loss layer in network '" + networkName + "'.");
        }
    }

    return rawLossRoots;
}

std::vector<ParameterReference> Network::getTrainableParameterReferences(bool trainingEnabledOnly) const {
    std::vector<ParameterReference> result;

    for (const std::shared_ptr<TrainableLayer>& trainableLayer : allTrainableLayersInNetwork) {
        if (trainableLayer == nullptr) {
            continue;
        }
        std::vector<ParameterReference> layerParameters = trainableLayer->getParameterReferences(/*trainableOnly=*/true, trainingEnabledOnly);
        result.insert(result.end(), layerParameters.begin(), layerParameters.end());
    }

    return result;
}


BoundParameter Network::resolveParameterReference(PlacedNetwork* placedNetwork, const ParameterReference& parameterReference) const {
    if (placedNetwork == nullptr) {
        throw runtime_error("Cannot resolve a ParameterReference without a placed network.");
    }
    if (!parameterReference.isInitialized()) {
        throw runtime_error("Cannot resolve an uninitialized ParameterReference.");
    }

    const uint64_t parameterizableId = parameterReference.getParameterizableId();
    const string& parameterName = parameterReference.getParameterName();
    for (const shared_ptr<TrainableLayer>& trainableLayer : allTrainableLayersInNetwork) {
        if (trainableLayer == nullptr || trainableLayer->getParameterizableId() != parameterizableId) {
            continue;
        }
        if (!trainableLayer->hasParameter(parameterName)) {
            throw runtime_error("ParameterReference points to layer id " + to_string(parameterizableId) +
                                " but parameter '" + parameterName + "' is not present on that layer.");
        }
        if (!trainableLayer->getParameterSpecification(parameterName)->isTrainable()) {
            throw runtime_error("ParameterReference points to layer id " + to_string(parameterizableId) +
                                " parameter '" + parameterName + "', but that parameter is not trainable.");
        }
        return trainableLayer->getBoundParameter(placedNetwork, parameterName);
    }

    throw runtime_error("ParameterReference points to unknown trainable layer id " + to_string(parameterizableId) +
                        " parameter '" + parameterName + "'.");
}

std::vector<BoundParameter> Network::resolveParameterReferences(PlacedNetwork* placedNetwork,
                                                                const std::vector<ParameterReference>& parameterReferences) const {
    std::vector<BoundParameter> result;
    result.reserve(parameterReferences.size());
    for (const ParameterReference& parameterReference : parameterReferences) {
        result.push_back(resolveParameterReference(placedNetwork, parameterReference));
    }
    return result;
}

void Network::freezeTraining() {
    for (auto& trainableLayer : allTrainableLayersInNetwork) {
        if (trainableLayer != nullptr) {
            trainableLayer->freezeTraining();
        }
    }
}

void Network::unfreezeTraining() {
    for (auto& trainableLayer : allTrainableLayersInNetwork) {
        if (trainableLayer != nullptr) {
            trainableLayer->unfreezeTraining();
        }
    }
}

// For future multi-gpu support, optimizers for the same layer on different GPU's will need to accumulate into a single weights memory
// and then broadcast the updated weights to the optimizers on the other gpus.
// FIXME: What is the right place for network/optimizer interaction code? Is it here? Is it in Optimizer?
//  I also have Optimizer::updateHyperParameters so currently there is code like that in both places.
//  This pattern needs to be revisited.
void Network::attachOptimizerToLayers(bool replaceIfExisting) {
    // Once the optimizer is distributed to the layers, the layers own the optimizer instances.
    // If additional layers are added later on, attachOptimizerToLayers(false) will need to be called before training more.
    THOR_THROW_IF_FALSE(defaultOptimizer != nullptr);

    for (shared_ptr<TrainableLayer> &trainableLayer : allTrainableLayersInNetwork) {
        if (replaceIfExisting or !trainableLayer->hasOptimizer())
            trainableLayer->attachDefaultOptimizer(defaultOptimizer);
    }
}

void Network::stampLayer(Tensor inputTensor,
                         const shared_ptr<Thor::Layer> layer,
                         uint32_t gpuNum,
                         uint32_t batchSize,
                         ThorImplementation::StampedNetwork &stampedNetwork,
                         const bool inferenceOnly) {
    ThorImplementation::TensorPlacement placement(TensorPlacement::MemDevices::GPU, gpuNum);
    shared_ptr<ThorImplementation::Layer> physicalDrivingLayer = stampedNetwork.apiTensorToPhysicalDrivingLayerShared[inputTensor];
    shared_ptr<Thor::Layer> apiDrivingLayer =
        apiTensorToApiDrivingLayer.count(inputTensor) == 0 ? nullptr : apiTensorToApiDrivingLayer[inputTensor];

    // If the api tensor has multiple loads and the physical driving layer is not a fanout,
    // then replace the physical driving layer with a newly stamped fanout
    uint32_t numLoadingLayers = apiTensorToApiLoadingLayers[inputTensor].size();
    THOR_THROW_IF_FALSE(numLoadingLayers > 0);
    shared_ptr<ThorImplementation::TensorFanout> implementationTensorFanout =
        dynamic_pointer_cast<ThorImplementation::TensorFanout>(physicalDrivingLayer);
    if (apiTensorToApiLoadingLayers[inputTensor].size() != 1) {
        if (implementationTensorFanout == nullptr) {
            implementationTensorFanout = make_shared<ThorImplementation::TensorFanout>();
            Layer::connectTwoLayers(physicalDrivingLayer, implementationTensorFanout, apiDrivingLayer, nullptr, inputTensor);
            physicalDrivingLayer = implementationTensorFanout;

            stampedNetwork.otherLayersShared.push_back(implementationTensorFanout);
            stampedNetwork.otherLayers.push_back(implementationTensorFanout.get());
            apiDrivingLayer = nullptr;
            stampedNetwork.apiTensorToPhysicalDrivingLayerShared[inputTensor] = physicalDrivingLayer;
            stampedNetwork.apiTensorToPhysicalDrivingLayer[inputTensor] = physicalDrivingLayer.get();

            if (DEBUG_STAMP) {
                printf("stamped tensor fanout - id %ld - driving %s - num output connections afterward %ld\n",
                       physicalDrivingLayer->getId(),
                       layer->getLayerType().c_str(),
                       implementationTensorFanout->getStreams().size());
                fflush(stdout);
            }
        } else {
            if (DEBUG_STAMP) {
                printf("connecting existing tensor fanout - id %ld - driving %s - num output connections before hand %ld\n",
                       physicalDrivingLayer->getId(),
                       layer->getLayerType().c_str(),
                       implementationTensorFanout->getStreams().size());
                fflush(stdout);
            }
        }
    }

    // Stamp the layer
    // Unless it was previously stamped on a prior pass, if so just connect the tensor.
    shared_ptr<ThorImplementation::Layer> implementationLayer = nullptr;
    // vector<Event> initializationReadyEvents;
    bool layerPreviouslyStamped = (stampedNetwork.apiLayerToPhysicalLayer.count(layer->getId()) == 1);
    // In case of a tensor fanout, there is no apiLayer...
    if (layerPreviouslyStamped) {
        implementationLayer = stampedNetwork.apiLayerToPhysicalLayerShared[layer->getId()];

        if (DEBUG_STAMP) {
            printf("connecting to %s\n", layer->getLayerType().c_str());
            fflush(stdout);
        }
    } else {
        implementationLayer = layer->stamp(placement, physicalDrivingLayer, apiDrivingLayer, inputTensor, inferenceOnly);
        shared_ptr<ThorImplementation::TrainableLayer> implementationTrainableLayer =
            dynamic_pointer_cast<ThorImplementation::TrainableLayer>(implementationLayer);
        if (implementationTrainableLayer != nullptr) {
            implementationTrainableLayer->setGradientUpdateStreamPool(stampedNetwork.gradientUpdateStreamPool);
        }
        stampedNetwork.apiLayerToPhysicalLayerShared[layer->getId()] = implementationLayer;
        stampedNetwork.apiLayerToPhysicalLayer[layer->getId()] = implementationLayer.get();
        stampedNetwork.physicalLayerToApiLayerShared[implementationLayer] = layer->getId();
        stampedNetwork.physicalLayerToApiLayer[implementationLayer.get()] = layer->getId();

        // stampedNetwork.recordIfParameterizable(layer, implementationLayer);

        if (DEBUG_STAMP) {
            printf("stamped %s (physical layer id = %ld, api layer id = %ld) driven by physical layer id = %ld\n",
                   layer->getLayerType().c_str(),
                   implementationLayer->getId(),
                   layer->getId(),
                   physicalDrivingLayer->getId());
            fflush(stdout);
        }
    }
    Layer::connectTwoLayers(physicalDrivingLayer, implementationLayer, apiDrivingLayer, layer, inputTensor);

    vector<Tensor> apiOutputTensors = layer->getAllOutputTensors();
    for (uint32_t i = 0; i < apiOutputTensors.size(); ++i) {
        stampedNetwork.apiTensorToPhysicalDrivingLayerShared[apiOutputTensors[i]] = implementationLayer;
        stampedNetwork.apiTensorToPhysicalDrivingLayer[apiOutputTensors[i]] = implementationLayer.get();
    }

    if (!layerPreviouslyStamped) {
        shared_ptr<ThorImplementation::TrainableLayer> implementationTrainableLayer =
            dynamic_pointer_cast<ThorImplementation::TrainableLayer>(implementationLayer);
        if (implementationTrainableLayer != nullptr) {
            stampedNetwork.trainableLayersShared.push_back(implementationTrainableLayer);
            stampedNetwork.trainableLayers.push_back(implementationTrainableLayer.get());
        } else {
            stampedNetwork.otherLayersShared.push_back(implementationLayer);
            stampedNetwork.otherLayers.push_back(implementationLayer.get());
        }
    }


    // eturn initializationReadyEvents;
}

void Network::stampNetworkOutput(Tensor inputTensor,
                                 const shared_ptr<Thor::NetworkOutput> networkOutput,
                                 uint32_t gpuNum,
                                 uint32_t batchSize,
                                 ThorImplementation::StampedNetwork &stampedNetwork,
                                 const bool inferenceOnly,
                                 bool networkOutputsOnGpu) {
    ThorImplementation::TensorPlacement placement(TensorPlacement::MemDevices::GPU, gpuNum);
    shared_ptr<ThorImplementation::Layer> physicalDrivingLayer = stampedNetwork.apiTensorToPhysicalDrivingLayerShared[inputTensor];
    shared_ptr<Thor::Layer> apiDrivingLayer =
        apiTensorToApiDrivingLayer.count(inputTensor) == 0 ? nullptr : apiTensorToApiDrivingLayer[inputTensor];

    // If the api tensor has multiple loads and the physical driving layer is not a fanout,
    // then replace the physical driving layer with a newly stamped fanout
    uint32_t numLoadingLayers = apiTensorToApiLoadingLayers[inputTensor].size();
    shared_ptr<ThorImplementation::TensorFanout> implementationTensorFanout =
        dynamic_pointer_cast<ThorImplementation::TensorFanout>(physicalDrivingLayer);
    THOR_THROW_IF_FALSE(numLoadingLayers > 0);
    if (apiTensorToApiLoadingLayers[inputTensor].size() != 1 && implementationTensorFanout == nullptr) {
        implementationTensorFanout = make_shared<ThorImplementation::TensorFanout>();
        Layer::connectTwoLayers(physicalDrivingLayer, implementationTensorFanout, apiDrivingLayer, nullptr, inputTensor);
        physicalDrivingLayer = implementationTensorFanout;

        stampedNetwork.otherLayersShared.push_back(implementationTensorFanout);
        stampedNetwork.otherLayers.push_back(implementationTensorFanout.get());
        apiDrivingLayer = nullptr;
        stampedNetwork.apiTensorToPhysicalDrivingLayerShared[inputTensor] = physicalDrivingLayer;
        stampedNetwork.apiTensorToPhysicalDrivingLayer[inputTensor] = physicalDrivingLayer.get();
        if (DEBUG_STAMP) {
            printf("stamped tensor fanout - network output\n");
            fflush(stdout);
        }
    }

    // Stamp ordinary inference/training outputs to CPU so PlacedNetwork.infer preserves its public API.
    // Ensemble inference can request GPU output stamping so member predictions can be aggregated on device
    // before a single final D2H copy is materialized.
    TensorPlacement outputPlacement = networkOutputsOnGpu
        ? TensorPlacement(TensorPlacement::MemDevices::GPU, gpuNum)
        : TensorPlacement(TensorPlacement::MemDevices::CPU);
    shared_ptr<ThorImplementation::Layer> implementationLayer =
        ((Layer *)networkOutput.get())->stamp(outputPlacement, physicalDrivingLayer, apiDrivingLayer, inputTensor, inferenceOnly);
    shared_ptr<ThorImplementation::NetworkOutput> implementationNetworkOutput =
        dynamic_pointer_cast<ThorImplementation::NetworkOutput>(implementationLayer);
    Layer::connectTwoLayers(physicalDrivingLayer, implementationNetworkOutput, apiDrivingLayer, networkOutput, inputTensor);
    THOR_THROW_IF_FALSE(implementationNetworkOutput != nullptr);
    stampedNetwork.outputsShared.push_back(implementationNetworkOutput);
    stampedNetwork.outputs.push_back(implementationNetworkOutput.get());
    stampedNetwork.outputNamedShared[implementationNetworkOutput->getName()] = implementationNetworkOutput;
    stampedNetwork.outputNamed[implementationNetworkOutput->getName()] = implementationNetworkOutput.get();
    if (DEBUG_STAMP) {
        printf("stamped network output\n");
        fflush(stdout);
    }

    stampedNetwork.apiLayerToPhysicalLayerShared[networkOutput->getId()] = implementationNetworkOutput;
    stampedNetwork.apiLayerToPhysicalLayer[networkOutput->getId()] = implementationNetworkOutput.get();
    stampedNetwork.physicalLayerToApiLayerShared[implementationNetworkOutput] = networkOutput->getId();
    stampedNetwork.physicalLayerToApiLayer[implementationNetworkOutput.get()] = networkOutput->getId();

    // stampedNetwork.recordIfParameterizable(networkOutput, implementationNetworkOutput);
}

bool Network::hasApiTensorByOriginalId(uint64_t originalId) const {
    return apiTensorByOriginalId.count(originalId) != 0;
}

Tensor Network::resolveApiTensorByOriginalId(uint64_t originalId) const {
    auto it = apiTensorByOriginalId.find(originalId);
    if (it == apiTensorByOriginalId.end()) {
        throw runtime_error("Tensor with original id " + to_string(originalId) + " does not belong to network '" + networkName + "'.");
    }
    return it->second;
}

Tensor Network::getApiTensorByOriginalId(uint64_t originalId) {
    auto existingIt = apiTensorByOriginalId.find(originalId);
    if (existingIt != apiTensorByOriginalId.end()) {
        return existingIt->second;
    }

    auto rememberIfMatches = [&](const Tensor& tensor) -> std::optional<Tensor> {
        if (tensor.isInitialized() && tensor.getOriginalId() == originalId) {
            apiTensorByOriginalId[originalId] = tensor;
            return tensor;
        }
        return std::nullopt;
    };

    auto rememberOptionalIfMatches = [&](const std::optional<Tensor>& tensor) -> std::optional<Tensor> {
        if (!tensor.has_value()) {
            return std::nullopt;
        }
        return rememberIfMatches(*tensor);
    };

    for (const std::shared_ptr<Layer>& layer : allLayersInNetworkList) {
        if (layer == nullptr) {
            continue;
        }

        if (std::optional<Tensor> found = rememberOptionalIfMatches(layer->getFeatureInput())) {
            return *found;
        }
        if (std::optional<Tensor> found = rememberOptionalIfMatches(layer->getFeatureOutput())) {
            return *found;
        }

        std::shared_ptr<NetworkInput> networkInput = std::dynamic_pointer_cast<NetworkInput>(layer);
        if (networkInput != nullptr && networkInput->hasPassThroughSource()) {
            if (std::optional<Tensor> found = rememberIfMatches(networkInput->getPassThroughSource())) {
                return *found;
            }
        }

        std::shared_ptr<Loss> loss = std::dynamic_pointer_cast<Loss>(layer);
        if (loss != nullptr) {
            for (const Tensor& tensor : loss->getLossInputTensors()) {
                if (std::optional<Tensor> found = rememberIfMatches(tensor)) {
                    return *found;
                }
            }
            if (std::optional<Tensor> found = rememberIfMatches(loss->getLoss())) {
                return *found;
            }
            continue;
        }

        std::shared_ptr<Metric> metric = std::dynamic_pointer_cast<Metric>(layer);
        if (metric != nullptr && metric->requiresLabels()) {
            if (std::optional<Tensor> found = rememberIfMatches(metric->getLabels())) {
                return *found;
            }
        }

        std::shared_ptr<CustomLayer> customLayer = std::dynamic_pointer_cast<CustomLayer>(layer);
        if (customLayer != nullptr) {
            for (const Tensor& tensor : customLayer->getFeatureInputs()) {
                if (std::optional<Tensor> found = rememberIfMatches(tensor)) {
                    return *found;
                }
            }
            for (const Tensor& tensor : customLayer->getFeatureOutputs()) {
                if (std::optional<Tensor> found = rememberIfMatches(tensor)) {
                    return *found;
                }
            }
            continue;
        }

        std::shared_ptr<Activation> activationLayer = std::dynamic_pointer_cast<Activation>(layer);
        if (activationLayer != nullptr && activationLayer->mustConnectAllInputsToDriveOutput()) {
            for (const Tensor& tensor : activationLayer->getFeatureInputs()) {
                if (std::optional<Tensor> found = rememberIfMatches(tensor)) {
                    return *found;
                }
            }
            for (const Tensor& tensor : activationLayer->getFeatureOutputs()) {
                if (std::optional<Tensor> found = rememberIfMatches(tensor)) {
                    return *found;
                }
            }
            continue;
        }

        std::shared_ptr<MultiConnectionLayer> multiConnectionLayer = std::dynamic_pointer_cast<MultiConnectionLayer>(layer);
        if (multiConnectionLayer != nullptr) {
            for (const Tensor& tensor : multiConnectionLayer->getFeatureInputs()) {
                if (std::optional<Tensor> found = rememberIfMatches(tensor)) {
                    return *found;
                }
            }
            for (const Tensor& tensor : multiConnectionLayer->getFeatureOutputs()) {
                if (std::optional<Tensor> found = rememberIfMatches(tensor)) {
                    return *found;
                }
            }
            continue;
        }
    }

    std::string message = "Tensor with original id " + std::to_string(originalId) + " does not belong to network '" + networkName + "'. Known tensor original ids:";
    for (const auto& [knownOriginalId, _] : apiTensorByOriginalId) {
        message += " " + std::to_string(knownOriginalId);
    }
    throw std::runtime_error(message);
}

}  // namespace Thor
