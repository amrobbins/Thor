#include "DeepLearning/Api/Network/Network.h"
#include "Utilities/Expression/CudaKernelSecurity.h"
#include <optional>
#include <algorithm>
#include <cctype>
#include <set>
#include <string_view>
#include "DeepLearning/Api/Layers/Learning/CustomLayer.h"
#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Parameter/ParameterSpecification.h"
#include "DeepLearning/Implementation/ThorError.h"
#include <cstdio>
#include <fstream>
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


bool hasSuffix(std::string_view value, std::string_view suffix) {
    return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool isSixDigits(std::string_view value) {
    if (value.size() != 6) {
        return false;
    }
    return std::all_of(value.begin(), value.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
}

std::vector<std::string> archiveNamesFromShardFilename(const std::string& filename) {
    constexpr std::string_view kSuffix = ".thor.tar";
    if (!hasSuffix(filename, kSuffix)) {
        return {};
    }

    std::vector<std::string> archiveNames;
    std::string archiveName = filename.substr(0, filename.size() - kSuffix.size());
    if (!archiveName.empty()) {
        archiveNames.push_back(archiveName);
    }

    const size_t dot = archiveName.rfind('.');
    if (dot != std::string::npos && isSixDigits(std::string_view(archiveName).substr(dot + 1))) {
        archiveName.erase(dot);
        if (!archiveName.empty()) {
            archiveNames.push_back(archiveName);
        }
    }
    return archiveNames;
}

bool hasArchiveShard0(const std::filesystem::path& directory, const std::string& archiveName) {
    return std::filesystem::exists(directory / (archiveName + ".thor.tar")) ||
           std::filesystem::exists(directory / (archiveName + ".000000.thor.tar"));
}

bool isThorModelJsonName(const std::string& pathInArchive) {
    constexpr std::string_view kSuffix = ".thor.json";
    return hasSuffix(pathInArchive, kSuffix) && pathInArchive.find('/') == std::string::npos;
}

struct NetworkArchiveSelection {
    std::string archiveName;
    std::string modelJsonFileName;
};

NetworkArchiveSelection selectNetworkArchiveForLoad(const std::filesystem::path& directory, const std::string& preferredArchiveName) {
    if (hasArchiveShard0(directory, preferredArchiveName)) {
        auto reader = std::make_shared<thor_file::TarReader>(preferredArchiveName, directory);
        const std::string preferredModelJsonFileName = preferredArchiveName + ".thor.json";
        if (reader->containsFile(preferredModelJsonFileName)) {
            return {preferredArchiveName, preferredModelJsonFileName};
        }

        std::vector<std::string> modelJsonFiles;
        for (const auto& [pathInArchive, _] : reader->getArchiveEntries()) {
            if (isThorModelJsonName(pathInArchive)) {
                modelJsonFiles.push_back(pathInArchive);
            }
        }
        if (modelJsonFiles.size() == 1) {
            return {preferredArchiveName, modelJsonFiles.front()};
        }
        throw std::runtime_error("Network::load: archive '" + preferredArchiveName + "' in " + directory.string() +
                                 " does not contain a unique top-level .thor.json model file");
    }

    std::set<std::string> candidateArchiveNames;
    if (std::filesystem::exists(directory)) {
        for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(directory)) {
            if (!entry.is_regular_file() && !entry.is_symlink()) {
                continue;
            }
            for (const std::string& archiveName : archiveNamesFromShardFilename(entry.path().filename().string())) {
                if (hasArchiveShard0(directory, archiveName)) {
                    candidateArchiveNames.insert(archiveName);
                }
            }
        }
    }

    std::vector<NetworkArchiveSelection> matches;
    std::vector<std::string> scanErrors;
    for (const std::string& candidateArchiveName : candidateArchiveNames) {
        try {
            auto reader = std::make_shared<thor_file::TarReader>(candidateArchiveName, directory);
            std::vector<std::string> modelJsonFiles;
            for (const auto& [pathInArchive, _] : reader->getArchiveEntries()) {
                if (isThorModelJsonName(pathInArchive)) {
                    modelJsonFiles.push_back(pathInArchive);
                }
            }
            if (modelJsonFiles.size() == 1) {
                matches.push_back({candidateArchiveName, modelJsonFiles.front()});
            }
        } catch (const std::exception& e) {
            scanErrors.push_back(candidateArchiveName + ": " + e.what());
        }
    }

    if (matches.size() == 1) {
        return matches.front();
    }
    if (matches.empty()) {
        std::string message = "Network::load: missing archive for network '" + preferredArchiveName + "' in " + directory.string() +
                              " and no alternate archive with a unique top-level .thor.json model file was found";
        if (!scanErrors.empty()) {
            message += "; scan errors:";
            for (const std::string& scanError : scanErrors) {
                message += " [" + scanError + "]";
            }
        }
        throw std::runtime_error(message);
    }

    std::string message = "Network::load: archive name is ambiguous for network '" + preferredArchiveName + "' in " +
                          directory.string() + "; candidates:";
    for (const NetworkArchiveSelection& match : matches) {
        message += " " + match.archiveName;
    }
    throw std::runtime_error(message);
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
Network::StatusCode Network::createDagAndFreeze() {
    if (!frozen) {
        StatusCode status = evaluateGraph();
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
                                          const bool inferenceOnly) {
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

    // Leave 100MB of headroom
    // FIXME: need to determine if this is the not the first instance and use shared weights and shared weights mem requirements
    if (MachineEvaluator::instance().getFreeMemBytes(gpuNum) < firstInstanceBytes + 100000000)
        return StatusCode::GPU_OUT_OF_MEMORY;

    // 1. Stamp (i.e. construct) all layers
    // 2. At the moment, I connect the layers upon stamping them, I think I should change that and stamp everything first,
    //    then for the next phase, connect them
    stampedNetwork.clear();
    try {
        // FIXME: need to throw GPU_OUT_OF_MEMORY when stamping and run out of memory

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
                stampNetworkOutput(inputTensor.value(), networkOutput, gpuNum, batchSize, stampedNetwork, inferenceOnly);
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
                initDoneEvents.insert(initDoneEvents.end(), make_move_iterator(layerEvents.begin()), make_move_iterator(layerEvents.end()));
            } else {
                vector<Event> layerEvents = layer->initialize(implementationLayer);
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

    } catch (GpuOutOfMemoryError ex) {
        stampedNetwork.clear();
        return StatusCode::GPU_OUT_OF_MEMORY;
    }

    stampedNetworks.push_back(stampedNetwork);

    if (archiveReader != nullptr)
        archiveReader->executeReadRequests();

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

    StatusCode dagStatus = createDagAndFreeze();
    if (dagStatus != StatusCode::SUCCESS) {
        printf("ERROR: evaluateGraph() returned %s\n", statusCodeToString(dagStatus).c_str());
        fflush(stdout);
    }

    return dagStatus;
}

shared_ptr<PlacedNetwork> Network::place(
    uint32_t batchSize, vector<Event> &initDoneEvents, bool inferenceOnly, vector<int32_t> forcedDevices, uint32_t forcedNumStampsPerGpu) {
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

    // FIXME: pull preOptimize into initialize
    for (uint32_t i = 0; i < devices.size(); ++i) {
        preOptimize(devices[i], batchSize);
    }
    for (uint32_t i = 0; i < devices.size(); ++i) {
        for (uint32_t j = 0; j < numStampsPerDevice[i]; ++j) {
            // FIXME: need to propagate inferenceOnly from here through to the API layer to the implementation layer
            StatusCode statusCode = stampNetwork(devices[i], initDoneEvents, batchSize, stampedNetworks, inferenceOnly);
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

    // For the initial implementation, I will just force GPU 0.
    // I will optimize from there, but I need changes elsewhere first anyway.
    Stream stream = Stream::getNextDownloadStream(0);
    json modelJson;
    modelJson["layers"] = json::array();
    uint32_t stampIndex = 0;
    for (const shared_ptr<Layer> &layer : allLayersInNetworkList) {
        modelJson["layers"].push_back(layer->serialize(archiveWriter, stream, saveOptimizerState, stampedNetworks[stampIndex]));
        stampIndex += 1;
        if (stampIndex >= stampedNetworks.size())
            stampIndex = 0;
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

    // First I must synchronize with all devices to make sure the final batch is completely finished updating the weights.
    uint32_t numGpus = MachineEvaluator::instance().getNumGpus();
    for (uint32_t gpu = 0; gpu < numGpus; ++gpu)
        Stream::deviceSynchronize(gpu);

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
    // Read the model json from the archive. Prefer the current network name for
    // backward-compatible behavior, but allow a load target with a different
    // in-memory name when the save directory contains exactly one Thor model.
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

    for (const json &layerJson : layers) {
        // printf("%s\n", layerJson.dump(4).c_str());
        Layer::deserialize(archiveReader, layerJson, this);
    }

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

// Determine the graph structure
// Tensors are the edges that connect the Layers which are nodes.
Network::StatusCode Network::evaluateGraph() {
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
 * A tensor has a dangling output when nothing is connected to read from it -> No Loader.
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

    orderedNetwork.clear();

    // Put all network inputs into the work queue
    for (auto it = network.begin(); it != network.end(); ++it) {
        shared_ptr<Layer> layer = *it;

        const shared_ptr<NetworkInput> networkInput = dynamic_pointer_cast<NetworkInput>(layer);
        if (networkInput) {
            Tensor outputTensor = layer->getFeatureOutput().value();
            vector<shared_ptr<Layer>> loadingLayers = apiTensorToApiLoadingLayers[outputTensor];
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
                                 const bool inferenceOnly) {
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

    // Stamp the network output
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    shared_ptr<ThorImplementation::Layer> implementationLayer =
        ((Layer *)networkOutput.get())->stamp(cpuPlacement, physicalDrivingLayer, apiDrivingLayer, inputTensor, inferenceOnly);
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
    if (apiTensorByOriginalId.count(originalId) == 0) {
        printf("Looking for tensor orig id %ld.\n I have these:\n\n", originalId);
        for (auto it = apiTensorByOriginalId.begin(); it != apiTensorByOriginalId.end(); ++it) {
            printf("tensor orig id %ld\n", it->first);
            fflush(stdout);
        }
    }
    THOR_THROW_IF_FALSE(apiTensorByOriginalId.count(originalId) != 0);
    return apiTensorByOriginalId[originalId];
}

}  // namespace Thor
