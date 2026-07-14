#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Diagnostics/TrainingDiagnostics.h"

#include "DeepLearning/Implementation/Layers/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Parameter/Parameterizable.h"
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"
#include "Utilities/Common/Event.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TarFile/TarReader.h"

#include <utility>
#include <exception>
#include <stdexcept>
#include <set>
#include <iterator>
#include <filesystem>
#include <optional>
#include <system_error>
#if THOR_ENABLE_BATCH_SUBMISSION_TIMING
#include <chrono>
#endif
#include <unordered_map>

using namespace std;
using json = nlohmann::json;

namespace Thor {

namespace {

#if THOR_ENABLE_BATCH_SUBMISSION_TIMING
using BatchTimingClock = std::chrono::high_resolution_clock;
using BatchTimingTimePoint = BatchTimingClock::time_point;

BatchTimingTimePoint timingNow(const ThorImplementation::BatchSubmissionTiming* submitTiming) {
    return submitTiming == nullptr ? BatchTimingTimePoint{} : BatchTimingClock::now();
}

uint64_t elapsedMicros(BatchTimingTimePoint start, BatchTimingTimePoint finish) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count());
}
#else
struct BatchTimingTimePoint {};

constexpr BatchTimingTimePoint timingNow(const ThorImplementation::BatchSubmissionTiming*) {
    return {};
}

constexpr uint64_t elapsedMicros(BatchTimingTimePoint, BatchTimingTimePoint) {
    return 0;
}
#endif

std::shared_ptr<ThorImplementation::PhysicalParameter> getPhysicalParameter(ThorImplementation::StampedNetwork& stampedNetwork,
                                                                            const ParameterReference& parameterReference) {
    std::shared_ptr<ThorImplementation::Layer> physicalLayer =
        stampedNetwork.getPhysicalLayerFromApiLayer(parameterReference.getParameterizableId());
    if (physicalLayer == nullptr) {
        throw std::runtime_error("Could not find physical layer for api layer id " +
                                 std::to_string(parameterReference.getParameterizableId()) +
                                 " while copying placed-network training state.");
    }

    std::shared_ptr<ThorImplementation::Parameterizable> physicalParameterizable =
        std::dynamic_pointer_cast<ThorImplementation::Parameterizable>(physicalLayer);
    if (physicalParameterizable == nullptr) {
        throw std::runtime_error("Physical layer for api layer id " +
                                 std::to_string(parameterReference.getParameterizableId()) +
                                 " is not parameterizable while copying placed-network training state.");
    }

    return physicalParameterizable->getParameter(parameterReference.getParameterName());
}

void copyTensorState(ThorImplementation::Tensor destination, ThorImplementation::Tensor source, Stream& stream, const std::string& description) {
    if (destination.getDescriptor() != source.getDescriptor()) {
        throw std::runtime_error("Cannot copy placed-network training state for " + description +
                                 ": source and destination tensor descriptors differ.");
    }
    destination.copyFromAsync(source, stream);
}

void copyOptimizerState(ThorImplementation::Optimizer& destination, ThorImplementation::Optimizer& source, Stream& stream, const std::string& description) {
    if (!destination.isCompiled() || !source.isCompiled()) {
        return;
    }

    const std::vector<std::string> parameterNames = destination.getOptimizerParameterNames();
    for (const std::string& parameterName : parameterNames) {
        ThorImplementation::Tensor destinationTensor = destination.getOptimizerParameterTensor(parameterName);
        ThorImplementation::Tensor sourceTensor = source.getOptimizerParameterTensor(parameterName);
        copyTensorState(destinationTensor, sourceTensor, stream, description + " optimizer parameter '" + parameterName + "'");
    }

    destination.restoreHyperParameters(source.getAllHyperParameters());
}

std::optional<std::string> stateMatchKeyForParameter(Network& network, const ParameterReference& reference) {
    const std::optional<std::string> cloneKey = network.getCloneSourceKeyForLayerId(reference.getParameterizableId());
    if (!cloneKey.has_value() || cloneKey->empty()) {
        return std::nullopt;
    }

    return cloneKey.value() + ":" + reference.getParameterName();
}

std::string sameNetworkStateKeyForParameter(const ParameterReference& reference) {
    return "layer" + std::to_string(reference.getParameterizableId()) + ":" + reference.getParameterName();
}

std::optional<std::string> sameNetworkStateKeyForSerializedParameter(const json& layerJson,
                                                                     const std::string& parameterName) {
    if (!layerJson.contains("layer_name") || !layerJson.at("layer_name").is_string()) {
        return std::nullopt;
    }

    const std::string layerName = layerJson.at("layer_name").get<std::string>();
    if (layerName.empty()) {
        return std::nullopt;
    }

    return layerName + ":" + parameterName;
}


bool hasArchiveShard0(const std::filesystem::path& directory, const std::string& archiveName) {
    return std::filesystem::exists(directory / (archiveName + ".thor.tar")) ||
           std::filesystem::exists(directory / (archiveName + ".000000.thor.tar"));
}

struct PlacedNetworkArchiveSelection {
    std::string archiveName;
    std::string modelJsonFileName;
};

PlacedNetworkArchiveSelection selectArchiveForPlacedStateLoad(const std::filesystem::path& directory,
                                                              const std::string& archiveName) {
    if (archiveName.empty()) {
        throw std::runtime_error("PlacedNetwork::loadMatchingTrainingStateFromArtifact requires a non-empty artifact network name.");
    }

    std::error_code errorCode;
    if (!std::filesystem::exists(directory, errorCode)) {
        throw std::runtime_error("PlacedNetwork::loadMatchingTrainingStateFromArtifact: directory does not exist: " + directory.string());
    }
    if (errorCode) {
        throw std::runtime_error("PlacedNetwork::loadMatchingTrainingStateFromArtifact: failed to inspect directory '" +
                                 directory.string() + "': " + errorCode.message());
    }
    if (!std::filesystem::is_directory(directory, errorCode)) {
        throw std::runtime_error("PlacedNetwork::loadMatchingTrainingStateFromArtifact: expected a directory containing archive '" +
                                 archiveName + ".thor.tar', got: " + directory.string());
    }
    if (errorCode) {
        throw std::runtime_error("PlacedNetwork::loadMatchingTrainingStateFromArtifact: failed to inspect directory '" +
                                 directory.string() + "': " + errorCode.message());
    }
    if (!hasArchiveShard0(directory, archiveName)) {
        throw std::runtime_error("PlacedNetwork::loadMatchingTrainingStateFromArtifact: expected archive '" + archiveName +
                                 ".thor.tar' or sharded archive '" + archiveName + ".000000.thor.tar' in directory " +
                                 directory.string() + ".");
    }

    const std::string modelJsonFileName = archiveName + ".thor.json";
    auto reader = std::make_shared<thor_file::TarReader>(archiveName, directory);
    if (!reader->containsFile(modelJsonFileName)) {
        throw std::runtime_error("PlacedNetwork::loadMatchingTrainingStateFromArtifact: archive '" + archiveName + "' in " +
                                 directory.string() + " is missing expected model file '" + modelJsonFileName + "'.");
    }

    return {archiveName, modelJsonFileName};
}

json readModelJsonFromArchive(thor_file::TarReader& archiveReader, const std::string& modelJsonFileName) {
    const uint32_t modelJsonNumBytes = static_cast<uint32_t>(archiveReader.getFileSize(modelJsonFileName));
    ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
    ThorImplementation::TensorDescriptor descriptor(ThorImplementation::DataType::UINT8, {modelJsonNumBytes});
    ThorImplementation::Tensor jsonTensor(cpuPlacement, descriptor);
    archiveReader.registerReadRequest(modelJsonFileName, jsonTensor);
    archiveReader.executeReadRequests();

    const char* jsonBytes = jsonTensor.getMemPtr<char>();
    return json::parse(jsonBytes, jsonBytes + modelJsonNumBytes);
}

struct ArchiveParameterState {
    std::string storageFile;
    json optimizerJson;
    std::string description;
};

std::unordered_map<size_t, std::string> cloneSourceKeyByLayerIndex(const json& modelJson) {
    std::unordered_map<size_t, std::string> result;
    if (!modelJson.contains("clone_source_keys")) {
        throw std::runtime_error(
            "PlacedNetwork::loadMatchingTrainingStateFromArtifact: artifact is missing clone_source_keys; "
            "phase handoff requires exact clone-source identity and will not fall back to positional matching.");
    }

    const json& cloneSourceKeys = modelJson.at("clone_source_keys");
    if (!cloneSourceKeys.is_array()) {
        throw std::runtime_error("PlacedNetwork::loadMatchingTrainingStateFromArtifact: clone_source_keys must be an array.");
    }
    for (const json& entry : cloneSourceKeys) {
        const size_t layerIndex = entry.at("layer_index").get<size_t>();
        const std::string key = entry.at("key").get<std::string>();
        if (!key.empty()) {
            result[layerIndex] = key;
        }
    }
    return result;
}

std::optional<std::string> stateMatchKeyForSerializedParameter(size_t layerIndex,
                                                               const std::unordered_map<size_t, std::string>& cloneKeys,
                                                               const std::string& parameterName) {
    auto cloneIt = cloneKeys.find(layerIndex);
    if (cloneIt == cloneKeys.end() || cloneIt->second.empty()) {
        return std::nullopt;
    }

    return cloneIt->second + ":" + parameterName;
}

std::unordered_map<std::string, ArchiveParameterState> archiveParameterStateByKey(const json& modelJson) {
    const json& layers = modelJson.at("layers");
    if (!layers.is_array()) {
        throw std::runtime_error("PlacedNetwork::loadMatchingTrainingStateFromArtifact: model layers must be an array.");
    }

    const std::unordered_map<size_t, std::string> cloneKeys = cloneSourceKeyByLayerIndex(modelJson);
    std::unordered_map<std::string, ArchiveParameterState> result;
    for (size_t layerIndex = 0; layerIndex < layers.size(); ++layerIndex) {
        const json& layerJson = layers[layerIndex];
        if (!layerJson.contains("parameters")) {
            continue;
        }
        const json& parametersJson = layerJson.at("parameters");
        if (!parametersJson.is_object()) {
            throw std::runtime_error("PlacedNetwork::loadMatchingTrainingStateFromArtifact: layer parameters must be an object.");
        }

        for (const auto& [parameterName, parameterJson] : parametersJson.items()) {
            const char* storageFileKey = nullptr;
            if (parameterJson.contains("storage_file")) {
                storageFileKey = "storage_file";
            } else if (parameterJson.contains("parameter_storage")) {
                storageFileKey = "parameter_storage";
            }
            if (storageFileKey == nullptr) {
                continue;
            }

            ArchiveParameterState state;
            state.storageFile = parameterJson.at(storageFileKey).get<std::string>();
            if (parameterJson.contains("optimizer_override")) {
                state.optimizerJson = parameterJson.at("optimizer_override");
            } else if (parameterJson.contains("optimizer")) {
                state.optimizerJson = parameterJson.at("optimizer");
            }
            state.description = "artifact layer " + std::to_string(layerIndex) + " parameter '" + parameterName + "'";

            const std::optional<std::string> key = stateMatchKeyForSerializedParameter(layerIndex, cloneKeys, parameterName);
            if (!key.has_value()) {
                continue;
            }
            auto insertResult = result.emplace(key.value(), state);
            if (!insertResult.second) {
                throw std::runtime_error("PlacedNetwork::loadMatchingTrainingStateFromArtifact: duplicate clone-source parameter key '" +
                                         key.value() + "' in serialized artifact.");
            }
        }
    }
    return result;
}

std::unordered_map<std::string, ArchiveParameterState> archiveParameterStateBySameNetworkKey(const json& modelJson) {
    const json& layers = modelJson.at("layers");
    if (!layers.is_array()) {
        throw std::runtime_error("PlacedNetwork::loadTrainingStateFromSameNetworkArtifact: model layers must be an array.");
    }

    std::unordered_map<std::string, ArchiveParameterState> result;
    for (size_t layerIndex = 0; layerIndex < layers.size(); ++layerIndex) {
        const json& layerJson = layers[layerIndex];
        if (!layerJson.contains("parameters")) {
            continue;
        }
        const json& parametersJson = layerJson.at("parameters");
        if (!parametersJson.is_object()) {
            throw std::runtime_error("PlacedNetwork::loadTrainingStateFromSameNetworkArtifact: layer parameters must be an object.");
        }

        for (const auto& [parameterName, parameterJson] : parametersJson.items()) {
            const char* storageFileKey = nullptr;
            if (parameterJson.contains("storage_file")) {
                storageFileKey = "storage_file";
            } else if (parameterJson.contains("parameter_storage")) {
                storageFileKey = "parameter_storage";
            }
            if (storageFileKey == nullptr) {
                continue;
            }

            const std::optional<std::string> key = sameNetworkStateKeyForSerializedParameter(layerJson, parameterName);
            if (!key.has_value()) {
                throw std::runtime_error(
                    "PlacedNetwork::loadTrainingStateFromSameNetworkArtifact: serialized trainable parameter '" +
                    parameterName + "' in artifact layer " + std::to_string(layerIndex) +
                    " is missing layer_name; same-network restore requires exact API layer identity.");
            }

            ArchiveParameterState state;
            state.storageFile = parameterJson.at(storageFileKey).get<std::string>();
            if (parameterJson.contains("optimizer_override")) {
                state.optimizerJson = parameterJson.at("optimizer_override");
            } else if (parameterJson.contains("optimizer")) {
                state.optimizerJson = parameterJson.at("optimizer");
            }
            state.description = "artifact layer " + std::to_string(layerIndex) + " parameter '" + parameterName + "'";

            auto insertResult = result.emplace(key.value(), state);
            if (!insertResult.second) {
                throw std::runtime_error(
                    "PlacedNetwork::loadTrainingStateFromSameNetworkArtifact: duplicate API-layer parameter key '" +
                    key.value() + "' in serialized artifact.");
            }
        }
    }
    return result;
}

void registerArchiveTensorReadWithSizeCheck(thor_file::TarReader& archiveReader,
                                            const std::string& fileName,
                                            ThorImplementation::Tensor destination,
                                            const std::string& description) {
    if (fileName.empty()) {
        throw std::runtime_error("Cannot load " + description + " from an empty artifact file name.");
    }
    const uint64_t archiveBytes = archiveReader.getFileSize(fileName);
    const uint64_t destinationBytes = destination.getArraySizeInBytes();
    if (archiveBytes != destinationBytes) {
        throw std::runtime_error("Cannot load " + description + " from artifact file '" + fileName +
                                 "': archive tensor byte size " + std::to_string(archiveBytes) +
                                 " does not match destination tensor byte size " + std::to_string(destinationBytes) + ".");
    }
    archiveReader.registerReadRequest(fileName, destination);
}

std::optional<std::string> optimizerStateFileForParameter(const json& optimizerJson, const std::string& parameterName) {
    if (!optimizerJson.is_object()) {
        return std::nullopt;
    }

    const std::string exactKey = parameterName + "_tensor";
    if (optimizerJson.contains(exactKey) && optimizerJson.at(exactKey).is_string()) {
        return optimizerJson.at(exactKey).get<std::string>();
    }

    // Muon may serialize the selected fallback optimizer state as a nested optimizer.
    static const std::vector<std::string> nestedOptimizerKeys = {"fallback_optimizer_state", "fallback_optimizer"};
    for (const std::string& nestedKey : nestedOptimizerKeys) {
        if (!optimizerJson.contains(nestedKey)) {
            continue;
        }
        std::optional<std::string> nestedFile = optimizerStateFileForParameter(optimizerJson.at(nestedKey), parameterName);
        if (nestedFile.has_value()) {
            return nestedFile;
        }
    }

    return std::nullopt;
}


std::unordered_map<std::string, float> optimizerHyperParametersFromJson(const json& optimizerJson) {
    std::unordered_map<std::string, float> result;
    if (!optimizerJson.is_object()) {
        return result;
    }

    const auto readNumber = [&](const std::string& key) {
        if (!optimizerJson.contains(key)) {
            return;
        }
        const json& value = optimizerJson.at(key);
        if (value.is_number_float() || value.is_number_integer() || value.is_number_unsigned()) {
            result[key] = value.get<float>();
        } else if (value.is_boolean()) {
            result[key] = value.get<bool>() ? 1.0f : 0.0f;
        }
    };

    // Only restore true runtime state. Static architecture/config hyperparameters
    // are captured when the optimizer expression is compiled and must continue to
    // come from the destination API optimizer.
    readNumber("t");
    readNumber("epoch");
    readNumber("currentBatch");
    readNumber("current_batch");
    readNumber("currentLearningRate");
    readNumber("current_learning_rate");

    const auto alias = [&](const std::string& serializedName, const std::string& runtimeName) {
        auto it = result.find(serializedName);
        if (it != result.end() && result.find(runtimeName) == result.end()) {
            result[runtimeName] = it->second;
        }
    };
    alias("current_learning_rate", "currentLearningRate");
    alias("current_batch", "currentBatch");

    for (const std::string& nestedKey : {std::string("fallback_optimizer_state"), std::string("fallback_optimizer")}) {
        if (!optimizerJson.contains(nestedKey)) {
            continue;
        }
        std::unordered_map<std::string, float> nested = optimizerHyperParametersFromJson(optimizerJson.at(nestedKey));
        for (const auto& [key, value] : nested) {
            result.emplace(key, value);
        }
    }

    return result;
}

void registerOptimizerStateReadRequests(thor_file::TarReader& archiveReader,
                                        const json& sourceOptimizerJson,
                                        ThorImplementation::Optimizer& destinationOptimizer,
                                        const std::string& description) {
    if (!sourceOptimizerJson.is_object() || !destinationOptimizer.isCompiled()) {
        return;
    }

    destinationOptimizer.restoreHyperParameters(optimizerHyperParametersFromJson(sourceOptimizerJson));

    for (const std::string& parameterName : destinationOptimizer.getOptimizerParameterNames()) {
        if (parameterName == "weights") {
            continue;
        }

        const std::optional<std::string> maybeFile = optimizerStateFileForParameter(sourceOptimizerJson, parameterName);
        if (!maybeFile.has_value()) {
            continue;
        }
        if (!archiveReader.containsFile(maybeFile.value())) {
            throw std::runtime_error("Cannot load " + description + " optimizer parameter '" + parameterName +
                                     "': artifact is missing file '" + maybeFile.value() + "'.");
        }

        ThorImplementation::Tensor destinationTensor = destinationOptimizer.getOptimizerParameterTensor(parameterName);
        registerArchiveTensorReadWithSizeCheck(archiveReader,
                                               maybeFile.value(),
                                               destinationTensor,
                                               description + " optimizer parameter '" + parameterName + "'");
    }
}

}  // namespace

PlacedNetwork::~PlacedNetwork() {
    for (uint32_t i = 0; i < stampedNetworks.size(); ++i) {
        // Calls parentCleanup then cleanUp then clears all the shared pointers:
        stampedNetworks[i].clearNoThrow();
    }
    stampedNetworks.clear();

    // Notify the startup coordinator only after all placement-owned GPU tensors
    // have actually been destroyed. A waiting model may retry immediately after
    // this reset returns.
    deviceModelResidencyLease.reset();
}

std::vector<Event> PlacedNetwork::getSynchronizeEvents() const {
    std::vector<Event> events;
    for (const ThorImplementation::StampedNetwork& stampedNetwork : stampedNetworks) {
        std::vector<Event> stampEvents = stampedNetwork.getSynchronizeEvents();
        events.insert(events.end(),
                      std::make_move_iterator(stampEvents.begin()),
                      std::make_move_iterator(stampEvents.end()));
    }
    return events;
}

void PlacedNetwork::synchronize() const {
    std::vector<Event> events = getSynchronizeEvents();
    for (Event& event : events) {
        event.synchronize();
    }
}

void PlacedNetwork::releaseGpuResources() {
    if (stampedNetworks.empty()) {
        // A second release is a no-op. Resetting an already-empty lease also
        // makes partially constructed/test placements deterministic.
        deviceModelResidencyLease.reset();
        return;
    }

    // Do not use cudaDeviceSynchronize here. Every physical layer exposes the
    // streams/events owned by this placement, so unrelated models on the same
    // device can continue running while this placement drains.
    synchronize();

    std::exception_ptr firstCleanupFailure;
    for (ThorImplementation::StampedNetwork& stampedNetwork : stampedNetworks) {
        try {
            stampedNetwork.clear();
        } catch (...) {
            if (firstCleanupFailure == nullptr) {
                firstCleanupFailure = std::current_exception();
            }
        }
    }
    stampedNetworks.clear();

    // Wake a capacity waiter only after every placement-owned stamp has been
    // torn down. This is independent of the PlacedNetwork shared_ptr count.
    deviceModelResidencyLease.reset();

    if (firstCleanupFailure != nullptr) {
        std::rethrow_exception(firstCleanupFailure);
    }
}

void PlacedNetwork::synchronizeDevices() const {
    std::set<uint32_t> gpuNums;
    for (const ThorImplementation::StampedNetwork& stampedNetwork : stampedNetworks) {
        gpuNums.insert(stampedNetwork.getGpuNum());
    }
    for (uint32_t gpuNum : gpuNums) {
        Stream::deviceSynchronize(static_cast<int>(gpuNum));
    }
}

void PlacedNetwork::save(const std::string &directory, bool overwrite, bool saveOptimizerState) {
    network.save(stampedNetworks, directory, overwrite, saveOptimizerState);
}

std::map<std::string, ThorImplementation::Tensor> PlacedNetwork::infer(std::map<std::string, ThorImplementation::Tensor> batchInputs,
                                                                       uint64_t stampIndex) {
    std::map<std::string, ThorImplementation::Tensor> batchOutputs;
    std::map<std::string, Event> outputReadyEvents;
    Event done = submitBatch(stampIndex, std::move(batchInputs), batchOutputs, outputReadyEvents, true);
    done.synchronize();
    return batchOutputs;
}

std::map<std::string, ThorImplementation::Tensor> PlacedNetwork::infer(const Batch& batchInputs, uint64_t stampIndex) {
    std::map<std::string, ThorImplementation::Tensor> batchOutputs;
    std::map<std::string, Event> outputReadyEvents;
    Event done = submitBatch(stampIndex, batchInputs, batchOutputs, outputReadyEvents, true);
    done.synchronize();
    return batchOutputs;
}

Event PlacedNetwork::submitBatch(uint64_t stampIndex,
                                 std::map<std::string, ThorImplementation::Tensor> batchInputs,
                                 std::map<std::string, ThorImplementation::Tensor>& batchOutputs,
                                 std::map<std::string, Event>& outputReadyEvents,
                                 bool isInferenceOnly,
                                 Event* reusableProcessingFinishedEvent,
                                 bool waitForOutputsOnProcessingStream,
                                 ThorImplementation::BatchSubmissionTiming* submitTiming,
                                 std::optional<uint32_t> outputSlotIndex) {
    THOR_THROW_IF_FALSE(stampIndex < stampedNetworks.size());
    const auto totalStart = timingNow(submitTiming);
    if (!isInferenceOnly) {
        const auto activeRootsStart = timingNow(submitTiming);
        std::vector<Tensor> activeRawLossRoots = network.getRawLossTensorsForTrainingRoots(network.getLossRootTensors());
        const auto activeRootsFinish = timingNow(submitTiming);
        const auto setActiveRootsStart = timingNow(submitTiming);
        stampedNetworks[stampIndex].setActiveTrainingLossRoots(activeRawLossRoots);
        const auto setActiveRootsFinish = timingNow(submitTiming);
        if (submitTiming != nullptr) {
            submitTiming->activeObjectiveRootsMicros += elapsedMicros(activeRootsStart, activeRootsFinish);
            submitTiming->setActiveObjectiveRootsMicros += elapsedMicros(setActiveRootsStart, setActiveRootsFinish);
            submitTiming->activeObjectiveRootCount += activeRawLossRoots.size();
        }
    }
    const auto sendBatchStart = timingNow(submitTiming);
    Event processingFinishedEvent = stampedNetworks[stampIndex].sendBatch(std::move(batchInputs),
                                                                           batchOutputs,
                                                                           outputReadyEvents,
                                                                           isInferenceOnly,
                                                                           reusableProcessingFinishedEvent,
                                                                           waitForOutputsOnProcessingStream,
                                                                           submitTiming,
                                                                           outputSlotIndex);
    const auto sendBatchFinish = timingNow(submitTiming);
    if (submitTiming != nullptr) {
        submitTiming->sendBatchMicros += elapsedMicros(sendBatchStart, sendBatchFinish);
        submitTiming->totalMicros += elapsedMicros(totalStart, sendBatchFinish);
    }
    return processingFinishedEvent;
}

Event PlacedNetwork::submitBatch(uint64_t stampIndex,
                                 std::map<std::string, ThorImplementation::Tensor> batchInputs,
                                 const std::map<std::string, Event>& inputReadyEvents,
                                 std::map<std::string, ThorImplementation::Tensor>& batchOutputs,
                                 std::map<std::string, Event>& outputReadyEvents,
                                 bool isInferenceOnly,
                                 Event* reusableProcessingFinishedEvent,
                                 bool waitForOutputsOnProcessingStream,
                                 ThorImplementation::BatchSubmissionTiming* submitTiming,
                                 std::optional<uint32_t> outputSlotIndex) {
    THOR_THROW_IF_FALSE(stampIndex < stampedNetworks.size());
    const auto totalStart = timingNow(submitTiming);
    if (!isInferenceOnly) {
        const auto activeRootsStart = timingNow(submitTiming);
        std::vector<Tensor> activeRawLossRoots = network.getRawLossTensorsForTrainingRoots(network.getLossRootTensors());
        const auto activeRootsFinish = timingNow(submitTiming);
        const auto setActiveRootsStart = timingNow(submitTiming);
        stampedNetworks[stampIndex].setActiveTrainingLossRoots(activeRawLossRoots);
        const auto setActiveRootsFinish = timingNow(submitTiming);
        if (submitTiming != nullptr) {
            submitTiming->activeObjectiveRootsMicros += elapsedMicros(activeRootsStart, activeRootsFinish);
            submitTiming->setActiveObjectiveRootsMicros += elapsedMicros(setActiveRootsStart, setActiveRootsFinish);
            submitTiming->activeObjectiveRootCount += activeRawLossRoots.size();
        }
    }
    const auto sendBatchStart = timingNow(submitTiming);
    Event processingFinishedEvent = stampedNetworks[stampIndex].sendBatch(std::move(batchInputs),
                                                                           inputReadyEvents,
                                                                           batchOutputs,
                                                                           outputReadyEvents,
                                                                           isInferenceOnly,
                                                                           reusableProcessingFinishedEvent,
                                                                           waitForOutputsOnProcessingStream,
                                                                           submitTiming,
                                                                           outputSlotIndex);
    const auto sendBatchFinish = timingNow(submitTiming);
    if (submitTiming != nullptr) {
        submitTiming->sendBatchMicros += elapsedMicros(sendBatchStart, sendBatchFinish);
        submitTiming->totalMicros += elapsedMicros(totalStart, sendBatchFinish);
    }
    return processingFinishedEvent;
}

Event PlacedNetwork::submitBatch(uint64_t stampIndex,
                                 std::map<std::string, ThorImplementation::Tensor> batchInputs,
                                 std::map<std::string, ThorImplementation::Tensor>& batchOutputs,
                                 std::map<std::string, Event>& outputReadyEvents,
                                 bool isInferenceOnly,
                                 const std::vector<Tensor>& activeTrainingLossRoots,
                                 Event* reusableProcessingFinishedEvent,
                                 bool waitForOutputsOnProcessingStream,
                                 ThorImplementation::BatchSubmissionTiming* submitTiming,
                                 std::optional<uint32_t> outputSlotIndex) {
    THOR_THROW_IF_FALSE(stampIndex < stampedNetworks.size());
    const auto totalStart = timingNow(submitTiming);
    if (!isInferenceOnly) {
        const auto activeRootsStart = timingNow(submitTiming);
        std::vector<Tensor> activeRawLossRoots = network.getRawLossTensorsForTrainingRoots(activeTrainingLossRoots);
        const auto activeRootsFinish = timingNow(submitTiming);
        const auto setActiveRootsStart = timingNow(submitTiming);
        stampedNetworks[stampIndex].setActiveTrainingLossRoots(activeRawLossRoots);
        const auto setActiveRootsFinish = timingNow(submitTiming);
        if (submitTiming != nullptr) {
            submitTiming->activeObjectiveRootsMicros += elapsedMicros(activeRootsStart, activeRootsFinish);
            submitTiming->setActiveObjectiveRootsMicros += elapsedMicros(setActiveRootsStart, setActiveRootsFinish);
            submitTiming->activeObjectiveRootCount += activeRawLossRoots.size();
        }
    }
    const auto sendBatchStart = timingNow(submitTiming);
    Event processingFinishedEvent = stampedNetworks[stampIndex].sendBatch(std::move(batchInputs),
                                                                           batchOutputs,
                                                                           outputReadyEvents,
                                                                           isInferenceOnly,
                                                                           reusableProcessingFinishedEvent,
                                                                           waitForOutputsOnProcessingStream,
                                                                           submitTiming,
                                                                           outputSlotIndex);
    const auto sendBatchFinish = timingNow(submitTiming);
    if (submitTiming != nullptr) {
        submitTiming->sendBatchMicros += elapsedMicros(sendBatchStart, sendBatchFinish);
        submitTiming->totalMicros += elapsedMicros(totalStart, sendBatchFinish);
    }
    return processingFinishedEvent;
}

Event PlacedNetwork::submitBatch(uint64_t stampIndex,
                                 const Batch& batchInputs,
                                 std::map<std::string, ThorImplementation::Tensor>& batchOutputs,
                                 std::map<std::string, Event>& outputReadyEvents,
                                 bool isInferenceOnly,
                                 Event* reusableProcessingFinishedEvent,
                                 bool waitForOutputsOnProcessingStream,
                                 ThorImplementation::BatchSubmissionTiming* submitTiming,
                                 std::optional<uint32_t> outputSlotIndex) {
    THOR_THROW_IF_FALSE(stampIndex < stampedNetworks.size());
    const auto totalStart = timingNow(submitTiming);
    if (!isInferenceOnly) {
        const auto activeRootsStart = timingNow(submitTiming);
        std::vector<Tensor> activeRawLossRoots = network.getRawLossTensorsForTrainingRoots(network.getLossRootTensors());
        const auto activeRootsFinish = timingNow(submitTiming);
        const auto setActiveRootsStart = timingNow(submitTiming);
        stampedNetworks[stampIndex].setActiveTrainingLossRoots(activeRawLossRoots);
        const auto setActiveRootsFinish = timingNow(submitTiming);
        if (submitTiming != nullptr) {
            submitTiming->activeObjectiveRootsMicros += elapsedMicros(activeRootsStart, activeRootsFinish);
            submitTiming->setActiveObjectiveRootsMicros += elapsedMicros(setActiveRootsStart, setActiveRootsFinish);
            submitTiming->activeObjectiveRootCount += activeRawLossRoots.size();
        }
    }
    const auto sendBatchStart = timingNow(submitTiming);
    Event processingFinishedEvent = stampedNetworks[stampIndex].sendBatch(batchInputs,
                                                                           batchOutputs,
                                                                           outputReadyEvents,
                                                                           isInferenceOnly,
                                                                           reusableProcessingFinishedEvent,
                                                                           waitForOutputsOnProcessingStream,
                                                                           submitTiming,
                                                                           outputSlotIndex);
    const auto sendBatchFinish = timingNow(submitTiming);
    if (submitTiming != nullptr) {
        submitTiming->sendBatchMicros += elapsedMicros(sendBatchStart, sendBatchFinish);
        submitTiming->totalMicros += elapsedMicros(totalStart, sendBatchFinish);
    }
    return processingFinishedEvent;
}

Event PlacedNetwork::submitBatch(uint64_t stampIndex,
                                 const Batch& batchInputs,
                                 std::map<std::string, ThorImplementation::Tensor>& batchOutputs,
                                 std::map<std::string, Event>& outputReadyEvents,
                                 bool isInferenceOnly,
                                 const std::vector<Tensor>& activeTrainingLossRoots,
                                 Event* reusableProcessingFinishedEvent,
                                 bool waitForOutputsOnProcessingStream,
                                 ThorImplementation::BatchSubmissionTiming* submitTiming,
                                 std::optional<uint32_t> outputSlotIndex) {
    THOR_THROW_IF_FALSE(stampIndex < stampedNetworks.size());
    const auto totalStart = timingNow(submitTiming);
    if (!isInferenceOnly) {
        const auto activeRootsStart = timingNow(submitTiming);
        std::vector<Tensor> activeRawLossRoots = network.getRawLossTensorsForTrainingRoots(activeTrainingLossRoots);
        const auto activeRootsFinish = timingNow(submitTiming);
        const auto setActiveRootsStart = timingNow(submitTiming);
        stampedNetworks[stampIndex].setActiveTrainingLossRoots(activeRawLossRoots);
        const auto setActiveRootsFinish = timingNow(submitTiming);
        if (submitTiming != nullptr) {
            submitTiming->activeObjectiveRootsMicros += elapsedMicros(activeRootsStart, activeRootsFinish);
            submitTiming->setActiveObjectiveRootsMicros += elapsedMicros(setActiveRootsStart, setActiveRootsFinish);
            submitTiming->activeObjectiveRootCount += activeRawLossRoots.size();
        }
    }
    const auto sendBatchStart = timingNow(submitTiming);
    Event processingFinishedEvent = stampedNetworks[stampIndex].sendBatch(batchInputs,
                                                                           batchOutputs,
                                                                           outputReadyEvents,
                                                                           isInferenceOnly,
                                                                           reusableProcessingFinishedEvent,
                                                                           waitForOutputsOnProcessingStream,
                                                                           submitTiming,
                                                                           outputSlotIndex);
    const auto sendBatchFinish = timingNow(submitTiming);
    if (submitTiming != nullptr) {
        submitTiming->sendBatchMicros += elapsedMicros(sendBatchStart, sendBatchFinish);
        submitTiming->totalMicros += elapsedMicros(totalStart, sendBatchFinish);
    }
    return processingFinishedEvent;
}

std::vector<uint64_t> PlacedNetwork::getActiveTrainingRawLossOriginalIdsForDebug(uint64_t stampIndex) const {
    THOR_THROW_IF_FALSE(stampIndex < stampedNetworks.size());
    return stampedNetworks[stampIndex].getActiveTrainingRawLossOriginalIdsForDebug();
}

void PlacedNetwork::configureBatchInputSources(
    const std::map<std::string, BatchFieldSourceDescription> &sourcesByNetworkInput) {
    for (ThorImplementation::StampedNetwork &stampedNetwork : stampedNetworks) {
        for (const std::shared_ptr<ThorImplementation::NetworkInput> &input : stampedNetwork.getInputs()) {
            THOR_THROW_IF_FALSE(input != nullptr);
            if (input->isPassThrough()) {
                continue;
            }
            const auto source = sourcesByNetworkInput.find(input->getName());
            input->configureBatchInputSource(
                source == sourcesByNetworkInput.end()
                    ? BatchFieldSourceDescription::materialized()
                    : source->second);
        }
    }
}

void PlacedNetwork::configureBatchInputPlacements(
    const std::map<std::string, std::optional<ThorImplementation::TensorPlacement>> &placementsByNetworkInput) {
    std::map<std::string, BatchFieldSourceDescription> sources;
    for (const auto& [inputName, placement] : placementsByNetworkInput) {
        sources.emplace(inputName, BatchFieldSourceDescription::materialized(placement));
    }
    configureBatchInputSources(sources);
}

void PlacedNetwork::preallocateInputSlots(uint32_t numSlots) {
    THOR_THROW_IF_FALSE(numSlots >= 1);
    for (ThorImplementation::StampedNetwork& stampedNetwork : stampedNetworks) {
        stampedNetwork.preallocateInputSlots(numSlots);
    }
}

void PlacedNetwork::preallocateOutputSlots(uint32_t numSlots) {
    THOR_THROW_IF_FALSE(numSlots >= 1);
    for (ThorImplementation::StampedNetwork& stampedNetwork : stampedNetworks) {
        stampedNetwork.preallocateOutputSlots(numSlots);
    }
}

void PlacedNetwork::extendOutputWritableEvents(uint64_t stampIndex, Event event, std::optional<uint32_t> outputSlotIndex) {
    THOR_THROW_IF_FALSE(stampIndex < stampedNetworks.size());
    stampedNetworks[stampIndex].extendOutputWritableEvents(event, outputSlotIndex);
}

void PlacedNetwork::copyTrainingStateFrom(PlacedNetwork& source) {
    if (source.getNumStamps() != getNumStamps()) {
        throw std::runtime_error("Cannot copy placed-network training state between networks with different stamp counts.");
    }

    const std::vector<ParameterReference> parameterReferences = getTrainableParameterReferences(/*trainingEnabledOnly=*/false);
    if (parameterReferences.empty()) {
        return;
    }

    std::vector<Stream> copyStreams;
    for (uint64_t stampIndex = 0; stampIndex < stampedNetworks.size(); ++stampIndex) {
        ThorImplementation::StampedNetwork& destinationStamp = stampedNetworks[stampIndex];
        ThorImplementation::StampedNetwork& sourceStamp = source.getStampedNetwork(stampIndex);

        for (const ParameterReference& parameterReference : parameterReferences) {
            std::shared_ptr<ThorImplementation::PhysicalParameter> destinationParameter =
                getPhysicalParameter(destinationStamp, parameterReference);
            std::shared_ptr<ThorImplementation::PhysicalParameter> sourceParameter =
                getPhysicalParameter(sourceStamp, parameterReference);

            if (destinationParameter == nullptr || sourceParameter == nullptr) {
                throw std::runtime_error("Cannot copy placed-network training state for parameter '" +
                                         parameterReference.getParameterName() + "': missing physical parameter.");
            }
            if (!destinationParameter->getStorage().has_value() || !sourceParameter->getStorage().has_value()) {
                throw std::runtime_error("Cannot copy placed-network training state for parameter '" +
                                         parameterReference.getParameterName() + "': parameter storage is not initialized.");
            }

            const std::string description = "api layer " + std::to_string(parameterReference.getParameterizableId()) +
                                            " parameter '" + parameterReference.getParameterName() + "'";
            ThorImplementation::Tensor destinationStorage = destinationParameter->getStorage().value();
            Stream copyStream = Stream::getNextUploadStream(destinationStorage.getPlacement().getDeviceNum());
            copyTensorState(destinationStorage, sourceParameter->getStorage().value(), copyStream, description);

            if (destinationParameter->hasOptimizer() && sourceParameter->hasOptimizer()) {
                copyOptimizerState(*destinationParameter->getOptimizer(),
                                         *sourceParameter->getOptimizer(),
                                         copyStream,
                                         description);
            }
            copyStreams.push_back(copyStream);
        }
    }

    for (Stream& copyStream : copyStreams) {
        copyStream.synchronize();
    }
}

void PlacedNetwork::copyMatchingTrainingStateFrom(PlacedNetwork& source) {
    if (source.getNumStamps() != getNumStamps()) {
        throw std::runtime_error("Cannot copy matching placed-network training state between networks with different stamp counts.");
    }

    const std::vector<ParameterReference> destinationParameterReferences =
        getTrainableParameterReferences(/*trainingEnabledOnly=*/false);
    if (destinationParameterReferences.empty()) {
        return;
    }

    std::unordered_map<std::string, ParameterReference> sourceParameterByStateKey;
    for (const ParameterReference& sourceReference : source.getTrainableParameterReferences(/*trainingEnabledOnly=*/false)) {
        const std::optional<std::string> sourceKey = stateMatchKeyForParameter(source.network, sourceReference);
        if (!sourceKey.has_value()) {
            continue;
        }
        auto insertResult = sourceParameterByStateKey.emplace(sourceKey.value(), sourceReference);
        if (!insertResult.second) {
            throw std::runtime_error("Cannot copy matching placed-network training state: duplicate clone-source parameter key '" +
                                     sourceKey.value() + "' in source network.");
        }
    }

    if (sourceParameterByStateKey.empty()) {
        throw std::runtime_error(
            "Cannot copy matching placed-network training state: source network has no clone-source keyed trainable parameters.");
    }

    std::vector<Stream> copyStreams;
    for (uint64_t stampIndex = 0; stampIndex < stampedNetworks.size(); ++stampIndex) {
        ThorImplementation::StampedNetwork& destinationStamp = stampedNetworks[stampIndex];
        ThorImplementation::StampedNetwork& sourceStamp = source.getStampedNetwork(stampIndex);

        for (const ParameterReference& destinationReference : destinationParameterReferences) {
            const std::optional<std::string> destinationKey = stateMatchKeyForParameter(network, destinationReference);
            if (!destinationKey.has_value()) {
                continue;
            }

            auto sourceIt = sourceParameterByStateKey.find(destinationKey.value());
            if (sourceIt == sourceParameterByStateKey.end()) {
                continue;
            }
            const std::string matchedKey = destinationKey.value();

            std::shared_ptr<ThorImplementation::PhysicalParameter> destinationParameter =
                getPhysicalParameter(destinationStamp, destinationReference);
            std::shared_ptr<ThorImplementation::PhysicalParameter> sourceParameter =
                getPhysicalParameter(sourceStamp, sourceIt->second);

            if (destinationParameter == nullptr || sourceParameter == nullptr) {
                throw std::runtime_error("Cannot copy matching placed-network training state for parameter '" +
                                         destinationReference.getParameterName() + "': missing physical parameter.");
            }
            if (!destinationParameter->getStorage().has_value() || !sourceParameter->getStorage().has_value()) {
                throw std::runtime_error("Cannot copy matching placed-network training state for parameter '" +
                                         destinationReference.getParameterName() + "': parameter storage is not initialized.");
            }

            const std::string description = "state key '" + matchedKey + "' parameter '" +
                                            destinationReference.getParameterName() + "'";
            ThorImplementation::Tensor destinationStorage = destinationParameter->getStorage().value();
            Stream copyStream = Stream::getNextUploadStream(destinationStorage.getPlacement().getDeviceNum());
            copyTensorState(destinationStorage, sourceParameter->getStorage().value(), copyStream, description);

            if (destinationParameter->hasOptimizer() && sourceParameter->hasOptimizer()) {
                copyOptimizerState(*destinationParameter->getOptimizer(),
                                         *sourceParameter->getOptimizer(),
                                         copyStream,
                                         description);
            }
            copyStreams.push_back(copyStream);
        }
    }

    if (copyStreams.empty()) {
        throw std::runtime_error(
            "Cannot copy matching placed-network training state: no destination parameters matched clone-source keyed source parameters.");
    }

    for (Stream& copyStream : copyStreams) {
        copyStream.synchronize();
    }
}


void PlacedNetwork::loadTrainingStateFromSameNetworkArtifact(const std::string& artifactDirectory,
                                                                     const std::string& artifactNetworkName) {
    const std::vector<ParameterReference> destinationParameterReferences =
        getTrainableParameterReferences(/*trainingEnabledOnly=*/false);
    if (destinationParameterReferences.empty()) {
        return;
    }

    const PlacedNetworkArchiveSelection archiveSelection =
        selectArchiveForPlacedStateLoad(std::filesystem::path(artifactDirectory), artifactNetworkName);
    auto archiveReader = std::make_shared<thor_file::TarReader>(archiveSelection.archiveName, std::filesystem::path(artifactDirectory));
    const json modelJson = readModelJsonFromArchive(*archiveReader, archiveSelection.modelJsonFileName);
    const std::unordered_map<std::string, ArchiveParameterState> sourceParameterByStateKey =
        archiveParameterStateBySameNetworkKey(modelJson);
    if (sourceParameterByStateKey.empty()) {
        throw std::runtime_error(
            "PlacedNetwork::loadTrainingStateFromSameNetworkArtifact: artifact contains no API-layer keyed trainable parameter state.");
    }

    uint64_t registeredStateLoads = 0;
    for (uint64_t stampIndex = 0; stampIndex < stampedNetworks.size(); ++stampIndex) {
        ThorImplementation::StampedNetwork& destinationStamp = stampedNetworks[stampIndex];

        for (const ParameterReference& destinationReference : destinationParameterReferences) {
            const std::string destinationKey = sameNetworkStateKeyForParameter(destinationReference);

            auto sourceIt = sourceParameterByStateKey.find(destinationKey);
            if (sourceIt == sourceParameterByStateKey.end()) {
                throw std::runtime_error(
                    "PlacedNetwork::loadTrainingStateFromSameNetworkArtifact: artifact has no saved state for exact API-layer parameter key '" +
                    destinationKey + "'.");
            }

            std::shared_ptr<ThorImplementation::PhysicalParameter> destinationParameter =
                getPhysicalParameter(destinationStamp, destinationReference);
            if (destinationParameter == nullptr) {
                throw std::runtime_error("Cannot load same-network placed-network training state for parameter '" +
                                         destinationReference.getParameterName() + "': missing destination physical parameter.");
            }
            if (!destinationParameter->getStorage().has_value()) {
                throw std::runtime_error("Cannot load same-network placed-network training state for parameter '" +
                                         destinationReference.getParameterName() + "': destination parameter storage is not initialized.");
            }

            const std::string description = "API-layer state key '" + destinationKey + "' parameter '" +
                                            destinationReference.getParameterName() + "'";
            ThorImplementation::Tensor destinationStorage = destinationParameter->getStorage().value();
            registerArchiveTensorReadWithSizeCheck(*archiveReader,
                                                   sourceIt->second.storageFile,
                                                   destinationStorage,
                                                   description);
            ++registeredStateLoads;

            if (destinationParameter->hasOptimizer()) {
                registerOptimizerStateReadRequests(*archiveReader,
                                                   sourceIt->second.optimizerJson,
                                                   *destinationParameter->getOptimizer(),
                                                   description);
            }
        }
    }

    if (registeredStateLoads == 0) {
        throw std::runtime_error(
            "PlacedNetwork::loadTrainingStateFromSameNetworkArtifact: no destination parameters were loaded from artifact.");
    }

    archiveReader->executeReadRequests();
}

void PlacedNetwork::loadMatchingTrainingStateFromArtifact(const std::string& artifactDirectory,
                                                          const std::string& artifactNetworkName) {
    const std::vector<ParameterReference> destinationParameterReferences =
        getTrainableParameterReferences(/*trainingEnabledOnly=*/false);
    if (destinationParameterReferences.empty()) {
        return;
    }

    const PlacedNetworkArchiveSelection archiveSelection =
        selectArchiveForPlacedStateLoad(std::filesystem::path(artifactDirectory), artifactNetworkName);
    auto archiveReader = std::make_shared<thor_file::TarReader>(archiveSelection.archiveName, std::filesystem::path(artifactDirectory));
    const json modelJson = readModelJsonFromArchive(*archiveReader, archiveSelection.modelJsonFileName);
    const std::unordered_map<std::string, ArchiveParameterState> sourceParameterByStateKey =
        archiveParameterStateByKey(modelJson);
    if (sourceParameterByStateKey.empty()) {
        throw std::runtime_error(
            "PlacedNetwork::loadMatchingTrainingStateFromArtifact: artifact contains no clone-source keyed trainable parameter state.");
    }

    uint64_t registeredStateLoads = 0;
    for (uint64_t stampIndex = 0; stampIndex < stampedNetworks.size(); ++stampIndex) {
        ThorImplementation::StampedNetwork& destinationStamp = stampedNetworks[stampIndex];

        for (const ParameterReference& destinationReference : destinationParameterReferences) {
            const std::optional<std::string> destinationKey = stateMatchKeyForParameter(network, destinationReference);
            if (!destinationKey.has_value()) {
                continue;
            }

            auto sourceIt = sourceParameterByStateKey.find(destinationKey.value());
            if (sourceIt == sourceParameterByStateKey.end()) {
                continue;
            }
            const std::string matchedKey = destinationKey.value();

            std::shared_ptr<ThorImplementation::PhysicalParameter> destinationParameter =
                getPhysicalParameter(destinationStamp, destinationReference);
            if (destinationParameter == nullptr) {
                throw std::runtime_error("Cannot load matching placed-network training state for parameter '" +
                                         destinationReference.getParameterName() + "': missing destination physical parameter.");
            }
            if (!destinationParameter->getStorage().has_value()) {
                throw std::runtime_error("Cannot load matching placed-network training state for parameter '" +
                                         destinationReference.getParameterName() + "': destination parameter storage is not initialized.");
            }

            const std::string description = "state key '" + matchedKey + "' parameter '" +
                                            destinationReference.getParameterName() + "'";
            ThorImplementation::Tensor destinationStorage = destinationParameter->getStorage().value();
            registerArchiveTensorReadWithSizeCheck(*archiveReader,
                                                   sourceIt->second.storageFile,
                                                   destinationStorage,
                                                   description);
            ++registeredStateLoads;

            if (destinationParameter->hasOptimizer()) {
                registerOptimizerStateReadRequests(*archiveReader,
                                                   sourceIt->second.optimizerJson,
                                                   *destinationParameter->getOptimizer(),
                                                   description);
            }
        }
    }

    if (registeredStateLoads == 0) {
        throw std::runtime_error(
            "PlacedNetwork::loadMatchingTrainingStateFromArtifact: no destination parameters matched clone-source keyed artifact parameters.");
    }

    archiveReader->executeReadRequests();
}

std::vector<ParameterReference> PlacedNetwork::getTrainableParameterReferences(bool trainingEnabledOnly) {
    return network.getTrainableParameterReferences(trainingEnabledOnly);
}

BoundParameter PlacedNetwork::resolveParameterReference(const ParameterReference& parameterReference) {
    return network.resolveParameterReference(this, parameterReference);
}

std::vector<BoundParameter> PlacedNetwork::resolveParameterReferences(const std::vector<ParameterReference>& parameterReferences) {
    return network.resolveParameterReferences(this, parameterReferences);
}

bool PlacedNetwork::hasApiTensor(const Tensor& tensor) {
    return tensor.isInitialized() && network.hasApiTensorByOriginalId(tensor.getOriginalId());
}

Tensor PlacedNetwork::resolveApiTensor(const Tensor& tensor) {
    if (!tensor.isInitialized()) {
        throw std::runtime_error("Cannot resolve an uninitialized Tensor against a placed network.");
    }
    return network.resolveApiTensorByOriginalId(tensor.getOriginalId());
}

std::vector<Tensor> PlacedNetwork::resolveApiTensors(const std::vector<Tensor>& tensors) {
    std::vector<Tensor> resolved;
    resolved.reserve(tensors.size());
    for (const Tensor& tensor : tensors) {
        resolved.push_back(resolveApiTensor(tensor));
    }
    return resolved;
}

bool PlacedNetwork::hasNetworkInput(const std::string& name) {
    if (stampedNetworks.empty()) {
        return false;
    }
    if (stampedNetworks[0].raggedInputNamedShared.count(name) != 0) {
        return true;
    }
    for (const auto& [raggedName, binding] : stampedNetworks[0].raggedInputNamedShared) {
        (void)raggedName;
        if (name == binding.valuesInputName || name == binding.offsetsInputName) {
            return false;
        }
    }
    return stampedNetworks[0].inputNamedShared.count(name) != 0;
}

std::vector<std::string> PlacedNetwork::getNetworkInputNames(uint64_t stampIndex) {
    if (stampIndex >= stampedNetworks.size()) {
        throw std::runtime_error("PlacedNetwork stamp index out of range while listing network inputs.");
    }
    std::set<std::string> raggedPhysicalNames;
    for (const auto& [name, binding] : stampedNetworks[stampIndex].raggedInputNamedShared) {
        (void)name;
        raggedPhysicalNames.insert(binding.valuesInputName);
        raggedPhysicalNames.insert(binding.offsetsInputName);
    }

    std::vector<std::string> names;
    names.reserve(stampedNetworks[stampIndex].inputNamedShared.size() + stampedNetworks[stampIndex].raggedInputNamedShared.size());
    for (const auto& [name, input] : stampedNetworks[stampIndex].inputNamedShared) {
        (void)input;
        if (raggedPhysicalNames.count(name) == 0) {
            names.push_back(name);
        }
    }
    for (const auto& [name, binding] : stampedNetworks[stampIndex].raggedInputNamedShared) {
        (void)binding;
        names.push_back(name);
    }
    return names;
}

}  // namespace Thor
