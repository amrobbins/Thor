#include "DeepLearning/Api/Training/TrainingRuns.h"

#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Learning/CustomLayer.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "DeepLearning/Api/Data/BatchSession.h"
#include "DeepLearning/Api/Data/TrainingData.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Training/Observers/TrainingRunsStatsReporter.h"
#include "DeepLearning/Api/Training/Observers/TrainingStatsSink.h"
#include "DeepLearning/Api/Training/PhaseGraphConnector.h"
#include "DeepLearning/Api/Training/TrainingProgram.h"
#include "DeepLearning/Implementation/Training/DeviceStartupCoordinator.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <iomanip>
#include <limits>
#include <thread>
#include <cctype>
#include <utility>
#include <optional>

namespace Thor {

namespace {

constexpr int ENSEMBLE_MANIFEST_FIRST_ARTIFACT_VERSION = 1;
constexpr int ENSEMBLE_MANIFEST_CURRENT_ARTIFACT_VERSION = ENSEMBLE_MANIFEST_FIRST_ARTIFACT_VERSION;

std::shared_ptr<PlacedNetwork> placeInferenceNetworkWithSerializedStartup(
    Network& network,
    uint32_t batchSize,
    bool networkOutputsOnGpu = false) {
    constexpr int startupDeviceNum = 0;
    ThorImplementation::DeviceStartupGuard startupGuard =
        ThorImplementation::acquireDeviceStartupGuard(startupDeviceNum);

    for (;;) {
        std::shared_ptr<PlacedNetwork> placed;
        std::exception_ptr startupFailure;
        try {
            std::vector<Event> initDoneEvents;
            placed = network.place(
                batchSize,
                initDoneEvents,
                /*inferenceOnly=*/true,
                /*forcedDevices=*/{},
                /*forcedNumStampsPerGpu=*/0,
                networkOutputsOnGpu);
            THOR_THROW_IF_FALSE(placed->getNumStamps() == 1);
            THOR_THROW_IF_FALSE(
                placed->getStampedNetwork(0).getGpuNum() == startupDeviceNum);
            for (Event& event : initDoneEvents) {
                event.synchronize();
            }

            // Synchronous evaluator inference otherwise allocates NetworkInput
            // slot zero lazily on the first batch. Make that allocation part of
            // the serialized and memory-admitted startup transaction.
            placed->preallocateOutputSlots(1);
            placed->preallocateInputSlots(1);
            startupGuard.complete(*placed);
            return placed;
        } catch (...) {
            startupFailure = std::current_exception();
        }

        if (placed != nullptr) {
            try {
                placed->synchronize();
            } catch (...) {
            }
            placed.reset();
        }

        if (!ThorImplementation::isDeviceStartupMemoryFailure(
                startupFailure) ||
            startupGuard.getRetryableLoadedModelCount() == 0) {
            std::rethrow_exception(startupFailure);
        }
        startupGuard.waitForModelRelease();
    }
}

std::string normalizedOutputPathForCollisionCheck(const std::string& path) {
    std::filesystem::path outputPath(path);
    if (outputPath.empty()) {
        outputPath = std::filesystem::path(".");
    }
    if (!outputPath.is_absolute()) {
        outputPath = std::filesystem::current_path() / outputPath;
    }
    return outputPath.lexically_normal().string();
}

std::filesystem::path selectedTrainingArtifactModelDirectory(const std::filesystem::path& artifactRoot) {
    const std::filesystem::path bestDirectory = artifactRoot / "best";
    if (std::filesystem::exists(bestDirectory)) {
        return bestDirectory;
    }
    const std::filesystem::path latestDirectory = artifactRoot / "latest";
    if (std::filesystem::exists(latestDirectory)) {
        return latestDirectory;
    }
    return artifactRoot;
}

LineStatsColorMode combinedTrainingRunsColorMode(const std::vector<TrainingRunsSpec>& runs) {
    bool sawAuto = false;
    for (const TrainingRunsSpec& spec : runs) {
        if (spec.trainer == nullptr) {
            continue;
        }
        const LineStatsColorMode mode = spec.trainer->getRuntimeConfig().statsColorMode;
        if (mode == LineStatsColorMode::ALWAYS) {
            return LineStatsColorMode::ALWAYS;
        }
        if (mode == LineStatsColorMode::AUTO) {
            sawAuto = true;
        }
    }
    return sawAuto ? LineStatsColorMode::AUTO : LineStatsColorMode::NEVER;
}

std::vector<TrainingRunInputSignature> collectNetworkInputSignature(const std::shared_ptr<Network>& network) {
    if (network == nullptr) {
        return {};
    }

    std::vector<TrainingRunInputSignature> signature;
    const uint32_t numLayers = network->getNumLayers();
    signature.reserve(numLayers);
    for (uint32_t i = 0; i < numLayers; ++i) {
        std::shared_ptr<NetworkInput> input = std::dynamic_pointer_cast<NetworkInput>(network->getLayer(i));
        if (input == nullptr) {
            continue;
        }

        TrainingRunInputSignature item;
        item.inputName = input->getName();
        item.dimensions = input->getDimensions();
        item.dataType = ThorImplementation::TensorDescriptor::getElementTypeName(input->getDataType());
        item.dimensionsIncludeBatch = input->dimensionsIncludeBatch();
        signature.push_back(std::move(item));
    }

    return signature;
}

std::vector<TrainingRunOutputSignature> collectNetworkOutputSignature(const std::shared_ptr<Network>& network) {
    if (network == nullptr) {
        return {};
    }

    std::vector<TrainingRunOutputSignature> signature;
    const uint32_t numLayers = network->getNumLayers();
    signature.reserve(numLayers);
    for (uint32_t i = 0; i < numLayers; ++i) {
        std::shared_ptr<NetworkOutput> output = std::dynamic_pointer_cast<NetworkOutput>(network->getLayer(i));
        if (output == nullptr) {
            continue;
        }

        std::optional<Tensor> tensor = output->getFeatureOutput();
        if (!tensor.has_value()) {
            tensor = output->getFeatureInput();
        }
        if (!tensor.has_value()) {
            continue;
        }

        TrainingRunOutputSignature item;
        item.outputName = output->getName();
        item.dimensions = tensor->getDimensions();
        item.dataType = ThorImplementation::TensorDescriptor::getElementTypeName(tensor->getDataType());
        signature.push_back(std::move(item));
    }

    return signature;
}

std::vector<TrainingRunInputSignature> collectMergedNetworkInputSignature(
    const std::vector<std::shared_ptr<Network>>& networks,
    const std::string& context) {
    std::vector<TrainingRunInputSignature> merged;
    std::map<std::string, size_t> indexByName;
    for (const std::shared_ptr<Network>& network : networks) {
        for (const TrainingRunInputSignature& item : collectNetworkInputSignature(network)) {
            auto [it, inserted] = indexByName.emplace(item.inputName, merged.size());
            if (inserted) {
                merged.push_back(item);
                continue;
            }
            if (!merged[it->second].compatibleWith(item)) {
                throw std::runtime_error(context + " has incompatible NetworkInput signatures for '" + item.inputName + "'.");
            }
        }
    }
    return merged;
}

std::vector<TrainingRunOutputSignature> collectMergedNetworkOutputSignature(
    const std::vector<std::shared_ptr<Network>>& networks,
    const std::string& context) {
    std::vector<TrainingRunOutputSignature> merged;
    std::map<std::string, size_t> indexByName;
    for (const std::shared_ptr<Network>& network : networks) {
        for (const TrainingRunOutputSignature& item : collectNetworkOutputSignature(network)) {
            auto [it, inserted] = indexByName.emplace(item.outputName, merged.size());
            if (inserted) {
                merged.push_back(item);
                continue;
            }
            if (!merged[it->second].compatibleWith(item)) {
                throw std::runtime_error(context + " has incompatible NetworkOutput signatures for '" + item.outputName + "'.");
            }
        }
    }
    return merged;
}

std::vector<NetworkLossReference> collectNetworkReportableLosses(
    const std::vector<std::shared_ptr<Network>>& networks) {
    std::vector<NetworkLossReference> losses;
    for (const std::shared_ptr<Network>& network : networks) {
        if (network == nullptr) {
            continue;
        }
        const std::vector<NetworkLossReference> networkLosses = network->getReportableLosses();
        losses.insert(losses.end(), networkLosses.begin(), networkLosses.end());
    }
    return losses;
}

std::vector<NetworkMetricReference> collectNetworkReportableMetrics(
    const std::vector<std::shared_ptr<Network>>& networks) {
    std::vector<NetworkMetricReference> metrics;
    for (const std::shared_ptr<Network>& network : networks) {
        if (network == nullptr) {
            continue;
        }
        const std::vector<NetworkMetricReference> networkMetrics = network->getReportableMetrics();
        metrics.insert(metrics.end(), networkMetrics.begin(), networkMetrics.end());
    }
    return metrics;
}

std::vector<std::string> filterRequestedLossNamesToAvailable(const std::vector<NetworkLossReference>& availableLosses,
                                                            const std::vector<std::string>& requestedLossNames) {
    if (requestedLossNames.empty()) {
        return {};
    }
    std::set<std::string> availableNames;
    for (const NetworkLossReference& loss : availableLosses) {
        availableNames.insert(loss.lossName);
    }
    std::vector<std::string> filtered;
    filtered.reserve(requestedLossNames.size());
    for (const std::string& requestedName : requestedLossNames) {
        if (availableNames.count(requestedName) != 0) {
            filtered.push_back(requestedName);
        }
    }
    return filtered;
}

std::vector<std::string> filterRequestedMetricNamesToAvailable(const std::vector<NetworkMetricReference>& availableMetrics,
                                                              const std::vector<std::string>& requestedMetricNames) {
    if (requestedMetricNames.empty()) {
        return {};
    }
    std::set<std::string> availableNames;
    for (const NetworkMetricReference& metric : availableMetrics) {
        availableNames.insert(metric.metricName);
    }
    std::vector<std::string> filtered;
    filtered.reserve(requestedMetricNames.size());
    for (const std::string& requestedName : requestedMetricNames) {
        if (availableNames.count(requestedName) != 0) {
            filtered.push_back(requestedName);
        }
    }
    return filtered;
}

std::string dimensionsToString(const std::vector<uint64_t>& dimensions) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < dimensions.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << dimensions[i];
    }
    out << "]";
    return out.str();
}

std::string inputSignatureToString(const std::vector<TrainingRunInputSignature>& signature) {
    std::ostringstream out;
    out << "{";
    for (size_t i = 0; i < signature.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << signature[i].inputName << ":" << dimensionsToString(signature[i].dimensions) << ":" << signature[i].dataType
            << (signature[i].dimensionsIncludeBatch ? ":batch_included" : ":batch_excluded");
    }
    out << "}";
    return out.str();
}

std::string outputSignatureToString(const std::vector<TrainingRunOutputSignature>& signature) {
    std::ostringstream out;
    out << "{";
    for (size_t i = 0; i < signature.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << signature[i].outputName << ":" << dimensionsToString(signature[i].dimensions) << ":" << signature[i].dataType;
    }
    out << "}";
    return out.str();
}

bool inputSignaturesCompatible(const std::vector<TrainingRunInputSignature>& lhs,
                               const std::vector<TrainingRunInputSignature>& rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }

    std::map<std::string, const TrainingRunInputSignature*> lhsByName;
    for (const TrainingRunInputSignature& item : lhs) {
        if (!lhsByName.emplace(item.inputName, &item).second) {
            return false;
        }
    }

    std::set<std::string> seenRhsNames;
    for (const TrainingRunInputSignature& rhsItem : rhs) {
        if (!seenRhsNames.insert(rhsItem.inputName).second) {
            return false;
        }
        auto lhsIt = lhsByName.find(rhsItem.inputName);
        if (lhsIt == lhsByName.end() || !lhsIt->second->compatibleWith(rhsItem)) {
            return false;
        }
    }
    return true;
}

bool outputSignaturesCompatible(const std::vector<TrainingRunOutputSignature>& lhs,
                                const std::vector<TrainingRunOutputSignature>& rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }

    std::map<std::string, const TrainingRunOutputSignature*> lhsByName;
    for (const TrainingRunOutputSignature& item : lhs) {
        if (!lhsByName.emplace(item.outputName, &item).second) {
            return false;
        }
    }

    std::set<std::string> seenRhsNames;
    for (const TrainingRunOutputSignature& rhsItem : rhs) {
        if (!seenRhsNames.insert(rhsItem.outputName).second) {
            return false;
        }
        auto lhsIt = lhsByName.find(rhsItem.outputName);
        if (lhsIt == lhsByName.end() || !lhsIt->second->compatibleWith(rhsItem)) {
            return false;
        }
    }
    return true;
}

bool outputSignatureHasPredictionTensor(const std::vector<TrainingRunOutputSignature>& signature) {
    return std::any_of(signature.begin(), signature.end(), [](const TrainingRunOutputSignature& item) {
        return item.outputName != "loss";
    });
}

bool trainingProgramHasActivePhaseForValidation(const TrainingProgram& program) {
    for (const std::shared_ptr<TrainingStep>& step : program.getSteps()) {
        if (step == nullptr || !step->isInitialized()) {
            continue;
        }
        if (!step->getActivePhaseNetworkSpecs().empty()) {
            return true;
        }
    }
    return false;
}

std::string jsonEscape(const std::string& value) {
    std::ostringstream out;
    for (unsigned char ch : value) {
        switch (ch) {
            case '\\':
                out << "\\\\";
                break;
            case '"':
                out << "\\\"";
                break;
            case '\b':
                out << "\\b";
                break;
            case '\f':
                out << "\\f";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                if (ch < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned int>(ch)
                        << std::dec << std::setfill(' ');
                } else {
                    out << static_cast<char>(ch);
                }
                break;
        }
    }
    return out.str();
}

void writeJsonString(std::ostream& out, const std::string& value) {
    out << '"' << jsonEscape(value) << '"';
}

void writeJsonStringArray(std::ostream& out, const std::vector<std::string>& values, const std::string& indent) {
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << "\n" << indent;
        writeJsonString(out, values[i]);
    }
    if (!values.empty()) {
        out << "\n" << indent.substr(0, indent.size() >= 2 ? indent.size() - 2 : 0);
    }
    out << "]";
}

void writeOptionalUint64Json(std::ostream& out, const std::optional<uint64_t>& value) {
    if (value.has_value()) {
        out << value.value();
    } else {
        out << "null";
    }
}

void writeOptionalDoubleJson(std::ostream& out, const std::optional<double>& value) {
    if (value.has_value() && std::isfinite(value.value())) {
        out << std::setprecision(17) << value.value();
    } else {
        out << "null";
    }
}

void eraseNames(std::vector<std::string>& names, const std::vector<std::string>& namesToErase) {
    if (names.empty() || namesToErase.empty()) {
        return;
    }
    const std::set<std::string> erased(namesToErase.begin(), namesToErase.end());
    names.erase(std::remove_if(names.begin(), names.end(), [&](const std::string& name) { return erased.count(name) != 0; }),
                names.end());
}

void appendNameIfMissing(std::vector<std::string>& names, const std::string& name) {
    if (name.empty()) {
        return;
    }
    if (std::find(names.begin(), names.end(), name) == names.end()) {
        names.push_back(name);
    }
}

struct SavedNetworkArtifactRef {
    std::filesystem::path directory;
    std::string networkName;
};

std::vector<std::string> deployableOutputNamesForSavedEnsemble(Network& referenceMember,
                                                               const std::vector<TrainingRunOutputSignature>& signature) {
    std::set<std::string> reportOnlyOutputNames;
    reportOnlyOutputNames.insert("loss");
    for (const NetworkLossReference& loss : referenceMember.getReportableLosses()) {
        reportOnlyOutputNames.insert(loss.lossName);
    }
    for (const NetworkMetricReference& metric : referenceMember.getReportableMetrics()) {
        reportOnlyOutputNames.insert(metric.metricName);
    }

    std::vector<std::string> names;
    names.reserve(signature.size());
    for (const TrainingRunOutputSignature& item : signature) {
        if (reportOnlyOutputNames.count(item.outputName) == 0) {
            names.push_back(item.outputName);
        }
    }
    return names;
}

std::string safeNetworkNameForSavedEnsemble(std::string_view ensembleGroup) {
    std::string name = "ensemble";
    if (!ensembleGroup.empty()) {
        name += "_";
    }
    bool wroteAnyNameChar = false;
    for (unsigned char ch : ensembleGroup) {
        if (std::isalnum(ch) || ch == '-' || ch == '_' || ch == '.') {
            name.push_back(static_cast<char>(ch));
            wroteAnyNameChar = true;
        } else {
            name.push_back('_');
            wroteAnyNameChar = true;
        }
    }
    if (!wroteAnyNameChar && ensembleGroup.empty()) {
        name += "model";
    }
    return name;
}

void removePathIfExistsForEnsembleSave(const std::filesystem::path& path) {
    std::error_code errorCode;
    if (!std::filesystem::exists(path, errorCode) && !errorCode) {
        return;
    }
    errorCode.clear();
    std::filesystem::remove_all(path, errorCode);
    if (errorCode) {
        throw std::runtime_error("Failed to remove ensemble artifact path '" + path.string() + "': " + errorCode.message());
    }
}

std::string resolvedEnsembleAggregation(const TrainingEnsembleResult& ensemble, std::string aggregation) {
    if (aggregation.empty() || aggregation == "auto") {
        const bool allCompletedWeightsUnit = std::all_of(ensemble.members.begin(), ensemble.members.end(), [](const TrainingEnsembleMemberResult& member) {
            return member.status != TrainingRunStatus::COMPLETED || member.weight == 1.0;
        });
        return allCompletedWeightsUnit ? "mean" : "weighted_mean";
    }
    if (aggregation != "mean" && aggregation != "weighted_mean") {
        throw std::runtime_error("TrainingRunsResult.save_ensemble aggregation must be 'auto', 'mean', or 'weighted_mean'.");
    }
    if (aggregation == "mean") {
        const bool allCompletedWeightsUnit = std::all_of(ensemble.members.begin(), ensemble.members.end(), [](const TrainingEnsembleMemberResult& member) {
            return member.status != TrainingRunStatus::COMPLETED || member.weight == 1.0;
        });
        if (!allCompletedWeightsUnit) {
            throw std::runtime_error("TrainingRunsResult.save_ensemble aggregation='mean' cannot be used with non-unit completed ensemble weights; "
                                     "use aggregation='weighted_mean' or aggregation='auto'.");
        }
    }
    return aggregation;
}

}  // namespace

const char* trainingRunsFailurePolicyName(TrainingRunsFailurePolicy policy) {
    switch (policy) {
        case TrainingRunsFailurePolicy::CONTINUE:
            return "continue";
        case TrainingRunsFailurePolicy::CANCEL_SIBLINGS:
            return "cancel_siblings";
        default:
            return "unknown";
    }
}

namespace {

const TrainingRunInputSignature* findInputSignatureItem(const std::vector<TrainingRunInputSignature>& signature,
                                                       const std::string& inputName) {
    auto it = std::find_if(signature.begin(), signature.end(), [&](const TrainingRunInputSignature& item) {
        return item.inputName == inputName;
    });
    return it == signature.end() ? nullptr : &*it;
}

const TrainingRunOutputSignature* findOutputSignatureItem(const std::vector<TrainingRunOutputSignature>& signature,
                                                         const std::string& outputName) {
    auto it = std::find_if(signature.begin(), signature.end(), [&](const TrainingRunOutputSignature& item) {
        return item.outputName == outputName;
    });
    return it == signature.end() ? nullptr : &*it;
}

std::string inputSignatureItemToString(const TrainingRunInputSignature& item) {
    std::ostringstream out;
    out << item.inputName << ":" << dimensionsToString(item.dimensions) << ":" << item.dataType
        << (item.dimensionsIncludeBatch ? ":batch_included" : ":batch_excluded");
    return out.str();
}

std::string outputSignatureItemToString(const TrainingRunOutputSignature& item) {
    std::ostringstream out;
    out << item.outputName << ":" << dimensionsToString(item.dimensions) << ":" << item.dataType;
    return out.str();
}

struct ResolvedEnsembleLoss {
    std::string lossName{};
    std::string predictionOutputName{};
    std::string targetInputName{};
    std::optional<std::string> weightInputName{};
    std::string lossType{};
    double lossWeight = 1.0;
    std::optional<double> quantile{};
};

struct ResolvedEnsembleMetric {
    std::string metricName{};
    std::string predictionOutputName{};
    std::optional<std::string> targetInputName{};
    std::optional<std::string> inputSourceName{};
    std::string metricType{};
};

bool quantilesCompatible(std::optional<double> lhs, std::optional<double> rhs) {
    if (!lhs.has_value() && !rhs.has_value()) {
        return true;
    }
    if (!lhs.has_value() || !rhs.has_value()) {
        return false;
    }
    return std::fabs(lhs.value() - rhs.value()) <= 1.0e-6;
}

bool trainingRunsReportableLossesCompatible(const NetworkLossReference& lhs, const NetworkLossReference& rhs) {
    if (lhs.lossName != rhs.lossName) {
        return false;
    }
    if (std::fabs(lhs.lossWeight - rhs.lossWeight) > 1.0e-6) {
        return false;
    }
    if (lhs.predictionOutputName != rhs.predictionOutputName || lhs.targetInputName != rhs.targetInputName) {
        return false;
    }
    if (lhs.weightInputName != rhs.weightInputName) {
        return false;
    }
    if (lhs.lossLayerType != rhs.lossLayerType) {
        return false;
    }
    if (!quantilesCompatible(lhs.quantile, rhs.quantile)) {
        return false;
    }
    return true;
}

std::string trainingRunsReportableLossDescription(const NetworkLossReference& reference) {
    std::ostringstream out;
    out << "loss_name='" << reference.lossName << "'";
    if (!reference.predictionOutputName.empty()) {
        out << ", prediction_output_name='" << reference.predictionOutputName << "'";
    }
    out << ", target_input_name='" << reference.targetInputName << "'";
    if (reference.weightInputName.has_value()) {
        out << ", weight_input_name='" << *reference.weightInputName << "'";
    }
    if (reference.quantile.has_value()) {
        out << ", quantile=" << *reference.quantile;
    }
    out << ", loss_layer_type='" << reference.lossLayerType << "'";
    out << ", loss_weight=" << reference.lossWeight;
    return out.str();
}


std::map<std::string, NetworkLossReference> trainingRunsReportableLossesByName(
    const std::vector<NetworkLossReference>& reportableLosses,
    const std::string& context) {
    std::map<std::string, NetworkLossReference> byName;
    for (const NetworkLossReference& reference : reportableLosses) {
        auto [it, inserted] = byName.emplace(reference.lossName, reference);
        if (!inserted && !trainingRunsReportableLossesCompatible(it->second, reference)) {
            throw std::runtime_error(context + " found ambiguous graph loss name '" + reference.lossName +
                                     "'. Multiple source losses with that report name have different wiring/configuration: " +
                                     trainingRunsReportableLossDescription(it->second) + "; " +
                                     trainingRunsReportableLossDescription(reference) +
                                     ". Give graph losses unique NetworkOutput names before ensemble evaluation.");
        }
    }
    return byName;
}

std::vector<ResolvedEnsembleLoss> resolveTrainingRunsReportedLosses(
    const std::vector<NetworkLossReference>& reportableLosses,
    const std::vector<std::string>& requestedLossNames,
    const std::string& context) {
    const std::map<std::string, NetworkLossReference> byName = trainingRunsReportableLossesByName(reportableLosses, context);
    if (byName.empty()) {
        if (requestedLossNames.empty()) {
            return {};
        }
        throw std::runtime_error(context + " did not find any reportable graph losses.");
    }

    std::vector<std::string> selectedNames;
    if (requestedLossNames.empty()) {
        selectedNames.reserve(byName.size());
        for (const auto& [lossName, _] : byName) {
            (void)_;
            selectedNames.push_back(lossName);
        }
    } else {
        std::set<std::string> seen;
        for (const std::string& lossName : requestedLossNames) {
            if (lossName.empty()) {
                throw std::runtime_error(context + " contains an empty loss report name.");
            }
            if (!seen.insert(lossName).second) {
                throw std::runtime_error(context + " contains duplicate loss report name '" + lossName + "'.");
            }
            if (byName.find(lossName) == byName.end()) {
                std::ostringstream oss;
                oss << context << " requested loss report '" << lossName << "', but no graph loss with that name exists. Available losses:";
                for (const auto& [availableName, reference] : byName) {
                    oss << " " << availableName << "(" << trainingRunsReportableLossDescription(reference) << ")";
                }
                throw std::runtime_error(oss.str());
            }
            selectedNames.push_back(lossName);
        }
    }

    std::vector<ResolvedEnsembleLoss> resolved;
    resolved.reserve(selectedNames.size());
    for (const std::string& lossName : selectedNames) {
        const NetworkLossReference& reference = byName.at(lossName);
        ResolvedEnsembleLoss loss;
        loss.lossName = reference.lossName;
        loss.predictionOutputName = reference.predictionOutputName;
        loss.targetInputName = reference.targetInputName;
        loss.weightInputName = reference.weightInputName;
        loss.lossType = reference.lossLayerType;
        loss.lossWeight = reference.lossWeight;
        loss.quantile = reference.quantile;
        resolved.push_back(std::move(loss));
    }
    return resolved;
}

bool trainingRunsReportableMetricsCompatible(const NetworkMetricReference& lhs, const NetworkMetricReference& rhs) {
    return lhs.metricName == rhs.metricName && lhs.predictionOutputName == rhs.predictionOutputName &&
           lhs.targetInputName == rhs.targetInputName && lhs.inputSourceName == rhs.inputSourceName &&
           lhs.metricLayerType == rhs.metricLayerType;
}

std::string trainingRunsReportableMetricDescription(const NetworkMetricReference& reference) {
    std::ostringstream out;
    out << "metric_name='" << reference.metricName << "'";
    if (!reference.predictionOutputName.empty()) {
        out << ", prediction_output_name='" << reference.predictionOutputName << "'";
    }
    if (reference.targetInputName.has_value()) {
        out << ", target_input_name='" << *reference.targetInputName << "'";
    }
    if (reference.inputSourceName.has_value()) {
        out << ", input_source_name='" << *reference.inputSourceName << "'";
    }
    out << ", metric_layer_type='" << reference.metricLayerType << "'";
    return out.str();
}

std::map<std::string, NetworkMetricReference> trainingRunsReportableMetricsByName(
    const std::vector<NetworkMetricReference>& reportableMetrics,
    const std::string& context) {
    std::map<std::string, NetworkMetricReference> byName;
    for (const NetworkMetricReference& reference : reportableMetrics) {
        auto [it, inserted] = byName.emplace(reference.metricName, reference);
        if (!inserted && !trainingRunsReportableMetricsCompatible(it->second, reference)) {
            throw std::runtime_error(context + " found ambiguous graph metric name '" + reference.metricName +
                                     "'. Multiple source metrics with that report name have different wiring/configuration: " +
                                     trainingRunsReportableMetricDescription(it->second) + "; " +
                                     trainingRunsReportableMetricDescription(reference) +
                                     ". Give graph metrics unique NetworkOutput names before ensemble evaluation.");
        }
    }
    return byName;
}

std::vector<ResolvedEnsembleMetric> resolveTrainingRunsReportedMetrics(
    const std::vector<NetworkMetricReference>& reportableMetrics,
    const std::vector<std::string>& requestedMetricNames,
    const std::string& context) {
    const std::map<std::string, NetworkMetricReference> byName = trainingRunsReportableMetricsByName(reportableMetrics, context);
    if (byName.empty()) {
        if (requestedMetricNames.empty()) {
            return {};
        }
        throw std::runtime_error(context + " did not find any reportable graph metrics.");
    }

    std::vector<std::string> selectedNames;
    if (requestedMetricNames.empty()) {
        selectedNames.reserve(byName.size());
        for (const auto& [metricName, _] : byName) {
            (void)_;
            selectedNames.push_back(metricName);
        }
    } else {
        std::set<std::string> seen;
        for (const std::string& metricName : requestedMetricNames) {
            if (metricName.empty()) {
                throw std::runtime_error(context + " contains an empty metric report name.");
            }
            if (!seen.insert(metricName).second) {
                throw std::runtime_error(context + " contains duplicate metric report name '" + metricName + "'.");
            }
            if (byName.find(metricName) == byName.end()) {
                std::ostringstream oss;
                oss << context << " requested metric report '" << metricName << "', but no graph metric with that name exists. Available metrics:";
                for (const auto& [availableName, reference] : byName) {
                    oss << " " << availableName << "(" << trainingRunsReportableMetricDescription(reference) << ")";
                }
                throw std::runtime_error(oss.str());
            }
            selectedNames.push_back(metricName);
        }
    }

    std::vector<ResolvedEnsembleMetric> resolved;
    resolved.reserve(selectedNames.size());
    for (const std::string& metricName : selectedNames) {
        const NetworkMetricReference& reference = byName.at(metricName);
        ResolvedEnsembleMetric metric;
        metric.metricName = reference.metricName;
        metric.predictionOutputName = reference.predictionOutputName;
        metric.targetInputName = reference.targetInputName;
        metric.inputSourceName = reference.inputSourceName;
        metric.metricType = reference.metricLayerType;
        resolved.push_back(std::move(metric));
    }
    return resolved;
}

bool trainingRunsLossCanParticipateInComposedEvaluation(const ResolvedEnsembleLoss& loss) {
    (void)loss;
    // Loss reportability is established by the source network exposing the loss
    // tensor through a NetworkOutput.  Whether that loss can be represented in a
    // particular composed evaluator is decided when the evaluator is built from
    // the tensors that composition actually contains.
    return true;
}

bool trainingRunsMetricCanParticipateInComposedEvaluation(const ResolvedEnsembleMetric& metric) {
    return !metric.predictionOutputName.empty() || metric.inputSourceName.has_value();
}


struct TrainingRunsReportNameSelections {
    std::optional<std::vector<std::string>> lossNames{};
    std::optional<std::vector<std::string>> metricNames{};
};

std::string trainingRunsAvailableReportsDescription(const std::map<std::string, NetworkLossReference>& lossByName,
                                                    const std::map<std::string, NetworkMetricReference>& metricByName) {
    std::ostringstream oss;
    oss << " Available reports:";
    for (const auto& [name, reference] : lossByName) {
        oss << " " << name << "(loss: " << trainingRunsReportableLossDescription(reference) << ")";
    }
    for (const auto& [name, reference] : metricByName) {
        oss << " " << name << "(metric: " << trainingRunsReportableMetricDescription(reference) << ")";
    }
    return oss.str();
}

TrainingRunsReportNameSelections splitTrainingRunsRequestedReportsByKind(
    const std::vector<NetworkLossReference>& reportableLosses,
    const std::vector<NetworkMetricReference>& reportableMetrics,
    const std::vector<std::string>& requestedReportNames,
    const std::string& context) {
    TrainingRunsReportNameSelections selections;
    if (requestedReportNames.empty()) {
        return selections;
    }

    selections.lossNames = std::vector<std::string>{};
    selections.metricNames = std::vector<std::string>{};
    const std::map<std::string, NetworkLossReference> lossByName = trainingRunsReportableLossesByName(reportableLosses, context);
    const std::map<std::string, NetworkMetricReference> metricByName = trainingRunsReportableMetricsByName(reportableMetrics, context);

    std::set<std::string> seen;
    for (const std::string& reportName : requestedReportNames) {
        if (reportName.empty()) {
            throw std::runtime_error(context + " contains an empty report name.");
        }
        if (!seen.insert(reportName).second) {
            throw std::runtime_error(context + " contains duplicate report name '" + reportName + "'.");
        }
        const bool isLoss = lossByName.find(reportName) != lossByName.end();
        const bool isMetric = metricByName.find(reportName) != metricByName.end();
        if (isLoss && isMetric) {
            throw std::runtime_error(context + " requested report '" + reportName +
                                     "', but that name is ambiguous because both a graph loss and a graph metric use it. "
                                     "Give the NetworkOutput reports unique names.");
        }
        if (!isLoss && !isMetric) {
            throw std::runtime_error(context + " requested report '" + reportName +
                                     "', but no reportable graph loss or metric with that name exists." +
                                     trainingRunsAvailableReportsDescription(lossByName, metricByName));
        }
        if (isLoss) {
            selections.lossNames->push_back(reportName);
        } else {
            selections.metricNames->push_back(reportName);
        }
    }
    return selections;
}

std::vector<ResolvedEnsembleLoss> resolveTrainingRunsSelectedLossReports(
    const std::vector<NetworkLossReference>& reportableLosses,
    const std::optional<std::vector<std::string>>& requestedLossNames,
    const std::string& context) {
    if (requestedLossNames.has_value() && requestedLossNames->empty()) {
        return {};
    }
    static const std::vector<std::string> all;
    return resolveTrainingRunsReportedLosses(reportableLosses, requestedLossNames.has_value() ? *requestedLossNames : all, context);
}

std::vector<ResolvedEnsembleMetric> resolveTrainingRunsSelectedMetricReports(
    const std::vector<NetworkMetricReference>& reportableMetrics,
    const std::optional<std::vector<std::string>>& requestedMetricNames,
    const std::string& context) {
    if (requestedMetricNames.has_value() && requestedMetricNames->empty()) {
        return {};
    }
    static const std::vector<std::string> all;
    return resolveTrainingRunsReportedMetrics(reportableMetrics, requestedMetricNames.has_value() ? *requestedMetricNames : all, context);
}

bool trainingRunsReportNameExists(const std::vector<NetworkLossReference>& reportableLosses,
                                  const std::vector<NetworkMetricReference>& reportableMetrics,
                                  const std::string& reportName) {
    for (const NetworkLossReference& loss : reportableLosses) {
        if (loss.lossName == reportName) {
            return true;
        }
    }
    for (const NetworkMetricReference& metric : reportableMetrics) {
        if (metric.metricName == reportName) {
            return true;
        }
    }
    return false;
}

std::vector<std::string> filterRequestedReportNamesToAvailable(const std::vector<NetworkLossReference>& reportableLosses,
                                                               const std::vector<NetworkMetricReference>& reportableMetrics,
                                                               const std::vector<std::string>& requestedReportNames) {
    if (requestedReportNames.empty()) {
        return {};
    }
    std::vector<std::string> filtered;
    filtered.reserve(requestedReportNames.size());
    for (const std::string& reportName : requestedReportNames) {
        if (trainingRunsReportNameExists(reportableLosses, reportableMetrics, reportName)) {
            filtered.push_back(reportName);
        }
    }
    return filtered;
}

}  // namespace

bool TrainingEnsembleResult::allCompleted() const {
    return !members.empty() && std::all_of(members.begin(), members.end(), [](const TrainingEnsembleMemberResult& member) {
        return member.status == TrainingRunStatus::COMPLETED;
    });
}

bool TrainingEnsembleResult::anyFailed() const {
    return std::any_of(members.begin(), members.end(), [](const TrainingEnsembleMemberResult& member) {
        return member.status == TrainingRunStatus::FAILED || member.status == TrainingRunStatus::OUT_OF_MEMORY ||
               member.status == TrainingRunStatus::INTERRUPTED;
    });
}

size_t TrainingEnsembleResult::successfulModels() const {
    return static_cast<size_t>(std::count_if(members.begin(), members.end(), [](const TrainingEnsembleMemberResult& member) {
        return member.status == TrainingRunStatus::COMPLETED;
    }));
}

bool TrainingEnsembleResult::hasEnoughSuccessfulModels() const {
    return successfulModels() >= requiredSuccessfulModels();
}

double TrainingEnsembleResult::totalWeight() const {
    double total = 0.0;
    for (const TrainingEnsembleMemberResult& member : members) {
        total += member.weight;
    }
    return total;
}

std::map<std::string, size_t> TrainingEnsembleResult::statusCounts() const {
    std::map<std::string, size_t> counts;
    for (const TrainingEnsembleMemberResult& member : members) {
        counts[trainingRunStatusName(member.status)] += 1;
    }
    return counts;
}

bool TrainingRunsResult::allCompleted() const {
    return std::all_of(results.begin(), results.end(), [](const TrainingRunResult& result) { return result.completed(); });
}

bool TrainingRunsResult::anyFailed() const {
    return std::any_of(results.begin(), results.end(), [](const TrainingRunResult& result) { return result.failed(); });
}

bool TrainingRunsResult::anyCancelled() const {
    return std::any_of(results.begin(), results.end(), [](const TrainingRunResult& result) { return result.cancelled(); });
}

namespace {

void saveEnsembleNetworkArtifact(const TrainingEnsembleResult& ensembleResult,
                                 const std::vector<SavedNetworkArtifactRef>& memberArtifacts,
                                 const std::string& aggregation,
                                 const std::filesystem::path& artifactDirectory,
                                 bool overwriteNetworkArchive);

}  // namespace

std::map<std::string, size_t> TrainingRunsResult::statusCounts() const {
    std::map<std::string, size_t> counts;
    for (const TrainingRunResult& result : results) {
        counts[trainingRunStatusName(result.status)] += 1;
    }
    return counts;
}

const TrainingRunResult& TrainingRunsResult::at(size_t index) const {
    if (index >= results.size()) {
        throw std::out_of_range("TrainingRunsResult index is out of range.");
    }
    return results[index];
}

const TrainingRunResult& TrainingRunsResult::at(std::string_view runName) const {
    const auto it = std::find_if(results.begin(), results.end(), [runName](const TrainingRunResult& result) {
        return result.runName == runName;
    });
    if (it == results.end()) {
        throw std::out_of_range("TrainingRunsResult does not contain run name '" + std::string(runName) + "'.");
    }
    return *it;
}

const TrainingEnsembleResult& TrainingRunsResult::ensemble(std::string_view ensembleGroup) const {
    const auto it = std::find_if(ensembles_.begin(), ensembles_.end(), [ensembleGroup](const TrainingEnsembleResult& result) {
        return result.ensembleGroup == ensembleGroup;
    });
    if (it == ensembles_.end()) {
        throw std::out_of_range("TrainingRunsResult does not contain ensemble group '" + std::string(ensembleGroup) + "'.");
    }
    return *it;
}

std::string TrainingRunsResult::saveEnsemble(std::string_view ensembleGroup,
                                             const std::string& directory,
                                             std::string aggregation,
                                             bool overwrite) const {
    if (directory.empty()) {
        throw std::runtime_error("TrainingRunsResult.save_ensemble directory must not be empty.");
    }

    const TrainingEnsembleResult& ensembleResult = ensemble(ensembleGroup);
    if (ensembleResult.empty()) {
        throw std::runtime_error("TrainingRunsResult.save_ensemble cannot save empty ensemble group '" +
                                 std::string(ensembleGroup) + "'.");
    }
    if (!ensembleResult.hasEnoughSuccessfulModels()) {
        throw std::runtime_error("TrainingRunsResult.save_ensemble requires ensemble group '" + std::string(ensembleGroup) +
                                 "' to have at least " + std::to_string(ensembleResult.requiredSuccessfulModels()) +
                                 " completed member(s); only " + std::to_string(ensembleResult.successfulModels()) +
                                 " completed.");
    }

    std::map<std::string, const TrainingRunResult*> resultByRunName;
    for (const TrainingRunResult& result : results) {
        resultByRunName.emplace(result.runName, &result);
    }

    const std::string resolvedAggregation = resolvedEnsembleAggregation(ensembleResult, std::move(aggregation));
    const std::filesystem::path artifactDirectory(directory);
    std::error_code errorCode;
    if (std::filesystem::exists(artifactDirectory, errorCode) && !overwrite) {
        throw std::runtime_error("TrainingRunsResult.save_ensemble output directory already exists: " + artifactDirectory.string());
    }
    if (errorCode) {
        throw std::runtime_error("Failed to inspect ensemble output directory '" + artifactDirectory.string() + "': " + errorCode.message());
    }
    if (overwrite) {
        removePathIfExistsForEnsembleSave(artifactDirectory);
    }

    std::vector<SavedNetworkArtifactRef> completedMemberArtifacts;
    completedMemberArtifacts.reserve(ensembleResult.members.size());
    for (const TrainingEnsembleMemberResult& member : ensembleResult.members) {
        if (member.status != TrainingRunStatus::COMPLETED) {
            continue;
        }
        const auto resultIt = resultByRunName.find(member.runName);
        if (resultIt == resultByRunName.end()) {
            throw std::runtime_error("TrainingRunsResult.save_ensemble could not find run result for member '" + member.runName + "'.");
        }
        const TrainingRunResult& result = *resultIt->second;
        if (!result.savedModelDirectory.has_value() || result.savedModelDirectory->empty()) {
            throw std::runtime_error("TrainingRunsResult.save_ensemble requires member '" + member.runName +
                                     "' to have a trainer save_model_dir / saved_model_dir artifact.");
        }
        if (!result.savedModelNetworkName.has_value() || result.savedModelNetworkName->empty()) {
            throw std::runtime_error("TrainingRunsResult.save_ensemble requires member '" + member.runName +
                                     "' to record the saved model network name.");
        }
        completedMemberArtifacts.push_back(
            SavedNetworkArtifactRef{selectedTrainingArtifactModelDirectory(*result.savedModelDirectory), *result.savedModelNetworkName});
    }

    // save_ensemble produces a regular Thor Network artifact.  Member artifacts
    // and manifests are not copied into the deployable artifact.
    saveEnsembleNetworkArtifact(ensembleResult,
                                completedMemberArtifacts,
                                resolvedAggregation,
                                artifactDirectory,
                                /*overwriteNetworkArchive=*/true);

    return artifactDirectory.string();
}

TrainingRuns::TrainingRuns(std::vector<TrainingRunsSpec> runs,
                           TrainingRunsFailurePolicy failurePolicy,
                           double maxSummaryLogsPerSecond,
                           std::optional<size_t> maxParallelRuns,
                           std::map<std::string, size_t> minSuccessfulModels)
    : runs(std::move(runs)),
      failurePolicy(failurePolicy),
      maxSummaryLogsPerSecond(maxSummaryLogsPerSecond),
      maxParallelRuns(maxParallelRuns),
      minSuccessfulModels(std::move(minSuccessfulModels)) {
    if (!std::isfinite(maxSummaryLogsPerSecond) || maxSummaryLogsPerSecond < 0.0) {
        throw std::runtime_error("TrainingRuns maxSummaryLogsPerSecond must be finite and >= 0.");
    }
    if (maxParallelRuns.has_value() && maxParallelRuns.value() == 0) {
        throw std::runtime_error("TrainingRuns maxParallelRuns must be >= 1 when specified.");
    }
    validateRunSpecs();
    validateMinSuccessfulModels();
}

size_t TrainingRuns::getEffectiveMaxParallelRuns() const {
    if (!maxParallelRuns.has_value()) {
        return runs.size();
    }
    return std::min(maxParallelRuns.value(), runs.size());
}

bool TrainingRuns::hasEnsembleGroups() const {
    return std::any_of(runs.begin(), runs.end(), [](const TrainingRunsSpec& spec) { return spec.ensembleGroup.has_value(); });
}

void TrainingRuns::validateEnsembleArtifactsForFit(const TrainingRunsEvaluationOptions& evaluationOptions) const {
    if (!hasEnsembleGroups()) {
        return;
    }
    if (!evaluationOptions.evaluateTrainingPopulation && evaluationOptions.testData == nullptr) {
        return;
    }

    for (const TrainingRunsSpec& spec : runs) {
        if (!spec.ensembleGroup.has_value()) {
            continue;
        }
        const std::optional<std::string>& saveModelDirectory = spec.trainer->getSaveModelDirectory();
        if (!saveModelDirectory.has_value()) {
            throw std::runtime_error(
                "TrainingRuns ensemble evaluation requires run '" + spec.runName +
                "' to configure trainer save_model_dir so its trained model artifact can be reloaded for post-fit ensemble inference.");
        }
    }
}


TrainingRunsResult TrainingRuns::fit(uint32_t epochs) { return fit(TrainerFitOptions{epochs}); }

TrainingRunsResult TrainingRuns::fit(uint32_t epochs, std::shared_ptr<const TrainingData> testData) {
    TrainingRunsEvaluationOptions evaluationOptions;
    evaluationOptions.testData = std::move(testData);
    return fit(TrainerFitOptions{epochs}, evaluationOptions);
}

TrainingRunsResult TrainingRuns::fit(const TrainerFitOptions& options) {
    return fit(options, TrainingRunsSessionOptions{});
}

TrainingRunsResult TrainingRuns::fit(const TrainerFitOptions& options, const TrainingRunsEvaluationOptions& evaluationOptions) {
    TrainingRunsSessionOptions sessionOptions;
    sessionOptions.evaluation = evaluationOptions;
    return fit(options, sessionOptions);
}

TrainingRunsResult TrainingRuns::fit(const TrainerFitOptions& options, const TrainingRunsSessionOptions& sessionOptions) {
    restartConditions = sessionOptions.restartConditions;
    earlyCompletionRules = sessionOptions.earlyCompletionRules;
    reports = sessionOptions.reports;

    const TrainingRunsEvaluationOptions& evaluationOptions = sessionOptions.evaluation;

    validateFitOptions(options);
    validateRestartConditions();
    validateEarlyCompletionRules();
    validateReportedLosses();
    validateReportedMetrics();
    if (evaluationOptions.testData != nullptr) {
        validateTestData(*evaluationOptions.testData);
    }
    validateEnsembleArtifactsForFit(evaluationOptions);

    TrainingCancellationSource cancellationSource;
    std::vector<TrainingRunResult> results;
    results.reserve(runs.size());
    for (const TrainingRunsSpec& spec : runs) {
        TrainingRunResult result;
        result.runName = spec.runName;
        result.ensembleGroup = spec.ensembleGroup;
        result.ensembleWeight = spec.ensembleWeight;
        result.status = TrainingRunStatus::NOT_STARTED;
        results.push_back(std::move(result));
    }

    auto statsReporter =
        std::make_shared<TrainingRunsStatsReporter>(stdout, combinedTrainingRunsColorMode(runs), maxSummaryLogsPerSecond);
    for (const TrainingRunsSpec& spec : runs) {
        const TrainingRuntimeConfig& runtime = spec.trainer->getRuntimeConfig();
        statsReporter->configureRun(spec.runName,
                                    TrainingRunsStatsReporter::RunConfig{runtime.statsIntervalSeconds,
                                                                         spec.ensembleGroup,
                                                                         spec.ensembleWeight,
                                                                         reportedScalarTensorNamesForSpec(spec)});
    }

    const size_t maxActiveRuns = getEffectiveMaxParallelRuns();
    std::mutex resultMutex;
    std::mutex schedulingMutex;
    std::condition_variable schedulingChanged;
    std::vector<std::thread> workers(runs.size());
    std::vector<bool> workerStarted(runs.size(), false);
    std::vector<bool> workerFinished(runs.size(), false);
    std::vector<bool> workerJoined(runs.size(), false);
    std::atomic_bool cancellationRequestedByFailure{false};

    auto launchWorker = [&](size_t i) {
        workerStarted[i] = true;
        workers[i] = std::thread([this,
                                  &options,
                                  &cancellationSource,
                                  &results,
                                  &resultMutex,
                                  &schedulingMutex,
                                  &schedulingChanged,
                                  &workerFinished,
                                  &cancellationRequestedByFailure,
                                  statsReporter,
                                  i]() {
            TrainingRunResult result;
            result.runName = runs[i].runName;
            result.ensembleGroup = runs[i].ensembleGroup;
            result.ensembleWeight = runs[i].ensembleWeight;
            result.status = TrainingRunStatus::RUNNING;

            {
                std::lock_guard<std::mutex> lock(resultMutex);
                results[i] = result;
            }
            statsReporter->markRunStarting(runs[i].runName);

            try {
                TrainingStatsSinkObserver observer(statsReporter, runs[i].runName);
                const std::vector<std::string> reportedScalarTensorNames = reportedScalarTensorNamesForSpec(runs[i]);
                const std::set<std::string> additionalScalarTensorsToReport(reportedScalarTensorNames.begin(), reportedScalarTensorNames.end());
                result = runs[i].trainer->fitTrainingRun(runs[i].runName,
                                                          options,
                                                          observer,
                                                          cancellationSource.token(),
                                                          restartConditionsForRun(runs[i]),
                                                          earlyCompletionPoliciesForRun(runs[i]),
                                                          additionalScalarTensorsToReport);
            } catch (...) {
                result = TrainingRunResult::fromException(runs[i].runName, std::current_exception());
            }
            result.ensembleGroup = runs[i].ensembleGroup;
            result.ensembleWeight = runs[i].ensembleWeight;
            if (!result.savedModelDirectory.has_value()) {
                result.savedModelDirectory = runs[i].trainer->getSaveModelDirectory();
            }
            if (!result.savedModelNetworkName.has_value() && runs[i].trainer->getNetwork() != nullptr) {
                result.savedModelNetworkName = runs[i].trainer->getNetwork()->getNetworkName();
            }
            if (result.completed() && result.savedModelDirectory.has_value()) {
                // TrainingRuns consumes trained members through saved artifacts for
                // later phase handoff and ensemble composition. Once the artifact
                // is finalized, keep the CPU-side trainer/spec but release the
                // completed member placement so finished folds do not accumulate
                // unnecessary GPU residency while siblings or composed evaluators
                // run.
                runs[i].trainer->releasePlacedNetworkAfterLastFit();
            }

            bool shouldCancelSiblings = false;
            {
                std::lock_guard<std::mutex> lock(resultMutex);
                results[i] = result;
                shouldCancelSiblings = failurePolicy == TrainingRunsFailurePolicy::CANCEL_SIBLINGS &&
                                       result.failed() && failedRunShouldTriggerCancellation(i, results);
            }
            if (shouldCancelSiblings) {
                bool expected = false;
                if (cancellationRequestedByFailure.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
                    cancellationSource.requestCancellation();
                }
            }

            statsReporter->markRunFinished(result);
            {
                std::lock_guard<std::mutex> lock(schedulingMutex);
                workerFinished[i] = true;
            }
            schedulingChanged.notify_one();
        });
    };

    try {
        size_t nextRunIndex = 0;
        size_t activeRuns = 0;
        size_t completedOrSkippedRuns = 0;

        while (completedOrSkippedRuns < runs.size()) {
            while (nextRunIndex < runs.size() && activeRuns < maxActiveRuns && !cancellationSource.token().isCancellationRequested()) {
                launchWorker(nextRunIndex);
                ++nextRunIndex;
                ++activeRuns;
            }

            if (cancellationSource.token().isCancellationRequested() && failurePolicy == TrainingRunsFailurePolicy::CANCEL_SIBLINGS) {
                std::lock_guard<std::mutex> lock(schedulingMutex);
                for (size_t i = 0; i < runs.size(); ++i) {
                    if (workerStarted[i]) {
                        continue;
                    }
                    TrainingRunResult result;
                    result.runName = runs[i].runName;
                    result.ensembleGroup = runs[i].ensembleGroup;
                    result.ensembleWeight = runs[i].ensembleWeight;
                    result.savedModelDirectory = runs[i].trainer->getSaveModelDirectory();
                    if (runs[i].trainer->getNetwork() != nullptr) {
                        result.savedModelNetworkName = runs[i].trainer->getNetwork()->getNetworkName();
                    }
                    result.status = TrainingRunStatus::CANCELLED;
                    result.exception = TrainingRunExceptionSummary{"TrainingCancelled", "cancelled before launch by sibling failure"};
                    {
                        std::lock_guard<std::mutex> resultLock(resultMutex);
                        if (results[i].status == TrainingRunStatus::NOT_STARTED) {
                            results[i] = result;
                        }
                    }
                    statsReporter->markRunFinished(result);
                    workerStarted[i] = true;
                    workerFinished[i] = true;
                    workerJoined[i] = true;
                    ++completedOrSkippedRuns;
                }
                nextRunIndex = runs.size();
            }

            bool joinedAny = false;
            for (size_t i = 0; i < workers.size(); ++i) {
                bool shouldJoin = false;
                {
                    std::lock_guard<std::mutex> lock(schedulingMutex);
                    shouldJoin = workerStarted[i] && workerFinished[i] && !workerJoined[i];
                    if (shouldJoin) {
                        workerJoined[i] = true;
                    }
                }
                if (shouldJoin) {
                    if (workers[i].joinable()) {
                        workers[i].join();
                    }
                    --activeRuns;
                    ++completedOrSkippedRuns;
                    joinedAny = true;
                }
            }

            if (!joinedAny) {
                std::unique_lock<std::mutex> lock(schedulingMutex);
                schedulingChanged.wait(lock, [&]() {
                    if (cancellationSource.token().isCancellationRequested() && failurePolicy == TrainingRunsFailurePolicy::CANCEL_SIBLINGS) {
                        return true;
                    }
                    for (size_t i = 0; i < workerFinished.size(); ++i) {
                        if (workerStarted[i] && workerFinished[i] && !workerJoined[i]) {
                            return true;
                        }
                    }
                    return false;
                });
            }
        }
    } catch (...) {
        cancellationSource.requestCancellation();
        for (std::thread& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        statsReporter->close();
        throw;
    }

    statsReporter->flush();

    std::map<std::string, TrainingEnsembleResult> ensembleResultsByGroup = buildEnsembleResultsByGroup(results);
    if (evaluationOptions.evaluateTrainingPopulation) {
        evaluateEnsembles(results, ensembleResultsByGroup);
    }
    if (evaluationOptions.testData != nullptr) {
        evaluateEnsemblesOnTestData(results, ensembleResultsByGroup, evaluationOptions.testData);
    }

    statsReporter->flush();
    statsReporter->emitFinalReport(results);

    std::vector<TrainingEnsembleResult> ensembleResults;
    ensembleResults.reserve(ensembleResultsByGroup.size());
    for (auto& entry : ensembleResultsByGroup) {
        ensembleResults.push_back(std::move(entry.second));
    }

    statsReporter->flush();
    statsReporter->emitEnsembleReport(ensembleResults);
    statsReporter->close();

    return TrainingRunsResult(std::move(results), std::move(ensembleResults));
}

std::shared_ptr<Network> TrainingRuns::validationNetworkForSpec(const TrainingRunsSpec& spec) const {
    if (spec.trainer == nullptr) {
        return nullptr;
    }

    std::shared_ptr<Network> trainerNetwork = spec.trainer->getNetwork();
    std::shared_ptr<TrainingProgram> program = spec.trainer->trainingProgram;
    if (program == nullptr || !program->isInitialized() || !trainingProgramHasActivePhaseForValidation(*program)) {
        return trainerNetwork;
    }

    if (program->getNumSteps() != 1) {
        throw std::runtime_error("TrainingRuns run '" + spec.runName +
                                 "' uses TrainingPhase objects, but ensemble/report validation currently supports exactly one TrainingStep.");
    }

    const TrainingStep& step = program->getStep(0);
    const std::string context = "TrainingRuns run '" + spec.runName + "' TrainingPhase validation";
    std::vector<PhaseGraphNetworkSpec> phaseSpecs = step.getActivePhaseNetworkSpecs();
    if (phaseSpecs.empty()) {
        return trainerNetwork;
    }

    PhaseGraphComposeOptions composeOptions;
    const std::string baseName = trainerNetwork == nullptr ? spec.runName : trainerNetwork->getNetworkName();
    composeOptions.networkName = baseName + "_training_runs_validation_phases";
    composeOptions.inferenceOnly = false;
    composeOptions.exposePhaseOutputsAsNetworkOutputs = true;

    ComposedPhaseGraph graph = buildComposedPhaseGraphByName(phaseSpecs, composeOptions);
    if (graph.network == nullptr) {
        throw std::runtime_error(context + " produced a null validation Network.");
    }
    return graph.network;
}

std::vector<std::shared_ptr<Network>> TrainingRuns::reportingValidationNetworksForSpec(const TrainingRunsSpec& spec) const {
    if (spec.trainer == nullptr) {
        return {};
    }

    std::shared_ptr<TrainingProgram> program = spec.trainer->trainingProgram;
    if (program == nullptr || !program->isInitialized()) {
        std::shared_ptr<Network> network = validationNetworkForSpec(spec);
        return network == nullptr ? std::vector<std::shared_ptr<Network>>{} : std::vector<std::shared_ptr<Network>>{network};
    }

    std::vector<std::shared_ptr<Network>> phaseNetworks;
    bool sawPhaseBackedStep = false;
    for (const std::shared_ptr<TrainingStep>& step : program->getSteps()) {
        if (step == nullptr || !step->isInitialized()) {
            continue;
        }
        const std::vector<std::shared_ptr<TrainingPhase>>& phases = step->getPhases();
        if (phases.empty()) {
            continue;
        }
        sawPhaseBackedStep = true;
        for (const std::shared_ptr<TrainingPhase>& phase : phases) {
            if (phase == nullptr || !phase->isInitialized() || phase->getNetwork() == nullptr) {
                throw std::runtime_error("TrainingRuns run '" + spec.runName + "' has an invalid TrainingPhase in its reporting validation view.");
            }
            phaseNetworks.push_back(phase->getNetwork());
        }
    }

    if (sawPhaseBackedStep) {
        return phaseNetworks;
    }

    std::shared_ptr<Network> network = validationNetworkForSpec(spec);
    return network == nullptr ? std::vector<std::shared_ptr<Network>>{} : std::vector<std::shared_ptr<Network>>{network};
}

std::vector<NetworkLossReference> TrainingRuns::reportableLossesForSpec(const TrainingRunsSpec& spec) const {
    return collectNetworkReportableLosses(reportingValidationNetworksForSpec(spec));
}

std::vector<NetworkMetricReference> TrainingRuns::reportableMetricsForSpec(const TrainingRunsSpec& spec) const {
    return collectNetworkReportableMetrics(reportingValidationNetworksForSpec(spec));
}

std::vector<std::string> TrainingRuns::reportOrderForGroup(std::string_view ensembleGroup) const {
    const auto it = reports.find(std::string(ensembleGroup));
    return it == reports.end() ? std::vector<std::string>{} : it->second;
}

std::vector<std::string> TrainingRuns::reportedScalarTensorNamesForSpec(const TrainingRunsSpec& spec) const {
    std::shared_ptr<Network> activeNetwork = validationNetworkForSpec(spec);
    if (spec.trainer == nullptr || activeNetwork == nullptr) {
        return {};
    }

    std::vector<std::string> requestedReportNames;
    if (spec.ensembleGroup.has_value()) {
        const auto groupIt = reports.find(*spec.ensembleGroup);
        if (groupIt != reports.end()) {
            requestedReportNames = groupIt->second;
        }
    }
    if (!spec.ensembleGroup.has_value()) {
        const auto runIt = reports.find(spec.runName);
        if (runIt != reports.end()) {
            requestedReportNames = runIt->second;
        }
    }

    const std::vector<NetworkLossReference> activeLosses = activeNetwork->getReportableLosses();
    const std::vector<NetworkMetricReference> activeMetrics = activeNetwork->getReportableMetrics();
    const std::vector<std::string> activeRequestedReportNames =
        filterRequestedReportNamesToAvailable(activeLosses, activeMetrics, requestedReportNames);
    if (!requestedReportNames.empty() && activeRequestedReportNames.empty()) {
        return {};
    }

    const TrainingRunsReportNameSelections selections = splitTrainingRunsRequestedReportsByKind(
        activeLosses,
        activeMetrics,
        requestedReportNames.empty() ? requestedReportNames : activeRequestedReportNames,
        "TrainingRuns reports for run '" + spec.runName + "'");

    std::vector<std::string> names;
    if (!selections.lossNames.has_value() && !selections.metricNames.has_value()) {
        const std::vector<ResolvedEnsembleLoss> resolvedLosses = resolveTrainingRunsReportedLosses(
            activeLosses, {}, "TrainingRuns reports for run '" + spec.runName + "'");
        const std::vector<ResolvedEnsembleMetric> resolvedMetrics = resolveTrainingRunsReportedMetrics(
            activeMetrics, {}, "TrainingRuns reports for run '" + spec.runName + "'");
        names.reserve(resolvedLosses.size() + resolvedMetrics.size());
        for (const ResolvedEnsembleLoss& loss : resolvedLosses) {
            names.push_back(loss.lossName);
        }
        for (const ResolvedEnsembleMetric& metric : resolvedMetrics) {
            names.push_back(metric.metricName);
        }
        return names;
    }

    names.reserve(activeRequestedReportNames.size());
    for (const std::string& reportName : activeRequestedReportNames) {
        if (trainingRunsReportNameExists(activeLosses, activeMetrics, reportName)) {
            names.push_back(reportName);
        }
    }
    return names;
}


void TrainingRuns::validateRunSpecs() const {
    if (runs.empty()) {
        throw std::runtime_error("TrainingRuns requires at least one run.");
    }

    std::set<std::string> runNames;
    std::set<const Trainer*> trainers;
    std::map<std::string, std::string> saveModelDirectories;
    struct EnsembleValidationState {
        std::string firstRunName{};
        std::vector<TrainingRunInputSignature> inputSignature{};
        std::vector<TrainingRunOutputSignature> outputSignature{};
        bool reportsGraphLosses = false;
    };
    std::map<std::string, EnsembleValidationState> ensembleSignatures;

    for (size_t i = 0; i < runs.size(); ++i) {
        const TrainingRunsSpec& spec = runs[i];
        if (spec.runName.empty()) {
            throw std::runtime_error("TrainingRuns run at index " + std::to_string(i) + " has an empty name.");
        }
        if (spec.trainer == nullptr) {
            throw std::runtime_error("TrainingRuns run '" + spec.runName + "' has a null trainer.");
        }
        if (!std::isfinite(spec.ensembleWeight) || spec.ensembleWeight <= 0.0) {
            throw std::runtime_error("TrainingRuns run '" + spec.runName + "' has invalid ensemble_weight; it must be finite and > 0.");
        }
        if (spec.ensembleGroup.has_value() && spec.ensembleGroup->empty()) {
            throw std::runtime_error("TrainingRuns run '" + spec.runName + "' has an empty ensemble_group.");
        }
        if (!runNames.insert(spec.runName).second) {
            throw std::runtime_error("TrainingRuns contains duplicate run name '" + spec.runName + "'.");
        }
        if (!trainers.insert(spec.trainer.get()).second) {
            throw std::runtime_error("TrainingRuns run '" + spec.runName + "' reuses a Trainer that is already present.");
        }

        const std::optional<std::string>& saveModelDirectory = spec.trainer->getSaveModelDirectory();
        if (saveModelDirectory.has_value()) {
            if (saveModelDirectory->empty()) {
                throw std::runtime_error("TrainingRuns run '" + spec.runName + "' has an empty trainer save_model_dir.");
            }
            const std::string normalizedDirectory = normalizedOutputPathForCollisionCheck(*saveModelDirectory);
            auto [it, inserted] = saveModelDirectories.emplace(normalizedDirectory, spec.runName);
            if (!inserted) {
                throw std::runtime_error("TrainingRuns save_model_dir collision: runs '" + it->second + "' and '" + spec.runName +
                                         "' both save model output to '" + normalizedDirectory +
                                         "'. Give each trainer a distinct save_model_dir or disable saving for one of them.");
            }
        }

        if (spec.ensembleGroup.has_value()) {
            const std::vector<TrainingRunInputSignature> inputSignature = collectNetworkInputSignature(validationNetworkForSpec(spec));
            const std::vector<TrainingRunOutputSignature> outputSignature = collectNetworkOutputSignature(validationNetworkForSpec(spec));
            if (inputSignature.empty()) {
                throw std::runtime_error("TrainingRuns run '" + spec.runName + "' is in ensemble_group '" + *spec.ensembleGroup +
                                         "' but its network has no NetworkInput layers to validate for ensemble evaluation.");
            }
            if (outputSignature.empty()) {
                throw std::runtime_error("TrainingRuns run '" + spec.runName + "' is in ensemble_group '" + *spec.ensembleGroup +
                                         "' but its network has no NetworkOutput layers to ensemble.");
            }
            if (!outputSignatureHasPredictionTensor(outputSignature)) {
                throw std::runtime_error("TrainingRuns run '" + spec.runName + "' is in ensemble_group '" + *spec.ensembleGroup +
                                         "' but its network has no non-loss NetworkOutput prediction tensor to ensemble.");
            }

            const std::vector<NetworkLossReference> reportableLosses = reportableLossesForSpec(spec);
            const std::vector<NetworkMetricReference> reportableMetrics = reportableMetricsForSpec(spec);
            const auto reportsIt = reports.find(*spec.ensembleGroup);
            const std::vector<std::string> requestedReportNames = reportsIt == reports.end() ? std::vector<std::string>{} : reportsIt->second;
            const TrainingRunsReportNameSelections reportSelections = splitTrainingRunsRequestedReportsByKind(
                reportableLosses,
                reportableMetrics,
                requestedReportNames,
                "TrainingRuns reports for ensemble_group '" + *spec.ensembleGroup + "' run '" + spec.runName + "'");
            const bool reportsGraphLosses = !resolveTrainingRunsSelectedLossReports(
                reportableLosses,
                reportSelections.lossNames,
                "TrainingRuns reports for ensemble_group '" + *spec.ensembleGroup + "' run '" + spec.runName + "'")
                                                 .empty();

            auto [it, inserted] = ensembleSignatures.emplace(
                *spec.ensembleGroup, EnsembleValidationState{spec.runName, inputSignature, outputSignature, reportsGraphLosses});
            if (!inserted && !inputSignaturesCompatible(it->second.inputSignature, inputSignature)) {
                throw std::runtime_error("TrainingRuns ensemble_group '" + *spec.ensembleGroup + "' has incompatible input signatures: run '" +
                                         it->second.firstRunName + "' has " + inputSignatureToString(it->second.inputSignature) +
                                         ", but run '" + spec.runName + "' has " + inputSignatureToString(inputSignature) + ".");
            }
            if (!inserted && it->second.reportsGraphLosses != reportsGraphLosses) {
                throw std::runtime_error("TrainingRuns ensemble_group '" + *spec.ensembleGroup +
                                         "' mixes runs with and without reportable graph losses; run '" + it->second.firstRunName +
                                         "' and run '" + spec.runName + "' must use compatible loss reporting configuration.");
            }
            if (!inserted && !it->second.reportsGraphLosses && !outputSignaturesCompatible(it->second.outputSignature, outputSignature)) {
                throw std::runtime_error("TrainingRuns ensemble_group '" + *spec.ensembleGroup + "' has incompatible output signatures: run '" +
                                         it->second.firstRunName + "' has " + outputSignatureToString(it->second.outputSignature) +
                                         ", but run '" + spec.runName + "' has " + outputSignatureToString(outputSignature) + ".");
            }
        }
    }
}


void TrainingRuns::validateMinSuccessfulModels() const {
    if (minSuccessfulModels.empty()) {
        return;
    }

    std::map<std::string, size_t> ensembleGroupSizes;
    for (const TrainingRunsSpec& spec : runs) {
        if (spec.ensembleGroup.has_value()) {
            ensembleGroupSizes[*spec.ensembleGroup] += 1;
        }
    }

    for (const auto& [groupName, minimum] : minSuccessfulModels) {
        if (groupName.empty()) {
            throw std::runtime_error("TrainingRuns min_successful_models contains an empty ensemble_group name.");
        }
        const auto groupIt = ensembleGroupSizes.find(groupName);
        if (groupIt == ensembleGroupSizes.end()) {
            throw std::runtime_error("TrainingRuns min_successful_models targets unknown ensemble_group '" + groupName + "'.");
        }
        if (minimum == 0) {
            throw std::runtime_error("TrainingRuns min_successful_models for ensemble_group '" + groupName + "' must be >= 1.");
        }
        if (minimum > groupIt->second) {
            throw std::runtime_error("TrainingRuns min_successful_models for ensemble_group '" + groupName + "' is " +
                                     std::to_string(minimum) + ", but the ensemble only has " +
                                     std::to_string(groupIt->second) + " member(s).");
        }
    }
}

size_t TrainingRuns::minSuccessfulModelsForGroup(std::string_view ensembleGroup, size_t defaultValue) const {
    const auto it = minSuccessfulModels.find(std::string(ensembleGroup));
    if (it == minSuccessfulModels.end()) {
        return defaultValue;
    }
    return it->second;
}

bool TrainingRuns::failedRunShouldTriggerCancellation(size_t runIndex, const std::vector<TrainingRunResult>& results) const {
    if (runIndex >= runs.size()) {
        return true;
    }
    const TrainingRunsSpec& failedSpec = runs[runIndex];
    if (!failedSpec.ensembleGroup.has_value()) {
        return true;
    }

    const auto minimumIt = minSuccessfulModels.find(*failedSpec.ensembleGroup);
    if (minimumIt == minSuccessfulModels.end()) {
        return true;
    }

    const size_t requiredSuccesses = minimumIt->second;
    size_t possibleSuccesses = 0;
    for (size_t i = 0; i < runs.size(); ++i) {
        const TrainingRunsSpec& spec = runs[i];
        if (!spec.ensembleGroup.has_value() || *spec.ensembleGroup != *failedSpec.ensembleGroup) {
            continue;
        }
        if (i >= results.size()) {
            possibleSuccesses += 1;
            continue;
        }
        const TrainingRunStatus status = results[i].status;
        if (status == TrainingRunStatus::COMPLETED || status == TrainingRunStatus::RUNNING ||
            status == TrainingRunStatus::NOT_STARTED) {
            possibleSuccesses += 1;
        }
    }

    return possibleSuccesses < requiredSuccesses;
}



void TrainingRuns::validateRestartConditions() const {
    std::set<std::string> runNames;
    std::set<std::string> ensembleGroups;
    for (const TrainingRunsSpec& run : runs) {
        runNames.insert(run.runName);
        if (run.ensembleGroup.has_value()) {
            ensembleGroups.insert(*run.ensembleGroup);
        }
    }

    for (size_t i = 0; i < restartConditions.size(); ++i) {
        const TrainingRunsRestartPolicy& condition = restartConditions[i];
        const bool hasRunName = condition.runName.has_value();
        const bool hasEnsembleGroup = condition.ensembleGroup.has_value();
        if (hasRunName && hasEnsembleGroup) {
            throw std::runtime_error("TrainingRuns restart_condition at index " + std::to_string(i) +
                                     " cannot specify both run_name and ensemble_group.");
        }
        if (condition.progressCheckEpochs == 0) {
            throw std::runtime_error("TrainingRuns restart_condition at index " + std::to_string(i) +
                                     " must have progress_check_epochs >= 1.");
        }
        if (!std::isfinite(condition.progressImprovementMinPercentage) || condition.progressImprovementMinPercentage < 0.0 || condition.progressImprovementMinPercentage > 100.0) {
            throw std::runtime_error("TrainingRuns restart_condition at index " + std::to_string(i) +
                                     " must have progress_improvement_min_percentage in [0, 100].");
        }

        if (!hasRunName && !hasEnsembleGroup) {
            // Untargeted policies are global and apply to every run.
            continue;
        }

        if (hasRunName) {
            if (condition.runName->empty()) {
                throw std::runtime_error("TrainingRuns restart_condition at index " + std::to_string(i) + " has an empty run_name.");
            }
            if (runNames.count(*condition.runName) == 0) {
                throw std::runtime_error("TrainingRuns restart_condition targets unknown run_name '" + *condition.runName + "'.");
            }
            continue;
        }

        if (condition.ensembleGroup->empty()) {
            throw std::runtime_error("TrainingRuns restart_condition at index " + std::to_string(i) + " has an empty ensemble_group.");
        }
        if (ensembleGroups.count(*condition.ensembleGroup) == 0) {
            throw std::runtime_error("TrainingRuns restart_condition targets unknown ensemble_group '" + *condition.ensembleGroup + "'.");
        }
    }
}

std::vector<TrainingRestartCondition> TrainingRuns::restartConditionsForRun(const TrainingRunsSpec& run) const {
    std::vector<TrainingRestartCondition> matches;
    for (const TrainingRunsRestartPolicy& condition : restartConditions) {
        if (!condition.runName.has_value() && !condition.ensembleGroup.has_value()) {
            matches.push_back(condition.withoutTarget());
        } else if (condition.runName.has_value() && *condition.runName == run.runName) {
            matches.push_back(condition.withoutTarget());
        } else if (condition.ensembleGroup.has_value() && run.ensembleGroup.has_value() && *condition.ensembleGroup == *run.ensembleGroup) {
            matches.push_back(condition.withoutTarget());
        }
    }
    return matches;
}

void TrainingRuns::validateEarlyCompletionRules() const {
    std::set<std::string> runNames;
    std::set<std::string> ensembleGroups;
    for (const TrainingRunsSpec& run : runs) {
        runNames.insert(run.runName);
        if (run.ensembleGroup.has_value()) {
            ensembleGroups.insert(*run.ensembleGroup);
        }
    }

    for (size_t i = 0; i < earlyCompletionRules.size(); ++i) {
        const TrainingRunsEarlyCompletionRule& rule = earlyCompletionRules[i];
        const bool hasRunName = rule.runName.has_value();
        const bool hasEnsembleGroup = rule.ensembleGroup.has_value();
        if (hasRunName == hasEnsembleGroup) {
            throw std::runtime_error("TrainingRuns early_completion_rule at index " + std::to_string(i) +
                                     " must specify exactly one of run_name or ensemble_group.");
        }
        if (!rule.completionCondition) {
            throw std::runtime_error("TrainingRuns early_completion_rule at index " + std::to_string(i) +
                                     " must have a completion_condition.");
        }

        if (hasRunName) {
            if (rule.runName->empty()) {
                throw std::runtime_error("TrainingRuns early_completion_rule at index " + std::to_string(i) + " has an empty run_name.");
            }
            if (runNames.count(*rule.runName) == 0) {
                throw std::runtime_error("TrainingRuns early_completion_rule targets unknown run_name '" + *rule.runName + "'.");
            }
            continue;
        }

        if (rule.ensembleGroup->empty()) {
            throw std::runtime_error("TrainingRuns early_completion_rule at index " + std::to_string(i) + " has an empty ensemble_group.");
        }
        if (ensembleGroups.count(*rule.ensembleGroup) == 0) {
            throw std::runtime_error("TrainingRuns early_completion_rule targets unknown ensemble_group '" + *rule.ensembleGroup + "'.");
        }
    }
}

std::vector<TrainingEarlyCompletionPolicy> TrainingRuns::earlyCompletionPoliciesForRun(const TrainingRunsSpec& run) const {
    std::vector<TrainingEarlyCompletionPolicy> matches;
    for (const TrainingRunsEarlyCompletionRule& rule : earlyCompletionRules) {
        if (rule.runName.has_value() && *rule.runName == run.runName) {
            matches.push_back(rule.toEarlyCompletionPolicy());
        } else if (rule.ensembleGroup.has_value() && run.ensembleGroup.has_value() && *rule.ensembleGroup == *run.ensembleGroup) {
            matches.push_back(rule.toEarlyCompletionPolicy());
        }
    }
    return matches;
}

void TrainingRuns::validateReportedLosses() const {
    struct EnsembleMemberSignatureState {
        std::string runName{};
        std::vector<TrainingRunInputSignature> inputSignature{};
        std::vector<TrainingRunOutputSignature> outputSignature{};
        std::vector<std::shared_ptr<Network>> reportingNetworks{};
        std::optional<std::vector<NetworkLossReference>> reportableLosses{};
    };

    std::map<std::string, std::vector<EnsembleMemberSignatureState>> membersByGroup;
    for (const TrainingRunsSpec& spec : runs) {
        if (!spec.ensembleGroup.has_value()) {
            continue;
        }
        EnsembleMemberSignatureState state;
        state.runName = spec.runName;
        state.reportingNetworks = reportingValidationNetworksForSpec(spec);
        const std::string context = "TrainingRuns reports for ensemble_group '" + *spec.ensembleGroup + "' run '" + spec.runName + "'";
        state.inputSignature = collectMergedNetworkInputSignature(state.reportingNetworks, context);
        state.outputSignature = collectMergedNetworkOutputSignature(state.reportingNetworks, context);
        membersByGroup[*spec.ensembleGroup].push_back(std::move(state));
    }

    auto reportableLossesForMember = [](EnsembleMemberSignatureState& member) -> const std::vector<NetworkLossReference>& {
        if (!member.reportableLosses.has_value()) {
            member.reportableLosses = collectNetworkReportableLosses(member.reportingNetworks);
        }
        return *member.reportableLosses;
    };

    auto reportableMetricsForMember = [](EnsembleMemberSignatureState& member) -> std::vector<NetworkMetricReference> {
        return collectNetworkReportableMetrics(member.reportingNetworks);
    };

    for (auto& [groupName, members] : membersByGroup) {
        if (members.empty()) {
            continue;
        }
        EnsembleMemberSignatureState& referenceMember = members.front();
        const auto reportsIt = reports.find(groupName);
        const std::vector<std::string> requestedReportNames = reportsIt == reports.end() ? std::vector<std::string>{} : reportsIt->second;
        const std::string context = "TrainingRuns reports for ensemble_group '" + groupName + "'";
        const std::vector<NetworkLossReference>& referenceAvailableLosses = reportableLossesForMember(referenceMember);
        const std::vector<NetworkMetricReference> referenceAvailableMetrics = reportableMetricsForMember(referenceMember);
        const TrainingRunsReportNameSelections referenceSelections = splitTrainingRunsRequestedReportsByKind(
            referenceAvailableLosses,
            referenceAvailableMetrics,
            requestedReportNames,
            context + " reference run '" + referenceMember.runName + "'");
        const std::vector<ResolvedEnsembleLoss> referenceLosses = resolveTrainingRunsSelectedLossReports(
            referenceAvailableLosses, referenceSelections.lossNames, context + " reference run '" + referenceMember.runName + "'");

        for (const ResolvedEnsembleLoss& resolved : referenceLosses) {
            const TrainingRunOutputSignature* referenceOutput = nullptr;
            if (!resolved.predictionOutputName.empty()) {
                referenceOutput = findOutputSignatureItem(referenceMember.outputSignature, resolved.predictionOutputName);
                if (referenceOutput == nullptr) {
                    throw std::runtime_error(context + " resolved loss '" + resolved.lossName + "' to prediction output '" +
                                             resolved.predictionOutputName + "', but reference run '" + referenceMember.runName +
                                             "' has outputs " + outputSignatureToString(referenceMember.outputSignature) + ".");
                }
            }
            const TrainingRunInputSignature* referenceTarget = findInputSignatureItem(referenceMember.inputSignature, resolved.targetInputName);
            if (referenceTarget == nullptr) {
                throw std::runtime_error(context + " resolved loss '" + resolved.lossName + "' to target input '" +
                                         resolved.targetInputName + "', but reference run '" + referenceMember.runName +
                                         "' has inputs " + inputSignatureToString(referenceMember.inputSignature) + ".");
            }
            const TrainingRunInputSignature* referenceWeight = nullptr;
            if (resolved.weightInputName.has_value()) {
                referenceWeight = findInputSignatureItem(referenceMember.inputSignature, *resolved.weightInputName);
                if (referenceWeight == nullptr) {
                    throw std::runtime_error(context + " resolved loss '" + resolved.lossName + "' to weight input '" +
                                             *resolved.weightInputName + "', but reference run '" + referenceMember.runName +
                                             "' has inputs " + inputSignatureToString(referenceMember.inputSignature) + ".");
                }
            }

            for (size_t memberIndex = 1; memberIndex < members.size(); ++memberIndex) {
                EnsembleMemberSignatureState& member = members[memberIndex];
                const std::string memberContext = context + " run '" + member.runName + "'";
                const std::vector<NetworkLossReference>& memberAvailableLosses = reportableLossesForMember(member);
                const std::vector<NetworkMetricReference> memberAvailableMetrics = reportableMetricsForMember(member);
                const TrainingRunsReportNameSelections memberSelections = splitTrainingRunsRequestedReportsByKind(
                    memberAvailableLosses,
                    memberAvailableMetrics,
                    requestedReportNames,
                    memberContext);
                const std::vector<ResolvedEnsembleLoss> memberLosses =
                    resolveTrainingRunsSelectedLossReports(memberAvailableLosses, memberSelections.lossNames, memberContext);
                auto memberResolvedIt = std::find_if(memberLosses.begin(), memberLosses.end(), [&](const ResolvedEnsembleLoss& candidate) {
                    return candidate.lossName == resolved.lossName;
                });
                if (memberResolvedIt == memberLosses.end()) {
                    throw std::runtime_error(memberContext + " is missing graph loss '" + resolved.lossName + "'.");
                }
                NetworkLossReference reference;
                reference.lossName = resolved.lossName;
                reference.predictionOutputName = resolved.predictionOutputName;
                reference.targetInputName = resolved.targetInputName;
                reference.weightInputName = resolved.weightInputName;
                reference.lossLayerType = resolved.lossType;
                reference.lossWeight = resolved.lossWeight;
                reference.quantile = resolved.quantile;
                NetworkLossReference memberReference;
                memberReference.lossName = memberResolvedIt->lossName;
                memberReference.predictionOutputName = memberResolvedIt->predictionOutputName;
                memberReference.targetInputName = memberResolvedIt->targetInputName;
                memberReference.weightInputName = memberResolvedIt->weightInputName;
                memberReference.lossLayerType = memberResolvedIt->lossType;
                memberReference.lossWeight = memberResolvedIt->lossWeight;
                memberReference.quantile = memberResolvedIt->quantile;
                if (!trainingRunsReportableLossesCompatible(reference, memberReference)) {
                    throw std::runtime_error(memberContext + " resolved graph loss '" + resolved.lossName +
                                             "' differently than reference run '" + referenceMember.runName + "'. Reference: " +
                                             trainingRunsReportableLossDescription(reference) + "; member: " +
                                             trainingRunsReportableLossDescription(memberReference) + ".");
                }

                if (referenceOutput != nullptr) {
                    const TrainingRunOutputSignature* memberOutput = findOutputSignatureItem(member.outputSignature, resolved.predictionOutputName);
                    if (memberOutput == nullptr || !referenceOutput->compatibleWith(*memberOutput)) {
                        throw std::runtime_error(memberContext + " has incompatible prediction output '" + resolved.predictionOutputName + "'.");
                    }
                }
                const TrainingRunInputSignature* memberTarget = findInputSignatureItem(member.inputSignature, resolved.targetInputName);
                if (memberTarget == nullptr || !referenceTarget->compatibleWith(*memberTarget)) {
                    throw std::runtime_error(memberContext + " has incompatible target input '" + resolved.targetInputName + "'.");
                }
                if (referenceWeight != nullptr) {
                    const TrainingRunInputSignature* memberWeight = findInputSignatureItem(member.inputSignature, *resolved.weightInputName);
                    if (memberWeight == nullptr || !referenceWeight->compatibleWith(*memberWeight)) {
                        throw std::runtime_error(memberContext + " has incompatible weight input '" + *resolved.weightInputName + "'.");
                    }
                }
            }
        }
    }
}

void TrainingRuns::validateReportedMetrics() const {
    struct MemberMetricState {
        std::string runName{};
        std::optional<std::string> ensembleGroup{};
        std::vector<TrainingRunInputSignature> inputSignature{};
        std::vector<TrainingRunOutputSignature> outputSignature{};
        std::vector<std::shared_ptr<Network>> reportingNetworks{};
        std::optional<std::vector<NetworkMetricReference>> reportableMetrics{};
    };

    std::set<std::string> runNames;
    std::set<std::string> ensembleGroups;
    std::map<std::string, std::vector<MemberMetricState*>> membersByGroup;
    std::vector<MemberMetricState> memberStates;
    memberStates.reserve(runs.size());
    for (const TrainingRunsSpec& spec : runs) {
        runNames.insert(spec.runName);
        if (spec.ensembleGroup.has_value()) {
            ensembleGroups.insert(*spec.ensembleGroup);
        }
        MemberMetricState state;
        state.runName = spec.runName;
        state.ensembleGroup = spec.ensembleGroup;
        state.reportingNetworks = reportingValidationNetworksForSpec(spec);
        const std::string context = "TrainingRuns reports for run '" + spec.runName + "'";
        state.inputSignature = collectMergedNetworkInputSignature(state.reportingNetworks, context);
        state.outputSignature = collectMergedNetworkOutputSignature(state.reportingNetworks, context);
        memberStates.push_back(std::move(state));
    }
    for (MemberMetricState& state : memberStates) {
        if (state.ensembleGroup.has_value()) {
            membersByGroup[*state.ensembleGroup].push_back(&state);
        }
    }

    for (const auto& [targetName, requestedReportNames] : reports) {
        if (targetName.empty()) {
            throw std::runtime_error("TrainingRuns reports contains an empty run_name/ensemble_group name.");
        }
        if (runNames.count(targetName) == 0 && ensembleGroups.count(targetName) == 0) {
            throw std::runtime_error("TrainingRuns reports targets unknown run_name or ensemble_group '" + targetName + "'.");
        }
        std::set<std::string> seen;
        for (const std::string& reportName : requestedReportNames) {
            if (reportName.empty()) {
                throw std::runtime_error("TrainingRuns reports for '" + targetName + "' contains an empty report name.");
            }
            if (!seen.insert(reportName).second) {
                throw std::runtime_error("TrainingRuns reports for '" + targetName + "' contains duplicate report name '" + reportName + "'.");
            }
        }
    }

    auto reportableMetricsForMember = [](MemberMetricState& member) -> const std::vector<NetworkMetricReference>& {
        if (!member.reportableMetrics.has_value()) {
            member.reportableMetrics = collectNetworkReportableMetrics(member.reportingNetworks);
        }
        return *member.reportableMetrics;
    };

    for (MemberMetricState& state : memberStates) {
        std::vector<std::string> requestedReportNames;
        if (state.ensembleGroup.has_value()) {
            const auto groupIt = reports.find(*state.ensembleGroup);
            if (groupIt != reports.end()) {
                requestedReportNames = groupIt->second;
            }
        }
        const auto runIt = reports.find(state.runName);
        if (!state.ensembleGroup.has_value() && runIt != reports.end()) {
            requestedReportNames = runIt->second;
        }
        const std::string context = "TrainingRuns reports for run '" + state.runName + "'";
        const std::vector<NetworkMetricReference>& availableMetrics = reportableMetricsForMember(state);
        const TrainingRunsReportNameSelections selections = splitTrainingRunsRequestedReportsByKind(
            collectNetworkReportableLosses(state.reportingNetworks),
            availableMetrics,
            requestedReportNames,
            context);
        const std::vector<ResolvedEnsembleMetric> resolvedMetrics = resolveTrainingRunsSelectedMetricReports(
            availableMetrics, selections.metricNames, context);
        for (const ResolvedEnsembleMetric& resolved : resolvedMetrics) {
            if (!resolved.predictionOutputName.empty()) {
                const TrainingRunOutputSignature* referenceOutput = findOutputSignatureItem(state.outputSignature, resolved.predictionOutputName);
                if (referenceOutput == nullptr) {
                    throw std::runtime_error(context + " resolved metric '" + resolved.metricName + "' to prediction output '" +
                                             resolved.predictionOutputName + "', but run has outputs " + outputSignatureToString(state.outputSignature) + ".");
                }
            }
            if (resolved.targetInputName.has_value() && findInputSignatureItem(state.inputSignature, *resolved.targetInputName) == nullptr) {
                throw std::runtime_error(context + " resolved metric '" + resolved.metricName + "' to target input '" +
                                         *resolved.targetInputName + "', but run has inputs " + inputSignatureToString(state.inputSignature) + ".");
            }
            if (resolved.inputSourceName.has_value() && findInputSignatureItem(state.inputSignature, *resolved.inputSourceName) == nullptr) {
                throw std::runtime_error(context + " resolved metric '" + resolved.metricName + "' to input source '" +
                                         *resolved.inputSourceName + "', but run has inputs " + inputSignatureToString(state.inputSignature) + ".");
            }
        }
    }

    for (auto& [groupName, members] : membersByGroup) {
        if (members.empty()) {
            continue;
        }
        MemberMetricState& referenceMember = *members.front();
        const auto reportsIt = reports.find(groupName);
        const std::vector<std::string> requestedReportNames = reportsIt == reports.end() ? std::vector<std::string>{} : reportsIt->second;
        const std::string context = "TrainingRuns reports for ensemble_group '" + groupName + "'";
        const std::vector<NetworkMetricReference>& referenceAvailableMetrics = reportableMetricsForMember(referenceMember);
        const TrainingRunsReportNameSelections referenceSelections = splitTrainingRunsRequestedReportsByKind(
            collectNetworkReportableLosses(referenceMember.reportingNetworks),
            referenceAvailableMetrics,
            requestedReportNames,
            context + " reference run '" + referenceMember.runName + "'");
        const std::vector<ResolvedEnsembleMetric> referenceMetrics = resolveTrainingRunsSelectedMetricReports(
            referenceAvailableMetrics, referenceSelections.metricNames, context + " reference run '" + referenceMember.runName + "'");

        for (const ResolvedEnsembleMetric& resolved : referenceMetrics) {
            const TrainingRunOutputSignature* referenceOutput = resolved.predictionOutputName.empty()
                ? nullptr
                : findOutputSignatureItem(referenceMember.outputSignature, resolved.predictionOutputName);
            const TrainingRunInputSignature* referenceTarget = resolved.targetInputName.has_value()
                ? findInputSignatureItem(referenceMember.inputSignature, *resolved.targetInputName)
                : nullptr;
            const TrainingRunInputSignature* referenceInputSource = resolved.inputSourceName.has_value()
                ? findInputSignatureItem(referenceMember.inputSignature, *resolved.inputSourceName)
                : nullptr;
            for (size_t memberIndex = 1; memberIndex < members.size(); ++memberIndex) {
                MemberMetricState& member = *members[memberIndex];
                const std::string memberContext = context + " run '" + member.runName + "'";
                const std::vector<NetworkMetricReference>& memberAvailableMetrics = reportableMetricsForMember(member);
                const TrainingRunsReportNameSelections memberSelections = splitTrainingRunsRequestedReportsByKind(
                    collectNetworkReportableLosses(member.reportingNetworks),
                    memberAvailableMetrics,
                    requestedReportNames,
                    memberContext);
                const std::vector<ResolvedEnsembleMetric> memberMetrics = resolveTrainingRunsSelectedMetricReports(
                    memberAvailableMetrics, memberSelections.metricNames, memberContext);
                auto memberResolvedIt = std::find_if(memberMetrics.begin(), memberMetrics.end(), [&](const ResolvedEnsembleMetric& candidate) {
                    return candidate.metricName == resolved.metricName;
                });
                if (memberResolvedIt == memberMetrics.end()) {
                    throw std::runtime_error(memberContext + " is missing graph metric '" + resolved.metricName + "'.");
                }
                NetworkMetricReference reference;
                reference.metricName = resolved.metricName;
                reference.predictionOutputName = resolved.predictionOutputName;
                reference.targetInputName = resolved.targetInputName;
                reference.inputSourceName = resolved.inputSourceName;
                reference.metricLayerType = resolved.metricType;
                NetworkMetricReference memberReference;
                memberReference.metricName = memberResolvedIt->metricName;
                memberReference.predictionOutputName = memberResolvedIt->predictionOutputName;
                memberReference.targetInputName = memberResolvedIt->targetInputName;
                memberReference.inputSourceName = memberResolvedIt->inputSourceName;
                memberReference.metricLayerType = memberResolvedIt->metricType;
                if (!trainingRunsReportableMetricsCompatible(reference, memberReference)) {
                    throw std::runtime_error(memberContext + " resolved graph metric '" + resolved.metricName +
                                             "' differently than reference run '" + referenceMember.runName + "'. Reference: " +
                                             trainingRunsReportableMetricDescription(reference) + "; member: " +
                                             trainingRunsReportableMetricDescription(memberReference) + ".");
                }

                if (!resolved.predictionOutputName.empty()) {
                    const TrainingRunOutputSignature* memberOutput = findOutputSignatureItem(member.outputSignature, resolved.predictionOutputName);
                    if (referenceOutput != nullptr && (memberOutput == nullptr || !referenceOutput->compatibleWith(*memberOutput))) {
                        throw std::runtime_error(memberContext + " has incompatible prediction output '" + resolved.predictionOutputName + "'.");
                    }
                }
                if (referenceTarget != nullptr) {
                    const TrainingRunInputSignature* memberTarget = findInputSignatureItem(member.inputSignature, *resolved.targetInputName);
                    if (memberTarget == nullptr || !referenceTarget->compatibleWith(*memberTarget)) {
                        throw std::runtime_error(memberContext + " has incompatible target input '" + *resolved.targetInputName + "'.");
                    }
                }
                if (referenceInputSource != nullptr) {
                    const TrainingRunInputSignature* memberInputSource = findInputSignatureItem(member.inputSignature, *resolved.inputSourceName);
                    if (memberInputSource == nullptr || !referenceInputSource->compatibleWith(*memberInputSource)) {
                        throw std::runtime_error(memberContext + " has incompatible input source '" + *resolved.inputSourceName + "'.");
                    }
                }
            }
        }
    }
}

std::vector<TrainingNamedMetricResult> TrainingRuns::namedGraphMetricResultsForGroup(std::string_view ensembleGroup) const {
    std::shared_ptr<Network> representativeNetwork;
    for (const TrainingRunsSpec& run : runs) {
        if (run.ensembleGroup.has_value() && std::string_view(run.ensembleGroup.value()) == ensembleGroup && run.trainer != nullptr &&
            validationNetworkForSpec(run) != nullptr) {
            representativeNetwork = validationNetworkForSpec(run);
            break;
        }
    }
    if (representativeNetwork == nullptr) {
        return {};
    }

    const std::vector<std::string> requestedReportNames = reportOrderForGroup(ensembleGroup);
    const std::vector<NetworkLossReference> activeLosses = representativeNetwork->getReportableLosses();
    const std::vector<NetworkMetricReference> activeMetrics = representativeNetwork->getReportableMetrics();
    const std::vector<std::string> activeRequestedReportNames =
        filterRequestedReportNamesToAvailable(activeLosses, activeMetrics, requestedReportNames);
    if (!requestedReportNames.empty() && activeRequestedReportNames.empty()) {
        return {};
    }
    const TrainingRunsReportNameSelections selections = splitTrainingRunsRequestedReportsByKind(
        activeLosses,
        activeMetrics,
        requestedReportNames.empty() ? requestedReportNames : activeRequestedReportNames,
        "TrainingRuns reports for ensemble_group '" + std::string(ensembleGroup) + "'");
    const std::vector<ResolvedEnsembleMetric> resolvedMetrics = resolveTrainingRunsSelectedMetricReports(
        activeMetrics,
        selections.metricNames,
        "TrainingRuns reports for ensemble_group '" + std::string(ensembleGroup) + "'");

    std::vector<TrainingNamedMetricResult> results;
    results.reserve(resolvedMetrics.size());
    for (const ResolvedEnsembleMetric& metric : resolvedMetrics) {
        if (!trainingRunsMetricCanParticipateInComposedEvaluation(metric)) {
            continue;
        }
        TrainingNamedMetricResult result;
        result.name = metric.metricName;
        results.push_back(std::move(result));
    }
    return results;
}

std::vector<TrainingNamedMetricResult> TrainingRuns::namedMetricResultsForGroup(std::string_view ensembleGroup) const {
    std::shared_ptr<Network> representativeNetwork;
    for (const TrainingRunsSpec& run : runs) {
        if (run.ensembleGroup.has_value() && std::string_view(run.ensembleGroup.value()) == ensembleGroup && run.trainer != nullptr &&
            validationNetworkForSpec(run) != nullptr) {
            representativeNetwork = validationNetworkForSpec(run);
            break;
        }
    }
    if (representativeNetwork == nullptr) {
        return {};
    }

    const std::vector<std::string> requestedReportNames = reportOrderForGroup(ensembleGroup);
    const std::vector<NetworkLossReference> activeLosses = representativeNetwork->getReportableLosses();
    const std::vector<NetworkMetricReference> activeMetrics = representativeNetwork->getReportableMetrics();
    const std::vector<std::string> activeRequestedReportNames =
        filterRequestedReportNamesToAvailable(activeLosses, activeMetrics, requestedReportNames);
    if (!requestedReportNames.empty() && activeRequestedReportNames.empty()) {
        return {};
    }
    const TrainingRunsReportNameSelections selections = splitTrainingRunsRequestedReportsByKind(
        activeLosses,
        activeMetrics,
        requestedReportNames.empty() ? requestedReportNames : activeRequestedReportNames,
        "TrainingRuns reports for ensemble_group '" + std::string(ensembleGroup) + "'");
    const std::vector<ResolvedEnsembleLoss> resolvedLosses = resolveTrainingRunsSelectedLossReports(
        activeLosses,
        selections.lossNames,
        "TrainingRuns reports for ensemble_group '" + std::string(ensembleGroup) + "'");

    std::vector<TrainingNamedMetricResult> results;
    results.reserve(resolvedLosses.size());
    for (const ResolvedEnsembleLoss& loss : resolvedLosses) {
        if (!trainingRunsLossCanParticipateInComposedEvaluation(loss)) {
            continue;
        }
        TrainingNamedMetricResult result;
        result.name = loss.lossName;
        results.push_back(std::move(result));
    }
    return results;
}

std::vector<TrainingEnsembleResult> TrainingRuns::buildEnsembleResults(const std::vector<TrainingRunResult>& results) const {
    std::map<std::string, TrainingEnsembleResult> byGroup = buildEnsembleResultsByGroup(results);
    std::vector<TrainingEnsembleResult> ensembles;
    ensembles.reserve(byGroup.size());
    for (auto& entry : byGroup) {
        ensembles.push_back(std::move(entry.second));
    }
    return ensembles;
}

std::map<std::string, TrainingEnsembleResult> TrainingRuns::buildEnsembleResultsByGroup(const std::vector<TrainingRunResult>& results) const {
    std::map<std::string, TrainingEnsembleResult> byGroup;
    std::map<std::string, const TrainingRunResult*> resultByRunName;
    for (const TrainingRunResult& result : results) {
        resultByRunName.emplace(result.runName, &result);
    }

    for (const TrainingRunsSpec& spec : runs) {
        if (!spec.ensembleGroup.has_value()) {
            continue;
        }

        TrainingEnsembleResult& ensemble = byGroup[*spec.ensembleGroup];
        if (ensemble.ensembleGroup.empty()) {
            ensemble.ensembleGroup = *spec.ensembleGroup;
            ensemble.inputSignature = collectNetworkInputSignature(validationNetworkForSpec(spec));
            ensemble.outputSignature = collectNetworkOutputSignature(validationNetworkForSpec(spec));
            ensemble.minSuccessfulModels = minSuccessfulModelsForGroup(*spec.ensembleGroup, 0);
            ensemble.namedMetrics = namedMetricResultsForGroup(*spec.ensembleGroup);
            ensemble.namedGraphMetrics = namedGraphMetricResultsForGroup(*spec.ensembleGroup);
            ensemble.reportOrder = reportOrderForGroup(*spec.ensembleGroup);
        }

        const auto resultIt = resultByRunName.find(spec.runName);
        if (resultIt == resultByRunName.end()) {
            continue;
        }
        const TrainingRunResult& result = *resultIt->second;

        TrainingEnsembleMemberResult member;
        member.runName = spec.runName;
        member.weight = spec.ensembleWeight;
        member.status = result.status;
        member.finalTrainingLoss = result.finalLossForPhase(TrainingEventPhase::TRAIN);
        member.finalValidationLoss = result.finalLossForPhase(TrainingEventPhase::VALIDATE);
        member.finalTestLoss = result.finalLossForPhase(TrainingEventPhase::TEST);
        ensemble.members.push_back(std::move(member));
    }

    return byGroup;
}

namespace {

struct EnsembleMemberSpecRef {
    size_t runIndex = 0;
    const TrainingRunsSpec* spec = nullptr;
    const TrainingRunResult* result = nullptr;
};

std::map<std::string, std::vector<EnsembleMemberSpecRef>> completedEnsembleMembersByGroup(
    const std::vector<TrainingRunsSpec>& runs,
    const std::vector<TrainingRunResult>& results) {
    std::map<std::string, std::vector<EnsembleMemberSpecRef>> byGroup;
    for (size_t i = 0; i < runs.size(); ++i) {
        if (!runs[i].ensembleGroup.has_value()) {
            continue;
        }
        if (i >= results.size() || results[i].status != TrainingRunStatus::COMPLETED) {
            continue;
        }
        byGroup[*runs[i].ensembleGroup].push_back(EnsembleMemberSpecRef{i, &runs[i], &results[i]});
    }
    return byGroup;
}


std::optional<double> weightedAverage(const std::vector<std::optional<double>>& values, const std::vector<double>& weights) {
    if (values.empty() || values.size() != weights.size()) {
        return std::nullopt;
    }
    double weightedSum = 0.0;
    double weightSum = 0.0;
    for (size_t i = 0; i < values.size(); ++i) {
        if (!values[i].has_value() || !std::isfinite(weights[i]) || weights[i] <= 0.0) {
            return std::nullopt;
        }
        weightedSum += weights[i] * values[i].value();
        weightSum += weights[i];
    }
    if (weightSum <= 0.0) {
        return std::nullopt;
    }
    return weightedSum / weightSum;
}

std::optional<double> weightedLossSumFromWeightedLossValues(const std::vector<std::optional<double>>& weightedLossValues) {
    if (weightedLossValues.empty()) {
        return std::nullopt;
    }
    double lossSum = 0.0;
    for (const std::optional<double>& value : weightedLossValues) {
        if (!value.has_value() || !std::isfinite(value.value())) {
            return std::nullopt;
        }
        lossSum += value.value();
    }
    return lossSum;
}

ThorImplementation::DynamicExpression makeWeightedMeanExpression(const std::vector<std::string>& inputNames,
                                                                 const std::vector<double>& weights,
                                                                 const std::string& outputName,
                                                                 DataType dataType) {
    if (inputNames.empty() || inputNames.size() != weights.size()) {
        throw std::runtime_error("TrainingRuns ensemble accumulator requires one input name per member weight.");
    }
    double weightSum = 0.0;
    std::optional<ThorImplementation::Expression> expr;
    for (size_t i = 0; i < inputNames.size(); ++i) {
        if (!std::isfinite(weights[i]) || weights[i] <= 0.0) {
            throw std::runtime_error("TrainingRuns ensemble accumulator weights must be finite and positive.");
        }
        weightSum += weights[i];
        ThorImplementation::Expression term =
            ThorImplementation::Expression::input(inputNames[i], dataType, dataType) * static_cast<float>(weights[i]);
        expr = expr.has_value() ? (expr.value() + term) : term;
    }
    if (weightSum <= 0.0 || !expr.has_value()) {
        throw std::runtime_error("TrainingRuns ensemble accumulator weights must have a positive sum.");
    }
    ThorImplementation::Expression output = expr.value() * static_cast<float>(1.0 / weightSum);
    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{outputName, output}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}


struct TrainingRunsComposedEnsembleEvaluator {
    std::shared_ptr<Network> network = nullptr;
    std::map<std::string, Tensor> sharedInputTensorsByName{};
    std::vector<std::map<std::string, Tensor>> memberOutputTensorsByName{};
    std::map<std::string, Tensor> averagedOutputTensorsByName{};
    std::map<std::string, Tensor> lossOutputTensorsByName{};
    std::map<std::string, Tensor> metricOutputTensorsByName{};
    std::vector<std::string> externalInputNames{};
};

std::string trainingRunsMemberScopedName(size_t memberIndex, const std::string& name) {
    return "member_" + std::to_string(memberIndex) + "/" + name;
}

std::string trainingRunsComposedAccumulatorInputName(const std::string& outputName, size_t memberIndex) {
    return "member_" + std::to_string(memberIndex) + "_" + outputName;
}

std::map<std::string, std::shared_ptr<NetworkInput>> apiNetworkInputsByName(Network& network, bool includePassThroughInputs = true) {
    std::map<std::string, std::shared_ptr<NetworkInput>> inputsByName;
    const uint32_t layerCount = network.getNumLayers();
    for (uint32_t layerIndex = 0; layerIndex < layerCount; ++layerIndex) {
        std::shared_ptr<NetworkInput> input = std::dynamic_pointer_cast<NetworkInput>(network.getLayer(layerIndex));
        if (input == nullptr) {
            continue;
        }
        if (!includePassThroughInputs && input->hasPassThroughSource()) {
            continue;
        }
        if (!inputsByName.emplace(input->getName(), input).second) {
            throw std::runtime_error("TrainingRuns composed ensemble evaluator found duplicate API NetworkInput named '" + input->getName() +
                                     "' in network '" + network.getNetworkName() + "'.");
        }
    }
    return inputsByName;
}

std::map<std::string, std::shared_ptr<NetworkOutput>> apiNetworkOutputsByName(Network& network) {
    std::map<std::string, std::shared_ptr<NetworkOutput>> outputsByName;
    const uint32_t layerCount = network.getNumLayers();
    for (uint32_t layerIndex = 0; layerIndex < layerCount; ++layerIndex) {
        std::shared_ptr<NetworkOutput> output = std::dynamic_pointer_cast<NetworkOutput>(network.getLayer(layerIndex));
        if (output == nullptr) {
            continue;
        }
        if (!outputsByName.emplace(output->getName(), output).second) {
            throw std::runtime_error("TrainingRuns composed ensemble evaluator found duplicate API NetworkOutput named '" + output->getName() +
                                     "' in network '" + network.getNetworkName() + "'.");
        }
    }
    return outputsByName;
}

std::shared_ptr<NetworkOutput> requiredApiNetworkOutputByName(Network& network,
                                                              const std::map<std::string, std::shared_ptr<NetworkOutput>>& outputsByName,
                                                              const std::string& outputName,
                                                              const std::string& context) {
    auto outputIt = outputsByName.find(outputName);
    if (outputIt == outputsByName.end() || outputIt->second == nullptr) {
        throw std::runtime_error(context + " is missing API NetworkOutput '" + outputName + "' in network '" +
                                 network.getNetworkName() + "'.");
    }
    if (!outputIt->second->getFeatureInput().has_value()) {
        throw std::runtime_error(context + " API NetworkOutput '" + outputName + "' has no feature input.");
    }
    return outputIt->second;
}

std::shared_ptr<NetworkInput> requiredApiNetworkInputByName(Network& network,
                                                            const std::map<std::string, std::shared_ptr<NetworkInput>>& inputsByName,
                                                            const std::string& inputName,
                                                            const std::string& context) {
    auto inputIt = inputsByName.find(inputName);
    if (inputIt == inputsByName.end() || inputIt->second == nullptr) {
        throw std::runtime_error(context + " is missing API NetworkInput '" + inputName + "' in network '" +
                                 network.getNetworkName() + "'.");
    }
    if (!inputIt->second->getFeatureOutput().has_value()) {
        throw std::runtime_error(context + " API NetworkInput '" + inputName + "' has no feature output.");
    }
    return inputIt->second;
}

void validateComposedEvaluatorMemberInputsCompatible(const NetworkInput& referenceInput,
                                                     const NetworkInput& memberInput,
                                                     size_t memberIndex,
                                                     const std::string& inputName) {
    if (referenceInput.getDimensions() != memberInput.getDimensions() || referenceInput.getDataType() != memberInput.getDataType() ||
        referenceInput.dimensionsIncludeBatch() != memberInput.dimensionsIncludeBatch()) {
        throw std::runtime_error("TrainingRuns composed ensemble evaluator member " + std::to_string(memberIndex) +
                                 " has incompatible descriptor for input '" + inputName + "'.");
    }
}

TrainingRunsComposedEnsembleEvaluator buildTrainingRunsComposedEnsembleEvaluatorThroughAccumulator(
    const std::vector<std::shared_ptr<Network>>& memberNetworks,
    const std::vector<double>& weights,
    const std::vector<std::string>& outputNames,
    bool exposeAveragedOutputsAsNetworkOutputs = true,
    const std::string& networkName = "training_runs_composed_ensemble_evaluator",
    std::optional<std::vector<std::string>> sharedInputNamesOverride = std::nullopt) {
    if (memberNetworks.empty()) {
        throw std::runtime_error("TrainingRuns composed ensemble evaluator requires at least one member network.");
    }
    if (weights.size() != memberNetworks.size()) {
        throw std::runtime_error("TrainingRuns composed ensemble evaluator requires one ensemble weight per member network.");
    }
    // Reports that summarize only external inputs, such as Mean(labels), do not
    // require any member output to be averaged.  In that case this helper builds
    // the evaluator input-distribution shell and returns without cloning members.
    const bool hasOutputsToAverage = !outputNames.empty();
    for (double weight : weights) {
        if (!std::isfinite(weight) || weight <= 0.0) {
            throw std::runtime_error("TrainingRuns composed ensemble evaluator weights must be finite and positive.");
        }
    }
    for (size_t memberIndex = 0; memberIndex < memberNetworks.size(); ++memberIndex) {
        if (memberNetworks[memberIndex] == nullptr) {
            throw std::runtime_error("TrainingRuns composed ensemble evaluator received a null member network at index " +
                                     std::to_string(memberIndex) + ".");
        }
    }

    Network& referenceMember = *memberNetworks.front();
    std::vector<std::string> referenceInputNames;
    const bool requireExactSharedInferenceInputSet = !sharedInputNamesOverride.has_value();
    if (sharedInputNamesOverride.has_value()) {
        referenceInputNames = std::move(*sharedInputNamesOverride);
    } else {
        referenceInputNames = referenceMember.getInferenceNetworkInputNames();
        eraseNames(referenceInputNames, referenceMember.getTrainingOnlyNetworkInputNames());
    }
    if (referenceInputNames.empty()) {
        throw std::runtime_error("TrainingRuns composed ensemble evaluator requires at least one shared member inference input.");
    }
    const std::map<std::string, std::shared_ptr<NetworkInput>> referenceInputsByName =
        apiNetworkInputsByName(referenceMember, /*includePassThroughInputs=*/false);

    TrainingRunsComposedEnsembleEvaluator evaluator;
    evaluator.network = std::make_shared<Network>(networkName);
    evaluator.memberOutputTensorsByName.resize(memberNetworks.size());

    std::set<std::string> seenReferenceInputNames;
    for (const std::string& inputName : referenceInputNames) {
        if (!seenReferenceInputNames.insert(inputName).second) {
            throw std::runtime_error("TrainingRuns composed ensemble evaluator found duplicate shared input name '" + inputName + "'.");
        }
        std::shared_ptr<NetworkInput> referenceInput =
            requiredApiNetworkInputByName(referenceMember, referenceInputsByName, inputName, "TrainingRuns composed ensemble evaluator reference member");
        NetworkInput evaluatorInput = NetworkInput::Builder()
                                          .network(*evaluator.network)
                                          .name(inputName)
                                          .dimensions(referenceInput->getDimensions())
                                          .dataType(referenceInput->getDataType())
                                          .dimensionsIncludeBatch(referenceInput->dimensionsIncludeBatch())
                                          .build();
        evaluator.sharedInputTensorsByName[inputName] = evaluatorInput.getFeatureOutput().value();
        evaluator.externalInputNames.push_back(inputName);
    }

    std::set<std::string> referenceInputNameSet(referenceInputNames.begin(), referenceInputNames.end());
    if (!hasOutputsToAverage) {
        return evaluator;
    }

    std::vector<std::string> memberCloneInputNames = referenceInputNames;
    if (!requireExactSharedInferenceInputSet) {
        memberCloneInputNames = referenceMember.getInferenceNetworkInputNamesForOutputs(outputNames);
        for (const std::string& inputName : memberCloneInputNames) {
            if (referenceInputNameSet.count(inputName) == 0) {
                throw std::runtime_error("TrainingRuns composed ensemble evaluator internal error: clone input '" + inputName +
                                         "' is missing from the shared input set.");
            }
        }
    }

    for (size_t memberIndex = 0; memberIndex < memberNetworks.size(); ++memberIndex) {
        Network& memberNetwork = *memberNetworks[memberIndex];
        const std::map<std::string, std::shared_ptr<NetworkInput>> memberInputsByName =
            apiNetworkInputsByName(memberNetwork, /*includePassThroughInputs=*/false);
        std::vector<std::string> memberInputNames;
        if (requireExactSharedInferenceInputSet) {
            memberInputNames = memberNetwork.getInferenceNetworkInputNames();
            eraseNames(memberInputNames, memberNetwork.getTrainingOnlyNetworkInputNames());
        } else {
            memberInputNames.reserve(memberInputsByName.size());
            for (const auto& [inputName, _] : memberInputsByName) {
                (void)_;
                memberInputNames.push_back(inputName);
            }
        }
        std::set<std::string> memberInputNameSet(memberInputNames.begin(), memberInputNames.end());
        if (memberInputNames.size() != memberInputNameSet.size()) {
            throw std::runtime_error("TrainingRuns composed ensemble evaluator member " + std::to_string(memberIndex) +
                                     " has duplicate input names.");
        }
        if (requireExactSharedInferenceInputSet) {
            if (memberInputNameSet != referenceInputNameSet) {
                throw std::runtime_error("TrainingRuns composed ensemble evaluator members have incompatible inference input names.");
            }
        } else {
            for (const std::string& inputName : referenceInputNames) {
                if (memberInputNameSet.count(inputName) == 0) {
                    throw std::runtime_error("TrainingRuns composed ensemble evaluator member " + std::to_string(memberIndex) +
                                             " is missing required shared input '" + inputName + "'.");
                }
            }
        }

        ApiTensorRemap remap;
        for (const std::string& inputName : memberCloneInputNames) {
            std::shared_ptr<NetworkInput> referenceInput =
                requiredApiNetworkInputByName(referenceMember, referenceInputsByName, inputName, "TrainingRuns composed ensemble evaluator reference member");
            std::shared_ptr<NetworkInput> memberInput =
                requiredApiNetworkInputByName(memberNetwork, memberInputsByName, inputName, "TrainingRuns composed ensemble evaluator member");
            validateComposedEvaluatorMemberInputsCompatible(*referenceInput, *memberInput, memberIndex, inputName);

            const Tensor& sharedInputTensor = evaluator.sharedInputTensorsByName.at(inputName);
            NetworkInput memberPassThroughInput = NetworkInput::Builder()
                                                      .network(*evaluator.network)
                                                      .name(trainingRunsMemberScopedName(memberIndex, inputName))
                                                      .passThroughSource(sharedInputTensor)
                                                      .build();
            remap.map(memberInput->getFeatureOutput().value(), memberPassThroughInput.getFeatureOutput().value());
        }

        ApiSubgraphCloneOptions cloneOptions;
        cloneOptions.namePrefix = trainingRunsMemberScopedName(memberIndex, "");
        ApiSubgraphCloneResult cloneResult = evaluator.network->cloneInferenceSubgraphInto(memberNetwork, outputNames, remap, cloneOptions);
        evaluator.memberOutputTensorsByName[memberIndex] = std::move(cloneResult.outputTensorsByName);
    }

    for (size_t outputIndex = 0; outputIndex < outputNames.size(); ++outputIndex) {
        (void)outputIndex;
        const std::string& outputName = outputNames[outputIndex];
        const auto referenceOutputIt = evaluator.memberOutputTensorsByName[0].find(outputName);
        if (referenceOutputIt == evaluator.memberOutputTensorsByName[0].end()) {
            throw std::runtime_error("TrainingRuns composed ensemble evaluator member 0 did not produce requested output '" + outputName + "'.");
        }
        Tensor referenceOutput = referenceOutputIt->second;
        const std::vector<uint64_t> dimensions = referenceOutput.getDimensions();
        const DataType dataType = referenceOutput.getDataType();

        CustomLayer::TensorMap inputInterface;
        std::vector<std::string> accumulatorInputNames;
        accumulatorInputNames.reserve(memberNetworks.size());
        for (size_t memberIndex = 0; memberIndex < memberNetworks.size(); ++memberIndex) {
            const auto outputIt = evaluator.memberOutputTensorsByName[memberIndex].find(outputName);
            if (outputIt == evaluator.memberOutputTensorsByName[memberIndex].end()) {
                throw std::runtime_error("TrainingRuns composed ensemble evaluator member " + std::to_string(memberIndex) +
                                         " did not produce requested output '" + outputName + "'.");
            }
            const Tensor& memberOutput = outputIt->second;
            if (memberOutput.getDimensions() != dimensions || memberOutput.getDataType() != dataType) {
                throw std::runtime_error("TrainingRuns composed ensemble evaluator members have incompatible descriptors for output '" +
                                         outputName + "'.");
            }
            const std::string accumulatorInputName = trainingRunsComposedAccumulatorInputName(outputName, memberIndex);
            accumulatorInputNames.push_back(accumulatorInputName);
            inputInterface.emplace(accumulatorInputName, memberOutput);
        }

        CustomLayer accumulator = CustomLayer::Builder()
                                      .network(*evaluator.network)
                                      .expression(makeWeightedMeanExpression(accumulatorInputNames, weights, outputName, dataType))
                                      .inputNames(accumulatorInputNames)
                                      .outputNames({outputName})
                                      .inputInterface(inputInterface)
                                      .build();
        Tensor averagedOutput = accumulator.getOutput(outputName);
        evaluator.averagedOutputTensorsByName[outputName] = averagedOutput;
        if (exposeAveragedOutputsAsNetworkOutputs) {
            NetworkOutput::Builder().network(*evaluator.network).name(outputName).inputTensor(averagedOutput).dataType(dataType).build();
        }
    }

    return evaluator;
}

std::shared_ptr<Network> buildSingleMemberEnsembleNetworkArtifact(Network& memberNetwork,
                                                               const std::vector<std::string>& outputNames,
                                                               const std::vector<std::string>& deployableInputNames,
                                                               const std::string& networkName) {
    if (outputNames.empty()) {
        throw std::runtime_error("TrainingRunsResult.save_ensemble single-member artifact requires at least one deployable output.");
    }
    if (deployableInputNames.empty()) {
        throw std::runtime_error("TrainingRunsResult.save_ensemble single-member artifact requires at least one deployable input.");
    }

    auto ensembleNetwork = std::make_shared<Network>(networkName);
    const std::map<std::string, std::shared_ptr<NetworkInput>> memberInputsByName =
        apiNetworkInputsByName(memberNetwork, /*includePassThroughInputs=*/false);

    ApiTensorRemap remap;
    std::set<std::string> seenInputNames;
    for (const std::string& inputName : deployableInputNames) {
        if (!seenInputNames.insert(inputName).second) {
            throw std::runtime_error("TrainingRunsResult.save_ensemble single-member artifact found duplicate deployable input name '" +
                                     inputName + "'.");
        }
        std::shared_ptr<NetworkInput> memberInput =
            requiredApiNetworkInputByName(memberNetwork, memberInputsByName, inputName,
                                          "TrainingRunsResult.save_ensemble single-member artifact member");
        NetworkInput ensembleInput = NetworkInput::Builder()
                                         .network(*ensembleNetwork)
                                         .name(inputName)
                                         .dimensions(memberInput->getDimensions())
                                         .dataType(memberInput->getDataType())
                                         .dimensionsIncludeBatch(memberInput->dimensionsIncludeBatch())
                                         .build();
        remap.map(memberInput->getFeatureOutput().value(), ensembleInput.getFeatureOutput().value());
    }

    ApiSubgraphCloneOptions cloneOptions;
    cloneOptions.inferenceOnly = true;
    ApiSubgraphCloneResult cloneResult = ensembleNetwork->cloneInferenceSubgraphInto(memberNetwork, outputNames, remap, cloneOptions);

    std::set<std::string> seenOutputNames;
    for (const std::string& outputName : outputNames) {
        if (!seenOutputNames.insert(outputName).second) {
            throw std::runtime_error("TrainingRunsResult.save_ensemble single-member artifact found duplicate deployable output name '" +
                                     outputName + "'.");
        }
        const auto outputIt = cloneResult.outputTensorsByName.find(outputName);
        if (outputIt == cloneResult.outputTensorsByName.end()) {
            throw std::runtime_error("TrainingRunsResult.save_ensemble single-member artifact did not clone output '" + outputName + "'.");
        }
        const Tensor& outputTensor = outputIt->second;
        NetworkOutput::Builder()
            .network(*ensembleNetwork)
            .name(outputName)
            .inputTensor(outputTensor)
            .dataType(outputTensor.getDataType())
            .build();
    }

    return ensembleNetwork;
}

void saveEnsembleNetworkArtifact(const TrainingEnsembleResult& ensembleResult,
                                 const std::vector<SavedNetworkArtifactRef>& memberArtifacts,
                                 const std::string& aggregation,
                                 const std::filesystem::path& artifactDirectory,
                                 bool overwriteNetworkArchive) {
    if (memberArtifacts.empty()) {
        throw std::runtime_error("TrainingRunsResult.save_ensemble cannot save an ensemble network with no completed member artifacts.");
    }

    std::vector<std::shared_ptr<Network>> memberNetworks;
    memberNetworks.reserve(memberArtifacts.size());
    for (const SavedNetworkArtifactRef& memberArtifact : memberArtifacts) {
        if (memberArtifact.networkName.empty()) {
            throw std::runtime_error("TrainingRunsResult.save_ensemble member artifact is missing its saved model network name.");
        }
        auto memberNetwork = std::make_shared<Network>(memberArtifact.networkName);
        memberNetwork->load(memberArtifact.directory.string());
        memberNetworks.push_back(std::move(memberNetwork));
    }

    std::vector<double> weights;
    weights.reserve(memberArtifacts.size());
    for (const TrainingEnsembleMemberResult& member : ensembleResult.members) {
        if (member.status != TrainingRunStatus::COMPLETED) {
            continue;
        }
        weights.push_back(aggregation == "mean" ? 1.0 : member.weight);
    }
    if (weights.size() != memberArtifacts.size()) {
        throw std::runtime_error("TrainingRunsResult.save_ensemble internal error while collecting completed member weights.");
    }

    const std::vector<std::string> outputNames = deployableOutputNamesForSavedEnsemble(*memberNetworks.front(), ensembleResult.outputSignature);
    if (outputNames.empty()) {
        throw std::runtime_error("TrainingRunsResult.save_ensemble could not determine any deployable prediction output names for ensemble group '" +
                                 ensembleResult.ensembleGroup + "'.");
    }
    const std::vector<std::string> deployableInputNames = memberNetworks.front()->getInferenceNetworkInputNamesForOutputs(outputNames);
    if (deployableInputNames.empty()) {
        throw std::runtime_error("TrainingRunsResult.save_ensemble could not determine any deployable input names for ensemble group '" +
                                 ensembleResult.ensembleGroup + "'.");
    }

    if (memberNetworks.size() == 1) {
        std::shared_ptr<Network> singleMemberEnsembleNetwork = buildSingleMemberEnsembleNetworkArtifact(
            *memberNetworks.front(),
            outputNames,
            deployableInputNames,
            safeNetworkNameForSavedEnsemble(ensembleResult.ensembleGroup));
        std::shared_ptr<PlacedNetwork> placed =
            placeInferenceNetworkWithSerializedStartup(
                *singleMemberEnsembleNetwork,
                /*batchSize=*/1);
        placed->save(artifactDirectory.string(), overwriteNetworkArchive, /*saveOptimizerState=*/false);
        return;
    }

    TrainingRunsComposedEnsembleEvaluator evaluator = buildTrainingRunsComposedEnsembleEvaluatorThroughAccumulator(
        memberNetworks,
        weights,
        outputNames,
        /*exposeAveragedOutputsAsNetworkOutputs=*/true,
        safeNetworkNameForSavedEnsemble(ensembleResult.ensembleGroup),
        std::optional<std::vector<std::string>>{deployableInputNames});
    std::shared_ptr<PlacedNetwork> placed =
        placeInferenceNetworkWithSerializedStartup(
            *evaluator.network,
            /*batchSize=*/1);
    placed->save(artifactDirectory.string(), overwriteNetworkArchive, /*saveOptimizerState=*/false);
}

std::optional<Tensor> existingTrainingRunsEvaluatorInputTensorForGraphInput(
    const TrainingRunsComposedEnsembleEvaluator& evaluator,
    const std::string& inputName) {
    auto existingIt = evaluator.sharedInputTensorsByName.find(inputName);
    if (existingIt == evaluator.sharedInputTensorsByName.end()) {
        return std::nullopt;
    }
    return existingIt->second;
}

Tensor cloneTrainingRunsEvaluatorLossFromReference(TrainingRunsComposedEnsembleEvaluator& evaluator,
                                                  Network& referenceMember,
                                                  const std::map<std::string, std::shared_ptr<NetworkInput>>& referenceInputsByName,
                                                  const std::map<std::string, std::shared_ptr<NetworkOutput>>& referenceOutputsByName,
                                                  const ResolvedEnsembleLoss& loss,
                                                  const Tensor& averagedPredictions,
                                                  const Tensor& labels,
                                                  std::optional<Tensor> exampleWeights) {
    std::shared_ptr<NetworkOutput> predictionOutput = requiredApiNetworkOutputByName(
        referenceMember,
        referenceOutputsByName,
        loss.predictionOutputName,
        "TrainingRuns composed ensemble evaluator reference prediction output for reported loss '" + loss.lossName + "'");
    std::shared_ptr<NetworkOutput> lossOutput = requiredApiNetworkOutputByName(
        referenceMember,
        referenceOutputsByName,
        loss.lossName,
        "TrainingRuns composed ensemble evaluator reference loss output for reported loss '" + loss.lossName + "'");
    std::shared_ptr<NetworkInput> labelInput = requiredApiNetworkInputByName(
        referenceMember,
        referenceInputsByName,
        loss.targetInputName,
        "TrainingRuns composed ensemble evaluator reference label input for reported loss '" + loss.lossName + "'");

    ApiTensorRemap remap;
    remap.map(predictionOutput->getFeatureInput().value(), averagedPredictions);
    remap.map(labelInput->getFeatureOutput().value(), labels);
    if (loss.weightInputName.has_value()) {
        if (!exampleWeights.has_value()) {
            throw std::runtime_error("TrainingRuns composed ensemble evaluator did not build weight input '" + *loss.weightInputName +
                                     "' for reported loss '" + loss.lossName + "'.");
        }
        std::shared_ptr<NetworkInput> weightInput = requiredApiNetworkInputByName(
            referenceMember,
            referenceInputsByName,
            *loss.weightInputName,
            "TrainingRuns composed ensemble evaluator reference weight input for reported loss '" + loss.lossName + "'");
        remap.map(weightInput->getFeatureOutput().value(), *exampleWeights);
    }

    ApiSubgraphCloneOptions cloneOptions;
    cloneOptions.namePrefix = "reported_loss/" + loss.lossName + "/";
    cloneOptions.inferenceOnly = false;
    ApiSubgraphCloneResult cloneResult = evaluator.network->cloneInferenceSubgraphInto(referenceMember, {loss.lossName}, remap, cloneOptions);
    auto lossIt = cloneResult.outputTensorsByName.find(loss.lossName);
    if (lossIt == cloneResult.outputTensorsByName.end()) {
        throw std::runtime_error("TrainingRuns composed ensemble evaluator did not clone loss output '" + loss.lossName + "'.");
    }
    (void)lossOutput;
    return lossIt->second;
}

Tensor cloneTrainingRunsEvaluatorMetricFromReference(TrainingRunsComposedEnsembleEvaluator& evaluator,
                                                    Network& referenceMember,
                                                    const std::map<std::string, std::shared_ptr<NetworkInput>>& referenceInputsByName,
                                                    const std::map<std::string, std::shared_ptr<NetworkOutput>>& referenceOutputsByName,
                                                    const ResolvedEnsembleMetric& metric,
                                                    std::optional<Tensor> averagedPredictions,
                                                    std::optional<Tensor> labels,
                                                    std::optional<Tensor> inputSource) {
    std::shared_ptr<NetworkOutput> metricOutput = requiredApiNetworkOutputByName(
        referenceMember,
        referenceOutputsByName,
        metric.metricName,
        "TrainingRuns composed ensemble evaluator reference metric output for reported metric '" + metric.metricName + "'");

    ApiTensorRemap remap;
    if (!metric.predictionOutputName.empty()) {
        if (!averagedPredictions.has_value()) {
            throw std::runtime_error("TrainingRuns composed ensemble evaluator did not build averaged prediction output '" +
                                     metric.predictionOutputName + "' for reported metric '" + metric.metricName + "'.");
        }
        std::shared_ptr<NetworkOutput> predictionOutput = requiredApiNetworkOutputByName(
            referenceMember,
            referenceOutputsByName,
            metric.predictionOutputName,
            "TrainingRuns composed ensemble evaluator reference prediction output for reported metric '" + metric.metricName + "'");
        remap.map(predictionOutput->getFeatureInput().value(), *averagedPredictions);
    }
    if (metric.inputSourceName.has_value()) {
        if (!inputSource.has_value()) {
            throw std::runtime_error("TrainingRuns composed ensemble evaluator did not build input source '" + *metric.inputSourceName +
                                     "' for reported metric '" + metric.metricName + "'.");
        }
        std::shared_ptr<NetworkInput> sourceInput = requiredApiNetworkInputByName(
            referenceMember,
            referenceInputsByName,
            *metric.inputSourceName,
            "TrainingRuns composed ensemble evaluator reference input source for reported metric '" + metric.metricName + "'");
        remap.map(sourceInput->getFeatureOutput().value(), *inputSource);
    }
    if (metric.targetInputName.has_value()) {
        if (!labels.has_value()) {
            throw std::runtime_error("TrainingRuns composed ensemble evaluator did not build target input '" + *metric.targetInputName +
                                     "' for reported metric '" + metric.metricName + "'.");
        }
        std::shared_ptr<NetworkInput> labelInput = requiredApiNetworkInputByName(
            referenceMember,
            referenceInputsByName,
            *metric.targetInputName,
            "TrainingRuns composed ensemble evaluator reference target input for reported metric '" + metric.metricName + "'");
        remap.map(labelInput->getFeatureOutput().value(), *labels);
    }

    ApiSubgraphCloneOptions cloneOptions;
    cloneOptions.namePrefix = "reported_metric/" + metric.metricName + "/";
    cloneOptions.inferenceOnly = false;
    ApiSubgraphCloneResult cloneResult = evaluator.network->cloneInferenceSubgraphInto(referenceMember, {metric.metricName}, remap, cloneOptions);
    auto metricIt = cloneResult.outputTensorsByName.find(metric.metricName);
    if (metricIt == cloneResult.outputTensorsByName.end()) {
        throw std::runtime_error("TrainingRuns composed ensemble evaluator did not clone metric output '" + metric.metricName + "'.");
    }
    (void)metricOutput;
    return metricIt->second;
}

TrainingRunsComposedEnsembleEvaluator buildTrainingRunsComposedEnsembleEvaluatorForReports(
    const std::vector<std::shared_ptr<Network>>& memberNetworks,
    const std::vector<double>& weights,
    std::vector<ResolvedEnsembleLoss>& losses,
    std::vector<ResolvedEnsembleMetric>& metrics) {
    if (losses.empty() && metrics.empty()) {
        throw std::runtime_error("TrainingRuns composed ensemble evaluator requires at least one graph loss or metric to report.");
    }

    std::vector<std::string> outputNames;
    std::set<std::string> seenOutputs;
    for (const ResolvedEnsembleLoss& loss : losses) {
        if (!loss.predictionOutputName.empty() && seenOutputs.insert(loss.predictionOutputName).second) {
            outputNames.push_back(loss.predictionOutputName);
        }
    }
    for (const ResolvedEnsembleMetric& metric : metrics) {
        if (!metric.predictionOutputName.empty() && seenOutputs.insert(metric.predictionOutputName).second) {
            outputNames.push_back(metric.predictionOutputName);
        }
    }

    Network& referenceMember = *memberNetworks.front();
    // If no member prediction outputs are needed, this evaluator is for source-only
    // reports such as Mean(labels).  Do not seed the shared inputs with the
    // deployable inference inputs in that case: they would be unused in the
    // report-only graph and fail graph validation as dangling NetworkInput
    // outputs.  The loss/metric wiring below appends exactly the external inputs
    // that the requested reports consume.
    std::vector<std::string> sharedInputNames = outputNames.empty()
        ? std::vector<std::string>{}
        : referenceMember.getInferenceNetworkInputNamesForOutputs(outputNames);
    for (const ResolvedEnsembleLoss& loss : losses) {
        appendNameIfMissing(sharedInputNames, loss.targetInputName);
        if (loss.weightInputName.has_value()) {
            appendNameIfMissing(sharedInputNames, *loss.weightInputName);
        }
    }
    for (const ResolvedEnsembleMetric& metric : metrics) {
        if (metric.targetInputName.has_value()) {
            appendNameIfMissing(sharedInputNames, *metric.targetInputName);
        }
        if (metric.inputSourceName.has_value()) {
            appendNameIfMissing(sharedInputNames, *metric.inputSourceName);
        }
    }

    TrainingRunsComposedEnsembleEvaluator evaluator =
        buildTrainingRunsComposedEnsembleEvaluatorThroughAccumulator(
            memberNetworks,
            weights,
            outputNames,
            /*exposeAveragedOutputsAsNetworkOutputs=*/false,
            "training_runs_composed_ensemble_evaluator",
            std::optional<std::vector<std::string>>{sharedInputNames});

    const std::map<std::string, std::shared_ptr<NetworkInput>> referenceInputsByName =
        apiNetworkInputsByName(referenceMember, /*includePassThroughInputs=*/false);
    const std::map<std::string, std::shared_ptr<NetworkOutput>> referenceOutputsByName = apiNetworkOutputsByName(referenceMember);

    std::set<std::string> exposedReportOutputs;
    std::vector<ResolvedEnsembleLoss> activeLosses;
    activeLosses.reserve(losses.size());
    for (const ResolvedEnsembleLoss& loss : losses) {
        if (loss.predictionOutputName.empty()) {
            // The source network exposes this loss, but its prediction side is
            // not represented by an averaged ensemble output in this composition.
            // In this composed evaluator the loss does not exist, so do not
            // expose or report it.
            continue;
        }
        const auto predictionIt = evaluator.averagedOutputTensorsByName.find(loss.predictionOutputName);
        if (predictionIt == evaluator.averagedOutputTensorsByName.end()) {
            continue;
        }
        if (!exposedReportOutputs.insert(loss.lossName).second) {
            throw std::runtime_error("TrainingRuns composed ensemble evaluator cannot expose duplicate report output '" + loss.lossName + "'.");
        }
        std::optional<Tensor> labels = existingTrainingRunsEvaluatorInputTensorForGraphInput(evaluator, loss.targetInputName);
        if (!labels.has_value()) {
            continue;
        }
        std::optional<Tensor> weightsTensor;
        if (loss.weightInputName.has_value()) {
            weightsTensor = existingTrainingRunsEvaluatorInputTensorForGraphInput(evaluator, *loss.weightInputName);
            if (!weightsTensor.has_value()) {
                continue;
            }
        }
        Tensor lossTensor = cloneTrainingRunsEvaluatorLossFromReference(
            evaluator, referenceMember, referenceInputsByName, referenceOutputsByName, loss, predictionIt->second, *labels, weightsTensor);
        evaluator.lossOutputTensorsByName[loss.lossName] = lossTensor;
        NetworkOutput::Builder().network(*evaluator.network).name(loss.lossName).inputTensor(lossTensor).dataType(DataType::FP32).build();
        activeLosses.push_back(loss);
    }

    std::vector<ResolvedEnsembleMetric> activeMetrics;
    activeMetrics.reserve(metrics.size());
    for (const ResolvedEnsembleMetric& metric : metrics) {
        std::optional<Tensor> averagedPredictions;
        if (!metric.predictionOutputName.empty()) {
            const auto predictionIt = evaluator.averagedOutputTensorsByName.find(metric.predictionOutputName);
            if (predictionIt == evaluator.averagedOutputTensorsByName.end()) {
                // The metric's prediction source is not present in this composed
                // evaluator.  In this composition the metric does not exist, so do
                // not expose or report it.
                continue;
            }
            averagedPredictions = predictionIt->second;
        }

        std::optional<Tensor> labels;
        if (metric.targetInputName.has_value()) {
            labels = existingTrainingRunsEvaluatorInputTensorForGraphInput(evaluator, *metric.targetInputName);
            if (!labels.has_value()) {
                // Label-aware metrics are only valid in compositions that already
                // carry that target input.  Do not invent an input just to make a
                // metric reportable.
                continue;
            }
        }

        std::optional<Tensor> inputSource;
        if (metric.inputSourceName.has_value()) {
            inputSource = existingTrainingRunsEvaluatorInputTensorForGraphInput(evaluator, *metric.inputSourceName);
            if (!inputSource.has_value()) {
                continue;
            }
        } else if (metric.predictionOutputName.empty()) {
            // The user explicitly exposed this metric in the source network, so it
            // remains reportable for that network.  This composed evaluator has no
            // tensor that can be remapped to the metric's source, so the metric is
            // not part of this composition.
            continue;
        }

        if (!exposedReportOutputs.insert(metric.metricName).second) {
            throw std::runtime_error("TrainingRuns composed ensemble evaluator cannot expose duplicate report output '" + metric.metricName + "'.");
        }
        Tensor metricTensor = cloneTrainingRunsEvaluatorMetricFromReference(
            evaluator, referenceMember, referenceInputsByName, referenceOutputsByName, metric, averagedPredictions, labels, inputSource);
        evaluator.metricOutputTensorsByName[metric.metricName] = metricTensor;
        NetworkOutput::Builder().network(*evaluator.network).name(metric.metricName).inputTensor(metricTensor).dataType(DataType::FP32).build();
        activeMetrics.push_back(metric);
    }
    losses = std::move(activeLosses);
    metrics = std::move(activeMetrics);
    return evaluator;
}

struct TrainingRunsComposedEvaluatorArtifacts {
    uint64_t batchSize = 0;
    std::vector<std::shared_ptr<Network>> memberNetworks{};
    std::vector<double> weights{};
    std::vector<ResolvedEnsembleLoss> losses{};
    std::vector<ResolvedEnsembleMetric> metrics{};
    TrainingRunsComposedEnsembleEvaluator evaluator{};
    std::shared_ptr<PlacedNetwork> placedEvaluator = nullptr;
};

TrainingRunsComposedEvaluatorArtifacts loadTrainingRunsComposedEvaluatorArtifacts(
    const std::vector<EnsembleMemberSpecRef>& members,
    uint64_t batchSize,
    const std::vector<std::string>& requestedReportNames,
    const std::string& context) {
    if (batchSize == 0) {
        throw std::runtime_error(context + " cannot place composed ensemble evaluator for batch_size=0.");
    }
    if (members.empty()) {
        throw std::runtime_error(context + " requires at least one completed ensemble member.");
    }

    TrainingRunsComposedEvaluatorArtifacts artifacts;
    artifacts.batchSize = batchSize;
    artifacts.memberNetworks.reserve(members.size());
    artifacts.weights.reserve(members.size());

    for (const EnsembleMemberSpecRef& member : members) {
        if (member.spec == nullptr || member.spec->trainer == nullptr) {
            throw std::runtime_error(context + " requires a trainer for every ensemble member.");
        }
        const std::optional<std::string> artifactDir =
            member.result != nullptr && member.result->savedModelDirectory.has_value()
                ? member.result->savedModelDirectory
                : member.spec->trainer->getSaveModelDirectory();
        if (!artifactDir.has_value()) {
            throw std::runtime_error(context + " ensemble member '" + member.spec->runName +
                                     "' does not have a saved model artifact directory for ensemble evaluation.");
        }
        if (member.result == nullptr || !member.result->savedModelNetworkName.has_value() || member.result->savedModelNetworkName->empty()) {
            throw std::runtime_error(context + " ensemble member '" + member.spec->runName +
                                     "' does not have the exact saved model network name required for strict artifact loading.");
        }
        auto loadedNetwork = std::make_shared<Network>(*member.result->savedModelNetworkName);
        loadedNetwork->load(selectedTrainingArtifactModelDirectory(*artifactDir).string());
        artifacts.memberNetworks.push_back(std::move(loadedNetwork));
        artifacts.weights.push_back(member.spec->ensembleWeight);
    }

    const std::vector<NetworkLossReference> referenceAvailableLosses = artifacts.memberNetworks.front()->getReportableLosses();
    const std::vector<NetworkMetricReference> referenceAvailableMetrics = artifacts.memberNetworks.front()->getReportableMetrics();
    const std::vector<std::string> activeRequestedReportNames =
        filterRequestedReportNamesToAvailable(referenceAvailableLosses, referenceAvailableMetrics, requestedReportNames);
    if (requestedReportNames.empty() || !activeRequestedReportNames.empty()) {
        const TrainingRunsReportNameSelections selections = splitTrainingRunsRequestedReportsByKind(
            referenceAvailableLosses,
            referenceAvailableMetrics,
            requestedReportNames.empty() ? requestedReportNames : activeRequestedReportNames,
            context);
        artifacts.losses = resolveTrainingRunsSelectedLossReports(referenceAvailableLosses, selections.lossNames, context);
        artifacts.metrics = resolveTrainingRunsSelectedMetricReports(referenceAvailableMetrics, selections.metricNames, context);
        artifacts.metrics.erase(
            std::remove_if(artifacts.metrics.begin(),
                           artifacts.metrics.end(),
                           [](const ResolvedEnsembleMetric& metric) {
                               return !trainingRunsMetricCanParticipateInComposedEvaluation(metric);
                           }),
            artifacts.metrics.end());
    }
    if (artifacts.losses.empty() && artifacts.metrics.empty()) {
        return artifacts;
    }

    for (size_t memberIndex = 1; memberIndex < artifacts.memberNetworks.size(); ++memberIndex) {
        const std::vector<NetworkLossReference> memberAvailableLosses = artifacts.memberNetworks[memberIndex]->getReportableLosses();
        const std::vector<NetworkMetricReference> memberAvailableMetrics = artifacts.memberNetworks[memberIndex]->getReportableMetrics();
        const TrainingRunsReportNameSelections memberSelections = splitTrainingRunsRequestedReportsByKind(
            memberAvailableLosses,
            memberAvailableMetrics,
            requestedReportNames.empty() ? requestedReportNames : activeRequestedReportNames,
            context + " member " + std::to_string(memberIndex));
        const std::vector<ResolvedEnsembleLoss> memberLosses = resolveTrainingRunsSelectedLossReports(
            memberAvailableLosses,
            memberSelections.lossNames,
            context + " member " + std::to_string(memberIndex));
        for (const ResolvedEnsembleLoss& referenceLoss : artifacts.losses) {
            const auto memberLossIt = std::find_if(memberLosses.begin(), memberLosses.end(), [&](const ResolvedEnsembleLoss& candidate) {
                return candidate.lossName == referenceLoss.lossName;
            });
            if (memberLossIt == memberLosses.end()) {
                throw std::runtime_error(context + " member " + std::to_string(memberIndex) +
                                         " is missing graph loss '" + referenceLoss.lossName + "'.");
            }
            NetworkLossReference reference;
            reference.lossName = referenceLoss.lossName;
            reference.predictionOutputName = referenceLoss.predictionOutputName;
            reference.targetInputName = referenceLoss.targetInputName;
            reference.weightInputName = referenceLoss.weightInputName;
            reference.lossLayerType = referenceLoss.lossType;
            reference.lossWeight = referenceLoss.lossWeight;
            reference.quantile = referenceLoss.quantile;
            NetworkLossReference memberReference;
            memberReference.lossName = memberLossIt->lossName;
            memberReference.predictionOutputName = memberLossIt->predictionOutputName;
            memberReference.targetInputName = memberLossIt->targetInputName;
            memberReference.weightInputName = memberLossIt->weightInputName;
            memberReference.lossLayerType = memberLossIt->lossType;
            memberReference.lossWeight = memberLossIt->lossWeight;
            memberReference.quantile = memberLossIt->quantile;
            if (!trainingRunsReportableLossesCompatible(reference, memberReference)) {
                throw std::runtime_error(context + " member " + std::to_string(memberIndex) +
                                         " resolved graph loss '" + referenceLoss.lossName +
                                         "' differently than the reference member. Reference: " +
                                         trainingRunsReportableLossDescription(reference) + "; member: " +
                                         trainingRunsReportableLossDescription(memberReference) + ".");
            }
        }

        const std::vector<ResolvedEnsembleMetric> memberMetrics = resolveTrainingRunsSelectedMetricReports(
            memberAvailableMetrics,
            memberSelections.metricNames,
            context + " member " + std::to_string(memberIndex));
        for (const ResolvedEnsembleMetric& referenceMetric : artifacts.metrics) {
            const auto memberMetricIt = std::find_if(memberMetrics.begin(), memberMetrics.end(), [&](const ResolvedEnsembleMetric& candidate) {
                return candidate.metricName == referenceMetric.metricName;
            });
            if (memberMetricIt == memberMetrics.end()) {
                throw std::runtime_error(context + " member " + std::to_string(memberIndex) +
                                         " is missing graph metric '" + referenceMetric.metricName + "'.");
            }
            NetworkMetricReference reference;
            reference.metricName = referenceMetric.metricName;
            reference.predictionOutputName = referenceMetric.predictionOutputName;
            reference.targetInputName = referenceMetric.targetInputName;
            reference.inputSourceName = referenceMetric.inputSourceName;
            reference.metricLayerType = referenceMetric.metricType;
            NetworkMetricReference memberReference;
            memberReference.metricName = memberMetricIt->metricName;
            memberReference.predictionOutputName = memberMetricIt->predictionOutputName;
            memberReference.targetInputName = memberMetricIt->targetInputName;
            memberReference.inputSourceName = memberMetricIt->inputSourceName;
            memberReference.metricLayerType = memberMetricIt->metricType;
            if (!trainingRunsReportableMetricsCompatible(reference, memberReference)) {
                throw std::runtime_error(context + " member " + std::to_string(memberIndex) +
                                         " resolved graph metric '" + referenceMetric.metricName +
                                         "' differently than the reference member. Reference: " +
                                         trainingRunsReportableMetricDescription(reference) + "; member: " +
                                         trainingRunsReportableMetricDescription(memberReference) + ".");
            }
        }
    }

    artifacts.evaluator = buildTrainingRunsComposedEnsembleEvaluatorForReports(
        artifacts.memberNetworks, artifacts.weights, artifacts.losses, artifacts.metrics);
    if (artifacts.losses.empty() && artifacts.metrics.empty()) {
        return artifacts;
    }
    artifacts.placedEvaluator = placeInferenceNetworkWithSerializedStartup(
        *artifacts.evaluator.network,
        static_cast<uint32_t>(batchSize));
    return artifacts;
}

uint64_t batchRowsForEvaluatorInputs(const Batch& batch, const std::vector<std::string>& inputNames, const std::string& context) {
    if (inputNames.empty()) {
        throw std::runtime_error(context + " has no evaluator inputs from which to determine batch size.");
    }
    const std::string& inputName = inputNames.front();
    if (!batch.contains(inputName)) {
        throw std::runtime_error(context + " is missing evaluator input '" + inputName + "'.");
    }
    const BatchValue& value = batch.at(inputName);
    if (std::holds_alternative<ThorImplementation::Tensor>(value)) {
        const ThorImplementation::Tensor& tensor = std::get<ThorImplementation::Tensor>(value);
        const std::vector<uint64_t> dimensions = tensor.getDimensions();
        if (dimensions.empty()) {
            throw std::runtime_error(context + " evaluator input '" + inputName + "' has empty dimensions.");
        }
        return dimensions.front();
    }
    if (std::holds_alternative<ThorImplementation::RaggedTensor>(value)) {
        return std::get<ThorImplementation::RaggedTensor>(value).getBatchSize();
    }
    throw std::runtime_error(context + " evaluator input '" + inputName + "' has an unsupported value type.");
}

ThorImplementation::Tensor tensorOnCpuForLossReadback(ThorImplementation::Tensor tensor) {
    if (tensor.getPlacement().getMemDevice() == ThorImplementation::TensorPlacement::MemDevices::CPU) {
        return tensor;
    }
    ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
    ThorImplementation::Tensor cpuTensor(cpuPlacement, tensor.getDescriptor());
    Stream stream = Stream::getNextDownloadStream(tensor.getPlacement().getDeviceNum());
    cpuTensor.copyFromAsync(tensor, stream);
    stream.synchronize();
    return cpuTensor;
}

double tensorElementAsDoubleForLossReadback(const ThorImplementation::Tensor& tensor, uint64_t index, const std::string& context) {
    using ThorImplementation::DataType;
    switch (tensor.getDataType()) {
        case DataType::FP16:
            return static_cast<double>(__half2float(tensor.getMemPtr<half>()[index]));
        case DataType::BF16:
            return static_cast<double>(__bfloat162float(tensor.getMemPtr<__nv_bfloat16>()[index]));
        case DataType::FP32:
            return static_cast<double>(tensor.getMemPtr<float>()[index]);
        case DataType::FP64:
            return tensor.getMemPtr<double>()[index];
        default:
            throw std::runtime_error(context + " produced a non-floating-point loss tensor.");
    }
}

double tensorMeanAsDouble(ThorImplementation::Tensor tensor, const std::string& context) {
    ThorImplementation::Tensor cpuTensor = tensorOnCpuForLossReadback(std::move(tensor));
    if (cpuTensor.getTotalNumElements() == 0) {
        throw std::runtime_error(context + " produced an empty loss tensor.");
    }
    double sum = 0.0;
    for (uint64_t i = 0; i < cpuTensor.getTotalNumElements(); ++i) {
        const double value = tensorElementAsDoubleForLossReadback(cpuTensor, i, context);
        if (!std::isfinite(value)) {
            throw std::runtime_error(context + " produced a non-finite loss value.");
        }
        sum += value;
    }
    return sum / static_cast<double>(cpuTensor.getTotalNumElements());
}

struct ComposedEnsembleEvaluationMetrics {
    std::map<std::string, std::optional<double>> lossValues{};
    std::map<std::string, std::optional<double>> metricValues{};
    std::optional<double> overallLoss{};
    uint64_t batches = 0;
    uint64_t rows = 0;
};

Batch inferenceBatchForInputBindings(const std::vector<std::string>& inputNames,
                                     const std::vector<TrainingInputBinding>& inputBindings,
                                     const Batch& sourceBatch,
                                     const std::string& missingInputContext);

ComposedEnsembleEvaluationMetrics evaluateComposedEnsembleReportsOnSession(
    const TrainingRunsComposedEvaluatorArtifacts& artifacts,
    BatchSession& session,
    ExampleType exampleType,
    const std::vector<TrainingInputBinding>& inputBindings) {
    ComposedEnsembleEvaluationMetrics metrics;
    if (artifacts.losses.empty() && artifacts.metrics.empty()) {
        return metrics;
    }
    if (artifacts.placedEvaluator == nullptr) {
        throw std::runtime_error("TrainingRuns composed ensemble evaluator is not placed.");
    }
    if (session.getBatchSize() != artifacts.batchSize) {
        throw std::runtime_error("TrainingRuns composed ensemble evaluation requires session batch_size=" +
                                 std::to_string(artifacts.batchSize) + ", got " + std::to_string(session.getBatchSize()) + ".");
    }

    std::map<std::string, double> weightedLossSums;
    std::map<std::string, uint64_t> weightedLossRows;
    for (const ResolvedEnsembleLoss& loss : artifacts.losses) {
        weightedLossSums[loss.lossName] = 0.0;
        weightedLossRows[loss.lossName] = 0;
    }

    std::map<std::string, double> weightedMetricSums;
    std::map<std::string, uint64_t> weightedMetricRows;
    for (const ResolvedEnsembleMetric& metric : artifacts.metrics) {
        weightedMetricSums[metric.metricName] = 0.0;
        weightedMetricRows[metric.metricName] = 0;
    }

    const uint64_t batchesPerEpoch = session.getNumBatchesPerEpoch(exampleType);
    // Evaluation reports are full-population reports. Request every batch in the
    // split explicitly from this fresh session.
    for (uint64_t batchNum = 0; batchNum < batchesPerEpoch; ++batchNum) {
        uint64_t requestedBatchNum = batchNum;
        BatchLease lease = session.leaseBatch(exampleType, requestedBatchNum);
        if (requestedBatchNum != batchNum) {
            throw std::runtime_error("TrainingRuns composed ensemble evaluation session did not return requested batch " +
                                     std::to_string(batchNum) + " for full-population evaluation.");
        }
        const Batch& batch = lease.get();
        Batch evaluatorBatch = inferenceBatchForInputBindings(artifacts.evaluator.externalInputNames,
                                                              inputBindings,
                                                              batch,
                                                              "TrainingRuns composed ensemble evaluation batch");
        const uint64_t rows = batchRowsForEvaluatorInputs(evaluatorBatch,
                                                          artifacts.evaluator.externalInputNames,
                                                          "TrainingRuns composed ensemble evaluation batch");
        if (rows == 0) {
            continue;
        }
        std::map<std::string, ThorImplementation::Tensor> outputs = artifacts.placedEvaluator->infer(evaluatorBatch);
        for (const ResolvedEnsembleLoss& loss : artifacts.losses) {
            auto outputIt = outputs.find(loss.lossName);
            if (outputIt == outputs.end()) {
                throw std::runtime_error("TrainingRuns composed ensemble evaluator did not produce reported loss '" + loss.lossName + "'.");
            }
            const double lossValue = tensorMeanAsDouble(std::move(outputIt->second),
                                                        "TrainingRuns composed ensemble evaluator reported loss '" + loss.lossName + "'");
            weightedLossSums[loss.lossName] += lossValue * static_cast<double>(rows);
            weightedLossRows[loss.lossName] += rows;
        }
        for (const ResolvedEnsembleMetric& metric : artifacts.metrics) {
            auto outputIt = outputs.find(metric.metricName);
            if (outputIt == outputs.end()) {
                throw std::runtime_error("TrainingRuns composed ensemble evaluator did not produce reported metric '" + metric.metricName + "'.");
            }
            const double metricValue = tensorMeanAsDouble(std::move(outputIt->second),
                                                          "TrainingRuns composed ensemble evaluator reported metric '" + metric.metricName + "'");
            weightedMetricSums[metric.metricName] += metricValue * static_cast<double>(rows);
            weightedMetricRows[metric.metricName] += rows;
        }
        metrics.batches += 1;
        metrics.rows += rows;
    }

    std::vector<std::optional<double>> namedValuesForOverall;
    namedValuesForOverall.reserve(artifacts.losses.size());
    for (const ResolvedEnsembleLoss& loss : artifacts.losses) {
        auto rowsIt = weightedLossRows.find(loss.lossName);
        if (rowsIt == weightedLossRows.end() || rowsIt->second == 0) {
            metrics.lossValues[loss.lossName] = std::nullopt;
            namedValuesForOverall.push_back(std::nullopt);
            continue;
        }
        const double value = weightedLossSums.at(loss.lossName) / static_cast<double>(rowsIt->second);
        metrics.lossValues[loss.lossName] = value;
        namedValuesForOverall.push_back(value);
    }
    metrics.overallLoss = weightedLossSumFromWeightedLossValues(namedValuesForOverall);

    for (const ResolvedEnsembleMetric& metric : artifacts.metrics) {
        auto rowsIt = weightedMetricRows.find(metric.metricName);
        if (rowsIt == weightedMetricRows.end() || rowsIt->second == 0) {
            metrics.metricValues[metric.metricName] = std::nullopt;
            continue;
        }
        metrics.metricValues[metric.metricName] = weightedMetricSums.at(metric.metricName) / static_cast<double>(rowsIt->second);
    }
    return metrics;
}

void applyComposedEvaluationMetricsToEnsemble(TrainingEnsembleResult& ensemble,
                                              const ComposedEnsembleEvaluationMetrics& metrics,
                                              bool testPhase) {
    std::vector<std::optional<double>> namedValuesForOverall;
    namedValuesForOverall.reserve(ensemble.namedMetrics.size());
    for (TrainingNamedMetricResult& namedMetric : ensemble.namedMetrics) {
        auto valueIt = metrics.lossValues.find(namedMetric.name);
        const std::optional<double> value = valueIt == metrics.lossValues.end() ? std::optional<double>{} : valueIt->second;
        if (testPhase) {
            namedMetric.testValue = value;
        } else {
            namedMetric.trainValue = value;
        }
        namedValuesForOverall.push_back(value);
    }
    const std::optional<double> overall = weightedLossSumFromWeightedLossValues(namedValuesForOverall);
    if (testPhase) {
        ensemble.ensembleTestLoss = overall;
    } else {
        ensemble.ensembleTrainingLoss = overall;
    }

    for (TrainingNamedMetricResult& namedMetric : ensemble.namedGraphMetrics) {
        auto valueIt = metrics.metricValues.find(namedMetric.name);
        const std::optional<double> value = valueIt == metrics.metricValues.end() ? std::optional<double>{} : valueIt->second;
        if (testPhase) {
            namedMetric.testValue = value;
        } else {
            namedMetric.trainValue = value;
        }
    }
}

void retainNamedLossesAvailableInArtifacts(TrainingEnsembleResult& ensemble,
                                            const std::vector<ResolvedEnsembleLoss>& losses) {
    std::set<std::string> activeLossNames;
    for (const ResolvedEnsembleLoss& loss : losses) {
        activeLossNames.insert(loss.lossName);
    }
    ensemble.namedMetrics.erase(
        std::remove_if(ensemble.namedMetrics.begin(),
                       ensemble.namedMetrics.end(),
                       [&](const TrainingNamedMetricResult& metric) {
                           return activeLossNames.count(metric.name) == 0;
                       }),
        ensemble.namedMetrics.end());
}

void retainNamedGraphMetricsAvailableInArtifacts(TrainingEnsembleResult& ensemble,
                                                 const std::vector<ResolvedEnsembleMetric>& metrics) {
    std::set<std::string> activeMetricNames;
    for (const ResolvedEnsembleMetric& metric : metrics) {
        activeMetricNames.insert(metric.metricName);
    }
    ensemble.namedGraphMetrics.erase(
        std::remove_if(ensemble.namedGraphMetrics.begin(),
                       ensemble.namedGraphMetrics.end(),
                       [&](const TrainingNamedMetricResult& metric) {
                           return activeMetricNames.count(metric.name) == 0;
                       }),
        ensemble.namedGraphMetrics.end());
}


Batch inferenceBatchForInputBindings(const std::vector<std::string>& inputNames,
                                     const std::vector<TrainingInputBinding>& inputBindings,
                                     const Batch& sourceBatch,
                                     const std::string& missingInputContext) {
    std::map<std::string, std::string> batchInputByNetworkInput;
    for (const TrainingInputBinding& binding : inputBindings) {
        if (!binding.isInitialized()) {
            throw std::runtime_error(missingInputContext + " received an uninitialized dataset input binding.");
        }
        auto [it, inserted] = batchInputByNetworkInput.emplace(
            binding.getNetworkInputName(), binding.getBatchInputName());
        if (!inserted && it->second != binding.getBatchInputName()) {
            throw std::runtime_error(missingInputContext + " received conflicting dataset input bindings for NetworkInput '" +
                                     binding.getNetworkInputName() + "'.");
        }
    }

    Batch inferenceBatch;
    for (const std::string& inputName : inputNames) {
        const auto bindingIt = batchInputByNetworkInput.find(inputName);
        if (bindingIt == batchInputByNetworkInput.end()) {
            throw std::runtime_error(missingInputContext + " has no dataset binding for inference input '" + inputName + "'.");
        }
        const std::string& batchInputName = bindingIt->second;
        if (!sourceBatch.contains(batchInputName)) {
            throw std::runtime_error(missingInputContext + " is missing dataset field '" + batchInputName +
                                     "' bound to inference input '" + inputName + "'.");
        }
        const BatchValue& value = sourceBatch.at(batchInputName);
        if (std::holds_alternative<ThorImplementation::Tensor>(value)) {
            inferenceBatch.insert(inputName, std::get<ThorImplementation::Tensor>(value));
        } else if (std::holds_alternative<ThorImplementation::RaggedTensor>(value)) {
            inferenceBatch.insert(inputName, std::get<ThorImplementation::RaggedTensor>(value));
        } else {
            throw std::runtime_error(missingInputContext + " dataset field '" + batchInputName +
                                     "' bound to input '" + inputName + "' has an unsupported value type.");
        }
    }
    return inferenceBatch;
}

bool compiledDatasetInputBindingsEqual(const CompiledDatasetInputBindings& lhs,
                                       const CompiledDatasetInputBindings& rhs) {
    return lhs.trainingInputBindings == rhs.trainingInputBindings &&
           lhs.requiredFieldIds == rhs.requiredFieldIds;
}


}  // namespace


void TrainingRuns::validateTestData(const TrainingData& data) const {
    if (data.getBatching().getBatchSize() == 0) {
        throw std::runtime_error("TrainingRuns test_data has batch_size=0.");
    }
    data.requireNonEmptyPartition(ExampleType::TEST, "TrainingRuns test_data");
}

void TrainingRuns::evaluateEnsembles(std::vector<TrainingRunResult>& results,
                                     std::map<std::string, TrainingEnsembleResult>& ensembleResultsByGroup) const {
    if (ensembleResultsByGroup.empty()) {
        return;
    }

    const std::map<std::string, std::vector<EnsembleMemberSpecRef>> byGroup = completedEnsembleMembersByGroup(runs, results);
    for (const auto& [groupName, members] : byGroup) {
        if (members.empty()) {
            continue;
        }
        auto ensembleIt = ensembleResultsByGroup.find(groupName);
        if (ensembleIt == ensembleResultsByGroup.end()) {
            continue;
        }

        bool hasEvaluationData = false;
        for (const EnsembleMemberSpecRef& sourceMember : members) {
            if (sourceMember.spec != nullptr && sourceMember.spec->trainer != nullptr &&
                sourceMember.spec->trainer->trainingData != nullptr) {
                hasEvaluationData = true;
                break;
            }
        }
        if (!hasEvaluationData) {
            ensembleIt->second.ensembleTrainingLoss = std::nullopt;
            continue;
        }

        if (ensembleIt->second.namedMetrics.empty() && ensembleIt->second.namedGraphMetrics.empty()) {
            ensembleIt->second.ensembleTrainingLoss = std::nullopt;
            continue;
        }

        std::optional<TrainingRunsComposedEvaluatorArtifacts> composedArtifactsForCurrentBatchSize;
        uint64_t currentComposedArtifactsBatchSize = 0;
        bool retainedReportableArtifacts = false;
        auto composedArtifactsForBatchSize = [&](uint64_t batchSize) -> TrainingRunsComposedEvaluatorArtifacts& {
            if (!composedArtifactsForCurrentBatchSize.has_value() || currentComposedArtifactsBatchSize != batchSize) {
                // A placed composed evaluator owns GPU allocations for a concrete batch shape.  Keep only
                // the currently needed shape resident so a fold with a different validation population size
                // releases the previous evaluator before placing the next one.
                composedArtifactsForCurrentBatchSize.reset();
                currentComposedArtifactsBatchSize = 0;
                composedArtifactsForCurrentBatchSize.emplace(loadTrainingRunsComposedEvaluatorArtifacts(
                    members,
                    batchSize,
                    reportOrderForGroup(groupName),
                    "TrainingRuns composed ensemble training-population evaluation for ensemble_group '" + groupName +
                        "' batch_size=" + std::to_string(batchSize)));
                currentComposedArtifactsBatchSize = batchSize;
            }
            if (!retainedReportableArtifacts) {
                retainNamedLossesAvailableInArtifacts(ensembleIt->second, composedArtifactsForCurrentBatchSize->losses);
                retainNamedGraphMetricsAvailableInArtifacts(ensembleIt->second, composedArtifactsForCurrentBatchSize->metrics);
                retainedReportableArtifacts = true;
            }
            return *composedArtifactsForCurrentBatchSize;
        };

        std::map<std::string, std::vector<std::optional<double>>> sourcePopulationLossesByName;
        std::map<std::string, std::vector<std::optional<double>>> sourcePopulationMetricValuesByName;
        std::vector<double> sourcePopulationRowWeights;
        sourcePopulationRowWeights.reserve(members.size());
        for (const TrainingNamedMetricResult& metric : ensembleIt->second.namedMetrics) {
            sourcePopulationLossesByName[metric.name].reserve(members.size());
        }
        for (const TrainingNamedMetricResult& metric : ensembleIt->second.namedGraphMetrics) {
            sourcePopulationMetricValuesByName[metric.name].reserve(members.size());
        }

        for (const EnsembleMemberSpecRef& sourceMember : members) {
            if (sourceMember.spec == nullptr || sourceMember.spec->trainer == nullptr ||
                sourceMember.spec->trainer->trainingData == nullptr) {
                sourcePopulationRowWeights.push_back(0.0);
                for (const TrainingNamedMetricResult& metric : ensembleIt->second.namedMetrics) {
                    sourcePopulationLossesByName[metric.name].push_back(std::nullopt);
                }
                for (const TrainingNamedMetricResult& metric : ensembleIt->second.namedGraphMetrics) {
                    sourcePopulationMetricValuesByName[metric.name].push_back(std::nullopt);
                }
                continue;
            }

            const TrainingData& sourceData = *sourceMember.spec->trainer->trainingData;
            const CompiledDatasetInputBindings sourceBindings =
                sourceMember.spec->trainer->resolveDatasetInputsForData(sourceData, /*inferenceOnly=*/true);
            std::shared_ptr<BatchSession> sourceSession = sourceData.openSession(
                sourceMember.spec->trainer->getRuntimeConfig().maxInFlightBatches,
                sourceBindings.requiredFieldIds);
            TrainingRunsComposedEvaluatorArtifacts& sourceArtifacts = composedArtifactsForBatchSize(
                sourceSession->getBatchSize());
            ComposedEnsembleEvaluationMetrics sourceMetrics = evaluateComposedEnsembleReportsOnSession(
                sourceArtifacts, *sourceSession, ExampleType::VALIDATE, sourceBindings.trainingInputBindings);
            // The composed evaluator has already used ensemble member weights to
            // form predictions.  Across source validation splits, combine by
            // evaluated rows so ensemble_train_* is the validation-union report.
            sourcePopulationRowWeights.push_back(static_cast<double>(sourceMetrics.rows));
            for (const TrainingNamedMetricResult& metric : ensembleIt->second.namedMetrics) {
                auto valueIt = sourceMetrics.lossValues.find(metric.name);
                sourcePopulationLossesByName[metric.name].push_back(
                    valueIt == sourceMetrics.lossValues.end() ? std::optional<double>{} : valueIt->second);
            }
            for (const TrainingNamedMetricResult& metric : ensembleIt->second.namedGraphMetrics) {
                auto valueIt = sourceMetrics.metricValues.find(metric.name);
                sourcePopulationMetricValuesByName[metric.name].push_back(
                    valueIt == sourceMetrics.metricValues.end() ? std::optional<double>{} : valueIt->second);
            }
        }

        std::vector<std::optional<double>> namedTrainValuesForOverall;
        namedTrainValuesForOverall.reserve(ensembleIt->second.namedMetrics.size());
        for (TrainingNamedMetricResult& metric : ensembleIt->second.namedMetrics) {
            metric.trainValue = weightedAverage(sourcePopulationLossesByName[metric.name], sourcePopulationRowWeights);
            namedTrainValuesForOverall.push_back(metric.trainValue);
        }
        ensembleIt->second.ensembleTrainingLoss = weightedLossSumFromWeightedLossValues(namedTrainValuesForOverall);
        for (TrainingNamedMetricResult& metric : ensembleIt->second.namedGraphMetrics) {
            metric.trainValue = weightedAverage(sourcePopulationMetricValuesByName[metric.name], sourcePopulationRowWeights);
        }
    }
}

struct MemberGraphEvaluationMetrics {
    std::optional<double> loss{};
    std::map<std::string, std::optional<double>> lossValues{};
    std::map<std::string, std::optional<double>> metricValues{};
    uint64_t batches = 0;
    uint64_t rows = 0;
};

std::vector<MemberGraphEvaluationMetrics> evaluateMemberGraphReportsOnData(
    const std::vector<EnsembleMemberSpecRef>& members,
    const std::vector<CompiledDatasetInputBindings>& memberBindings,
    uint64_t batchSize,
    const std::vector<std::string>& requestedReportNames,
    const TrainingData& data,
    ExampleType exampleType,
    const std::string& context) {
    if (memberBindings.size() != members.size()) {
        throw std::runtime_error(context + " requires one compiled dataset binding set per member.");
    }
    std::vector<MemberGraphEvaluationMetrics> memberMetrics;
    memberMetrics.reserve(members.size());
    for (size_t memberIndex = 0; memberIndex < members.size(); ++memberIndex) {
        const EnsembleMemberSpecRef& member = members[memberIndex];
        MemberGraphEvaluationMetrics result;
        if (member.spec == nullptr || member.spec->trainer == nullptr) {
            memberMetrics.push_back(result);
            continue;
        }
        TrainingRunsComposedEvaluatorArtifacts artifacts = loadTrainingRunsComposedEvaluatorArtifacts(
            std::vector<EnsembleMemberSpecRef>{member},
            batchSize,
            requestedReportNames,
            context + " member '" + member.spec->runName + "'");
        if (!artifacts.losses.empty() || !artifacts.metrics.empty()) {
            std::shared_ptr<BatchSession> session = data.openSession(
                member.spec->trainer->getRuntimeConfig().maxInFlightBatches,
                memberBindings[memberIndex].requiredFieldIds);
            ComposedEnsembleEvaluationMetrics metrics = evaluateComposedEnsembleReportsOnSession(
                artifacts, *session, exampleType, memberBindings[memberIndex].trainingInputBindings);
            result.loss = metrics.overallLoss;
            result.lossValues = std::move(metrics.lossValues);
            result.metricValues = std::move(metrics.metricValues);
            result.batches = metrics.batches;
            result.rows = metrics.rows;
        }
        memberMetrics.push_back(std::move(result));
    }
    return memberMetrics;
}

void applyGraphEvaluationMemberTestStats(std::vector<TrainingRunResult>& results,
                                         TrainingEnsembleResult& ensemble,
                                         const std::vector<EnsembleMemberSpecRef>& members,
                                         const TrainingData& testData,
                                         const std::vector<MemberGraphEvaluationMetrics>& memberMetrics) {
    const uint64_t batchSize = testData.getBatching().getBatchSize();
    const uint64_t testExamples = testData.getSplits().getTest().size();
    const uint64_t stepsPerEpoch = (testExamples + batchSize - 1) / batchSize;

    for (size_t i = 0; i < members.size(); ++i) {
        if (members[i].spec == nullptr || members[i].spec->trainer == nullptr) {
            continue;
        }
        const MemberGraphEvaluationMetrics metrics = i < memberMetrics.size() ? memberMetrics[i] : MemberGraphEvaluationMetrics{};
        TrainingRunResult& runResult = results[members[i].runIndex];
        TrainingStatsSnapshot testStats;
        if (runResult.finalTestStats.has_value()) {
            testStats = *runResult.finalTestStats;
        }
        if (members[i].result != nullptr && members[i].result->savedModelNetworkName.has_value()) {
            testStats.networkName = *members[i].result->savedModelNetworkName;
        } else if (members[i].spec->trainer->getNetwork() != nullptr) {
            testStats.networkName = members[i].spec->trainer->getNetwork()->getNetworkName();
        } else {
            testStats.networkName = members[i].spec->runName;
        }
        testStats.datasetName = testData.getDatasetName();
        testStats.phase = TrainingEventPhase::TEST;
        testStats.epoch = 1;
        testStats.epochs = 1;
        testStats.step = metrics.batches;
        testStats.stepInEpoch = metrics.batches;
        testStats.stepsPerEpoch = stepsPerEpoch;
        testStats.batchSize = batchSize;
        testStats.samplesProcessed = metrics.rows;
        testStats.loss = metrics.loss;
        for (const auto& [name, value] : metrics.lossValues) {
            if (value.has_value()) {
                testStats.metrics[name] = *value;
            }
        }
        for (const auto& [name, value] : metrics.metricValues) {
            if (value.has_value()) {
                testStats.metrics[name] = *value;
            }
        }
        runResult.finalTestStats = std::move(testStats);
    }

    for (TrainingEnsembleMemberResult& memberResult : ensemble.members) {
        for (size_t i = 0; i < members.size(); ++i) {
            if (members[i].spec != nullptr && memberResult.runName == members[i].spec->runName) {
                const MemberGraphEvaluationMetrics metrics = i < memberMetrics.size() ? memberMetrics[i] : MemberGraphEvaluationMetrics{};
                memberResult.status = results[members[i].runIndex].status;
                memberResult.finalTestLoss = metrics.loss;
                memberResult.finalTestMetrics.clear();
                for (const auto& [name, value] : metrics.metricValues) {
                    if (value.has_value()) {
                        memberResult.finalTestMetrics[name] = *value;
                    }
                }
            }
        }
    }
}

void TrainingRuns::evaluateEnsemblesOnTestData(
    std::vector<TrainingRunResult>& results,
    std::map<std::string, TrainingEnsembleResult>& ensembleResultsByGroup,
    std::shared_ptr<const TrainingData> testData) const {
    if (testData == nullptr || ensembleResultsByGroup.empty()) {
        return;
    }

    const uint64_t batchSize = testData->getBatching().getBatchSize();
    const std::map<std::string, std::vector<EnsembleMemberSpecRef>> byGroup =
        completedEnsembleMembersByGroup(runs, results);
    for (const auto& [groupName, members] : byGroup) {
        if (members.empty()) {
            continue;
        }
        auto ensembleIt = ensembleResultsByGroup.find(groupName);
        if (ensembleIt == ensembleResultsByGroup.end()) {
            continue;
        }

        if (ensembleIt->second.namedMetrics.empty() && ensembleIt->second.namedGraphMetrics.empty()) {
            ensembleIt->second.ensembleTestLoss = std::nullopt;
            continue;
        }

        TrainingRunsComposedEvaluatorArtifacts composedArtifacts = loadTrainingRunsComposedEvaluatorArtifacts(
            members,
            batchSize,
            reportOrderForGroup(groupName),
            "TrainingRuns composed ensemble test evaluation for ensemble_group '" + groupName + "'");
        retainNamedLossesAvailableInArtifacts(ensembleIt->second, composedArtifacts.losses);
        retainNamedGraphMetricsAvailableInArtifacts(ensembleIt->second, composedArtifacts.metrics);

        std::vector<CompiledDatasetInputBindings> memberBindings;
        memberBindings.reserve(members.size());
        for (const EnsembleMemberSpecRef& member : members) {
            if (member.spec == nullptr || member.spec->trainer == nullptr) {
                throw std::runtime_error(
                    "TrainingRuns composed ensemble test evaluation requires a trainer for every completed member.");
            }
            memberBindings.push_back(
                member.spec->trainer->resolveDatasetInputsForData(*testData, /*inferenceOnly=*/true));
        }
        const CompiledDatasetInputBindings& composedBindings = memberBindings.front();
        for (size_t memberIndex = 1; memberIndex < memberBindings.size(); ++memberIndex) {
            if (!compiledDatasetInputBindingsEqual(composedBindings, memberBindings[memberIndex])) {
                throw std::runtime_error(
                    "TrainingRuns composed ensemble test evaluation requires every member to resolve the same "
                    "NetworkInput-to-dataset-field bindings for the shared test_data recipe.");
            }
        }

        std::shared_ptr<BatchSession> composedSession = testData->openSession(
            members.front().spec->trainer->getRuntimeConfig().maxInFlightBatches,
            composedBindings.requiredFieldIds);
        ComposedEnsembleEvaluationMetrics metrics = evaluateComposedEnsembleReportsOnSession(
            composedArtifacts, *composedSession, ExampleType::TEST, composedBindings.trainingInputBindings);
        applyComposedEvaluationMetricsToEnsemble(ensembleIt->second, metrics, /*testPhase=*/true);

        std::vector<MemberGraphEvaluationMetrics> memberMetrics = evaluateMemberGraphReportsOnData(
            members,
            memberBindings,
            batchSize,
            reportOrderForGroup(groupName),
            *testData,
            ExampleType::TEST,
            "TrainingRuns composed ensemble per-member test evaluation for ensemble_group '" + groupName + "'");
        applyGraphEvaluationMemberTestStats(results, ensembleIt->second, members, *testData, memberMetrics);
    }
}


void TrainingRuns::validateFitOptions(const TrainerFitOptions& options) const {
    for (const TrainingRunsSpec& spec : runs) {
        try {
            spec.trainer->validateFitOptions(options);
        } catch (const std::exception& e) {
            throw std::runtime_error("TrainingRuns run '" + spec.runName + "' has invalid fit options: " + e.what());
        }
    }
}

}  // namespace Thor
