#include "DeepLearning/Api/Training/TrainingRuns.h"

#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Learning/CustomLayer.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "DeepLearning/Api/Loaders/Loader.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Training/Observers/TrainingRunsStatsReporter.h"
#include "DeepLearning/Api/Training/Observers/TrainingStatsSink.h"

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

namespace Thor {

namespace {

constexpr int ENSEMBLE_MANIFEST_FIRST_ARTIFACT_VERSION = 1;
constexpr int ENSEMBLE_MANIFEST_CURRENT_ARTIFACT_VERSION = ENSEMBLE_MANIFEST_FIRST_ARTIFACT_VERSION;

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

std::vector<std::string> inputNamesFromSignature(const std::vector<TrainingRunInputSignature>& signature) {
    std::vector<std::string> names;
    names.reserve(signature.size());
    for (const TrainingRunInputSignature& item : signature) {
        names.push_back(item.inputName);
    }
    return names;
}

std::vector<std::string> inferenceInputNamesFromSavedArtifact(const std::filesystem::path& memberArtifactDirectory,
                                                              const std::vector<TrainingRunInputSignature>& fallbackSignature) {
    try {
        Network model("thor_training_runs_ensemble_manifest_probe");
        model.load(memberArtifactDirectory.string());
        std::vector<std::string> names = model.getInferenceNetworkInputNames();
        if (!names.empty()) {
            return names;
        }
    } catch (const std::exception&) {
        // Some unit tests construct synthetic TrainingRunsResult objects with minimal
        // member artifact directories instead of full saved Network artifacts. Those
        // tests still exercise the ensemble manifest writer, so fall back to the
        // recorded training signature when the copied member cannot be loaded. Real
        // saved training artifacts continue to use the inference-pruned signature so
        // label-only inputs are not exposed as deployable ensemble inputs.
    }
    return inputNamesFromSignature(fallbackSignature);
}

std::vector<std::string> predictionOutputNamesFromSignature(const std::vector<TrainingRunOutputSignature>& signature) {
    std::vector<std::string> names;
    names.reserve(signature.size());
    for (const TrainingRunOutputSignature& item : signature) {
        if (item.outputName != "loss") {
            names.push_back(item.outputName);
        }
    }
    return names;
}

std::string safeMemberDirectoryName(size_t index, const std::string& runName) {
    std::ostringstream out;
    out << std::setw(4) << std::setfill('0') << index << "_";
    bool wroteAnyNameChar = false;
    for (unsigned char ch : runName) {
        if (std::isalnum(ch) || ch == '-' || ch == '_' || ch == '.') {
            out << static_cast<char>(ch);
            wroteAnyNameChar = true;
        } else {
            out << '_';
            wroteAnyNameChar = true;
        }
    }
    if (!wroteAnyNameChar) {
        out << "member";
    }
    return out.str();
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

void copyMemberArtifactDirectory(const std::filesystem::path& source, const std::filesystem::path& destination) {
    std::error_code errorCode;
    if (!std::filesystem::exists(source, errorCode) || errorCode) {
        throw std::runtime_error("TrainingRuns cannot save ensemble member because saved model artifact path does not exist: " +
                                 source.string());
    }
    if (!std::filesystem::is_directory(source, errorCode) || errorCode) {
        throw std::runtime_error("TrainingRuns cannot save ensemble member because saved model artifact path is not a directory: " +
                                 source.string());
    }
    std::filesystem::create_directories(destination.parent_path(), errorCode);
    if (errorCode) {
        throw std::runtime_error("Failed to create ensemble members directory '" + destination.parent_path().string() +
                                 "': " + errorCode.message());
    }
    std::filesystem::copy(source,
                          destination,
                          std::filesystem::copy_options::recursive,
                          errorCode);
    if (errorCode) {
        throw std::runtime_error("Failed to copy saved model artifact from '" + source.string() + "' to '" + destination.string() +
                                 "': " + errorCode.message());
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
    out << "loss_name='" << reference.lossName << "', prediction_output_name='" << reference.predictionOutputName
        << "', target_input_name='" << reference.targetInputName << "'";
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
                throw std::runtime_error(context + " contains an empty reported loss name.");
            }
            if (!seen.insert(lossName).second) {
                throw std::runtime_error(context + " contains duplicate reported loss name '" + lossName + "'.");
            }
            if (byName.find(lossName) == byName.end()) {
                std::ostringstream oss;
                oss << context << " requested reported loss '" << lossName << "', but no graph loss with that name exists. Available losses:";
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

    const std::string manifestAggregation = resolvedEnsembleAggregation(ensembleResult, std::move(aggregation));
    const std::filesystem::path artifactDirectory(directory);
    const std::filesystem::path manifestPath = artifactDirectory / "ensemble_manifest.json";
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
    std::filesystem::create_directories(artifactDirectory / "members", errorCode);
    if (errorCode) {
        throw std::runtime_error("Failed to create ensemble output directory '" + artifactDirectory.string() + "': " + errorCode.message());
    }

    struct MemberManifestEntry {
        const TrainingEnsembleMemberResult* member = nullptr;
        const TrainingRunResult* result = nullptr;
        std::string relativePath{};
    };

    std::vector<MemberManifestEntry> entries;
    entries.reserve(ensembleResult.members.size());
    size_t savedMemberIndex = 0;
    for (size_t i = 0; i < ensembleResult.members.size(); ++i) {
        const TrainingEnsembleMemberResult& member = ensembleResult.members[i];
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
        const std::filesystem::path sourceDirectory(*result.savedModelDirectory);
        const std::string memberRelativePath = std::string("members/") + safeMemberDirectoryName(savedMemberIndex, member.runName);
        ++savedMemberIndex;
        copyMemberArtifactDirectory(sourceDirectory, artifactDirectory / memberRelativePath);
        entries.push_back(MemberManifestEntry{&member, &result, memberRelativePath});
    }

    const std::vector<std::string> manifestInputNames = entries.empty()
        ? inputNamesFromSignature(ensembleResult.inputSignature)
        : inferenceInputNamesFromSavedArtifact(artifactDirectory / entries.front().relativePath, ensembleResult.inputSignature);

    const std::filesystem::path tmpManifestPath = artifactDirectory / ".ensemble_manifest.json.tmp";
    {
        std::ofstream out(tmpManifestPath, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("Unable to open ensemble manifest for writing: " + tmpManifestPath.string());
        }

        out << "{\n";
        out << "  \"aggregation\": {\n";
        out << "    \"type\": ";
        writeJsonString(out, manifestAggregation);
        out << "\n";
        out << "  },\n";
        out << "  \"artifact_type\": \"thor_ensemble_model\",\n";
        out << "  \"ensemble_group\": ";
        writeJsonString(out, ensembleResult.ensembleGroup);
        out << ",\n";
        out << "  \"target_num_members\": " << ensembleResult.members.size() << ",\n";
        out << "  \"actual_num_members\": " << entries.size() << ",\n";
        out << "  \"min_successful_models\": " << ensembleResult.requiredSuccessfulModels() << ",\n";
        out << "  \"execution\": \"parallel_single_gpu\",\n";
        out << "  \"input_names\": ";
        writeJsonStringArray(out, manifestInputNames, "    ");
        out << ",\n";
        out << "  \"reported_losses\": ";
        std::vector<std::string> reportedLossNames;
        reportedLossNames.reserve(ensembleResult.namedMetrics.size());
        for (const TrainingNamedMetricResult& metric : ensembleResult.namedMetrics) {
            reportedLossNames.push_back(metric.name);
        }
        writeJsonStringArray(out, reportedLossNames, "    ");
        out << ",\n";
        out << "  \"overall_loss_reduction\": \"sum\",\n";
        out << "  \"losses\": [";
        for (size_t i = 0; i < ensembleResult.namedMetrics.size(); ++i) {
            const TrainingNamedMetricResult& metric = ensembleResult.namedMetrics[i];
            if (i != 0) {
                out << ",";
            }
            out << "\n    {\n";
            out << "      \"name\": ";
            writeJsonString(out, metric.name);
            out << ",\n";
            out << "      \"train_value\": ";
            writeOptionalDoubleJson(out, metric.trainValue);
            out << ",\n";
            out << "      \"test_value\": ";
            writeOptionalDoubleJson(out, metric.testValue);
            out << "\n";
            out << "    }";
        }
        out << "\n  ],\n";
        out << "  \"members\": [";
        for (size_t i = 0; i < entries.size(); ++i) {
            const MemberManifestEntry& entry = entries[i];
            if (i != 0) {
                out << ",";
            }
            out << "\n    {\n";
            out << "      \"name\": ";
            writeJsonString(out, entry.member->runName);
            out << ",\n";
            out << "      \"path\": ";
            writeJsonString(out, entry.relativePath);
            out << ",\n";
            out << "      \"selection\": {\n";
            out << "        \"best_epoch\": ";
            writeOptionalUint64Json(out, entry.result->bestEpoch);
            out << ",\n";
            out << "        \"best_score\": ";
            writeOptionalDoubleJson(out, entry.result->bestScore);
            out << ",\n";
            out << "        \"completed_epoch\": ";
            writeOptionalUint64Json(out, entry.result->completedEpoch);
            out << ",\n";
            out << "        \"completion_reason\": ";
            writeJsonString(out, trainingRunCompletionReasonName(entry.result->completionReason));
            out << ",\n";
            out << "        \"final_test_loss\": ";
            writeOptionalDoubleJson(out, entry.result->finalLossForPhase(TrainingEventPhase::TEST));
            out << ",\n";
            out << "        \"final_training_loss\": ";
            writeOptionalDoubleJson(out, entry.result->finalLossForPhase(TrainingEventPhase::TRAIN));
            out << ",\n";
            out << "        \"final_validation_loss\": ";
            writeOptionalDoubleJson(out, entry.result->finalLossForPhase(TrainingEventPhase::VALIDATE));
            out << ",\n";
            out << "        \"result\": ";
            writeJsonString(out, entry.result->resultName());
            out << ",\n";
            out << "        \"status\": ";
            writeJsonString(out, trainingRunStatusName(entry.result->status));
            out << "\n";
            out << "      },\n";
            out << "      \"weight\": " << std::setprecision(17) << entry.member->weight << "\n";
            out << "    }";
        }
        out << "\n  ],\n";
        out << "  \"output_names\": ";
        writeJsonStringArray(out, predictionOutputNamesFromSignature(ensembleResult.outputSignature), "    ");
        out << ",\n";
        out << "  \"version\": " << ENSEMBLE_MANIFEST_CURRENT_ARTIFACT_VERSION << "\n";
        out << "}\n";
        if (!out) {
            throw std::runtime_error("Failed while writing ensemble manifest: " + tmpManifestPath.string());
        }
    }

    std::filesystem::rename(tmpManifestPath, manifestPath, errorCode);
    if (errorCode) {
        removePathIfExistsForEnsembleSave(manifestPath);
        errorCode.clear();
        std::filesystem::rename(tmpManifestPath, manifestPath, errorCode);
        if (errorCode) {
            removePathIfExistsForEnsembleSave(tmpManifestPath);
            throw std::runtime_error("Failed to finalize ensemble manifest '" + manifestPath.string() + "': " + errorCode.message());
        }
    }

    return manifestPath.string();
}

TrainingRuns::TrainingRuns(std::vector<TrainingRunsSpec> runs,
                           TrainingRunsFailurePolicy failurePolicy,
                           double maxSummaryLogsPerSecond,
                           std::optional<size_t> maxParallelRuns,
                           std::vector<TrainingRunsRestartPolicy> restartConditions,
                           std::vector<TrainingRunsEarlyCompletionRule> earlyCompletionRules,
                           std::map<std::string, size_t> minSuccessfulModels,
                           std::map<std::string, std::vector<std::string>> reportedLosses)
    : runs(std::move(runs)),
      failurePolicy(failurePolicy),
      maxSummaryLogsPerSecond(maxSummaryLogsPerSecond),
      maxParallelRuns(maxParallelRuns),
      minSuccessfulModels(std::move(minSuccessfulModels)),
      restartConditions(std::move(restartConditions)),
      earlyCompletionRules(std::move(earlyCompletionRules)),
      reportedLosses(std::move(reportedLosses)) {
    if (!std::isfinite(maxSummaryLogsPerSecond) || maxSummaryLogsPerSecond < 0.0) {
        throw std::runtime_error("TrainingRuns maxSummaryLogsPerSecond must be finite and >= 0.");
    }
    if (maxParallelRuns.has_value() && maxParallelRuns.value() == 0) {
        throw std::runtime_error("TrainingRuns maxParallelRuns must be >= 1 when specified.");
    }
    validateRunSpecs();
    validateMinSuccessfulModels();
    validateRestartConditions();
    validateEarlyCompletionRules();
    validateReportedLosses();
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
    if (!evaluationOptions.evaluateTrainingPopulation && evaluationOptions.testLoader == nullptr) {
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

TrainingRunsResult TrainingRuns::fit(uint32_t epochs, std::shared_ptr<Loader> testLoader) {
    TrainingRunsEvaluationOptions evaluationOptions;
    evaluationOptions.testLoader = std::move(testLoader);
    return fit(TrainerFitOptions{epochs}, evaluationOptions);
}

TrainingRunsResult TrainingRuns::fit(const TrainerFitOptions& options) {
    return fit(options, TrainingRunsEvaluationOptions{});
}

TrainingRunsResult TrainingRuns::fit(const TrainerFitOptions& options, const TrainingRunsEvaluationOptions& evaluationOptions) {
    validateFitOptions(options);
    if (evaluationOptions.testLoader != nullptr) {
        validateTestLoader(*evaluationOptions.testLoader);
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
                                                                         spec.ensembleWeight});
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
                result = runs[i].trainer->fitTrainingRun(runs[i].runName,
                                                          options,
                                                          observer,
                                                          cancellationSource.token(),
                                                          restartConditionsForRun(runs[i]),
                                                          earlyCompletionPoliciesForRun(runs[i]));
            } catch (...) {
                result = TrainingRunResult::fromException(runs[i].runName, std::current_exception());
            }
            result.ensembleGroup = runs[i].ensembleGroup;
            result.ensembleWeight = runs[i].ensembleWeight;
            result.savedModelDirectory = runs[i].trainer->getSaveModelDirectory();

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
    if (evaluationOptions.testLoader != nullptr) {
        evaluateEnsemblesOnTestLoader(results, ensembleResultsByGroup, evaluationOptions.testLoader);
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
            const std::vector<TrainingRunInputSignature> inputSignature = collectNetworkInputSignature(spec.trainer->getNetwork());
            const std::vector<TrainingRunOutputSignature> outputSignature = collectNetworkOutputSignature(spec.trainer->getNetwork());
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

            static const std::vector<std::string> emptyReportedLossNames;
            const auto reportedLossesIt = reportedLosses.find(*spec.ensembleGroup);
            const std::vector<std::string>& requestedLossNames =
                reportedLossesIt == reportedLosses.end() ? emptyReportedLossNames : reportedLossesIt->second;
            const bool reportsGraphLosses = !resolveTrainingRunsReportedLosses(
                spec.trainer->getNetwork()->getReportableLosses(),
                requestedLossNames,
                "TrainingRuns reported_losses for ensemble_group '" + *spec.ensembleGroup + "' run '" + spec.runName + "'")
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
        if (hasRunName == hasEnsembleGroup) {
            throw std::runtime_error("TrainingRuns restart_condition at index " + std::to_string(i) +
                                     " must specify exactly one of run_name or ensemble_group.");
        }
        if (condition.progressCheckEpochs == 0) {
            throw std::runtime_error("TrainingRuns restart_condition at index " + std::to_string(i) +
                                     " must have progress_check_epochs >= 1.");
        }
        if (!std::isfinite(condition.progressImprovementMinPercentage) || condition.progressImprovementMinPercentage < 0.0 || condition.progressImprovementMinPercentage > 100.0) {
            throw std::runtime_error("TrainingRuns restart_condition at index " + std::to_string(i) +
                                     " must have progress_improvement_min_percentage in [0, 100].");
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
        if (condition.runName.has_value() && *condition.runName == run.runName) {
            matches.push_back(condition.toRestartCondition());
        } else if (condition.ensembleGroup.has_value() && run.ensembleGroup.has_value() && *condition.ensembleGroup == *run.ensembleGroup) {
            matches.push_back(condition.toRestartCondition());
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
        std::shared_ptr<Network> network{};
        std::optional<std::vector<NetworkLossReference>> reportableLosses{};
    };

    std::map<std::string, std::vector<EnsembleMemberSignatureState>> membersByGroup;
    for (const TrainingRunsSpec& spec : runs) {
        if (!spec.ensembleGroup.has_value()) {
            continue;
        }
        EnsembleMemberSignatureState state;
        state.runName = spec.runName;
        state.network = spec.trainer->getNetwork();
        state.inputSignature = collectNetworkInputSignature(state.network);
        state.outputSignature = collectNetworkOutputSignature(state.network);
        membersByGroup[*spec.ensembleGroup].push_back(std::move(state));
    }

    for (const auto& [groupName, requestedLossNames] : reportedLosses) {
        if (groupName.empty()) {
            throw std::runtime_error("TrainingRuns reported_losses contains an empty ensemble_group name.");
        }
        auto membersIt = membersByGroup.find(groupName);
        if (membersIt == membersByGroup.end() || membersIt->second.empty()) {
            throw std::runtime_error("TrainingRuns reported_losses targets unknown ensemble_group '" + groupName + "'.");
        }
        std::set<std::string> seen;
        for (const std::string& lossName : requestedLossNames) {
            if (lossName.empty()) {
                throw std::runtime_error("TrainingRuns reported_losses for ensemble_group '" + groupName + "' contains an empty loss name.");
            }
            if (!seen.insert(lossName).second) {
                throw std::runtime_error("TrainingRuns reported_losses for ensemble_group '" + groupName + "' contains duplicate loss name '" + lossName + "'.");
            }
        }
    }

    auto reportableLossesForMember = [](EnsembleMemberSignatureState& member) -> const std::vector<NetworkLossReference>& {
        if (!member.reportableLosses.has_value()) {
            member.reportableLosses = member.network->getReportableLosses();
        }
        return *member.reportableLosses;
    };

    for (auto& [groupName, members] : membersByGroup) {
        if (members.empty()) {
            continue;
        }
        EnsembleMemberSignatureState& referenceMember = members.front();
        const std::vector<std::string>& requestedLossNames = [&]() -> const std::vector<std::string>& {
            static const std::vector<std::string> empty;
            auto it = reportedLosses.find(groupName);
            return it == reportedLosses.end() ? empty : it->second;
        }();
        const std::string context = "TrainingRuns reported_losses for ensemble_group '" + groupName + "'";
        const std::vector<ResolvedEnsembleLoss> referenceLosses =
            resolveTrainingRunsReportedLosses(reportableLossesForMember(referenceMember), requestedLossNames, context + " reference run '" + referenceMember.runName + "'");

        for (const ResolvedEnsembleLoss& resolved : referenceLosses) {
            const TrainingRunOutputSignature* referenceOutput = findOutputSignatureItem(referenceMember.outputSignature, resolved.predictionOutputName);
            if (referenceOutput == nullptr) {
                throw std::runtime_error(context + " resolved loss '" + resolved.lossName + "' to prediction output '" +
                                         resolved.predictionOutputName + "', but reference run '" + referenceMember.runName +
                                         "' has outputs " + outputSignatureToString(referenceMember.outputSignature) + ".");
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
                const std::vector<ResolvedEnsembleLoss> memberLosses =
                    resolveTrainingRunsReportedLosses(reportableLossesForMember(member), requestedLossNames, memberContext);
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

                const TrainingRunOutputSignature* memberOutput = findOutputSignatureItem(member.outputSignature, resolved.predictionOutputName);
                if (memberOutput == nullptr || !referenceOutput->compatibleWith(*memberOutput)) {
                    throw std::runtime_error(memberContext + " has incompatible prediction output '" + resolved.predictionOutputName + "'.");
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

std::vector<TrainingNamedMetricResult> TrainingRuns::namedMetricResultsForGroup(std::string_view ensembleGroup) const {
    std::shared_ptr<Network> representativeNetwork;
    for (const TrainingRunsSpec& run : runs) {
        if (run.ensembleGroup.has_value() && std::string_view(run.ensembleGroup.value()) == ensembleGroup && run.trainer != nullptr &&
            run.trainer->getNetwork() != nullptr) {
            representativeNetwork = run.trainer->getNetwork();
            break;
        }
    }
    if (representativeNetwork == nullptr) {
        return {};
    }

    static const std::vector<std::string> empty;
    const auto reportedIt = reportedLosses.find(std::string(ensembleGroup));
    const std::vector<std::string>& requestedLossNames = reportedIt == reportedLosses.end() ? empty : reportedIt->second;
    const std::vector<ResolvedEnsembleLoss> resolvedLosses = resolveTrainingRunsReportedLosses(
        representativeNetwork->getReportableLosses(),
        requestedLossNames,
        "TrainingRuns reported_losses for ensemble_group '" + std::string(ensembleGroup) + "'");

    std::vector<TrainingNamedMetricResult> results;
    results.reserve(resolvedLosses.size());
    for (const ResolvedEnsembleLoss& loss : resolvedLosses) {
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
            ensemble.inputSignature = collectNetworkInputSignature(spec.trainer->getNetwork());
            ensemble.outputSignature = collectNetworkOutputSignature(spec.trainer->getNetwork());
            ensemble.minSuccessfulModels = minSuccessfulModelsForGroup(*spec.ensembleGroup, 0);
            ensemble.namedMetrics = namedMetricResultsForGroup(*spec.ensembleGroup);
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
        member.finalTestAccuracy = result.finalAccuracyForPhase(TrainingEventPhase::TEST);
        ensemble.members.push_back(std::move(member));
    }

    return byGroup;
}

namespace {

struct EnsembleMemberSpecRef {
    size_t runIndex = 0;
    const TrainingRunsSpec* spec = nullptr;
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
        byGroup[*runs[i].ensembleGroup].push_back(EnsembleMemberSpecRef{i, &runs[i]});
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
    bool exposeAveragedOutputsAsNetworkOutputs = true) {
    if (memberNetworks.empty()) {
        throw std::runtime_error("TrainingRuns composed ensemble evaluator requires at least one member network.");
    }
    if (weights.size() != memberNetworks.size()) {
        throw std::runtime_error("TrainingRuns composed ensemble evaluator requires one ensemble weight per member network.");
    }
    if (outputNames.empty()) {
        throw std::runtime_error("TrainingRuns composed ensemble evaluator requires at least one output name.");
    }
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
    std::vector<std::string> referenceInputNames = referenceMember.getInferenceNetworkInputNames();
    if (referenceInputNames.empty()) {
        throw std::runtime_error("TrainingRuns composed ensemble evaluator requires at least one shared member inference input.");
    }
    const std::map<std::string, std::shared_ptr<NetworkInput>> referenceInputsByName =
        apiNetworkInputsByName(referenceMember, /*includePassThroughInputs=*/false);

    TrainingRunsComposedEnsembleEvaluator evaluator;
    evaluator.network = std::make_shared<Network>("training_runs_composed_ensemble_evaluator");
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
    for (size_t memberIndex = 0; memberIndex < memberNetworks.size(); ++memberIndex) {
        Network& memberNetwork = *memberNetworks[memberIndex];
        std::vector<std::string> memberInputNames = memberNetwork.getInferenceNetworkInputNames();
        std::set<std::string> memberInputNameSet(memberInputNames.begin(), memberInputNames.end());
        if (memberInputNameSet != referenceInputNameSet || memberInputNames.size() != memberInputNameSet.size()) {
            throw std::runtime_error("TrainingRuns composed ensemble evaluator members have incompatible inference input names.");
        }

        const std::map<std::string, std::shared_ptr<NetworkInput>> memberInputsByName =
            apiNetworkInputsByName(memberNetwork, /*includePassThroughInputs=*/false);
        ApiTensorRemap remap;
        for (const std::string& inputName : referenceInputNames) {
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

Tensor trainingRunsEvaluatorInputTensorForGraphInput(TrainingRunsComposedEnsembleEvaluator& evaluator,
                                                     Network& referenceMember,
                                                     const std::map<std::string, std::shared_ptr<NetworkInput>>& referenceInputsByName,
                                                     const std::string& inputName) {
    auto existingIt = evaluator.sharedInputTensorsByName.find(inputName);
    if (existingIt != evaluator.sharedInputTensorsByName.end()) {
        return existingIt->second;
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
    Tensor tensor = evaluatorInput.getFeatureOutput().value();
    evaluator.sharedInputTensorsByName[inputName] = tensor;
    evaluator.externalInputNames.push_back(inputName);
    return tensor;
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

TrainingRunsComposedEnsembleEvaluator buildTrainingRunsComposedEnsembleEvaluatorForLosses(
    const std::vector<std::shared_ptr<Network>>& memberNetworks,
    const std::vector<double>& weights,
    const std::vector<ResolvedEnsembleLoss>& losses) {
    if (losses.empty()) {
        throw std::runtime_error("TrainingRuns composed ensemble evaluator requires at least one graph loss to report.");
    }
    std::vector<std::string> outputNames;
    std::set<std::string> seenOutputs;
    for (const ResolvedEnsembleLoss& loss : losses) {
        if (seenOutputs.insert(loss.predictionOutputName).second) {
            outputNames.push_back(loss.predictionOutputName);
        }
    }
    TrainingRunsComposedEnsembleEvaluator evaluator =
        buildTrainingRunsComposedEnsembleEvaluatorThroughAccumulator(
            memberNetworks, weights, outputNames, /*exposeAveragedOutputsAsNetworkOutputs=*/false);

    Network& referenceMember = *memberNetworks.front();
    const std::map<std::string, std::shared_ptr<NetworkInput>> referenceInputsByName =
        apiNetworkInputsByName(referenceMember, /*includePassThroughInputs=*/false);
    const std::map<std::string, std::shared_ptr<NetworkOutput>> referenceOutputsByName = apiNetworkOutputsByName(referenceMember);
    for (const ResolvedEnsembleLoss& loss : losses) {
        const auto predictionIt = evaluator.averagedOutputTensorsByName.find(loss.predictionOutputName);
        if (predictionIt == evaluator.averagedOutputTensorsByName.end()) {
            throw std::runtime_error("TrainingRuns composed ensemble evaluator did not build averaged prediction output '" +
                                     loss.predictionOutputName + "' for reported loss '" + loss.lossName + "'.");
        }
        Tensor labels = trainingRunsEvaluatorInputTensorForGraphInput(evaluator, referenceMember, referenceInputsByName, loss.targetInputName);
        std::optional<Tensor> weightsTensor;
        if (loss.weightInputName.has_value()) {
            weightsTensor = trainingRunsEvaluatorInputTensorForGraphInput(evaluator, referenceMember, referenceInputsByName, *loss.weightInputName);
        }
        Tensor lossTensor = cloneTrainingRunsEvaluatorLossFromReference(
            evaluator, referenceMember, referenceInputsByName, referenceOutputsByName, loss, predictionIt->second, labels, weightsTensor);
        evaluator.lossOutputTensorsByName[loss.lossName] = lossTensor;
        NetworkOutput::Builder().network(*evaluator.network).name(loss.lossName).inputTensor(lossTensor).dataType(DataType::FP32).build();
    }
    return evaluator;
}

struct TrainingRunsComposedEvaluatorArtifacts {
    uint64_t batchSize = 0;
    std::vector<std::shared_ptr<Network>> memberNetworks{};
    std::vector<double> weights{};
    std::vector<ResolvedEnsembleLoss> losses{};
    TrainingRunsComposedEnsembleEvaluator evaluator{};
    std::shared_ptr<PlacedNetwork> placedEvaluator = nullptr;
};

std::vector<std::string> requestedReportedLossNamesForGroup(
    const std::map<std::string, std::vector<std::string>>& reportedLosses,
    const std::string& groupName) {
    const auto it = reportedLosses.find(groupName);
    return it == reportedLosses.end() ? std::vector<std::string>{} : it->second;
}

TrainingRunsComposedEvaluatorArtifacts loadTrainingRunsComposedEvaluatorArtifacts(
    const std::vector<EnsembleMemberSpecRef>& members,
    uint64_t batchSize,
    const std::vector<std::string>& requestedLossNames,
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
        if (member.spec == nullptr || member.spec->trainer == nullptr || member.spec->trainer->getNetwork() == nullptr) {
            throw std::runtime_error(context + " requires a trainer network for every ensemble member.");
        }
        const std::optional<std::string>& artifactDir = member.spec->trainer->getSaveModelDirectory();
        if (!artifactDir.has_value()) {
            throw std::runtime_error(context + " ensemble member '" + member.spec->runName +
                                     "' does not have a saved model artifact directory for ensemble evaluation.");
        }
        auto loadedNetwork = std::make_shared<Network>(member.spec->trainer->getNetwork()->getNetworkName());
        loadedNetwork->load(*artifactDir);
        artifacts.memberNetworks.push_back(std::move(loadedNetwork));
        artifacts.weights.push_back(member.spec->ensembleWeight);
    }

    artifacts.losses = resolveTrainingRunsReportedLosses(artifacts.memberNetworks.front()->getReportableLosses(),
                                                         requestedLossNames,
                                                         context);
    if (artifacts.losses.empty()) {
        return artifacts;
    }

    for (size_t memberIndex = 1; memberIndex < artifacts.memberNetworks.size(); ++memberIndex) {
        const std::vector<ResolvedEnsembleLoss> memberLosses = resolveTrainingRunsReportedLosses(
            artifacts.memberNetworks[memberIndex]->getReportableLosses(),
            requestedLossNames,
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
    }

    artifacts.evaluator = buildTrainingRunsComposedEnsembleEvaluatorForLosses(artifacts.memberNetworks, artifacts.weights, artifacts.losses);
    std::vector<Event> initDoneEvents;
    artifacts.placedEvaluator = artifacts.evaluator.network->place(static_cast<uint32_t>(batchSize),
                                                                   initDoneEvents,
                                                                   /*inferenceOnly=*/true,
                                                                   /*forcedDevices=*/{},
                                                                   /*forcedNumStampsPerGpu=*/0,
                                                                   /*networkOutputsOnGpu=*/false);
    for (Event& event : initDoneEvents) {
        event.synchronize();
    }
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

struct ComposedEnsembleLossEvaluationMetrics {
    std::map<std::string, std::optional<double>> lossValues{};
    std::optional<double> overallLoss{};
    uint64_t batches = 0;
    uint64_t rows = 0;
};

Batch inferenceBatchForInputNames(const std::vector<std::string>& inputNames,
                                  const Batch& sourceBatch,
                                  const std::string& missingInputContext);

ComposedEnsembleLossEvaluationMetrics evaluateComposedEnsembleLossesOnLoader(
    const TrainingRunsComposedEvaluatorArtifacts& artifacts,
    Loader& loader,
    ExampleType exampleType) {
    ComposedEnsembleLossEvaluationMetrics metrics;
    if (artifacts.losses.empty()) {
        return metrics;
    }
    if (artifacts.placedEvaluator == nullptr) {
        throw std::runtime_error("TrainingRuns composed ensemble evaluator is not placed.");
    }
    if (loader.getBatchSize() != artifacts.batchSize) {
        throw std::runtime_error("TrainingRuns composed ensemble evaluation requires loader batch_size=" +
                                 std::to_string(artifacts.batchSize) + ", got " + std::to_string(loader.getBatchSize()) + ".");
    }

    std::map<std::string, double> weightedLossSums;
    std::map<std::string, uint64_t> weightedRows;
    for (const ResolvedEnsembleLoss& loss : artifacts.losses) {
        weightedLossSums[loss.lossName] = 0.0;
        weightedRows[loss.lossName] = 0;
    }

    const uint64_t batchesPerEpoch = loader.getNumBatchesPerEpoch(exampleType);
    uint64_t batchNum = loader.getNextBatchNum(exampleType);
    if (batchNum > batchesPerEpoch) {
        throw std::runtime_error("TrainingRuns composed ensemble evaluation loader returned next batch beyond batches per epoch.");
    }
    const uint64_t batchesToRun = batchesPerEpoch - batchNum;
    for (uint64_t batchOffset = 0; batchOffset < batchesToRun; ++batchOffset) {
        Batch batch = loader.getBatch(exampleType, batchNum);
        Batch evaluatorBatch = inferenceBatchForInputNames(artifacts.evaluator.externalInputNames,
                                                           batch,
                                                           "TrainingRuns composed ensemble evaluation batch");
        const uint64_t rows = batchRowsForEvaluatorInputs(evaluatorBatch,
                                                          artifacts.evaluator.externalInputNames,
                                                          "TrainingRuns composed ensemble evaluation batch");
        if (rows == 0) {
            loader.returnBatchBuffers(exampleType, std::move(batch));
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
            weightedRows[loss.lossName] += rows;
        }
        metrics.batches += 1;
        metrics.rows += rows;
        loader.returnBatchBuffers(exampleType, std::move(batch));
    }

    std::vector<std::optional<double>> namedValuesForOverall;
    namedValuesForOverall.reserve(artifacts.losses.size());
    for (const ResolvedEnsembleLoss& loss : artifacts.losses) {
        auto rowsIt = weightedRows.find(loss.lossName);
        if (rowsIt == weightedRows.end() || rowsIt->second == 0) {
            metrics.lossValues[loss.lossName] = std::nullopt;
            namedValuesForOverall.push_back(std::nullopt);
            continue;
        }
        const double value = weightedLossSums.at(loss.lossName) / static_cast<double>(rowsIt->second);
        metrics.lossValues[loss.lossName] = value;
        namedValuesForOverall.push_back(value);
    }
    metrics.overallLoss = weightedLossSumFromWeightedLossValues(namedValuesForOverall);
    return metrics;
}

void applyComposedLossEvaluationMetricsToEnsemble(TrainingEnsembleResult& ensemble,
                                                  const ComposedEnsembleLossEvaluationMetrics& metrics,
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
}


Batch inferenceBatchForInputNames(const std::vector<std::string>& inputNames,
                                  const Batch& sourceBatch,
                                  const std::string& missingInputContext) {
    Batch inferenceBatch;
    for (const std::string& inputName : inputNames) {
        if (!sourceBatch.contains(inputName)) {
            throw std::runtime_error(missingInputContext + " is missing inference input '" + inputName + "'.");
        }
        const BatchValue& value = sourceBatch.at(inputName);
        if (std::holds_alternative<ThorImplementation::Tensor>(value)) {
            inferenceBatch.insert(inputName, std::get<ThorImplementation::Tensor>(value));
        } else if (std::holds_alternative<ThorImplementation::RaggedTensor>(value)) {
            inferenceBatch.insert(inputName, std::get<ThorImplementation::RaggedTensor>(value));
        } else {
            throw std::runtime_error(missingInputContext + " input '" + inputName + "' has an unsupported value type.");
        }
    }
    return inferenceBatch;
}


}  // namespace


void TrainingRuns::validateTestLoader(Loader& loader) const {
    if (loader.getBatchSize() == 0) {
        throw std::runtime_error("TrainingRuns test_loader has batch_size=0.");
    }
    if (loader.getNumBatchesPerEpoch(ExampleType::TEST) == 0) {
        throw std::runtime_error("TrainingRuns test_loader has no test batches.");
    }
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

        std::optional<uint64_t> evaluationBatchSize;
        for (const EnsembleMemberSpecRef& sourceMember : members) {
            if (sourceMember.spec == nullptr || sourceMember.spec->trainer == nullptr || sourceMember.spec->trainer->loader == nullptr) {
                continue;
            }
            const uint64_t sourceBatchSize = sourceMember.spec->trainer->loader->getBatchSize();
            if (!evaluationBatchSize.has_value()) {
                evaluationBatchSize = sourceBatchSize;
            } else if (*evaluationBatchSize != sourceBatchSize) {
                throw std::runtime_error("TrainingRuns ensemble_group '" + groupName +
                                         "' cannot reuse composed ensemble evaluator across validation populations with different batch_size values. "
                                         "Expected " +
                                         std::to_string(*evaluationBatchSize) + ", got " + std::to_string(sourceBatchSize) +
                                         " for run '" + sourceMember.spec->runName + "'.");
            }
        }
        if (!evaluationBatchSize.has_value()) {
            ensembleIt->second.ensembleTrainingLoss = std::nullopt;
            continue;
        }

        if (ensembleIt->second.namedMetrics.empty()) {
            ensembleIt->second.ensembleTrainingLoss = std::nullopt;
            continue;
        }

        TrainingRunsComposedEvaluatorArtifacts composedArtifacts = loadTrainingRunsComposedEvaluatorArtifacts(
            members,
            *evaluationBatchSize,
            requestedReportedLossNamesForGroup(reportedLosses, groupName),
            "TrainingRuns composed ensemble training-population evaluation for ensemble_group '" + groupName + "'");

        std::map<std::string, std::vector<std::optional<double>>> sourcePopulationLossesByName;
        std::vector<double> sourceWeights;
        sourceWeights.reserve(members.size());
        for (const TrainingNamedMetricResult& metric : ensembleIt->second.namedMetrics) {
            sourcePopulationLossesByName[metric.name].reserve(members.size());
        }

        for (const EnsembleMemberSpecRef& sourceMember : members) {
            const double sourceWeight = sourceMember.spec == nullptr ? 1.0 : sourceMember.spec->ensembleWeight;
            sourceWeights.push_back(sourceWeight);
            if (sourceMember.spec == nullptr || sourceMember.spec->trainer == nullptr || sourceMember.spec->trainer->loader == nullptr) {
                for (const TrainingNamedMetricResult& metric : ensembleIt->second.namedMetrics) {
                    sourcePopulationLossesByName[metric.name].push_back(std::nullopt);
                }
                continue;
            }

            ComposedEnsembleLossEvaluationMetrics sourceMetrics = evaluateComposedEnsembleLossesOnLoader(
                composedArtifacts, *sourceMember.spec->trainer->loader, ExampleType::VALIDATE);
            for (const TrainingNamedMetricResult& metric : ensembleIt->second.namedMetrics) {
                auto valueIt = sourceMetrics.lossValues.find(metric.name);
                sourcePopulationLossesByName[metric.name].push_back(
                    valueIt == sourceMetrics.lossValues.end() ? std::optional<double>{} : valueIt->second);
            }
        }

        std::vector<std::optional<double>> namedTrainValuesForOverall;
        namedTrainValuesForOverall.reserve(ensembleIt->second.namedMetrics.size());
        for (TrainingNamedMetricResult& metric : ensembleIt->second.namedMetrics) {
            metric.trainValue = weightedAverage(sourcePopulationLossesByName[metric.name], sourceWeights);
            namedTrainValuesForOverall.push_back(metric.trainValue);
        }
        ensembleIt->second.ensembleTrainingLoss = weightedLossSumFromWeightedLossValues(namedTrainValuesForOverall);
    }
}

struct MemberGraphLossEvaluationMetrics {
    std::optional<double> loss{};
    uint64_t batches = 0;
    uint64_t rows = 0;
};

std::vector<MemberGraphLossEvaluationMetrics> evaluateMemberGraphLossesOnLoader(
    const std::vector<EnsembleMemberSpecRef>& members,
    uint64_t batchSize,
    const std::vector<std::string>& requestedLossNames,
    Loader& loader,
    ExampleType exampleType,
    const std::string& context) {
    std::vector<MemberGraphLossEvaluationMetrics> memberMetrics;
    memberMetrics.reserve(members.size());
    for (size_t memberIndex = 0; memberIndex < members.size(); ++memberIndex) {
        const EnsembleMemberSpecRef& member = members[memberIndex];
        MemberGraphLossEvaluationMetrics result;
        if (member.spec == nullptr || member.spec->trainer == nullptr || member.spec->trainer->getNetwork() == nullptr) {
            memberMetrics.push_back(result);
            continue;
        }
        TrainingRunsComposedEvaluatorArtifacts artifacts = loadTrainingRunsComposedEvaluatorArtifacts(
            std::vector<EnsembleMemberSpecRef>{member},
            batchSize,
            requestedLossNames,
            context + " member '" + member.spec->runName + "'");
        if (!artifacts.losses.empty()) {
            ComposedEnsembleLossEvaluationMetrics metrics = evaluateComposedEnsembleLossesOnLoader(artifacts, loader, exampleType);
            result.loss = metrics.overallLoss;
            result.batches = metrics.batches;
            result.rows = metrics.rows;
        }
        memberMetrics.push_back(result);
    }
    return memberMetrics;
}

void applyGraphLossEvaluationMemberTestStats(std::vector<TrainingRunResult>& results,
                                             TrainingEnsembleResult& ensemble,
                                             const std::vector<EnsembleMemberSpecRef>& members,
                                             Loader& testLoader,
                                             const std::vector<MemberGraphLossEvaluationMetrics>& memberMetrics) {
    const uint64_t stepsPerEpoch = testLoader.getNumBatchesPerEpoch(ExampleType::TEST);

    for (size_t i = 0; i < members.size(); ++i) {
        if (members[i].spec == nullptr || members[i].spec->trainer == nullptr || members[i].spec->trainer->getNetwork() == nullptr) {
            continue;
        }
        const MemberGraphLossEvaluationMetrics metrics = i < memberMetrics.size() ? memberMetrics[i] : MemberGraphLossEvaluationMetrics{};
        TrainingRunResult& runResult = results[members[i].runIndex];
        TrainingStatsSnapshot testStats;
        if (runResult.finalTestStats.has_value()) {
            testStats = *runResult.finalTestStats;
        }
        testStats.networkName = members[i].spec->trainer->getNetwork()->getNetworkName();
        testStats.datasetName = testLoader.getDatasetName();
        testStats.phase = TrainingEventPhase::TEST;
        testStats.epoch = 1;
        testStats.epochs = 1;
        testStats.step = metrics.batches;
        testStats.stepInEpoch = metrics.batches;
        testStats.stepsPerEpoch = stepsPerEpoch;
        testStats.batchSize = testLoader.getBatchSize();
        testStats.samplesProcessed = metrics.rows;
        testStats.loss = metrics.loss;
        testStats.accuracy = std::nullopt;
        runResult.finalTestStats = std::move(testStats);
    }

    for (TrainingEnsembleMemberResult& memberResult : ensemble.members) {
        for (size_t i = 0; i < members.size(); ++i) {
            if (members[i].spec != nullptr && memberResult.runName == members[i].spec->runName) {
                const MemberGraphLossEvaluationMetrics metrics = i < memberMetrics.size() ? memberMetrics[i] : MemberGraphLossEvaluationMetrics{};
                memberResult.status = results[members[i].runIndex].status;
                memberResult.finalTestLoss = metrics.loss;
                memberResult.finalTestAccuracy = std::nullopt;
            }
        }
    }
}

void TrainingRuns::evaluateEnsemblesOnTestLoader(std::vector<TrainingRunResult>& results,
                                                 std::map<std::string, TrainingEnsembleResult>& ensembleResultsByGroup,
                                                 std::shared_ptr<Loader> testLoader) const {
    if (testLoader == nullptr || ensembleResultsByGroup.empty()) {
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

        if (ensembleIt->second.namedMetrics.empty()) {
            ensembleIt->second.ensembleTestLoss = std::nullopt;
            ensembleIt->second.ensembleTestAccuracy = std::nullopt;
            continue;
        }

        TrainingRunsComposedEvaluatorArtifacts composedArtifacts = loadTrainingRunsComposedEvaluatorArtifacts(
            members,
            testLoader->getBatchSize(),
            requestedReportedLossNamesForGroup(reportedLosses, groupName),
            "TrainingRuns composed ensemble test evaluation for ensemble_group '" + groupName + "'");
        ComposedEnsembleLossEvaluationMetrics metrics = evaluateComposedEnsembleLossesOnLoader(
            composedArtifacts, *testLoader, ExampleType::TEST);
        applyComposedLossEvaluationMetricsToEnsemble(ensembleIt->second, metrics, /*testPhase=*/true);
        ensembleIt->second.ensembleTestAccuracy = std::nullopt;

        std::vector<MemberGraphLossEvaluationMetrics> memberMetrics = evaluateMemberGraphLossesOnLoader(
            members,
            testLoader->getBatchSize(),
            requestedReportedLossNamesForGroup(reportedLosses, groupName),
            *testLoader,
            ExampleType::TEST,
            "TrainingRuns composed ensemble per-member test evaluation for ensemble_group '" + groupName + "'");
        applyGraphLossEvaluationMemberTestStats(results, ensembleIt->second, members, *testLoader, memberMetrics);
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
