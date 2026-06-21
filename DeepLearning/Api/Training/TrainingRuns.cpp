#include "DeepLearning/Api/Training/TrainingRuns.h"

#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
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
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

namespace Thor {

namespace {


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
    bool sawEnabledRun = false;
    for (const TrainingRunsSpec& spec : runs) {
        if (spec.trainer == nullptr || !spec.trainer->getRuntimeConfig().statsEnabled) {
            continue;
        }
        sawEnabledRun = true;
        const LineStatsColorMode mode = spec.trainer->getRuntimeConfig().statsColorMode;
        if (mode == LineStatsColorMode::ALWAYS) {
            return LineStatsColorMode::ALWAYS;
        }
        if (mode == LineStatsColorMode::AUTO) {
            sawAuto = true;
        }
    }
    if (!sawEnabledRun || sawAuto) {
        return LineStatsColorMode::AUTO;
    }
    return LineStatsColorMode::NEVER;
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

    std::sort(signature.begin(), signature.end(), [](const TrainingRunInputSignature& lhs, const TrainingRunInputSignature& rhs) {
        return lhs.inputName < rhs.inputName;
    });
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

    std::sort(signature.begin(), signature.end(), [](const TrainingRunOutputSignature& lhs, const TrainingRunOutputSignature& rhs) {
        return lhs.outputName < rhs.outputName;
    });
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
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (!lhs[i].compatibleWith(rhs[i])) {
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
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (!lhs[i].compatibleWith(rhs[i])) {
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

TrainingRuns::TrainingRuns(std::vector<TrainingRunsSpec> runs,
                           TrainingRunsFailurePolicy failurePolicy,
                           double maxSummaryLogsPerSecond,
                           std::optional<size_t> maxParallelRuns)
    : runs(std::move(runs)),
      failurePolicy(failurePolicy),
      maxSummaryLogsPerSecond(maxSummaryLogsPerSecond),
      maxParallelRuns(maxParallelRuns) {
    if (!std::isfinite(maxSummaryLogsPerSecond) || maxSummaryLogsPerSecond < 0.0) {
        throw std::runtime_error("TrainingRuns maxSummaryLogsPerSecond must be finite and >= 0.");
    }
    if (maxParallelRuns.has_value() && maxParallelRuns.value() == 0) {
        throw std::runtime_error("TrainingRuns maxParallelRuns must be >= 1 when specified.");
    }
    validateRunSpecs();
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
                                                                         runtime.statsEnabled,
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
                result = runs[i].trainer->fitTrainingRun(runs[i].runName, options, observer, cancellationSource.token());
            } catch (...) {
                result = TrainingRunResult::fromException(runs[i].runName, std::current_exception());
            }
            result.ensembleGroup = runs[i].ensembleGroup;
            result.ensembleWeight = runs[i].ensembleWeight;

            const bool shouldCancelSiblings = failurePolicy == TrainingRunsFailurePolicy::CANCEL_SIBLINGS && result.failed();
            if (shouldCancelSiblings) {
                bool expected = false;
                if (cancellationRequestedByFailure.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
                    cancellationSource.requestCancellation();
                }
            }

            statsReporter->markRunFinished(result);

            {
                std::lock_guard<std::mutex> lock(resultMutex);
                results[i] = std::move(result);
            }
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

            auto [it, inserted] = ensembleSignatures.emplace(
                *spec.ensembleGroup, EnsembleValidationState{spec.runName, inputSignature, outputSignature});
            if (!inserted && !inputSignaturesCompatible(it->second.inputSignature, inputSignature)) {
                throw std::runtime_error("TrainingRuns ensemble_group '" + *spec.ensembleGroup + "' has incompatible input signatures: run '" +
                                         it->second.firstRunName + "' has " + inputSignatureToString(it->second.inputSignature) +
                                         ", but run '" + spec.runName + "' has " + inputSignatureToString(inputSignature) + ".");
            }
            if (!inserted && !outputSignaturesCompatible(it->second.outputSignature, outputSignature)) {
                throw std::runtime_error("TrainingRuns ensemble_group '" + *spec.ensembleGroup + "' has incompatible output signatures: run '" +
                                         it->second.firstRunName + "' has " + outputSignatureToString(it->second.outputSignature) +
                                         ", but run '" + spec.runName + "' has " + outputSignatureToString(outputSignature) + ".");
            }
        }
    }
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

std::string choosePredictionOutputName(const std::vector<TrainingRunOutputSignature>& signature) {
    for (const TrainingRunOutputSignature& output : signature) {
        if (output.outputName == "scores" || output.outputName == "logits") {
            return output.outputName;
        }
    }
    for (const TrainingRunOutputSignature& output : signature) {
        if (output.outputName != "loss") {
            return output.outputName;
        }
    }
    throw std::runtime_error(
        "TrainingRuns ensemble evaluation requires a non-loss NetworkOutput prediction tensor such as 'scores', 'logits', or 'prediction'.");
}

ThorImplementation::Tensor tensorOnCpu(ThorImplementation::Tensor tensor) {
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

double tensorElementAsDouble(const ThorImplementation::Tensor& tensor, uint64_t index) {
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
            throw std::runtime_error("TrainingRuns ensemble evaluation currently supports floating-point prediction and label tensors only.");
    }
}

std::vector<double> tensorToDoubleVector(ThorImplementation::Tensor tensor) {
    ThorImplementation::Tensor cpuTensor = tensorOnCpu(std::move(tensor));
    std::vector<double> values;
    values.reserve(cpuTensor.getTotalNumElements());
    for (uint64_t i = 0; i < cpuTensor.getTotalNumElements(); ++i) {
        values.push_back(tensorElementAsDouble(cpuTensor, i));
    }
    return values;
}

uint64_t lastDimensionOrThrow(const ThorImplementation::Tensor& tensor, const std::string& name) {
    const std::vector<uint64_t> dims = tensor.getDimensions();
    if (dims.empty() || dims.back() == 0) {
        throw std::runtime_error("TrainingRuns ensemble evaluation tensor '" + name + "' has invalid dimensions.");
    }
    return dims.back();
}

struct PlacedEnsembleArtifacts {
    uint64_t batchSize = 0;
    std::vector<std::shared_ptr<PlacedNetwork>> placedMembers{};
    std::vector<double> weights{};
};

PlacedEnsembleArtifacts loadPlacedMemberArtifacts(const std::vector<EnsembleMemberSpecRef>& members, uint64_t batchSize) {
    if (batchSize == 0) {
        throw std::runtime_error("TrainingRuns ensemble evaluation cannot place saved model artifacts for batch_size=0.");
    }

    PlacedEnsembleArtifacts artifacts;
    artifacts.batchSize = batchSize;
    artifacts.placedMembers.reserve(members.size());
    artifacts.weights.reserve(members.size());
    for (const EnsembleMemberSpecRef& member : members) {
        const std::optional<std::string>& artifactDir = member.spec->trainer->getSaveModelDirectory();
        if (!artifactDir.has_value()) {
            throw std::runtime_error("TrainingRuns ensemble member '" + member.spec->runName +
                                     "' does not have a saved model artifact directory for ensemble evaluation.");
        }
        auto loadedNetwork = std::make_shared<Network>(member.spec->trainer->getNetwork()->getNetworkName());
        loadedNetwork->load(*artifactDir);
        std::vector<Event> initDoneEvents;
        std::shared_ptr<PlacedNetwork> placedNetwork = loadedNetwork->place(static_cast<uint32_t>(batchSize), initDoneEvents, /*inferenceOnly=*/true);
        for (Event& event : initDoneEvents) {
            event.synchronize();
        }
        artifacts.placedMembers.push_back(std::move(placedNetwork));
        artifacts.weights.push_back(member.spec->ensembleWeight);
    }
    return artifacts;
}

void validateEnsembleEvaluationBatchSize(const PlacedEnsembleArtifacts& artifacts, Loader& loader) {
    if (loader.getBatchSize() != artifacts.batchSize) {
        throw std::runtime_error("TrainingRuns ensemble evaluation currently requires all evaluated loaders for a placed ensemble group "
                                 "to use the same batch_size as the placed artifacts. Expected " +
                                 std::to_string(artifacts.batchSize) + ", got " + std::to_string(loader.getBatchSize()) + ".");
    }
}

std::optional<double> categoricalCrossEntropyFromWeightedMemberLogits(
    const std::vector<std::vector<double>>& memberLogits,
    const std::vector<double>& labels,
    const std::vector<double>& weights,
    uint64_t rows,
    uint64_t classes) {
    if (memberLogits.empty() || memberLogits.size() != weights.size()) {
        return std::nullopt;
    }
    const double weightSum = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (weightSum <= 0.0) {
        return std::nullopt;
    }

    double lossSum = 0.0;
    for (uint64_t row = 0; row < rows; ++row) {
        std::vector<double> probabilities(classes, 0.0);
        for (size_t memberIndex = 0; memberIndex < memberLogits.size(); ++memberIndex) {
            const std::vector<double>& logits = memberLogits[memberIndex];
            const uint64_t offset = row * classes;
            double maxLogit = logits[offset];
            for (uint64_t c = 1; c < classes; ++c) {
                maxLogit = std::max(maxLogit, logits[offset + c]);
            }
            double denom = 0.0;
            for (uint64_t c = 0; c < classes; ++c) {
                denom += std::exp(logits[offset + c] - maxLogit);
            }
            for (uint64_t c = 0; c < classes; ++c) {
                probabilities[c] += weights[memberIndex] * (std::exp(logits[offset + c] - maxLogit) / denom);
            }
        }
        for (double& probability : probabilities) {
            probability /= weightSum;
        }

        const uint64_t offset = row * classes;
        double sampleLoss = 0.0;
        for (uint64_t c = 0; c < classes; ++c) {
            sampleLoss += -labels[offset + c] * std::log(std::max(probabilities[c], 1.0e-12));
        }
        lossSum += sampleLoss;
    }
    return lossSum / static_cast<double>(rows);
}

std::optional<double> meanAbsoluteErrorFromWeightedMemberPredictions(
    const std::vector<std::vector<double>>& memberPredictions,
    const std::vector<double>& labels,
    const std::vector<double>& weights) {
    if (memberPredictions.empty() || memberPredictions.size() != weights.size()) {
        return std::nullopt;
    }
    const double weightSum = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (weightSum <= 0.0 || labels.empty()) {
        return std::nullopt;
    }

    double lossSum = 0.0;
    for (size_t i = 0; i < labels.size(); ++i) {
        double prediction = 0.0;
        for (size_t memberIndex = 0; memberIndex < memberPredictions.size(); ++memberIndex) {
            prediction += weights[memberIndex] * memberPredictions[memberIndex][i];
        }
        prediction /= weightSum;
        const double diff = prediction - labels[i];
        lossSum += std::abs(diff);
    }
    return lossSum / static_cast<double>(labels.size());
}


uint64_t argmaxRow(const std::vector<double>& values, uint64_t row, uint64_t width) {
    const uint64_t offset = row * width;
    uint64_t best = 0;
    double bestValue = values[offset];
    for (uint64_t c = 1; c < width; ++c) {
        const double value = values[offset + c];
        if (value > bestValue) {
            best = c;
            bestValue = value;
        }
    }
    return best;
}

struct PredictionEvaluationMetrics {
    std::optional<double> loss{};
    std::optional<double> accuracy{};
    std::vector<std::optional<double>> memberLosses{};
    std::vector<std::optional<double>> memberAccuracies{};
    uint64_t batches = 0;
    uint64_t rows = 0;
};

PredictionEvaluationMetrics evaluateEnsemblePredictionMetricsOnLoader(const PlacedEnsembleArtifacts& artifacts,
                                                                      Loader& loader,
                                                                      ExampleType exampleType,
                                                                      const TrainingEnsembleResult& ensembleTemplate) {
    PredictionEvaluationMetrics metrics;
    metrics.memberLosses.resize(artifacts.placedMembers.size());
    metrics.memberAccuracies.resize(artifacts.placedMembers.size());
    if (artifacts.placedMembers.empty()) {
        return metrics;
    }
    validateEnsembleEvaluationBatchSize(artifacts, loader);
    const std::string predictionOutputName = choosePredictionOutputName(ensembleTemplate.outputSignature);
    const bool categoricalPrediction = predictionOutputName == "scores" || predictionOutputName == "logits";
    const std::vector<double>& weights = artifacts.weights;
    const double weightSum = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (weightSum <= 0.0) {
        return metrics;
    }

    const uint64_t batchesPerEpoch = loader.getNumBatchesPerEpoch(exampleType);
    uint64_t batchNum = loader.getNextBatchNum(exampleType);
    if (batchNum > batchesPerEpoch) {
        throw std::runtime_error("TrainingRuns ensemble evaluation loader returned next batch beyond batches per epoch.");
    }
    const uint64_t batchesToRun = batchesPerEpoch - batchNum;
    double weightedLossSum = 0.0;
    uint64_t weightedRows = 0;
    uint64_t weightedElements = 0;
    uint64_t ensembleCorrect = 0;
    std::vector<double> memberLossSums(artifacts.placedMembers.size(), 0.0);
    std::vector<uint64_t> memberCorrect(artifacts.placedMembers.size(), 0);

    for (uint64_t batchOffset = 0; batchOffset < batchesToRun; ++batchOffset) {
        Batch batch = loader.getBatch(exampleType, batchNum);
        if (!batch.contains("labels") || !batch.isTensor("labels")) {
            throw std::runtime_error("TrainingRuns ensemble evaluation requires a dense 'labels' tensor in the evaluation batch.");
        }
        ThorImplementation::Tensor labelsTensor = batch.getTensor("labels");
        const uint64_t labelLastDim = lastDimensionOrThrow(labelsTensor, "labels");
        const uint64_t rows = labelsTensor.getTotalNumElements() / labelLastDim;
        if (rows == 0) {
            loader.returnBatchBuffers(exampleType, std::move(batch));
            continue;
        }
        const std::vector<double> labels = tensorToDoubleVector(labelsTensor);

        std::vector<std::vector<double>> memberPredictions;
        memberPredictions.reserve(artifacts.placedMembers.size());
        for (const std::shared_ptr<PlacedNetwork>& placedMember : artifacts.placedMembers) {
            std::map<std::string, ThorImplementation::Tensor> outputs = placedMember->infer(batch);
            auto outputIt = outputs.find(predictionOutputName);
            if (outputIt == outputs.end()) {
                throw std::runtime_error("TrainingRuns ensemble evaluation did not receive prediction output '" + predictionOutputName + "'.");
            }
            ThorImplementation::Tensor predictionTensor = outputIt->second;
            if (predictionTensor.getTotalNumElements() != labels.size()) {
                throw std::runtime_error("TrainingRuns ensemble prediction tensor and labels tensor have different element counts.");
            }
            if (lastDimensionOrThrow(predictionTensor, predictionOutputName) != labelLastDim) {
                throw std::runtime_error("TrainingRuns ensemble prediction tensor and labels tensor have incompatible class/value dimensions.");
            }
            memberPredictions.push_back(tensorToDoubleVector(predictionTensor));
        }

        if (categoricalPrediction) {
            for (uint64_t row = 0; row < rows; ++row) {
                std::vector<double> ensembleProbabilities(labelLastDim, 0.0);
                const uint64_t offset = row * labelLastDim;
                const uint64_t labelClass = argmaxRow(labels, row, labelLastDim);
                for (size_t memberIndex = 0; memberIndex < memberPredictions.size(); ++memberIndex) {
                    const std::vector<double>& logits = memberPredictions[memberIndex];
                    const uint64_t predictedClass = argmaxRow(logits, row, labelLastDim);
                    if (predictedClass == labelClass) {
                        memberCorrect[memberIndex] += 1;
                    }

                    double maxLogit = logits[offset];
                    for (uint64_t c = 1; c < labelLastDim; ++c) {
                        maxLogit = std::max(maxLogit, logits[offset + c]);
                    }
                    double denom = 0.0;
                    for (uint64_t c = 0; c < labelLastDim; ++c) {
                        denom += std::exp(logits[offset + c] - maxLogit);
                    }
                    for (uint64_t c = 0; c < labelLastDim; ++c) {
                        const double probability = std::exp(logits[offset + c] - maxLogit) / denom;
                        memberLossSums[memberIndex] += -labels[offset + c] * std::log(std::max(probability, 1.0e-12));
                        ensembleProbabilities[c] += weights[memberIndex] * probability;
                    }
                }
                for (double& probability : ensembleProbabilities) {
                    probability /= weightSum;
                }
                const uint64_t ensembleClass = argmaxRow(ensembleProbabilities, 0, labelLastDim);
                if (ensembleClass == labelClass) {
                    ensembleCorrect += 1;
                }
                double sampleLoss = 0.0;
                for (uint64_t c = 0; c < labelLastDim; ++c) {
                    sampleLoss += -labels[offset + c] * std::log(std::max(ensembleProbabilities[c], 1.0e-12));
                }
                weightedLossSum += sampleLoss;
            }
        } else {
            std::vector<double> ensemblePredictions(labels.size(), 0.0);
            for (size_t memberIndex = 0; memberIndex < memberPredictions.size(); ++memberIndex) {
                const std::vector<double>& predictions = memberPredictions[memberIndex];
                for (size_t i = 0; i < labels.size(); ++i) {
                    const double diff = predictions[i] - labels[i];
                    memberLossSums[memberIndex] += std::abs(diff);
                    ensemblePredictions[i] += weights[memberIndex] * predictions[i];
                }
            }
            for (size_t i = 0; i < labels.size(); ++i) {
                ensemblePredictions[i] /= weightSum;
                const double diff = ensemblePredictions[i] - labels[i];
                weightedLossSum += std::abs(diff);
            }
        }

        weightedRows += rows;
        weightedElements += labels.size();
        metrics.batches += 1;
        loader.returnBatchBuffers(exampleType, std::move(batch));
    }

    metrics.rows = weightedRows;
    if (weightedRows == 0) {
        return metrics;
    }
    if (categoricalPrediction) {
        metrics.loss = weightedLossSum / static_cast<double>(weightedRows);
        metrics.accuracy = static_cast<double>(ensembleCorrect) / static_cast<double>(weightedRows);
        for (size_t memberIndex = 0; memberIndex < artifacts.placedMembers.size(); ++memberIndex) {
            metrics.memberLosses[memberIndex] = memberLossSums[memberIndex] / static_cast<double>(weightedRows);
            metrics.memberAccuracies[memberIndex] = static_cast<double>(memberCorrect[memberIndex]) / static_cast<double>(weightedRows);
        }
    } else {
        // Non-classification ensemble loss is MAE. Accuracy is intentionally absent.
        if (weightedElements == 0) {
            return metrics;
        }
        metrics.loss = weightedLossSum / static_cast<double>(weightedElements);
        for (size_t memberIndex = 0; memberIndex < artifacts.placedMembers.size(); ++memberIndex) {
            metrics.memberLosses[memberIndex] = memberLossSums[memberIndex] / static_cast<double>(weightedElements);
        }
    }
    return metrics;
}

std::optional<double> evaluateEnsemblePredictionLossOnLoader(const PlacedEnsembleArtifacts& artifacts,
                                                             Loader& loader,
                                                             ExampleType exampleType,
                                                             const TrainingEnsembleResult& ensembleTemplate) {
    if (artifacts.placedMembers.empty()) {
        return std::nullopt;
    }
    validateEnsembleEvaluationBatchSize(artifacts, loader);
    const std::string predictionOutputName = choosePredictionOutputName(ensembleTemplate.outputSignature);
    const bool categoricalPrediction = predictionOutputName == "scores" || predictionOutputName == "logits";
    const std::vector<double>& weights = artifacts.weights;

    const uint64_t batchesPerEpoch = loader.getNumBatchesPerEpoch(exampleType);
    uint64_t batchNum = loader.getNextBatchNum(exampleType);
    if (batchNum > batchesPerEpoch) {
        throw std::runtime_error("TrainingRuns ensemble evaluation loader returned next batch beyond batches per epoch.");
    }
    const uint64_t batchesToRun = batchesPerEpoch - batchNum;
    double weightedLossSum = 0.0;
    uint64_t weightedRows = 0;

    for (uint64_t batchOffset = 0; batchOffset < batchesToRun; ++batchOffset) {
        Batch batch = loader.getBatch(exampleType, batchNum);
        if (!batch.contains("labels") || !batch.isTensor("labels")) {
            throw std::runtime_error("TrainingRuns ensemble evaluation requires a dense 'labels' tensor in the evaluation batch.");
        }
        ThorImplementation::Tensor labelsTensor = batch.getTensor("labels");
        const uint64_t labelLastDim = lastDimensionOrThrow(labelsTensor, "labels");
        const uint64_t rows = labelsTensor.getTotalNumElements() / labelLastDim;
        if (rows == 0) {
            loader.returnBatchBuffers(exampleType, std::move(batch));
            continue;
        }
        const std::vector<double> labels = tensorToDoubleVector(labelsTensor);

        std::vector<std::vector<double>> memberPredictions;
        memberPredictions.reserve(artifacts.placedMembers.size());
        for (const std::shared_ptr<PlacedNetwork>& placedMember : artifacts.placedMembers) {
            std::map<std::string, ThorImplementation::Tensor> outputs = placedMember->infer(batch);
            auto outputIt = outputs.find(predictionOutputName);
            if (outputIt == outputs.end()) {
                throw std::runtime_error("TrainingRuns ensemble evaluation did not receive prediction output '" + predictionOutputName + "'.");
            }
            ThorImplementation::Tensor predictionTensor = outputIt->second;
            if (predictionTensor.getTotalNumElements() != labels.size()) {
                throw std::runtime_error("TrainingRuns ensemble prediction tensor and labels tensor have different element counts.");
            }
            if (lastDimensionOrThrow(predictionTensor, predictionOutputName) != labelLastDim) {
                throw std::runtime_error("TrainingRuns ensemble prediction tensor and labels tensor have incompatible class/value dimensions.");
            }
            memberPredictions.push_back(tensorToDoubleVector(predictionTensor));
        }

        std::optional<double> batchLoss = categoricalPrediction
            ? categoricalCrossEntropyFromWeightedMemberLogits(memberPredictions, labels, weights, rows, labelLastDim)
            : meanAbsoluteErrorFromWeightedMemberPredictions(memberPredictions, labels, weights);
        if (!batchLoss.has_value()) {
            loader.returnBatchBuffers(exampleType, std::move(batch));
            return std::nullopt;
        }
        weightedLossSum += batchLoss.value() * static_cast<double>(rows);
        weightedRows += rows;
        loader.returnBatchBuffers(exampleType, std::move(batch));
    }

    if (weightedRows == 0) {
        return std::nullopt;
    }
    return weightedLossSum / static_cast<double>(weightedRows);
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
                                         "' cannot reuse placed ensemble artifacts across validation populations with different batch_size values. "
                                         "Expected " +
                                         std::to_string(*evaluationBatchSize) + ", got " + std::to_string(sourceBatchSize) +
                                         " for run '" + sourceMember.spec->runName + "'.");
            }
        }
        if (!evaluationBatchSize.has_value()) {
            ensembleIt->second.ensembleTrainingLoss = std::nullopt;
            continue;
        }

        PlacedEnsembleArtifacts placedArtifacts = loadPlacedMemberArtifacts(members, *evaluationBatchSize);
        std::vector<std::optional<double>> sourcePopulationLosses;
        std::vector<double> sourceWeights;
        sourcePopulationLosses.reserve(members.size());
        sourceWeights.reserve(members.size());
        for (const EnsembleMemberSpecRef& sourceMember : members) {
            if (sourceMember.spec == nullptr || sourceMember.spec->trainer == nullptr || sourceMember.spec->trainer->loader == nullptr) {
                sourcePopulationLosses.push_back(std::nullopt);
                sourceWeights.push_back(sourceMember.spec == nullptr ? 1.0 : sourceMember.spec->ensembleWeight);
                continue;
            }
            sourcePopulationLosses.push_back(evaluateEnsemblePredictionLossOnLoader(
                placedArtifacts, *sourceMember.spec->trainer->loader, ExampleType::VALIDATE, ensembleIt->second));
            sourceWeights.push_back(sourceMember.spec->ensembleWeight);
        }

        ensembleIt->second.ensembleTrainingLoss = weightedAverage(sourcePopulationLosses, sourceWeights);
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

        PlacedEnsembleArtifacts placedArtifacts = loadPlacedMemberArtifacts(members, testLoader->getBatchSize());
        PredictionEvaluationMetrics metrics = evaluateEnsemblePredictionMetricsOnLoader(
            placedArtifacts, *testLoader, ExampleType::TEST, ensembleIt->second);
        ensembleIt->second.ensembleTestLoss = metrics.loss;
        ensembleIt->second.ensembleTestAccuracy = metrics.accuracy;

        const uint64_t stepsPerEpoch = testLoader->getNumBatchesPerEpoch(ExampleType::TEST);
        for (size_t i = 0; i < members.size(); ++i) {
            TrainingRunResult& runResult = results[members[i].runIndex];
            TrainingStatsSnapshot testStats;
            if (runResult.finalTestStats.has_value()) {
                testStats = *runResult.finalTestStats;
            }
            testStats.networkName = members[i].spec->trainer->getNetwork()->getNetworkName();
            testStats.datasetName = testLoader->getDatasetName();
            testStats.phase = TrainingEventPhase::TEST;
            testStats.epoch = 1;
            testStats.epochs = 1;
            testStats.step = metrics.batches;
            testStats.stepInEpoch = metrics.batches;
            testStats.stepsPerEpoch = stepsPerEpoch;
            testStats.batchSize = testLoader->getBatchSize();
            testStats.samplesProcessed = metrics.rows;
            if (i < metrics.memberLosses.size()) {
                testStats.loss = metrics.memberLosses[i];
            }
            if (i < metrics.memberAccuracies.size()) {
                testStats.accuracy = metrics.memberAccuracies[i];
            }
            runResult.finalTestStats = std::move(testStats);
        }

        for (TrainingEnsembleMemberResult& memberResult : ensembleIt->second.members) {
            for (size_t i = 0; i < members.size(); ++i) {
                if (memberResult.runName == members[i].spec->runName) {
                    memberResult.status = results[members[i].runIndex].status;
                    memberResult.finalTestLoss = i < metrics.memberLosses.size() ? metrics.memberLosses[i] : std::optional<double>{};
                    memberResult.finalTestAccuracy = i < metrics.memberAccuracies.size() ? metrics.memberAccuracies[i] : std::optional<double>{};
                }
            }
        }
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
