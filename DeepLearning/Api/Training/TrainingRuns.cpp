#include "DeepLearning/Api/Training/TrainingRuns.h"

#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Training/Observers/TrainingRunsStatsReporter.h"
#include "DeepLearning/Api/Training/Observers/TrainingStatsSink.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
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

std::string outputSignatureToString(const std::vector<TrainingRunOutputSignature>& signature) {
    std::ostringstream out;
    out << "{";
    for (size_t i = 0; i < signature.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << signature[i].outputName << ":" << dimensionsToString(signature[i].dimensions);
    }
    out << "}";
    return out.str();
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

std::optional<double> finalLossForMemberPhase(const TrainingEnsembleMemberResult& member, TrainingEventPhase phase) {
    switch (phase) {
        case TrainingEventPhase::TRAIN:
            return member.finalTrainingLoss;
        case TrainingEventPhase::VALIDATE:
            return member.finalValidationLoss;
        case TrainingEventPhase::TEST:
            return member.finalTestLoss;
        case TrainingEventPhase::UNKNOWN:
        default:
            return std::nullopt;
    }
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

std::optional<double> TrainingEnsembleResult::weightedFinalLossForPhase(TrainingEventPhase phase) const {
    if (members.empty() || !allCompleted()) {
        return std::nullopt;
    }

    double weightedSum = 0.0;
    double weightSum = 0.0;
    for (const TrainingEnsembleMemberResult& member : members) {
        const std::optional<double> loss = finalLossForMemberPhase(member, phase);
        if (!loss.has_value()) {
            return std::nullopt;
        }
        weightedSum += member.weight * loss.value();
        weightSum += member.weight;
    }
    if (weightSum <= 0.0) {
        return std::nullopt;
    }
    return weightedSum / weightSum;
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

TrainingRunsResult TrainingRuns::fit(uint32_t epochs) { return fit(TrainerFitOptions{epochs}); }

TrainingRunsResult TrainingRuns::fit(const TrainerFitOptions& options) {
    validateFitOptions(options);

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

    std::vector<TrainingEnsembleResult> ensembleResults = buildEnsembleResults(results);

    statsReporter->flush();
    statsReporter->emitFinalReport(results);
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
        std::vector<TrainingRunOutputSignature> signature{};
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
            const std::string normalizedDirectory = normalizedOutputPathForCollisionCheck(*saveModelDirectory);
            auto [it, inserted] = saveModelDirectories.emplace(normalizedDirectory, spec.runName);
            if (!inserted) {
                throw std::runtime_error("TrainingRuns save_model_dir collision: runs '" + it->second + "' and '" + spec.runName +
                                         "' both save model output to '" + normalizedDirectory +
                                         "'. Give each trainer a distinct save_model_dir or disable saving for one of them.");
            }
        }

        if (spec.ensembleGroup.has_value()) {
            const std::vector<TrainingRunOutputSignature> signature = collectNetworkOutputSignature(spec.trainer->getNetwork());
            if (signature.empty()) {
                throw std::runtime_error("TrainingRuns run '" + spec.runName + "' is in ensemble_group '" + *spec.ensembleGroup +
                                         "' but its network has no NetworkOutput layers to ensemble.");
            }

            auto [it, inserted] = ensembleSignatures.emplace(*spec.ensembleGroup, EnsembleValidationState{spec.runName, signature});
            if (!inserted && !outputSignaturesCompatible(it->second.signature, signature)) {
                throw std::runtime_error("TrainingRuns ensemble_group '" + *spec.ensembleGroup + "' has incompatible output dimensions: run '" +
                                         it->second.firstRunName + "' has " + outputSignatureToString(it->second.signature) +
                                         ", but run '" + spec.runName + "' has " + outputSignatureToString(signature) + ".");
            }
        }
    }
}

std::vector<TrainingEnsembleResult> TrainingRuns::buildEnsembleResults(const std::vector<TrainingRunResult>& results) const {
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
        ensemble.members.push_back(std::move(member));
    }

    std::vector<TrainingEnsembleResult> ensembles;
    ensembles.reserve(byGroup.size());
    for (auto& entry : byGroup) {
        ensembles.push_back(std::move(entry.second));
    }
    return ensembles;
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
