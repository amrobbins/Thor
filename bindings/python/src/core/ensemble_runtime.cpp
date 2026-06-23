#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Ensemble/EnsembleWeightedMean.h"

namespace nb = nanobind;
using namespace nb::literals;

using PhysicalTensor = ThorImplementation::Tensor;
using TensorDescriptor = ThorImplementation::TensorDescriptor;
using TensorPlacement = ThorImplementation::TensorPlacement;

namespace {

std::vector<std::string> sortedOutputNames(const std::map<std::string, PhysicalTensor>& outputs) {
    std::vector<std::string> names;
    names.reserve(outputs.size());
    for (const auto& [name, _] : outputs) {
        names.push_back(name);
    }
    return names;
}

void validateOutputNames(const std::vector<std::map<std::string, PhysicalTensor>>& memberOutputs,
                         const std::vector<std::string>& outputNames) {
    if (memberOutputs.empty()) {
        throw std::runtime_error("ensemble GPU aggregation requires at least one member output map");
    }
    if (outputNames.empty()) {
        throw std::runtime_error("ensemble GPU aggregation requires at least one output name");
    }

    const std::vector<std::string> firstNames = sortedOutputNames(memberOutputs.front());
    for (const std::string& outputName : outputNames) {
        if (memberOutputs.front().count(outputName) == 0) {
            throw std::runtime_error("ensemble GPU aggregation output name '" + outputName + "' was not produced by the first member");
        }
    }

    for (size_t memberIndex = 1; memberIndex < memberOutputs.size(); ++memberIndex) {
        const std::vector<std::string> names = sortedOutputNames(memberOutputs[memberIndex]);
        if (names != firstNames) {
            throw std::runtime_error("ensemble GPU aggregation requires all members to produce the same output names");
        }
        for (const std::string& outputName : outputNames) {
            if (memberOutputs[memberIndex].count(outputName) == 0) {
                throw std::runtime_error("ensemble GPU aggregation output name '" + outputName + "' was not produced by all members");
            }
        }
    }

}

struct StagedEnsembleInputs {
    std::map<std::string, PhysicalTensor> tensors;
    std::map<std::string, Event> readyEvents;
};


std::string accumulatorInputName(size_t outputIndex, size_t memberIndex) {
    return "thor_ensemble_output_" + std::to_string(outputIndex) + "_member_" + std::to_string(memberIndex);
}

std::vector<std::string> resolveOutputNamesFromMember(const std::shared_ptr<Thor::PlacedNetwork>& placedMember,
                                                       std::vector<std::string> outputNames) {
    if (placedMember == nullptr) {
        throw nb::value_error("placed member must not be null");
    }
    ThorImplementation::StampedNetwork& stampedNetwork = placedMember->getStampedNetwork(0);
    if (outputNames.empty()) {
        outputNames = stampedNetwork.getNamedOutputNames();
    }
    if (outputNames.empty()) {
        throw std::runtime_error("ensemble member network does not expose any named outputs");
    }
    for (const std::string& outputName : outputNames) {
        if (stampedNetwork.getNamedOutput(outputName) == nullptr) {
            throw std::runtime_error("ensemble output '" + outputName + "' is not present in the reference member network");
        }
    }
    return outputNames;
}

nb::dict memberOutputSpecs(const std::shared_ptr<Thor::PlacedNetwork>& placedMember,
                           std::vector<std::string> outputNames) {
    outputNames = resolveOutputNamesFromMember(placedMember, std::move(outputNames));
    ThorImplementation::StampedNetwork& stampedNetwork = placedMember->getStampedNetwork(0);

    nb::dict result;
    for (const std::string& outputName : outputNames) {
        auto outputLayer = stampedNetwork.getNamedOutput(outputName);
        if (outputLayer == nullptr) {
            throw std::runtime_error("ensemble output '" + outputName + "' is not present in the reference member network");
        }
        std::optional<PhysicalTensor> tensor = outputLayer->getFeatureOutputForSlot(0);
        if (!tensor.has_value()) {
            throw std::runtime_error("ensemble output '" + outputName + "' does not have a physical tensor");
        }
        nb::dict spec;
        spec["dimensions"] = nb::cast(tensor.value().getDescriptor().getDimensions());
        spec["data_type"] = nb::cast(tensor.value().getDescriptor().getDataType());
        spec["device"] = tensor.value().getPlacement().getDeviceNum();
        result[nb::str(outputName.c_str())] = std::move(spec);
    }
    return result;
}

StagedEnsembleInputs stageInputsOnceForEnsemble(const std::shared_ptr<Thor::PlacedNetwork>& referenceMember,
                                                const std::map<std::string, PhysicalTensor>& batchInputs,
                                                int32_t device,
                                                Stream& stagingStream) {
    if (referenceMember == nullptr) {
        throw nb::value_error("reference placed member must not be null");
    }
    ThorImplementation::StampedNetwork& stampedNetwork = referenceMember->getStampedNetwork(0);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, device);

    StagedEnsembleInputs staged;
    for (const auto& [inputName, sourceTensor] : batchInputs) {
        auto inputLayer = stampedNetwork.getNamedInput(inputName);
        if (inputLayer == nullptr) {
            throw std::runtime_error("ensemble input '" + inputName + "' is not present in the reference member network");
        }
        std::optional<PhysicalTensor> expectedInput = inputLayer->getFeatureOutput();
        if (!expectedInput.has_value()) {
            throw std::runtime_error("ensemble input '" + inputName + "' does not have a physical feature output tensor");
        }

        const TensorDescriptor targetDescriptor = expectedInput.value().getDescriptor();
        if (sourceTensor.getDescriptor().getDimensions() != targetDescriptor.getDimensions()) {
            throw std::runtime_error("ensemble input '" + inputName + "' dimensions do not match the member NetworkInput");
        }
        if (expectedInput.value().getPlacement() != gpuPlacement) {
            throw std::runtime_error("ensemble input '" + inputName + "' reference member placement is not the aggregation GPU");
        }

        if (sourceTensor.getPlacement() == gpuPlacement && sourceTensor.getDescriptor() == targetDescriptor) {
            staged.tensors.emplace(inputName, sourceTensor);
        } else {
            PhysicalTensor stagedTensor(gpuPlacement, targetDescriptor);
            stagedTensor.copyFromAsync(sourceTensor, stagingStream);
            Event readyEvent = stagingStream.putEvent(false, false);
            staged.tensors.emplace(inputName, stagedTensor);
            staged.readyEvents.emplace(inputName, readyEvent);
        }
    }
    return staged;
}

std::map<std::string, PhysicalTensor> debugStageEnsembleInputsOnce(
    const std::shared_ptr<Thor::PlacedNetwork>& referenceMember,
    const std::map<std::string, PhysicalTensor>& batchInputs,
    int32_t device) {
    if (device < 0) {
        throw nb::value_error("device must be >= 0");
    }
    if (batchInputs.empty()) {
        throw nb::value_error("batch_inputs must not be empty");
    }
    Stream stagingStream(device, Stream::Priority::REGULAR);
    StagedEnsembleInputs staged;
    {
        nb::gil_scoped_release release;
        staged = stageInputsOnceForEnsemble(referenceMember, batchInputs, device, stagingStream);
        stagingStream.synchronize();
    }
    return staged.tensors;
}

std::map<std::string, PhysicalTensor> submitMembersAndAggregateGpuToCpu(
    std::vector<std::shared_ptr<Thor::PlacedNetwork>> placedMembers,
    const std::map<std::string, PhysicalTensor>& batchInputs,
    const std::vector<double>& weights,
    std::vector<std::string> outputNames,
    int32_t device) {
    if (device < 0) {
        throw nb::value_error("device must be >= 0");
    }
    if (placedMembers.empty()) {
        throw nb::value_error("placed_members must not be empty");
    }
    if (weights.size() != placedMembers.size()) {
        throw nb::value_error("weights must contain one value per placed member");
    }
    if (batchInputs.empty()) {
        throw nb::value_error("batch_inputs must not be empty");
    }
    for (const auto& placedMember : placedMembers) {
        if (placedMember == nullptr) {
            throw nb::value_error("placed_members must not contain null entries");
        }
    }
    for (double weight : weights) {
        if (!std::isfinite(weight) || weight <= 0.0) {
            throw nb::value_error("weights must be finite and positive");
        }
    }

    std::vector<std::map<std::string, PhysicalTensor>> memberOutputs(placedMembers.size());
    std::vector<std::map<std::string, Event>> memberOutputReadyEvents(placedMembers.size());
    std::vector<Event> memberProcessingFinishedEvents;
    memberProcessingFinishedEvents.reserve(placedMembers.size());

    StagedEnsembleInputs stagedInputs;
    {
        nb::gil_scoped_release release;
        Stream inputStagingStream(device, Stream::Priority::REGULAR);
        stagedInputs = stageInputsOnceForEnsemble(placedMembers.front(), batchInputs, device, inputStagingStream);
        TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, device);
        for (size_t memberIndex = 1; memberIndex < placedMembers.size(); ++memberIndex) {
            ThorImplementation::StampedNetwork& stampedNetwork = placedMembers[memberIndex]->getStampedNetwork(0);
            for (const auto& [inputName, stagedTensor] : stagedInputs.tensors) {
                auto inputLayer = stampedNetwork.getNamedInput(inputName);
                if (inputLayer == nullptr) {
                    throw std::runtime_error("ensemble input '" + inputName + "' is not present in all member networks");
                }
                std::optional<PhysicalTensor> expectedInput = inputLayer->getFeatureOutput();
                if (!expectedInput.has_value() || expectedInput.value().getDescriptor() != stagedTensor.getDescriptor() ||
                    expectedInput.value().getPlacement() != gpuPlacement) {
                    throw std::runtime_error("ensemble input '" + inputName + "' descriptors or placements differ across members");
                }
            }
        }

        for (size_t memberIndex = 0; memberIndex < placedMembers.size(); ++memberIndex) {
            // Stage each named user input once onto the ensemble GPU, then pass the
            // same staged tensor to every resident member.  Member NetworkInput
            // streams wait on the staged-input ready events, so this preserves
            // stream ordering without repeating H2D copies for each fold.
            std::map<std::string, PhysicalTensor> memberInputs(stagedInputs.tensors.begin(), stagedInputs.tensors.end());
            Event done = placedMembers[memberIndex]->submitBatch(
                /*stampIndex=*/0,
                std::move(memberInputs),
                stagedInputs.readyEvents,
                memberOutputs[memberIndex],
                memberOutputReadyEvents[memberIndex],
                /*isInferenceOnly=*/true,
                /*reusableProcessingFinishedEvent=*/nullptr,
                /*waitForOutputsOnProcessingStream=*/false);
            memberProcessingFinishedEvents.push_back(done);
        }
    }

    if (outputNames.empty()) {
        outputNames = sortedOutputNames(memberOutputs.front());
    }
    validateOutputNames(memberOutputs, outputNames);

    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, device);
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    Stream aggregationStream(device, Stream::Priority::REGULAR);
    std::map<std::string, PhysicalTensor> aggregatedOutputs;

    {
        nb::gil_scoped_release release;
        for (const std::string& outputName : outputNames) {
            const PhysicalTensor& first = memberOutputs.front().at(outputName);
            if (first.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
                throw std::runtime_error("ensemble member output '" + outputName + "' is not GPU-resident; "
                                         "place ensemble members with network_outputs_on_gpu=True");
            }
            if (first.getPlacement().getDeviceNum() != device) {
                throw std::runtime_error("ensemble member output '" + outputName + "' is on a different GPU than the aggregation device");
            }

            const TensorDescriptor descriptor = first.getDescriptor();
            std::vector<PhysicalTensor> sources;
            sources.reserve(memberOutputs.size());
            for (size_t memberIndex = 0; memberIndex < memberOutputs.size(); ++memberIndex) {
                const auto readyIt = memberOutputReadyEvents[memberIndex].find(outputName);
                if (readyIt == memberOutputReadyEvents[memberIndex].end()) {
                    throw std::runtime_error("ensemble member output '" + outputName + "' did not produce a ready event");
                }
                aggregationStream.waitEvent(readyIt->second);

                const PhysicalTensor& tensor = memberOutputs[memberIndex].at(outputName);
                if (tensor.getDescriptor() != descriptor) {
                    throw std::runtime_error("ensemble member output '" + outputName + "' descriptors do not match");
                }
                if (tensor.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
                    throw std::runtime_error("ensemble member output '" + outputName + "' is not GPU-resident");
                }
                if (tensor.getPlacement().getDeviceNum() != device) {
                    throw std::runtime_error("ensemble member output '" + outputName + "' is on a different GPU than the aggregation device");
                }
                sources.push_back(tensor);
            }

            PhysicalTensor aggregatedGpu(gpuPlacement, descriptor);
            ThorImplementation::Ensemble::launchWeightedMean(aggregatedGpu, sources, weights, aggregationStream);
            PhysicalTensor aggregatedCpu(cpuPlacement, descriptor);
            aggregatedCpu.copyFromAsync(aggregatedGpu, aggregationStream);
            aggregatedOutputs.emplace(outputName, aggregatedCpu);
        }
        aggregationStream.synchronize();
    }

    return aggregatedOutputs;
}


std::map<std::string, PhysicalTensor> submitMembersThenAccumulatorNetwork(
    std::vector<std::shared_ptr<Thor::PlacedNetwork>> placedMembers,
    std::shared_ptr<Thor::PlacedNetwork> accumulatorNetwork,
    const std::map<std::string, PhysicalTensor>& batchInputs,
    std::vector<std::string> outputNames,
    int32_t device) {
    if (device < 0) {
        throw nb::value_error("device must be >= 0");
    }
    if (placedMembers.empty()) {
        throw nb::value_error("placed_members must not be empty");
    }
    if (accumulatorNetwork == nullptr) {
        throw nb::value_error("accumulator_network must not be null");
    }
    if (batchInputs.empty()) {
        throw nb::value_error("batch_inputs must not be empty");
    }
    for (const auto& placedMember : placedMembers) {
        if (placedMember == nullptr) {
            throw nb::value_error("placed_members must not contain null entries");
        }
    }

    std::vector<std::map<std::string, PhysicalTensor>> memberOutputs(placedMembers.size());
    std::vector<std::map<std::string, Event>> memberOutputReadyEvents(placedMembers.size());
    std::vector<Event> memberProcessingFinishedEvents;
    memberProcessingFinishedEvents.reserve(placedMembers.size());

    StagedEnsembleInputs stagedInputs;
    {
        nb::gil_scoped_release release;
        Stream inputStagingStream(device, Stream::Priority::REGULAR);
        stagedInputs = stageInputsOnceForEnsemble(placedMembers.front(), batchInputs, device, inputStagingStream);
        TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, device);
        for (size_t memberIndex = 1; memberIndex < placedMembers.size(); ++memberIndex) {
            ThorImplementation::StampedNetwork& stampedNetwork = placedMembers[memberIndex]->getStampedNetwork(0);
            for (const auto& [inputName, stagedTensor] : stagedInputs.tensors) {
                auto inputLayer = stampedNetwork.getNamedInput(inputName);
                if (inputLayer == nullptr) {
                    throw std::runtime_error("ensemble input '" + inputName + "' is not present in all member networks");
                }
                std::optional<PhysicalTensor> expectedInput = inputLayer->getFeatureOutput();
                if (!expectedInput.has_value() || expectedInput.value().getDescriptor() != stagedTensor.getDescriptor() ||
                    expectedInput.value().getPlacement() != gpuPlacement) {
                    throw std::runtime_error("ensemble input '" + inputName + "' descriptors or placements differ across members");
                }
            }
        }

        for (size_t memberIndex = 0; memberIndex < placedMembers.size(); ++memberIndex) {
            std::map<std::string, PhysicalTensor> memberInputs(stagedInputs.tensors.begin(), stagedInputs.tensors.end());
            Event done = placedMembers[memberIndex]->submitBatch(
                /*stampIndex=*/0,
                std::move(memberInputs),
                stagedInputs.readyEvents,
                memberOutputs[memberIndex],
                memberOutputReadyEvents[memberIndex],
                /*isInferenceOnly=*/true,
                /*reusableProcessingFinishedEvent=*/nullptr,
                /*waitForOutputsOnProcessingStream=*/false);
            memberProcessingFinishedEvents.push_back(done);
        }
    }

    if (outputNames.empty()) {
        outputNames = sortedOutputNames(memberOutputs.front());
    }
    validateOutputNames(memberOutputs, outputNames);

    std::map<std::string, PhysicalTensor> accumulatorInputs;
    std::map<std::string, Event> accumulatorInputReadyEvents;
    for (size_t outputIndex = 0; outputIndex < outputNames.size(); ++outputIndex) {
        const std::string& outputName = outputNames[outputIndex];
        const PhysicalTensor& first = memberOutputs.front().at(outputName);
        if (first.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU ||
            first.getPlacement().getDeviceNum() != device) {
            throw std::runtime_error("ensemble member output '" + outputName + "' must be GPU-resident on the aggregation device");
        }
        const TensorDescriptor descriptor = first.getDescriptor();
        for (size_t memberIndex = 0; memberIndex < memberOutputs.size(); ++memberIndex) {
            const PhysicalTensor& tensor = memberOutputs[memberIndex].at(outputName);
            if (tensor.getDescriptor() != descriptor) {
                throw std::runtime_error("ensemble member output '" + outputName + "' descriptors do not match");
            }
            if (tensor.getPlacement() != first.getPlacement()) {
                throw std::runtime_error("ensemble member output '" + outputName + "' placements do not match");
            }
            const auto readyIt = memberOutputReadyEvents[memberIndex].find(outputName);
            if (readyIt == memberOutputReadyEvents[memberIndex].end()) {
                throw std::runtime_error("ensemble member output '" + outputName + "' did not produce a ready event");
            }
            const std::string inputName = accumulatorInputName(outputIndex, memberIndex);
            accumulatorInputs.emplace(inputName, tensor);
            accumulatorInputReadyEvents.emplace(inputName, readyIt->second);
        }
    }

    std::map<std::string, PhysicalTensor> accumulatorOutputs;
    std::map<std::string, Event> accumulatorOutputReadyEvents;
    Event accumulatorDone;
    {
        nb::gil_scoped_release release;
        accumulatorDone = accumulatorNetwork->submitBatch(
            /*stampIndex=*/0,
            std::move(accumulatorInputs),
            accumulatorInputReadyEvents,
            accumulatorOutputs,
            accumulatorOutputReadyEvents,
            /*isInferenceOnly=*/true,
            /*reusableProcessingFinishedEvent=*/nullptr,
            /*waitForOutputsOnProcessingStream=*/true);
        accumulatorDone.synchronize();
    }

    return accumulatorOutputs;
}

}  // namespace

void bind_ensemble_runtime(nb::module_& thor) {
    thor.def(
        "_aggregate_ensemble_outputs_gpu_to_cpu",
        [](std::vector<std::map<std::string, PhysicalTensor>> memberOutputs,
           std::vector<double> weights,
           std::vector<std::string> outputNames,
           int32_t device) {
            if (device < 0) {
                throw nb::value_error("device must be >= 0");
            }
            if (memberOutputs.empty()) {
                throw nb::value_error("member_outputs must not be empty");
            }
            if (weights.size() != memberOutputs.size()) {
                throw nb::value_error("weights must contain one value per member output map");
            }
            if (outputNames.empty()) {
                outputNames = sortedOutputNames(memberOutputs.front());
            }
            validateOutputNames(memberOutputs, outputNames);

            for (double weight : weights) {
                if (!std::isfinite(weight) || weight <= 0.0) {
                    throw nb::value_error("weights must be finite and positive");
                }
            }

            TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, device);
            TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
            Stream aggregationStream(device, Stream::Priority::REGULAR);
            std::map<std::string, PhysicalTensor> aggregatedOutputs;

            {
                nb::gil_scoped_release release;
                for (const std::string& outputName : outputNames) {
                    const PhysicalTensor& first = memberOutputs.front().at(outputName);
                    if (first.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
                        throw std::runtime_error("ensemble member output '" + outputName + "' is not GPU-resident; "
                                                 "place ensemble members with network_outputs_on_gpu=True");
                    }
                    if (first.getPlacement().getDeviceNum() != device) {
                        throw std::runtime_error("ensemble member output '" + outputName + "' is on a different GPU than the aggregation device");
                    }

                    const TensorDescriptor descriptor = first.getDescriptor();
                    std::vector<PhysicalTensor> sources;
                    sources.reserve(memberOutputs.size());
                    for (size_t memberIndex = 0; memberIndex < memberOutputs.size(); ++memberIndex) {
                        const PhysicalTensor& tensor = memberOutputs[memberIndex].at(outputName);
                        if (tensor.getDescriptor() != descriptor) {
                            throw std::runtime_error("ensemble member output '" + outputName + "' descriptors do not match");
                        }
                        if (tensor.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
                            throw std::runtime_error("ensemble member output '" + outputName + "' is not GPU-resident");
                        }
                        if (tensor.getPlacement().getDeviceNum() != device) {
                            throw std::runtime_error("ensemble member output '" + outputName + "' is on a different GPU than the aggregation device");
                        }
                        sources.push_back(tensor);
                    }

                    PhysicalTensor aggregatedGpu(gpuPlacement, descriptor);
                    ThorImplementation::Ensemble::launchWeightedMean(aggregatedGpu, sources, weights, aggregationStream);
                    PhysicalTensor aggregatedCpu(cpuPlacement, descriptor);
                    aggregatedCpu.copyFromAsync(aggregatedGpu, aggregationStream);
                    aggregatedOutputs.emplace(outputName, aggregatedCpu);
                }
                aggregationStream.synchronize();
            }
            return aggregatedOutputs;
        },
        "member_outputs"_a,
        "weights"_a,
        "output_names"_a = std::vector<std::string>{},
        "device"_a = 0,
        R"nbdoc(
Aggregate GPU-resident ensemble member outputs with a weighted mean on GPU, then
materialize exactly the aggregated outputs as CPU PhysicalTensors.

This is an internal bridge for thor.EnsembleModel. Member networks should be
placed with network_outputs_on_gpu=True so same-device NetworkOutput tensors alias
their producer tensors and remain on GPU until this aggregation finishes.
)nbdoc");

    thor.def(
        "_stage_ensemble_inputs_once_for_debug",
        [](std::shared_ptr<Thor::PlacedNetwork> referenceMember,
           std::map<std::string, PhysicalTensor> batchInputs,
           int32_t device) {
            return debugStageEnsembleInputsOnce(referenceMember, batchInputs, device);
        },
        "reference_member"_a,
        "batch_inputs"_a,
        "device"_a = 0,
        R"nbdoc(
Internal testing hook for the ensemble input-staging/fanout path.  It stages each
named input exactly once onto the reference member's GPU input placement and
returns the staged tensors after synchronizing the staging stream.
)nbdoc");

    thor.def(
        "_get_ensemble_member_output_specs",
        [](std::shared_ptr<Thor::PlacedNetwork> placedMember, std::vector<std::string> outputNames) {
            return memberOutputSpecs(placedMember, std::move(outputNames));
        },
        "placed_member"_a,
        "output_names"_a = std::vector<std::string>{},
        R"nbdoc(
Return physical output tensor specs for a placed ensemble member.  This is used
internally to build the ensemble output accumulator network before submitting the
member networks.
)nbdoc");

    thor.def(
        "_infer_ensemble_members_then_accumulator_network",
        [](std::vector<std::shared_ptr<Thor::PlacedNetwork>> placedMembers,
           std::shared_ptr<Thor::PlacedNetwork> accumulatorNetwork,
           std::map<std::string, PhysicalTensor> batchInputs,
           std::vector<std::string> outputNames,
           int32_t device) {
            return submitMembersThenAccumulatorNetwork(
                std::move(placedMembers), accumulatorNetwork, batchInputs, std::move(outputNames), device);
        },
        "placed_members"_a,
        "accumulator_network"_a,
        "batch_inputs"_a,
        "output_names"_a = std::vector<std::string>{},
        "device"_a = 0,
        R"nbdoc(
Submit all resident ensemble member networks, feed their GPU-resident named
outputs into a placed accumulator network using same-device NetworkInput aliasing,
and return the accumulator NetworkOutput tensors.
)nbdoc");

    thor.def(
        "_infer_ensemble_members_and_aggregate_gpu_to_cpu",
        [](std::vector<std::shared_ptr<Thor::PlacedNetwork>> placedMembers,
           std::map<std::string, PhysicalTensor> batchInputs,
           std::vector<double> weights,
           std::vector<std::string> outputNames,
           int32_t device) {
            return submitMembersAndAggregateGpuToCpu(std::move(placedMembers), batchInputs, weights, std::move(outputNames), device);
        },
        "placed_members"_a,
        "batch_inputs"_a,
        "weights"_a,
        "output_names"_a = std::vector<std::string>{},
        "device"_a = 0,
        R"nbdoc(
Submit all resident ensemble member networks from native code, aggregate their
GPU-resident outputs with a weighted mean on GPU, then materialize only the
combined outputs as CPU PhysicalTensors.

This removes Python ThreadPoolExecutor orchestration from thor.EnsembleModel.
Member networks still run on their own Thor/CUDA streams; the aggregation stream
waits on each member NetworkOutput ready event before launching aggregation.
)nbdoc");
}
