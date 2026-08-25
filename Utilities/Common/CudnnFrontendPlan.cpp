#include "Utilities/Common/CudnnFrontendPlan.h"

#include <cudnn_frontend.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace ThorImplementation {
namespace {

namespace fe = cudnn_frontend;

std::atomic<uint64_t> executable_preparation_count{0};

[[nodiscard]] std::vector<std::pair<int64_t, int64_t>> canonicalizeKnobs(
    std::vector<std::pair<int64_t, int64_t>> knobs) {
    std::sort(knobs.begin(), knobs.end());
    for (size_t i = 1; i < knobs.size(); ++i) {
        if (knobs[i - 1].first == knobs[i].first) {
            throw std::invalid_argument("cuDNN Frontend plan selection contains a duplicate knob type.");
        }
    }
    return knobs;
}

[[nodiscard]] bool structuredKnobsAreReplayable(const std::vector<std::pair<int64_t, int64_t>>& knobs) {
    for (const auto& [knob, _] : knobs) {
        cudnnBackendKnobType_t backend_type{};
        const auto frontend_type = static_cast<fe::KnobType_t>(knob);
        if (fe::convert_to_backend_knob_type(frontend_type, backend_type) != CUDNN_STATUS_SUCCESS) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] std::unordered_map<fe::KnobType_t, int64_t> materializeKnobs(
    const CudnnFrontendPlanSelection& selection) {
    std::unordered_map<fe::KnobType_t, int64_t> knobs;
    knobs.reserve(selection.knobs.size());
    for (const auto& [knob, value] : selection.knobs) {
        const auto [_, inserted] = knobs.emplace(static_cast<fe::KnobType_t>(knob), value);
        if (!inserted) {
            throw std::runtime_error("cuDNN Frontend plan selection contains a duplicate knob type.");
        }
    }
    return knobs;
}

void checkFrontendStatus(fe::error_t status, const std::string& message) {
    if (!status.is_good()) {
        throw std::runtime_error(message + ": " + status.get_message());
    }
}

[[nodiscard]] std::string operationPrefix(std::string_view operationName) {
    return operationName.empty() ? std::string("cuDNN Frontend operation")
                                 : std::string("cuDNN Frontend ") + std::string(operationName);
}

void serializeReplayToken(fe::graph::Graph& graph,
                          std::vector<uint8_t>& serializedPlan,
                          const std::string& operation) {
#if defined(CUDNN_FRONTEND_VERSION) && CUDNN_FRONTEND_VERSION >= 12700
    // Frontend 1.27+ can serialize only the finalized backend plan payload.
    checkFrontendStatus(graph.serialize(serializedPlan, false),
                        "Failed to serialize plan-only " + operation + " selection");
#else
    // Older supported Frontend releases serialize graph structure together with
    // the selected precompiled plan.  This remains immutable byte data rather
    // than live descriptor/execution state and is consumed only at preparation.
    checkFrontendStatus(graph.serialize(serializedPlan),
                        "Failed to serialize " + operation + " selection");
#endif
}

void deserializeReplayToken(fe::graph::Graph& graph,
                            cudnnHandle_t handle,
                            const std::vector<uint8_t>& serializedPlan,
                            const std::string& operation) {
#if defined(CUDNN_FRONTEND_VERSION) && CUDNN_FRONTEND_VERSION >= 12700
    checkFrontendStatus(graph.deserialize(handle, serializedPlan, true, false),
                        "Failed to deserialize exact-replay " + operation + " execution plan");
#else
    checkFrontendStatus(graph.deserialize(handle, serializedPlan),
                        "Failed to deserialize exact-replay " + operation + " execution plan");
#endif
}

}  // namespace

CudnnFrontendPlanSelection::CudnnFrontendPlanSelection(int64_t engineId,
                                                       std::vector<std::pair<int64_t, int64_t>> knobValues,
                                                       uint64_t expectedWorkspaceBytes,
                                                       std::vector<uint8_t> serializedPlan)
    : engine_id(engineId),
      knobs(canonicalizeKnobs(std::move(knobValues))),
      expected_workspace_bytes(expectedWorkspaceBytes),
      serialized_plan(std::move(serializedPlan)) {
    if (engine_id < 0 && serialized_plan.empty()) {
        throw std::invalid_argument(
            "cuDNN Frontend structured plan selection requires a non-negative engine id.");
    }
    if (serialized_plan.empty() && !structuredKnobsAreReplayable(knobs)) {
        throw std::invalid_argument(
            "cuDNN Frontend structured plan selection contains a knob type that the installed Frontend cannot replay; "
            "a serialized replay token is required.");
    }
}

CudnnFrontendExecutablePlan::CudnnFrontendExecutablePlan(std::shared_ptr<cudnn_frontend::graph::Graph> graph,
                                                         CudnnFrontendPlanSelection selection,
                                                         uint64_t workspaceBytes,
                                                         int64_t planIndex)
    : graph_(std::move(graph)),
      selection_(std::move(selection)),
      workspace_bytes_(workspaceBytes),
      plan_index_(planIndex) {
    if (!graph_) {
        throw std::invalid_argument("cuDNN Frontend executable plan requires a graph.");
    }
    if (graph_.use_count() != 1) {
        throw std::invalid_argument("cuDNN Frontend executable plan requires exclusive graph ownership.");
    }
    if (plan_index_ < 0) {
        throw std::invalid_argument("cuDNN Frontend executable plan requires a non-negative plan index.");
    }
    if (workspace_bytes_ != selection_.expected_workspace_bytes) {
        throw std::invalid_argument("cuDNN Frontend executable workspace does not match its selection recipe.");
    }
}

uintptr_t CudnnFrontendExecutablePlan::executableId() const noexcept {
    return reinterpret_cast<uintptr_t>(graph_.get());
}

void CudnnFrontendExecutablePlan::execute(cudnnHandle_t handle,
                                          std::unordered_map<int64_t, void*>& tensorPack,
                                          void* workspace) const {
    if (!graph_) {
        throw std::runtime_error("cuDNN Frontend executable plan lost its graph.");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("cuDNN Frontend executable plan requires a cuDNN handle.");
    }
    if (workspace_bytes_ > 0 && workspace == nullptr) {
        throw std::invalid_argument("cuDNN Frontend executable plan requires non-null workspace.");
    }
    auto status = graph_->execute(handle, tensorPack, workspace);
    if (!status.is_good()) {
        throw std::runtime_error("Failed to execute cuDNN Frontend local executable plan: " + status.get_message());
    }
}

CudnnFrontendPlanSelection cudnnFrontendPlanSelectionAtIndex(cudnn_frontend::graph::Graph& graph,
                                                              int64_t planIndex,
                                                              std::string_view operationName) {
    if (planIndex < 0) {
        throw std::invalid_argument(operationPrefix(operationName) + " selection requires a non-negative plan index.");
    }

    int64_t engine_id = -1;
    std::unordered_map<fe::KnobType_t, int64_t> knob_map;
    checkFrontendStatus(graph.get_engine_and_knobs_at_index(planIndex, engine_id, knob_map),
                        "Failed to query " + operationPrefix(operationName) + " engine/knob selection");
    if (engine_id < 0) {
        throw std::runtime_error(operationPrefix(operationName) + " returned a negative engine id.");
    }

    std::vector<std::pair<int64_t, int64_t>> knobs;
    knobs.reserve(knob_map.size());
    for (const auto& [knob, value] : knob_map) {
        knobs.emplace_back(static_cast<int64_t>(knob), value);
    }

    const int64_t reported_workspace_bytes = graph.get_workspace_size_plan_at_index(planIndex);
    if (reported_workspace_bytes < 0) {
        throw std::runtime_error(operationPrefix(operationName) + " returned a negative workspace size.");
    }
    const uint64_t workspace_bytes = static_cast<uint64_t>(reported_workspace_bytes);

    if (structuredKnobsAreReplayable(knobs)) {
        return CudnnFrontendPlanSelection(engine_id, std::move(knobs), workspace_bytes);
    }

    // Frontend 1.27 can expose backend knob kinds that convert_from_backend_knob_type()
    // collapses to NOT_SET.  Preserve the selected plan losslessly as an immutable
    // plan-only token instead of globally retaining any live graph/descriptor state.
    std::vector<uint8_t> serialized_plan;
    serializeReplayToken(graph, serialized_plan, operationPrefix(operationName));
    if (serialized_plan.empty()) {
        throw std::runtime_error(operationPrefix(operationName) +
                                 " plan-only serialization returned an empty replay token.");
    }
    return CudnnFrontendPlanSelection(engine_id, std::move(knobs), workspace_bytes, std::move(serialized_plan));
}

CudnnFrontendPlanSelection cudnnFrontendSelectedSerializedPlanSelection(cudnn_frontend::graph::Graph& graph,
                                                                         std::string_view operationName) {
    const std::string operation = operationPrefix(operationName);
    const int64_t reported_workspace_bytes = graph.get_workspace_size();
    if (reported_workspace_bytes < 0) {
        throw std::runtime_error(operation + " returned a negative workspace size for its selected plan.");
    }

    std::vector<uint8_t> serialized_plan;
    serializeReplayToken(graph, serialized_plan, operation);
    if (serialized_plan.empty()) {
        throw std::runtime_error(operation + " selected-plan serialization returned an empty replay token.");
    }

    // Empirical autotune can reorder execution_plans independently of the
    // heuristic engine-config vector.  Do not publish a potentially stale
    // engine/knob identity; the immutable serialized payload is the exact winner.
    return CudnnFrontendPlanSelection(-1,
                                      {},
                                      static_cast<uint64_t>(reported_workspace_bytes),
                                      std::move(serialized_plan));
}

CudnnFrontendExecutablePlan replayCudnnFrontendExecutablePlan(const CudnnFrontendGraphFactory& graphFactory,
                                                               const CudnnFrontendPlanSelection& selection,
                                                               cudnnHandle_t handle,
                                                               std::string_view operationName) {
    if (!graphFactory) {
        throw std::invalid_argument(operationPrefix(operationName) + " replay requires a graph factory.");
    }
    if (handle == nullptr) {
        throw std::invalid_argument(operationPrefix(operationName) + " replay requires a cuDNN handle.");
    }
    CudnnFrontendPlanSelection normalized_selection(selection.engine_id,
                                                    selection.knobs,
                                                    selection.expected_workspace_bytes,
                                                    selection.serialized_plan);

    executable_preparation_count.fetch_add(1, std::memory_order_relaxed);

    std::shared_ptr<fe::graph::Graph> graph;
    constexpr int64_t replay_plan_index = 0;

    if (normalized_selection.usesSerializedReplay()) {
        // The serialized token is immutable cache state.  Deserialization happens
        // only while preparing this operation and creates a new local Graph/plan.
        graph = std::make_shared<fe::graph::Graph>();
        deserializeReplayToken(*graph, handle, normalized_selection.serialized_plan, operationPrefix(operationName));
    } else {
        graph = graphFactory();
        if (!graph) {
            throw std::runtime_error(operationPrefix(operationName) + " graph factory returned null.");
        }
        if (graph.use_count() != 1) {
            throw std::runtime_error(operationPrefix(operationName) +
                                     " graph factory returned aliased state; executable graphs must be operation-local.");
        }

        checkFrontendStatus(graph->validate(), "Failed to validate exact-replay " + operationPrefix(operationName) + " graph");
        checkFrontendStatus(graph->build_operation_graph(handle),
                            "Failed to build exact-replay " + operationPrefix(operationName) + " operation graph");

        std::unordered_map<fe::KnobType_t, int64_t> knob_map = materializeKnobs(normalized_selection);
        checkFrontendStatus(graph->create_execution_plan(normalized_selection.engine_id, knob_map),
                            "Failed to recreate exact " + operationPrefix(operationName) +
                                " execution plan from engine/knob selection");
        checkFrontendStatus(graph->check_support(handle),
                            "Failed to check support for exact-replay " + operationPrefix(operationName) + " execution plan");
        checkFrontendStatus(graph->build_plan_at_index(handle, replay_plan_index),
                            "Failed to build exact-replay " + operationPrefix(operationName) + " execution plan");
    }

    if (!graph || graph.use_count() != 1) {
        throw std::runtime_error(operationPrefix(operationName) +
                                 " replay did not produce exclusively owned local executable state.");
    }

    const int64_t reported_workspace_bytes = graph->get_workspace_size_plan_at_index(replay_plan_index);
    if (reported_workspace_bytes < 0) {
        throw std::runtime_error(operationPrefix(operationName) + " exact replay returned a negative workspace size.");
    }
    const uint64_t workspace_bytes = static_cast<uint64_t>(reported_workspace_bytes);
    if (workspace_bytes != normalized_selection.expected_workspace_bytes) {
        std::ostringstream message;
        message << operationPrefix(operationName) << " exact replay changed workspace from "
                << normalized_selection.expected_workspace_bytes << " to " << workspace_bytes << " bytes.";
        throw std::runtime_error(message.str());
    }

    return CudnnFrontendExecutablePlan(
        std::move(graph), std::move(normalized_selection), workspace_bytes, replay_plan_index);
}

uint64_t cudnnFrontendExecutablePreparationCountForTests() noexcept {
    return executable_preparation_count.load(std::memory_order_relaxed);
}

}  // namespace ThorImplementation
