#pragma once

#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/TrainingDropoutControllable.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ThorImplementation {

// Execution semantics that belong to Attention itself rather than to whichever
// physical executor currently hosts it. A1 keeps the existing CustomLayer
// executor as a compatibility stamping scaffold while A2/A3 move forward/backward
// ownership into Attention-owned native plans. A5 executes that strict shared
// backward directly and no longer uses CustomLayer's split backward runtime.
class AttentionTrainingExecutionContract final {
   public:
    static constexpr bool kForwardReplayAllowed = false;
    static constexpr uint32_t kPhysicalForwardExecutionsPerApplication = 1;
    static constexpr uint32_t kPhysicalBackwardExecutionsPerApplication = 1;
    static constexpr bool kBackwardRequiresMatchingForwardState = true;

    AttentionTrainingExecutionContract(std::optional<DynamicExpressionVariantId> deterministicTrainingVariantId,
                                       bool trainingDropoutEnabled)
        : deterministicTrainingVariantId(deterministicTrainingVariantId),
          trainingDropoutEnabled(trainingDropoutEnabled) {}

    void setTrainingDropoutEnabled(bool enabled) { trainingDropoutEnabled = enabled; }

    [[nodiscard]] bool isTrainingDropoutEnabled() const { return trainingDropoutEnabled; }

    [[nodiscard]] DynamicExpressionVariantId activeTrainingVariant() const {
        if (!trainingDropoutEnabled && deterministicTrainingVariantId.has_value())
            return deterministicTrainingVariantId.value();
        return kPrimaryDynamicExpressionVariant;
    }

    [[nodiscard]] std::optional<DynamicExpressionVariantId> getDeterministicTrainingVariantId() const {
        return deterministicTrainingVariantId;
    }

   private:
    std::optional<DynamicExpressionVariantId> deterministicTrainingVariantId;
    bool trainingDropoutEnabled = true;
};

// Each application/variant owns the exact already-stamped real forward plan and,
// when trainable, the Q/K/V/O/stats retained from that plan. Different gradient
// destinations will branch inside one shared backward plan in A3 rather than
// creating additional differentiated plans.
struct AttentionNativeExecutionVariantPlans {
    std::shared_ptr<StampedExecutionPlan> forward;
    std::shared_ptr<StampedExecutionPlan> backward;
    std::optional<RetainedAttentionForwardValues> retained_forward;

    [[nodiscard]] bool hasForward() const { return forward != nullptr; }
    [[nodiscard]] bool hasBackward() const { return backward != nullptr; }
    [[nodiscard]] bool hasRetainedForward() const {
        return retained_forward.has_value() && retained_forward->complete();
    }
};

struct AttentionNativeApplicationExecutionPlans {
    std::unordered_map<DynamicExpressionVariantId, AttentionNativeExecutionVariantPlans> variants;
};

struct AttentionNativeExecutionPlans {
    std::unordered_map<uint32_t, AttentionNativeApplicationExecutionPlans> applications;

    [[nodiscard]] bool empty() const { return applications.empty(); }
    void clear() { applications.clear(); }

    [[nodiscard]] const AttentionNativeExecutionVariantPlans& at(uint32_t applicationIndex,
                                                                  DynamicExpressionVariantId variantId) const {
        auto appIt = applications.find(applicationIndex);
        if (appIt == applications.end()) {
            throw std::out_of_range("Attention native execution plans have no such application.");
        }
        auto variantIt = appIt->second.variants.find(variantId);
        if (variantIt == appIt->second.variants.end()) {
            throw std::out_of_range("Attention native execution plans have no such execution variant.");
        }
        return variantIt->second;
    }
};

// Attention owns its training execution contract and native forward/backward
// plans.  CustomLayer remains only the temporary expression/connection stamping
// scaffold; A5 no longer uses its split backward or optimizer-redifferentiation
// runtime paths for Attention.
class Attention final : public CustomLayer, public TrainingDropoutControllable {
   public:
    Attention(DynamicExpression expr,
              std::vector<std::string> inputNames,
              std::vector<std::string> outputNames,
              const TensorPlacement& placement,
              const std::vector<std::shared_ptr<PhysicalParameter>>& parameters,
              bool inferenceOnly,
              int64_t stampedId,
              std::vector<DeclaredOutputDescriptor> declaredOutputDescriptors,
              std::optional<DynamicExpressionVariantId> deterministicTrainingVariantId,
              bool trainingDropoutEnabled,
              std::vector<bool> inputDimensionsIncludeBatch = {},
              std::optional<uint32_t> fixedBatchCapacity = std::nullopt)
        : CustomLayer(std::move(expr),
                      inputNames,
                      outputNames,
                      placement,
                      parameters,
                      inferenceOnly,
                      stampedId,
                      std::move(declaredOutputDescriptors),
                      false,
                      false,
                      std::move(inputDimensionsIncludeBatch),
                      fixedBatchCapacity),
          trainingExecutionContract(deterministicTrainingVariantId, trainingDropoutEnabled) {
        applyTrainingExecutionVariant();
    }

    void setTrainingDropoutEnabled(bool enabled) override {
        trainingExecutionContract.setTrainingDropoutEnabled(enabled);
        applyTrainingExecutionVariant();
    }

    [[nodiscard]] bool isTrainingDropoutEnabled() const override {
        return trainingExecutionContract.isTrainingDropoutEnabled();
    }

    [[nodiscard]] const AttentionTrainingExecutionContract& getTrainingExecutionContract() const {
        return trainingExecutionContract;
    }

    // Native ownership is inspectable for architectural tests. A2 installs the
    // real forward/retained state, A3 installs the no-replay shared backward, and
    // A5 executes that one plan as Attention's training backward.
    [[nodiscard]] const AttentionNativeExecutionPlans& getNativeExecutionPlans() const {
        return nativeExecutionPlans;
    }

    void cleanup() override {
        nativeExecutionPlans.clear();
        CustomLayer::cleanup();
    }

   protected:
    bool wantsNativeSharedBackwardPlan() const override { return !isInferenceOnly(); }
    bool executesNativeSharedBackwardPlan() const override { return !isInferenceOnly(); }


    void onNativeSharedBackwardExecutionVariantStamped(
        uint32_t applicationIndex,
        DynamicExpressionVariantId variantId,
        const std::shared_ptr<StampedExecutionPlan>& backwardPlan) override {
        if (backwardPlan == nullptr) {
            throw std::runtime_error("Attention native shared backward ownership received a null plan.");
        }
        const AttentionNativeExecutionVariantPlans& nativeVariant =
            nativeExecutionPlans.at(applicationIndex, variantId);
        if (!nativeVariant.hasForward() || !nativeVariant.hasRetainedForward() ||
            !nativeVariant.retained_forward->forward_state) {
            throw std::runtime_error(
                "Attention native shared backward requires the exact retained state owned by its stamped real forward.");
        }
        const uintptr_t expectedForwardStateId =
            reinterpret_cast<uintptr_t>(nativeVariant.retained_forward->forward_state.get());
        if (expectedForwardStateId == 0) {
            throw std::runtime_error("Attention native shared backward retained-forward state has no identity.");
        }

        const std::vector<std::string> kinds = backwardPlan->stageKindNames();
        const size_t attentionForwardCount =
            static_cast<size_t>(std::count(kinds.begin(), kinds.end(), "Attention"));
        const size_t attentionBackwardCount =
            static_cast<size_t>(std::count(kinds.begin(), kinds.end(), "AttentionBackward"));
        if (attentionForwardCount != 0 || attentionBackwardCount != 1) {
            throw std::runtime_error(
                "Attention native shared backward must contain zero Attention forward stages and exactly one AttentionBackward stage.");
        }
        const auto diagnostics = backwardPlan->attentionBackwardStateDiagnostics();
        if (diagnostics.size() != 1 || !diagnostics.front().linked_forward_state ||
            diagnostics.front().linked_forward_state_id != expectedForwardStateId ||
            diagnostics.front().fallback_forward_executable_id != 0 ||
            diagnostics.front().fallback_forward_workspace_bytes != 0 ||
            diagnostics.front().fallback_forward_scratch_present) {
            throw std::runtime_error(
                "Attention native shared backward must be linked to the exact retained state owned by its real forward, with no "
                "fallback-forward executable or workspace.");
        }
        nativeExecutionPlans.applications[applicationIndex].variants[variantId].backward = backwardPlan;
    }

    void onForwardExecutionVariantStamped(uint32_t applicationIndex,
                                           DynamicExpressionVariantId variantId,
                                           const std::shared_ptr<StampedExecutionPlan>& forwardPlan,
                                           bool supportsBackward) override {
        if (forwardPlan == nullptr) {
            throw std::runtime_error("Attention native execution ownership received a null forward plan.");
        }

        AttentionNativeExecutionVariantPlans& nativeVariant =
            nativeExecutionPlans.applications[applicationIndex].variants[variantId];
        nativeVariant.forward = forwardPlan;
        nativeVariant.backward.reset();
        nativeVariant.retained_forward.reset();

        // Evaluation/inference forward needs no training state. Every execution
        // variant that can participate in training, however, must retain the exact
        // physical values consumed/produced by its one SDPA stage.
        if (isInferenceOnly() || !supportsBackward) {
            return;
        }

        std::vector<RetainedAttentionForwardValues> retained =
            forwardPlan->retainAttentionForwardValuesForBackward();
        if (retained.size() != 1) {
            throw std::runtime_error("Physical Attention forward must contain exactly one Attention stage; found " +
                                     std::to_string(retained.size()) + ".");
        }
        nativeVariant.retained_forward = std::move(retained.front());
    }

   private:
    void applyTrainingExecutionVariant() {
        setActiveTrainingExecutionVariant(trainingExecutionContract.activeTrainingVariant());
    }

    AttentionTrainingExecutionContract trainingExecutionContract;
    AttentionNativeExecutionPlans nativeExecutionPlans;
};

}  // namespace ThorImplementation
