#pragma once

#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/TrainingDropoutControllable.h"
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ThorImplementation {

// Expression-backed token-wise FullyConnected over packed ragged values.
//
// The mathematical graph is the same DynamicExpression used by the regular
// FullyConnected implementation (matmul -> bias -> activation -> optional
// epilogue). The row partition is carried as an explicit structural input.
// Same-partition ragged epilogue auxiliaries contribute packed value inputs but
// reuse that one placement-selected partition carrier. The carrier publishes the
// authoritative host partition and may also carry a managed [1] active count for
// device valuewise consumers. This wrapper owns no inactive-tail canonicalization.
class RaggedFullyConnected final : public CustomLayer, public TrainingDropoutControllable {
   public:
    static constexpr const char* ROW_PARTITION_INPUT_NAME = "row_partition";

    RaggedFullyConnected(DynamicExpression expression,
                         TensorPlacement placement,
                         std::vector<std::shared_ptr<PhysicalParameter>> physicalParameters,
                         bool inferenceOnly,
                         int64_t stampedId = -1,
                         bool useResidual = false,
                         std::vector<std::string> epilogueAuxInputNames = {},
                         std::optional<DynamicExpressionVariantId> deterministicTrainingVariantId = std::nullopt,
                         bool trainingDropoutEnabled = true);

    void setTrainingDropoutEnabled(bool enabled) override;
    [[nodiscard]] bool isTrainingDropoutEnabled() const override { return trainingDropoutEnabled; }

    std::string getType() override { return "RaggedFullyConnected"; }
    std::string getLayerType() override { return "RaggedFullyConnected"; }

   protected:
    void prepareApplicationOutputsForDownstream(uint32_t applicationIndex) override {
        propagateApplicationRowPartitionHostState(applicationIndex, 1);
    }

   private:
    std::optional<DynamicExpressionVariantId> deterministicTrainingVariantId;
    bool trainingDropoutEnabled = true;
};

}  // namespace ThorImplementation
