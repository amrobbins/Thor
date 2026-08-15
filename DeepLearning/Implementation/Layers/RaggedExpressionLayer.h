#pragma once

#include "DeepLearning/Implementation/Layers/CustomLayer.h"

#include <cstdint>
#include <vector>

namespace ThorImplementation {

// Internal execution specialization for a ragged-preserving DynamicExpression over
// packed ragged values. Input/output trailing value shapes may differ, but the row
// partition and packed-row capacity are preserved. The expression itself owns all mathematical computation and
// carries RAGGED_VALUEWISE_EXTENT using a structural offsets input. This class only
// preserves host-known active-row metadata and canonical zero padding so subsequent
// ragged physical layers can make host-side launch decisions without synchronizing.
class RaggedExpressionLayer final : public CustomLayer {
   public:
    RaggedExpressionLayer(DynamicExpression expression,
                          std::vector<std::string> inputNames,
                          std::vector<std::string> outputNames,
                          TensorPlacement placement,
                          bool inferenceOnly,
                          uint64_t fullCapacityRows,
                          uint64_t inputElementsPerValue,
                          uint64_t outputElementsPerValue,
                          uint32_t valuesInputPort = 0,
                          int64_t stampedId = -1);

    std::string getType() override { return "RaggedExpressionLayer"; }
    std::string getLayerType() override { return "RaggedExpressionLayer"; }

   protected:
    void beforeForwardExpressionRun(uint32_t connectionNumber, Stream& stream) override;
    void afterForwardExpressionRun(uint32_t connectionNumber, Stream& stream) override;
    void afterBackwardErrorExpressionRun(uint32_t connectionNumber, Stream& stream) override;

   private:
    [[nodiscard]] uint32_t applicationIndexForConnection(uint32_t connectionNumber) const;
    [[nodiscard]] uint64_t requireActiveRows(uint32_t applicationIndex) const;
    void zeroInactiveTail(Tensor tensor, uint64_t activeRows, uint64_t rowWidth, Stream stream) const;

    uint64_t fullCapacityRows;
    uint64_t inputElementsPerValue;
    uint64_t outputElementsPerValue;
    uint32_t valuesInputPort;
    uint32_t inputPortCount;
    uint32_t outputPortCount;
    std::vector<uint64_t> activeRowsByApplication;
};

}  // namespace ThorImplementation
