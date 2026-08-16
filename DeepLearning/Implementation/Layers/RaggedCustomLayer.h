#pragma once

#include "DeepLearning/Implementation/Layers/CustomLayer.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace ThorImplementation {

// Internal execution specialization for a ragged-preserving DynamicExpression over
// packed ragged values. Input/output trailing value shapes may differ, but the row
// partition and packed-row capacity are preserved. The expression itself owns all
// mathematical computation and carries RAGGED_VALUEWISE_EXTENT using a structural
// offsets input. This class only preserves host-known active-row metadata and
// canonical zero padding so subsequent ragged physical layers can make host-side
// launch decisions without synchronizing.
class RaggedCustomLayer final : public CustomLayer {
   public:
    static constexpr const char* RAGGED_OFFSETS_INPUT_NAME = "__thor_ragged_offsets";

    // Backward-compatible single-values-input/single-output form used by the
    // existing ragged API adapters.
    RaggedCustomLayer(DynamicExpression expression,
                      std::vector<std::string> inputNames,
                      std::vector<std::string> outputNames,
                      TensorPlacement placement,
                      bool inferenceOnly,
                      uint64_t fullCapacityRows,
                      uint64_t inputElementsPerValue,
                      uint64_t outputElementsPerValue,
                      uint32_t valuesInputPort = 0,
                      int64_t stampedId = -1);

    // Backward-compatible multi-values-input/single-output form used by
    // partition-preserving binary ragged expressions such as residual addition.
    RaggedCustomLayer(DynamicExpression expression,
                      std::vector<std::string> inputNames,
                      std::vector<std::string> outputNames,
                      TensorPlacement placement,
                      bool inferenceOnly,
                      uint64_t fullCapacityRows,
                      std::vector<uint64_t> inputElementsPerValue,
                      uint64_t outputElementsPerValue,
                      std::vector<uint32_t> valuesInputPorts,
                      int64_t stampedId = -1);

    // General physical CustomLayer form. Every packed-values input listed in
    // valuesInputPorts must carry the same host-known active-row count. Every
    // output is a packed ragged-values tensor that preserves that active-row
    // count, while outputElementsPerValue describes each output's physical row
    // width so its inactive capacity can be canonicalized independently.
    RaggedCustomLayer(DynamicExpression expression,
                      std::vector<std::string> inputNames,
                      std::vector<std::string> outputNames,
                      TensorPlacement placement,
                      std::vector<std::shared_ptr<PhysicalParameter>> parameters,
                      bool inferenceOnly,
                      uint64_t fullCapacityRows,
                      std::vector<uint64_t> inputElementsPerValue,
                      std::vector<uint64_t> outputElementsPerValue,
                      std::vector<uint32_t> valuesInputPorts,
                      int64_t stampedId = -1,
                      std::vector<DeclaredOutputDescriptor> declaredOutputDescriptors = {});

    std::string getType() override { return "RaggedCustomLayer"; }
    std::string getLayerType() override { return "RaggedCustomLayer"; }

   protected:
    void beforeForwardExpressionRun(uint32_t connectionNumber, Stream& stream) override;
    void afterForwardExpressionRun(uint32_t connectionNumber, Stream& stream) override;
    void afterBackwardErrorExpressionRun(uint32_t connectionNumber, Stream& stream) override;

   private:
    [[nodiscard]] uint32_t applicationIndexForConnection(uint32_t connectionNumber) const;
    [[nodiscard]] uint64_t requireActiveRows(uint32_t applicationIndex) const;
    void zeroInactiveTail(Tensor tensor, uint64_t activeRows, uint64_t rowWidth, Stream stream) const;

    uint64_t fullCapacityRows;
    std::vector<uint64_t> inputElementsPerValue;
    std::vector<uint64_t> outputElementsPerValue;
    std::vector<uint32_t> valuesInputPorts;
    uint32_t inputPortCount;
    uint32_t outputPortCount;
    std::vector<uint64_t> activeRowsByApplication;
};

}  // namespace ThorImplementation
