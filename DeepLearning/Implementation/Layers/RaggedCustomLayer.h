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
// offsets input. Runtime packed extent is obtained from that structural row
// partition, never from metadata attached to values tensors. This class only
// canonicalizes inactive capacity so subsequent ragged physical layers observe
// deterministic padding.
class RaggedCustomLayer final : public CustomLayer {
   public:
    static constexpr const char* RAGGED_OFFSETS_INPUT_NAME = "__thor_ragged_offsets";

    // Single-values-input/single-output form used by the existing ragged API
    // adapters. The structural offsets port is explicit so runtime extent never
    // has to be inferred from values-owned metadata.
    RaggedCustomLayer(DynamicExpression expression,
                      std::vector<std::string> inputNames,
                      std::vector<std::string> outputNames,
                      TensorPlacement placement,
                      bool inferenceOnly,
                      uint64_t fullCapacityRows,
                      uint64_t inputElementsPerValue,
                      uint64_t outputElementsPerValue,
                      uint32_t valuesInputPort,
                      uint32_t offsetsInputPort,
                      int64_t stampedId = -1);

    // Multi-values-input/single-output form used by partition-preserving binary
    // ragged expressions such as residual addition. Every values port shares the
    // explicitly identified structural offsets port.
    RaggedCustomLayer(DynamicExpression expression,
                      std::vector<std::string> inputNames,
                      std::vector<std::string> outputNames,
                      TensorPlacement placement,
                      bool inferenceOnly,
                      uint64_t fullCapacityRows,
                      std::vector<uint64_t> inputElementsPerValue,
                      uint64_t outputElementsPerValue,
                      std::vector<uint32_t> valuesInputPorts,
                      uint32_t offsetsInputPort,
                      int64_t stampedId = -1);

    // General physical CustomLayer form. Every packed-values input listed in
    // valuesInputPorts shares the single structural offsets input identified by
    // offsetsInputPort. Every output preserves that row partition, while
    // outputElementsPerValue describes each output's physical row width so its
    // inactive capacity can be canonicalized independently.
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
                      uint32_t offsetsInputPort,
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
    uint32_t offsetsInputPort;
    uint32_t inputPortCount;
    uint32_t outputPortCount;
    std::vector<uint64_t> activeRowsByApplication;
};

}  // namespace ThorImplementation
