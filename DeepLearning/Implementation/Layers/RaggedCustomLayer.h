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
// offsets input. Runtime packed extent is obtained directly from that structural
// row partition by the compiled Expression stages, never from host-side cached
// active-row metadata attached to this layer. Inactive packed capacity is not part
// of the logical result; active-aware Expression stages neither read nor canonicalize
// it, and callers must not rely on its incidental contents.
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
    // outputElementsPerValue describes each output's physical row width. Inactive
    // capacity remains outside the logical output contract.
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
};

}  // namespace ThorImplementation
