#pragma once

#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include <cstdint>
#include <vector>

namespace ThorImplementation {

// Expression-backed RMSNorm over packed ragged values.
//
// The row partition is an explicit structural input. Bucket selection and the
// exact pre-read sanitation required by cuDNN live in the stamped RMSNorm
// consumer stages; this wrapper does not canonicalize produced inactive storage.
class RaggedRMSNorm final : public CustomLayer {
   public:
    static constexpr const char* ROW_PARTITION_INPUT_NAME = "row_partition";

    RaggedRMSNorm(DynamicExpression expression,
                  std::vector<std::string> inputNames,
                  std::vector<std::string> outputNames,
                  TensorPlacement placement,
                  std::vector<std::shared_ptr<PhysicalParameter>> physicalParameters,
                  bool inferenceOnly,
                  int64_t stampedId = -1);

    std::string getType() override { return "RaggedRMSNorm"; }
    std::string getLayerType() override { return "RaggedRMSNorm"; }
};

}  // namespace ThorImplementation
