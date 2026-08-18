#pragma once

#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include <cstdint>
#include <vector>

namespace ThorImplementation {

// Expression-backed token-wise FullyConnected over packed ragged values.
//
// The mathematical graph is the same DynamicExpression used by the regular
// FullyConnected implementation (matmul -> bias -> activation). The row
// partition is carried as an explicit structural input. Packed MATMUL stages
// consume that runtime extent directly and sanitize only the selected bucket
// slack they are about to read. This wrapper therefore owns no inactive-tail
// canonicalization or host active-row cache.
class RaggedFullyConnected final : public CustomLayer {
   public:
    static constexpr const char* ROW_PARTITION_INPUT_NAME = "row_partition";

    RaggedFullyConnected(DynamicExpression expression,
                         TensorPlacement placement,
                         std::vector<std::shared_ptr<PhysicalParameter>> physicalParameters,
                         bool inferenceOnly,
                         int64_t stampedId = -1);

    std::string getType() override { return "RaggedFullyConnected"; }
    std::string getLayerType() override { return "RaggedFullyConnected"; }
};

}  // namespace ThorImplementation
