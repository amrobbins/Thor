#pragma once

#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/RaggedMatmulCapacityBuckets.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace ThorImplementation {

// Expression-backed token-wise FullyConnected over packed ragged values.
//
// The mathematical graph is the same DynamicExpression used by the regular
// FullyConnected implementation (matmul -> bias -> activation).  Packed-row
// bucketing is an annotation on the Expression MATMUL stage, not a parallel
// hand-written FC implementation.  This class only maintains the structural
// ragged invariants that are outside the value expression itself: host-known
// active-row metadata and canonical zero padding.
class RaggedFullyConnected final : public CustomLayer {
   public:
    RaggedFullyConnected(DynamicExpression expression,
                         TensorPlacement placement,
                         std::vector<std::shared_ptr<PhysicalParameter>> physicalParameters,
                         uint64_t inputFeatures,
                         uint64_t outputFeatures,
                         uint64_t fullCapacityRows,
                         DataType inputDataType,
                         DataType outputDataType,
                         bool inferenceOnly,
                         int64_t stampedId = -1);

    void backward(std::optional<Tensor> errorInput, uint32_t batchSize = 0) override;

    std::string getType() override { return "RaggedFullyConnected"; }
    std::string getLayerType() override { return "RaggedFullyConnected"; }

    [[nodiscard]] uint64_t selectedCapacityRows(uint64_t activeRows) const;

   protected:
    void beforeForwardExpressionRun(uint32_t connectionNumber, Stream& stream) override;
    void afterForwardExpressionRun(uint32_t connectionNumber, Stream& stream) override;
    void afterBackwardErrorExpressionRun(uint32_t connectionNumber, Stream& stream) override;

   private:
    uint64_t requireActiveRows(const Tensor& packedValues) const;
    void zeroRows(Tensor tensor, uint64_t firstRow, uint64_t endRow, uint64_t rowWidth, Stream stream) const;
    uint64_t activeRowsForConnection(uint32_t connectionNumber) const;

    uint64_t inputFeatures;
    uint64_t outputFeatures;
    uint64_t fullCapacityRows;
    DataType inputDataType;
    DataType outputDataType;
    std::vector<uint64_t> activeRowsByConnection;
};

}  // namespace ThorImplementation
