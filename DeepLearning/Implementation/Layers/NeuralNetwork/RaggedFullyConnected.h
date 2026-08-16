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
// FullyConnected implementation (matmul -> bias -> activation). The row
// partition is carried as an explicit structural input, and the host-known
// packed extent is obtained from its RowPartitionRuntime rather than from
// metadata owned by the values tensor.
//
// Packed Expression execution receives the same canonical offsets tensor as an
// explicit internal structural binding, so bucket selection never depends on
// values-owned ragged metadata.
class RaggedFullyConnected final : public CustomLayer {
   public:
    static constexpr const char* ROW_PARTITION_INPUT_NAME = "row_partition";

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
    [[nodiscard]] uint32_t applicationIndexForConnection(uint32_t connectionNumber) const;
    [[nodiscard]] uint64_t requireActiveRows(uint32_t applicationIndex) const;
    [[nodiscard]] Tensor packedValuesForApplication(uint32_t applicationIndex) const;
    void zeroRows(Tensor tensor, uint64_t firstRow, uint64_t endRow, uint64_t rowWidth, Stream stream) const;
    [[nodiscard]] uint64_t activeRowsForApplication(uint32_t applicationIndex) const;

    static constexpr uint32_t VALUES_INPUT_PORT = 0;
    static constexpr uint32_t ROW_PARTITION_INPUT_PORT = 1;
    static constexpr uint32_t INPUT_PORT_COUNT = 2;

    uint64_t inputFeatures;
    uint64_t outputFeatures;
    uint64_t fullCapacityRows;
    DataType inputDataType;
    DataType outputDataType;
    std::vector<uint64_t> activeRowsByApplication;
};

}  // namespace ThorImplementation
