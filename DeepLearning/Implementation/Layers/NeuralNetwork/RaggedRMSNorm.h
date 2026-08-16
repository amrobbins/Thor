#pragma once

#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include <cstdint>
#include <optional>
#include <vector>

namespace ThorImplementation {

// Expression-backed RMSNorm over packed ragged values.
//
// The row partition is an explicit structural input.  RowPartitionRuntime owns
// the host-visible active-row cache; packed values carry no ragged runtime
// contract. Packed Expression execution consumes the canonical offsets tensor
// directly as a structural runtime binding.
class RaggedRMSNorm final : public CustomLayer {
   public:
    static constexpr const char* ROW_PARTITION_INPUT_NAME = "row_partition";

    RaggedRMSNorm(DynamicExpression expression,
                  std::vector<std::string> inputNames,
                  std::vector<std::string> outputNames,
                  TensorPlacement placement,
                  std::vector<std::shared_ptr<PhysicalParameter>> physicalParameters,
                  uint64_t fullCapacityRows,
                  uint64_t elementsPerValue,
                  bool inferenceOnly,
                  int64_t stampedId = -1);

    void backward(std::optional<Tensor> errorInput, uint32_t batchSize = 0) override;

    std::string getType() override { return "RaggedRMSNorm"; }
    std::string getLayerType() override { return "RaggedRMSNorm"; }

    [[nodiscard]] uint64_t selectedCapacityRows(uint64_t activeRows) const;

   protected:
    void beforeForwardExpressionRun(uint32_t connectionNumber, Stream& stream) override;
    void afterForwardExpressionRun(uint32_t connectionNumber, Stream& stream) override;
    void afterBackwardErrorExpressionRun(uint32_t connectionNumber, Stream& stream) override;

   private:
    static constexpr uint32_t VALUES_INPUT_PORT = 0;
    static constexpr uint32_t ROW_PARTITION_INPUT_PORT = 1;
    static constexpr uint32_t INPUT_PORT_COUNT = 2;

    [[nodiscard]] uint32_t applicationIndexForConnection(uint32_t connectionNumber) const;
    [[nodiscard]] Tensor packedValuesForApplication(uint32_t applicationIndex) const;
    [[nodiscard]] uint64_t requireActiveRows(uint32_t applicationIndex) const;
    [[nodiscard]] uint64_t activeRowsForApplication(uint32_t applicationIndex) const;
    void validatePackedTensor(const Tensor& tensor, const char* what) const;
    void zeroInactiveTail(Tensor tensor, uint64_t activeRows, Stream stream) const;

    uint64_t fullCapacityRows;
    uint64_t elementsPerValue;
    std::vector<uint64_t> activeRowsByApplication;
};

}  // namespace ThorImplementation
