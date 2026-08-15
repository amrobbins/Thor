#pragma once

#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include <cstdint>
#include <optional>
#include <vector>

namespace ThorImplementation {

// Expression-backed RMSNorm over packed ragged values.
//
// RMSNorm itself remains an ordinary Expression RMSNORM stage. This wrapper owns
// only the ragged storage/runtime contract that is outside the mathematical
// expression: host-known active-row metadata and canonical zero padding. The
// RMSNORM stage uses its packed-row-capacity annotation to select a cached cuDNN
// row bucket at runtime.
class RaggedRMSNorm final : public CustomLayer {
   public:
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
    [[nodiscard]] uint64_t requireActiveRows(uint32_t connectionNumber) const;
    [[nodiscard]] uint64_t activeRowsForConnection(uint32_t connectionNumber) const;
    void validatePackedTensor(const Tensor& tensor, const char* what) const;
    void zeroInactiveTail(Tensor tensor, uint64_t activeRows, Stream stream) const;

    uint64_t fullCapacityRows;
    uint64_t elementsPerValue;
    std::vector<uint64_t> activeRowsByConnection;
};

}  // namespace ThorImplementation
