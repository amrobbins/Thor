#pragma once

#include "DeepLearning/Implementation/Layers/Optimizers/SparseRowGradient.h"
#include "DeepLearning/Implementation/Layers/TrainableLayer.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace ThorImplementation {

struct PreparedEmbeddingSparseGradient;

class Embedding final : public TrainableLayer {
   public:
    Embedding(TensorPlacement placement,
              std::vector<std::shared_ptr<PhysicalParameter>> parameters,
              uint64_t vocabularySize,
              uint64_t embeddingDim,
              TensorDescriptor::DataType weightsDataType,
              std::optional<uint64_t> paddingIndex,
              bool sparseGradients,
              bool inferenceOnly,
              int64_t stampedId = -1);

    void compileImpl() override;
    std::optional<Tensor> createFeatureOutputTensor() override;
    std::optional<Tensor> createErrorOutputTensor(bool backPropagateError, uint32_t connectionNumber) override;

    void backward(std::optional<Tensor> errorInput, uint32_t batchSize = 0) override;

    void computeFeatureOut(uint32_t connectionNumber) override;
    std::string getLayerType() override { return "Embedding"; }
    uint64_t flopCountForward() override { return 0; }
    uint64_t flopCountBackward() override { return 0; }

   private:
    Tensor weights() const;

    uint64_t vocabularySize;
    uint64_t embeddingDim;
    TensorDescriptor::DataType weightsDataType;
    std::optional<uint64_t> paddingIndex;
    bool sparseGradients;
    std::optional<SparseRowGradient> weightsSparseGradient;
    std::shared_ptr<PreparedEmbeddingSparseGradient> weightsSparseGradientProducer;
};

}  // namespace ThorImplementation
