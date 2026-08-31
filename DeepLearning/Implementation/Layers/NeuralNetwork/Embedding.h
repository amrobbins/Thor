#pragma once

#include "DeepLearning/Implementation/Layers/Optimizers/SparseRowGradient.h"
#include "DeepLearning/Implementation/Layers/TrainableLayer.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/CudaDriver/CudaGraph.h"
#include "Utilities/TensorOperations/Embedding/EmbeddingKernels.h"
#include "Utilities/TensorOperations/Embedding/EmbeddingSparseGradient.h"
#include "Utilities/TensorOperations/Ragged/RuntimeExtent.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <vector>

namespace ThorImplementation {

struct RaggedEmbeddingConfig {
    uint64_t batchSize = 0;
    uint64_t maxTotalValues = 0;
    uint64_t elementsPerValue = 1;
    DataType offsetsDataType = DataType::UINT32;
};

class Embedding final : public TrainableLayer {
   public:
    Embedding(TensorPlacement placement,
              std::vector<std::shared_ptr<PhysicalParameter>> parameters,
              uint64_t vocabularySize,
              uint64_t embeddingDim,
              DataType weightsDataType,
              std::optional<uint64_t> paddingIndex,
              bool sparseGradients,
              bool inferenceOnly,
              int64_t stampedId = -1,
              std::optional<RaggedEmbeddingConfig> raggedConfig = std::nullopt);

    void compileImpl() override;
    void initialize() override;
    void cleanup() override;
    std::optional<Tensor> createFeatureOutputTensor() override;
    std::optional<Tensor> createErrorOutputTensor(bool backPropagateError, uint32_t connectionNumber) override;

    std::optional<Tensor> connectToPreviousLayer(
        Layer* previousLayer, std::optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType = 0) override;
    void connectToNextLayer(Layer* nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) override;
    void forward(std::optional<Tensor> featureInput, bool validationPass, uint32_t batchSize = 0) override;

    void backward(std::optional<Tensor> errorInput, uint32_t batchSize = 0) override;

    void computeFeatureOut(uint32_t connectionNumber) override;
    std::string getLayerType() override { return "Embedding"; }
    uint64_t flopCountForward() override { return 0; }
    uint64_t flopCountBackward() override { return 0; }

   private:
    Tensor weights() const;
    bool isRagged() const { return raggedConfig.has_value(); }
    uint32_t raggedValuesSlot(uint32_t applicationIndex) const { return applicationIndex * 2; }
    uint32_t raggedOffsetsSlot(uint32_t applicationIndex) const { return applicationIndex * 2 + 1; }
    void ensureRaggedApplicationStorage(uint32_t applicationIndex);
    void resetRaggedForwardArrivalBookkeeping();
    uint32_t raggedApplicationCount() const;

    uint64_t vocabularySize;
    uint64_t embeddingDim;
    DataType weightsDataType;
    std::optional<uint64_t> paddingIndex;
    bool sparseGradients;
    std::optional<RaggedEmbeddingConfig> raggedConfig;
    std::vector<RaggedRuntimeExtent> raggedRuntimeExtents;
    std::vector<std::shared_ptr<PreparedEmbeddingForward>> preparedRaggedForwards;
    std::vector<std::set<uint64_t>> raggedAllForwardInputTensorIds;
    std::vector<std::set<uint64_t>> raggedWaitingForwardInputTensorIds;
    std::vector<uint32_t> raggedCurrentValidExampleCounts;
    std::vector<bool> raggedBatchCardinalitySet;
    std::vector<Event> raggedOffsetsReadyEvents;
    std::optional<SparseRowGradient> weightsSparseGradient;
    std::shared_ptr<PreparedEmbeddingSparseGradient> weightsSparseGradientProducer;
    std::optional<CapturedEmbeddingSparseGradient> weightsSparseGradientCapturedGraph;
    std::optional<CudaGraphExecutable> weightsSparseGradientGraphExecutable;
};

}  // namespace ThorImplementation
