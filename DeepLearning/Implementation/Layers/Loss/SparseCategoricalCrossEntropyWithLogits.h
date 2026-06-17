#pragma once

#include <optional>

#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Loss/LossWeight.h"
#include "Utilities/TensorOperations/Loss/SparseCategoricalCrossEntropyWithLogitsLoss.h"

namespace ThorImplementation {

class SparseCategoricalCrossEntropyWithLogits : public Loss {
   public:
    static constexpr int MASK_CONNECTION_TYPE = 9341;

    SparseCategoricalCrossEntropyWithLogits(DataType lossDataType,
                                            std::optional<float> lossWeight = std::nullopt,
                                            std::optional<uint32_t> ignoreIndex = std::nullopt);
    ~SparseCategoricalCrossEntropyWithLogits() override {}

    std::optional<Tensor> createFeatureOutputTensor() override;
    std::optional<Tensor> createErrorOutputTensor(bool backPropagateError) override;

    std::optional<Tensor> connectToPreviousLayer(Layer *previousLayer,
                                                 std::optional<Tensor> featureInput,
                                                 Stream stream,
                                                 bool backPropagateError,
                                                 int connectionType) override;

    std::optional<Tensor> connectToMaskInputLayer(Layer *maskLayer, std::optional<Tensor> mask, Stream maskStream);

    void initialize() override;
    void compileImpl() override;
    void infer(std::optional<Tensor> logits, std::optional<Tensor> loss, Stream stream) override;
    void backProp(std::optional<Tensor> labels, std::optional<Tensor> logits, std::optional<Tensor> lossGradient, Stream stream) override;
    void forward(std::optional<Tensor> inputTensor, bool validationPass, uint32_t batchSize = 0) override;
    void ensureNoDeviceCrossing() override;
    std::string getType() override;

   protected:
    void advanceDataIfReady(bool validationPass) override;

   private:
    static bool sparseLabelOrMaskDimensionsMatchFeaturePrefix(const std::vector<uint64_t> &candidateDimensions,
                                                              const std::vector<uint64_t> &featureInputDimensions);
    static std::vector<uint64_t> rawLossDimensionsForFeatureInput(const std::vector<uint64_t> &featureInputDimensions);
    void launchForCurrentTypes();

    std::optional<Tensor> maskInput;
    Stream maskStream;
    bool maskReceived = false;
    uint32_t numRows = 0;
    uint32_t numClasses = 0;
    std::optional<uint32_t> ignoreIndex;
    std::optional<float> lossWeight;
};

}  // namespace ThorImplementation
