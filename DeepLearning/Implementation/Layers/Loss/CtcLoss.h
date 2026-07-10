#pragma once

#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Loss/LossWeight.h"
#include "Utilities/TensorOperations/Loss/CtcLoss.h"

#include <memory>
#include <optional>

namespace ThorImplementation {

// cuDNN-backed CTC loss implementation layer.
//
// v1 policy:
//   * cuDNN only; no native or CPU fallback.
//   * dense activations/logits: physical [B, T, C], FP32, GPU.
//   * padded labels: physical [B, maxLabelLength], int32, GPU. The layer compacts
//     them to cuDNN's packed label list on device using label lengths.
//   * label lengths and input lengths: physical [B, 1], int32, GPU.
//   * raw per-sample loss output: [B, 1], FP32, GPU.
//   * gradient output matches [B, T, C], FP32, GPU.
//
// With CUDNN_LOSS_NORMALIZATION_SOFTMAX, cuDNN consumes unnormalized
// activations/logits, applies the CTC softmax normalization internally, and
// returns gradients with respect to those activations. This implementation layer
// intentionally exposes that cuDNN contract directly.
class CtcLoss : public Loss {
   public:
    static constexpr int LABEL_LENGTHS_CONNECTION_TYPE = 11941;
    static constexpr int INPUT_LENGTHS_CONNECTION_TYPE = 11942;

    explicit CtcLoss(uint32_t maxLabelLength,
                     CtcLossOobGradientMode oobGradientMode = CtcLossOobGradientMode::ZERO,
                     std::optional<float> lossWeight = std::nullopt);
    ~CtcLoss() override = default;

    std::optional<Tensor> createFeatureOutputTensor() override;
    std::optional<Tensor> createErrorOutputTensor(bool backPropagateError) override;

    std::optional<Tensor> connectToPreviousLayer(Layer* previousLayer,
                                                 std::optional<Tensor> featureInput,
                                                 Stream stream,
                                                 bool backPropagateError,
                                                 int connectionType) override;

    std::optional<Tensor> connectToLabelLengthsInputLayer(Layer* labelLengthsLayer,
                                                          std::optional<Tensor> labelLengths,
                                                          Stream labelLengthsStream);
    std::optional<Tensor> connectToInputLengthsInputLayer(Layer* inputLengthsLayer,
                                                          std::optional<Tensor> inputLengths,
                                                          Stream inputLengthsStream);

    void initialize() override;
    void compileImpl() override;
    void cleanup() override;

    void infer(std::optional<Tensor> probabilities, std::optional<Tensor> loss, Stream stream) override;
    void backProp(std::optional<Tensor> labels, std::optional<Tensor> probabilities, std::optional<Tensor> lossGradient, Stream stream) override;
    void forward(std::optional<Tensor> inputTensor, bool validationPass, uint32_t batchSize = 0) override;
    void ensureNoDeviceCrossing() override;
    std::string getType() override;
    std::vector<Event> getSynchronizeEvents() override;

    std::optional<Tensor> getLabelLengthsInput() const { return labelLengthsInput; }
    std::optional<Tensor> getInputLengthsInput() const { return inputLengthsInput; }
    size_t getWorkspaceSizeInBytesForTesting() const { return ctcPlan ? ctcPlan->getWorkspaceSizeInBytes() : 0; }

   protected:
    void advanceDataIfReady(bool validationPass) override;

   private:
    static std::vector<uint64_t> rawLossDimensionsForProbabilities(const std::vector<uint64_t>& probabilityDimensions);
    void validateConnectedDescriptors();
    void runCudnn(Stream stream);

    uint32_t maxLabelLength;
    CtcLossOobGradientMode oobGradientMode;
    std::optional<float> lossWeight;

    std::optional<Tensor> labelLengthsInput;
    std::optional<Tensor> inputLengthsInput;
    std::optional<Tensor> workspace;
    std::optional<Tensor> inferenceGradientScratch;
    std::optional<Tensor> packedLabels;

    Stream labelLengthsStream;
    Stream inputLengthsStream;

    bool labelLengthsReceived = false;
    bool inputLengthsReceived = false;

    uint32_t maxTimeSteps = 0;
    uint32_t ctcBatchSize = 0;
    uint32_t numClasses = 0;

    std::unique_ptr<CudnnCtcLossPlan> ctcPlan;
};

}  // namespace ThorImplementation
