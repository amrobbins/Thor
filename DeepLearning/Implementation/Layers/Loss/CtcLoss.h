#pragma once

#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Loss/LossWeight.h"
#include "Utilities/TensorOperations/Loss/CtcLoss.h"

#include <memory>
#include <optional>

namespace ThorImplementation {

// cuDNN-backed CTC loss implementation layer.
//
// Canonical policy:
//   * cuDNN only; no native or CPU fallback.
//   * dense activations/logits: physical [B, T, C], FP32, GPU.
//   * labels values: packed physical [maxTotalLabelValues], INT32, GPU.
//   * labels offsets: physical [B + 1], canonical UINT32/UINT64, GPU.
//   * label lengths are derived from offsets on device into a reusable INT32
//     buffer immediately before the cuDNN call.
//   * input lengths: physical [B, 1] (or [B]), INT32, GPU.
//   * raw per-sample loss output: [B, 1], FP32, GPU.
//   * gradient output matches [B, T, C], FP32, GPU.
//
// With CUDNN_LOSS_NORMALIZATION_SOFTMAX, cuDNN consumes unnormalized
// activations/logits, applies the CTC softmax normalization internally, and
// returns gradients with respect to those activations.
class CtcLoss : public Loss {
   public:
    static constexpr int LABEL_OFFSETS_CONNECTION_TYPE = 11941;
    static constexpr int INPUT_LENGTHS_CONNECTION_TYPE = 11942;

    explicit CtcLoss(CtcLossOobGradientMode oobGradientMode = CtcLossOobGradientMode::ZERO,
                     std::optional<float> lossWeight = std::nullopt);
    ~CtcLoss() override = default;

    std::optional<Tensor> createFeatureOutputTensor() override;
    std::optional<Tensor> createErrorOutputTensor(bool backPropagateError) override;

    std::optional<Tensor> connectToPreviousLayer(Layer* previousLayer,
                                                 std::optional<Tensor> featureInput,
                                                 Stream stream,
                                                 bool backPropagateError,
                                                 int connectionType) override;

    std::optional<Tensor> connectToLabelOffsetsInputLayer(Layer* labelOffsetsLayer,
                                                          std::optional<Tensor> labelOffsets,
                                                          Stream labelOffsetsStream);
    std::optional<Tensor> connectToInputLengthsInputLayer(Layer* inputLengthsLayer,
                                                          std::optional<Tensor> inputLengths,
                                                          Stream inputLengthsStream);

    void initialize() override;
    void compileImpl() override;
    void cleanup() override;

    void infer(std::optional<Tensor> probabilities, std::optional<Tensor> loss, Stream stream) override;
    void backProp(std::optional<Tensor> labels, std::optional<Tensor> probabilities, std::optional<Tensor> lossGradient, Stream stream) override;
    void forward(std::optional<Tensor> inputTensor, bool validationPass, uint32_t validExampleCount = 0) override;
    void ensureNoDeviceCrossing() override;
    std::string getType() override;
    std::vector<Stream> getProcessingStreams() override;
    std::vector<Event> getSynchronizeEvents() override;

    std::optional<Tensor> getLabelOffsetsInput() const { return labelOffsetsInput; }
    std::optional<Tensor> getGeneratedLabelLengthsForTesting() const { return generatedLabelLengths; }
    std::optional<Tensor> getLabelOffsetsValidationErrorBitsForTesting() const { return labelOffsetsValidationErrorBits; }
    std::optional<Tensor> getInputLengthsInput() const { return inputLengthsInput; }
    size_t getWorkspaceSizeInBytesForTesting() const { return ctcPlan ? ctcPlan->getWorkspaceSizeInBytes() : 0; }

   protected:
    void advanceDataIfReady(bool validationPass) override;

   private:
    static std::vector<uint64_t> rawLossDimensionsForProbabilities(const std::vector<uint64_t>& probabilityDimensions);
    void validateConnectedDescriptors();
    void runCudnn(Stream stream);

    uint32_t backendMaxLabelLength = 0;
    CtcLossOobGradientMode oobGradientMode;
    std::optional<float> lossWeight;

    std::optional<Tensor> labelOffsetsInput;
    std::optional<Tensor> inputLengthsInput;
    std::optional<Tensor> generatedLabelLengths;
    std::optional<Tensor> labelOffsetsValidationErrorBits;
    std::optional<Tensor> workspace;
    std::optional<Tensor> inferenceGradientScratch;

    Stream labelOffsetsStream;
    Stream inputLengthsStream;
    Event labelOffsetsReadyEvent;
    Event inputLengthsReadyEvent;
    Event auxiliaryInputsReusableEvent;

    bool labelOffsetsReceived = false;
    bool inputLengthsReceived = false;

    uint32_t maxTimeSteps = 0;
    uint32_t ctcBatchSize = 0;
    uint32_t numClasses = 0;
    uint64_t maxTotalLabelValues = 0;

    std::unique_ptr<CudnnCtcLossPlan> ctcPlan;
};

}  // namespace ThorImplementation
