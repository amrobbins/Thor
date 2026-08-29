#pragma once

#include <optional>
#include <vector>
#include "DeepLearning/Implementation/Layers/TrainableLayer.h"
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"
#include "DeepLearning/Implementation/ThorError.h"

namespace ThorImplementation {

/**
 * Parameter epsilon is used in the batch normalization formula and must be >= 0.00001.
 * Parameter exponentialRunningAverageFactor is the steady-state floor for the running-stat update factor. Training begins with an exact
 * cumulative moving average using factor 1 / numItemsObserved, then switches to the configured EMA factor once that factor is larger.
 * cuDNN applies the selected factor via: runningMean = runningMean*(1-factor) + newMean*factor.
 *
 * This layer will serialize processing of each input by synchronizing all streams with stream[0] when multiple connections are present, for
 * compatibility with cuDNN.
 *
 * Batch Normalization supports rank-2 [N,C], rank-4 [N,C,H,W], or rank-5 [N,C,D,H,W] input tensors.
 *
 * Full-capacity training batches use live batch statistics and update scale, bias, and running statistics. Validation, inference-only
 * execution, and partial-capacity training batches use the fixed running statistics. A partial-capacity training batch still propagates dx,
 * but it does not produce or apply BatchNormalization parameter gradients.
 */
class BatchNormalization : public TrainableLayer {
   public:
    ~BatchNormalization() override;

    BatchNormalization(const TensorPlacement& placement,
                       bool inferenceOnly,
                       uint64_t numItemsObserved,
                       std::optional<double> exponentialRunningAverageFactor = std::nullopt,
                       std::optional<double> epsilon = std::nullopt,
                       std::optional<DataType> storageDataType = std::nullopt,
                       std::vector<std::shared_ptr<PhysicalParameter>> physicalParameters = {},
                       int64_t stampedId = -1);

    double getExponentialRunningAverageFactor() const { return exponentialRunningAverageFactor; }
    void setExponentialRunningAverageFactor(double exponentialRunningAverageFactor) {
        THOR_THROW_IF_FALSE(exponentialRunningAverageFactor > 0.0);
        THOR_THROW_IF_FALSE(exponentialRunningAverageFactor <= 1.0);
        this->exponentialRunningAverageFactor = exponentialRunningAverageFactor;
    }

    double getEpsilon() const { return epsilon; }
    void setEpsilon(double epsilon) { this->epsilon = epsilon; }

    std::string getLayerType() override { return "BatchNormalization"; }

    uint64_t getNumItemsObserved() const { return itemsObserved; }
    void setNumItemsObserved(uint64_t numItemsObserved) { itemsObserved = numItemsObserved; }

    std::optional<Tensor> createFeatureOutputTensor() override;
    std::optional<Tensor> createErrorOutputTensor(bool backPropagateError, uint32_t connectionNumber) override;
    void forward(std::optional<Tensor> featureInput, bool isValidation, uint32_t validExampleCount = 0) override;

    uint64_t flopCountForward() override;
    uint64_t flopCountBackward() override;

    void cleanup() override;

   protected:
    void compileImpl() override;

   private:
    void computeFeatureOut(uint32_t connectionNumber) override;
    bool usesFusedBackwardImplementation() const override { return true; }
    std::optional<Event> computeErrorOutAccumulateWeightsGradienFused(uint32_t connectionNumber, bool clearWeightsGradientFirstIfFused) override;
    void accumulateWeightsGradient(uint32_t connectionNumber, bool clearGradientFirst) override;
    bool shouldApplyParameterUpdatesForBatch(uint32_t validExampleCount) const override;

    void runForward(std::optional<Tensor> inputTensor,
                    std::optional<Tensor> outputTensor,
                    Stream stream,
                    unsigned int connectionNumber,
                    Tensor weights,
                    std::optional<Tensor> biases);

   protected:
    Tensor weights;
    Tensor biases;
    Tensor resultRunningMean;
    Tensor resultRunningVariance;

   private:
    static const float ALPHA_NO_SCALE;
    static const float BETA_CLEAR;

    unsigned int batchSize = 0;
    unsigned int numChannels = 0;
    unsigned int depth = 0;
    unsigned int height = 0;
    unsigned int width = 0;
    size_t cudnnTensorRank = 4;

    double exponentialRunningAverageFactor;
    uint64_t itemsObserved = 0;
    double epsilon;

    std::vector<Tensor> resultSaveMean;
    std::vector<Tensor> resultSaveInvVariance;
    std::vector<bool> forwardUsedTrainingStatistics;
    std::vector<Tensor> scratchDScale;
    std::vector<Tensor> scratchDBias;

    // Since weights gradients and error gradient is a fused operation, then when back prop is pruned
    // we still need some valid chunk of memory to write values in, which we ignore.
    std::optional<Tensor> scratchErrorOutput = std::nullopt;
    // One stable dependency event per connection. The returned Event copy is
    // consumed immediately by TrainableLayer; subsequent passes re-record the
    // same cudaEvent_t after all waits for the previous generation were issued.
    std::vector<Event> backwardCompletionEvents;
};

}  // namespace ThorImplementation
