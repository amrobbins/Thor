#pragma once

#include "DeepLearning/Implementation/Initializers/Initializer.h"
#include "DeepLearning/Implementation/Layers/TrainableLayer.h"
#include "DeepLearning/Implementation/Parameter/Parameter.h"

// FIXME: More work to do on this, compare to old get it exact.

namespace ThorImplementation {

/**
 * Parameter epsilon is used in the batch normalization formula and must be >= 0.00001.
 * Parameter exponentialRunningAverageFactor is used to determine how long to remember past results and how much weight to give the newest
 * result, via equation: runningMean = runningMean*(1-exponentialRunningAverageFactor) + newMean*exponentialRunningAverageFactor
 *
 * This layer will serialize processing of each input by synchronizing all streams with stream[0] when multiple connections are present, for
 * compatibility with cuDNN.
 *
 * Batch Normalization supports 2d or 4d input tensor.
 *
 * Unlike CustomLayer-based layers, batch-normalization backward stays on the cuDNN fused path, so the error-output gradient and parameter
 * gradients are produced together by cudnnBatchNormalizationBackward().
 */
class BatchNormalization : public TrainableLayer {
   public:
    ~BatchNormalization() override;

    BatchNormalization(const TensorPlacement& placement,
                       bool inferenceOnly,
                       Optional<double> exponentialRunningAverageFactor = Optional<double>::empty(),
                       Optional<double> epsilon = Optional<double>::empty(),
                       Optional<TensorDescriptor::DataType> storageDataType = Optional<TensorDescriptor::DataType>::empty(),
                       int64_t stampedId = -1);

    double getExponentialRunningAverageFactor() const { return exponentialRunningAverageFactor; }
    void setExponentialRunningAverageFactor(double exponentialRunningAverageFactor) {
        this->exponentialRunningAverageFactor = exponentialRunningAverageFactor;
    }

    double getEpsilon() const { return epsilon; }
    void setEpsilon(double epsilon) { this->epsilon = epsilon; }

    std::string getLayerType() override { return "BatchNormalization"; }

    Optional<Tensor> createFeatureOutputTensor() override;
    Optional<Tensor> createErrorOutputTensor(bool backPropagateError, uint32_t connectionNumber) override;

    uint64_t flopCountForward() override;
    uint64_t flopCountBackward() override;

    void cleanup() override;

   protected:
    void compileImpl() override;

    std::shared_ptr<ThorImplementation::Initializer> resultRunningMeanInitializer = nullptr;
    std::shared_ptr<ThorImplementation::Initializer> resultRunningVarianceInitializer = nullptr;

   private:
    void computeFeatureOut(uint32_t connectionNumber) override;
    Optional<Event> computeErrorOut(uint32_t connectionNumber, bool clearWeightsGradientFirstIfFused) override;
    void accumulateWeightsGradient(uint32_t connectionNumber, bool clearGradientFirst) override;

    void runForward(Optional<Tensor> inputTensor,
                    Optional<Tensor> outputTensor,
                    Stream stream,
                    unsigned int connectionNumber,
                    Tensor weights,
                    Optional<Tensor> biases);

   protected:
    Tensor weights;
    Tensor biases;
    Tensor resultRunningMean;
    Tensor resultRunningVariance;

   private:
    static const float ALPHA_NO_SCALE;
    static const float BETA_CLEAR;
    static const float BETA_ACCUMULATE;

    unsigned int batchSize = 0;
    unsigned int numChannels = 0;
    unsigned int height = 0;
    unsigned int width = 0;

    double exponentialRunningAverageFactor;
    uint32_t itemsObserved = 0;
    std::vector<double> currentExponentialRunningAverageFactor;
    double epsilon;

    Optional<cudnnTensorDescriptor_t> featureInputDescriptor;
    Optional<cudnnTensorDescriptor_t> featureOutputDescriptor;
    Optional<cudnnTensorDescriptor_t> derivedBnDescriptor;

    std::vector<Tensor> resultSaveMean;
    std::vector<Tensor> resultSaveInvVariance;

    // Since weights gradients and error gradient is a fused operation, then when back prop is pruned
    // we still need some valid chunk of memory to write values in, which we ignore.
    Optional<Tensor> scratchErrorOutput = Optional<Tensor>::empty();
};

}  // namespace ThorImplementation
