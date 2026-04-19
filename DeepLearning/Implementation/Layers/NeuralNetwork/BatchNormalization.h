#pragma once

#include "DeepLearning/Implementation/Initializers/Initializer.h"
#include "DeepLearning/Implementation/Layers/TrainableLayer.h"
#include "DeepLearning/Implementation/Parameter/Parameter.h"

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

    BatchNormalization(bool training,
                       int64_t stampedId = -1,
                       Optional<double> exponentialRunningAverageFactor = Optional<double>::empty(),
                       Optional<double> epsilon = Optional<double>::empty());

    void setTrainingMode(bool training);
    bool getTrainingMode() const { return training; }

    double getExponentialRunningAverageFactor() const { return exponentialRunningAverageFactor; }
    void setExponentialRunningAverageFactor(double exponentialRunningAverageFactor) {
        this->exponentialRunningAverageFactor = exponentialRunningAverageFactor;
    }

    double getEpsilon() const { return epsilon; }
    void setEpsilon(double epsilon) { this->epsilon = epsilon; }

    Tensor getWeights() { return getParameterStorage("weights"); }
    Optional<Tensor> getBiases() { return Optional<Tensor>(getParameterStorage("biases")); }

    Tensor getResultRunningMean() { return resultRunningMean; }
    Tensor getResultRunningVariance() { return resultRunningVariance; }

    void setCurrentExponentialRunningAverageFactor(double value);

    void setInitializer(Tensor target, std::shared_ptr<ThorImplementation::Initializer> initializer) override;
    bool hasInitializer(Tensor target) override;
    Event initializeTensor(Tensor target) override;

    std::string getLayerType() override { return "BatchNormalization"; }
    std::string getType() override { return getLayerType(); }

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
    Optional<Event> computeErrorOut(uint32_t connectionNumber) override;
    void accumulateWeightsGradient(uint32_t connectionNumber, bool clearGradientFirst) override;

    void runForward(Optional<Tensor> inputTensor,
                    Optional<Tensor> outputTensor,
                    Stream stream,
                    unsigned int connectionNumber,
                    Tensor weights,
                    Optional<Tensor> biases);

    Optional<Stream> resolveStateStream(Optional<Stream> stream) const;
    void compileRunningStatInitializer(const std::shared_ptr<Initializer>& initializer, const Tensor& tensor);

   private:
    static const float ALPHA_NO_SCALE;
    static const float BETA_CLEAR;
    static const float BETA_ACCUMULATE;

    unsigned int batchSize = 0;
    unsigned int numChannels = 0;
    unsigned int height = 0;
    unsigned int width = 0;

    bool training;
    double exponentialRunningAverageFactor;
    uint32_t itemsObserved = 0;
    std::vector<double> currentExponentialRunningAverageFactor;
    double epsilon;

    Optional<cudnnTensorDescriptor_t> featureInputDescriptor;
    Optional<cudnnTensorDescriptor_t> featureOutputDescriptor;
    Optional<cudnnTensorDescriptor_t> derivedBnDescriptor;

    Tensor resultRunningMean;
    Tensor resultRunningVariance;
    std::vector<Tensor> resultSaveMean;
    std::vector<Tensor> resultSaveInvVariance;
    std::vector<Optional<Tensor>> scratchErrorOutputs;
};

}  // namespace ThorImplementation
