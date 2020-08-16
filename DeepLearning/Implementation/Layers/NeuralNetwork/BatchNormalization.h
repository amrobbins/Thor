#pragma once

#include "DeepLearning/Implementation/Layers/TrainableWeightsBiasesLayer.h"

/**
 * Parameter epsilon is used in the batch normalization formula and must be >= 0.00001.
 * Parameter exponentialRunningAverageFactor is used to determine how long to remember past results and how much weight to give the newest
 * result, via equation: runningMean = runningMean*(1-exponentialRunningAverageFactor) + newMean*exponentialRunningAverageFactor
 *
 * This layer will synchronize all streams with stream[0] when multiple connections are present, for compatibility with cuDNN.
 */

class BatchNormalization : public TrainableWeightsBiasesLayer {
   public:
    virtual ~BatchNormalization() {}

    // 2D Batch Normalization - e.g. after fully connected
    BatchNormalization(unsigned int batchSize,
                       unsigned int numFeatures,
                       bool training,
                       const bool inferenceOnly,
                       Optional<float> learningRate,
                       double exponentialRunningAverageFactor = 0.05,
                       double epsilon = 0.0001)
        : TrainableWeightsBiasesLayer(inferenceOnly, true, learningRate),
          batchSize(batchSize),
          numChannels(numFeatures),
          height(1),
          width(1),
          exponentialRunningAverageFactor(exponentialRunningAverageFactor),
          epsilon(epsilon) {
        setTrainingMode(training);
    }

    // 4D Batch Normalization - e.g. after convolution 2d
    BatchNormalization(unsigned int batchSize,
                       unsigned int numChannels,
                       unsigned int height,
                       unsigned int width,
                       bool training,
                       const bool inferenceOnly,
                       Optional<float> learningRate,
                       double exponentialRunningAverageFactor = 0.05,
                       double epsilon = 0.0001)
        : TrainableWeightsBiasesLayer(inferenceOnly, true, learningRate),
          batchSize(batchSize),
          numChannels(numChannels),
          height(height),
          width(width),
          exponentialRunningAverageFactor(exponentialRunningAverageFactor),
          epsilon(epsilon) {
        setTrainingMode(training);
    }

    void setTrainingMode(bool training) {
        assert(running == false);
        assert(inferenceOnly == false);
        this->training = training;
    }
    bool getTrainingMode() { return training; }

    double getExponentialRunningAverageFactor() { return exponentialRunningAverageFactor; }
    void setExponentialRunningAverageFactor(double exponentialRunningAverageFactor) {
        this->exponentialRunningAverageFactor = exponentialRunningAverageFactor;
    }

    double getEpsilon() { return epsilon; }
    void setEpsilon(double epsilon) { this->epsilon = epsilon; }

    virtual void compile() {
        cudnnStatus_t cudnnStatus;

        assert(!featureInputs.empty());
        assert(!featureOutputs.empty());
        assert(featureInputs.size() == featureOutputs.size());

        featureInputDescriptor = cudnnTensorDescriptor_t();
        cudnnStatus = cudnnCreateTensorDescriptor(&featureInputDescriptor.get());
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
        cudnnStatus =
            cudnnSetTensor4dDescriptor(featureInputDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize, numChannels, height, width);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

        featureOutputDescriptor = cudnnTensorDescriptor_t();
        cudnnStatus = cudnnCreateTensorDescriptor(&featureOutputDescriptor.get());
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
        cudnnStatus =
            cudnnSetTensor4dDescriptor(featureOutputDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize, numChannels, height, width);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

        derivedBnDescriptor = cudnnTensorDescriptor_t();
        cudnnStatus = cudnnCreateTensorDescriptor(&derivedBnDescriptor.get());
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
        cudnnStatus = cudnnDeriveBNTensorDescriptor(derivedBnDescriptor,
                                                    featureInputDescriptor,
                                                    height > 1 || width > 1 ? CUDNN_BATCHNORM_SPATIAL : CUDNN_BATCHNORM_PER_ACTIVATION);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

        vector<unsigned long> derivedBnTensorDimensions = {batchSize, numChannels, height, width};

        weights =
            Tensor(featureInputs[0].get().getPlacement(), TensorDescriptor(TensorDescriptor::DataType::FP32, derivedBnTensorDimensions));
        biases =
            Tensor(featureInputs[0].get().getPlacement(), TensorDescriptor(TensorDescriptor::DataType::FP32, derivedBnTensorDimensions));
        weightsGradient =
            Tensor(featureInputs[0].get().getPlacement(), TensorDescriptor(TensorDescriptor::DataType::FP32, derivedBnTensorDimensions));
        biasesGradient =
            Tensor(featureInputs[0].get().getPlacement(), TensorDescriptor(TensorDescriptor::DataType::FP32, derivedBnTensorDimensions));
        resultRunningMean =
            Tensor(featureInputs[0].get().getPlacement(), TensorDescriptor(TensorDescriptor::DataType::FP32, derivedBnTensorDimensions));
        resultRunningVariance =
            Tensor(featureInputs[0].get().getPlacement(), TensorDescriptor(TensorDescriptor::DataType::FP32, derivedBnTensorDimensions));

        for (unsigned int i = 0; i < featureInputs.size(); ++i) {
            resultSaveMean[i] = Tensor(featureInputs[0].get().getPlacement(),
                                       TensorDescriptor(TensorDescriptor::DataType::FP32, derivedBnTensorDimensions));
            resultSaveInvVariance[i] = Tensor(featureInputs[0].get().getPlacement(),
                                              TensorDescriptor(TensorDescriptor::DataType::FP32, derivedBnTensorDimensions));
        }

        Tensor oneInit_h =
            Tensor(TensorPlacement::MemDevices::CPU, TensorDescriptor(TensorDescriptor::DataType::FP32, derivedBnTensorDimensions));
        Tensor zeroInit_h =
            Tensor(TensorPlacement::MemDevices::CPU, TensorDescriptor(TensorDescriptor::DataType::FP32, derivedBnTensorDimensions));
        unsigned long numElements = oneInit_h.getDescriptor().getTotalNumElements();
        float *oneInitMem = (float *)oneInit_h.getMemPtr();
        float *zeroInitMem = (float *)zeroInit_h.getMemPtr();
        for (unsigned long i = 0; i < numElements; ++i) {
            oneInitMem[i] = 1.0f;
            zeroInitMem[i] = 0.0f;
        }
        weights.copyFromAsync(oneInit_h, streams[0]);
        biases.get().copyFromAsync(zeroInit_h, streams[0]);
        resultRunningMean.copyFromAsync(biases.get(), streams[0]);
        resultRunningVariance.copyFromAsync(biases.get(), streams[0]);

        if (streams.size() > 1) {
            Event initializedEvent = streams[0].putEvent();
            for (unsigned int i = 0; i < streams.size(); ++i)
                streams[i].waitEvent(initializedEvent);
        }
    }

    void cleanup() {
        cudnnStatus_t cudnnStatus;

        if (derivedBnDescriptor.isPresent()) {
            cudnnStatus = cudnnDestroyTensorDescriptor(derivedBnDescriptor.get());
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
            derivedBnDescriptor.clear();
        }

        if (featureInputDescriptor.isPresent()) {
            cudnnStatus = cudnnDestroyTensorDescriptor(featureInputDescriptor.get());
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
            featureInputDescriptor.clear();
        }

        if (featureOutputDescriptor.isPresent()) {
            cudnnStatus = cudnnDestroyTensorDescriptor(featureOutputDescriptor.get());
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
            featureOutputDescriptor.clear();
        }
    }

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream, unsigned int connectionNumber) {
        assert(inputTensor.isPresent());
        assert(outputTensor.isPresent());

        cudnnStatus_t cudnnStatus;

        if (training) {
            cudnnStatus =
                cudnnBatchNormalizationForwardTraining(streams[0].getCudnnHandle(),
                                                       height > 1 || width > 1 ? CUDNN_BATCHNORM_SPATIAL : CUDNN_BATCHNORM_PER_ACTIVATION,
                                                       &ALPHA_NO_SCALE,
                                                       &BETA_CLEAR,
                                                       featureInputDescriptor,
                                                       inputTensor.get().getMemPtr(),
                                                       featureOutputDescriptor,
                                                       outputTensor.get().getMemPtr(),
                                                       derivedBnDescriptor,
                                                       weights.getMemPtr(),
                                                       biases.get().getMemPtr(),
                                                       exponentialRunningAverageFactor,
                                                       resultRunningMean.getMemPtr(),
                                                       resultRunningVariance.getMemPtr(),
                                                       epsilon,
                                                       resultSaveMean[connectionNumber].getMemPtr(),
                                                       resultSaveInvVariance[connectionNumber].getMemPtr());
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
        } else {
            cudnnStatus =
                cudnnBatchNormalizationForwardInference(streams[0].getCudnnHandle(),
                                                        height > 1 || width > 1 ? CUDNN_BATCHNORM_SPATIAL : CUDNN_BATCHNORM_PER_ACTIVATION,
                                                        &ALPHA_NO_SCALE,
                                                        &BETA_CLEAR,
                                                        featureInputDescriptor,
                                                        inputTensor.get().getMemPtr(),
                                                        featureOutputDescriptor,
                                                        outputTensor.get().getMemPtr(),
                                                        derivedBnDescriptor,
                                                        weights.getMemPtr(),
                                                        biases.get().getMemPtr(),
                                                        resultRunningMean.getMemPtr(),
                                                        resultRunningVariance.getMemPtr(),
                                                        epsilon);
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
        }

        // Since running average is a single memory, multiple connections to this layer must serialize with each other.
        if (connectionNumber != 0)
            streams[connectionNumber].waitEvent(streams[0].putEvent());
    }

    virtual void backProp(Optional<Tensor> dataIn,
                          Optional<Tensor> errorIn,
                          Optional<Tensor> errorOut,
                          Stream gradientStream,
                          Optional<Stream> dataStream,
                          unsigned int connectionNumber,
                          bool accumulateGradient) {
        if (errorOut.isEmpty())
            return;
        assert(errorIn.isPresent());

        cudnnStatus_t cudnnStatus;
        cudnnStatus = cudnnBatchNormalizationBackward(streams[0].getCudnnHandle(),
                                                      height > 1 || width > 1 ? CUDNN_BATCHNORM_SPATIAL : CUDNN_BATCHNORM_PER_ACTIVATION,
                                                      &ALPHA_NO_SCALE,
                                                      &BETA_CLEAR,
                                                      &ALPHA_NO_SCALE,
                                                      accumulateGradient ? &BETA_ACCUMULATE : &BETA_CLEAR,
                                                      featureInputDescriptor,
                                                      dataIn.get().getMemPtr(),
                                                      featureOutputDescriptor,
                                                      errorIn.get().getMemPtr(),
                                                      featureInputDescriptor,
                                                      errorOut.get().getMemPtr(),
                                                      derivedBnDescriptor,
                                                      weights.getMemPtr(),
                                                      weightsGradient.get().getMemPtr(),
                                                      biasesGradient.get().getMemPtr(),
                                                      epsilon,
                                                      resultSaveMean[connectionNumber].getMemPtr(),
                                                      resultSaveInvVariance[connectionNumber].getMemPtr());
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

        // Since weights are a single memory, multiple connections to this layer must serialize with each other.
        // This is the normal case for gradients, so that accumulation may be used, however it differs in that the separate
        // gradientUpdateStream is not used. This is because cuDNN is not separating back propagation of errors and updating gradients.
        if (connectionNumber != 0)
            streams[connectionNumber].waitEvent(streams[0].putEvent());
    }

   private:
    static const float ALPHA_NO_SCALE;
    static const float BETA_CLEAR;
    static const float BETA_ACCUMULATE;

    const unsigned int batchSize;
    const unsigned int numChannels;
    const unsigned int height;
    const unsigned int width;

    bool training;
    double exponentialRunningAverageFactor;
    double epsilon;

    Optional<cudnnTensorDescriptor_t> featureInputDescriptor;
    Optional<cudnnTensorDescriptor_t> featureOutputDescriptor;
    Optional<cudnnTensorDescriptor_t> derivedBnDescriptor;

    Tensor resultRunningMean;
    Tensor resultRunningVariance;
    vector<Tensor> resultSaveMean;
    vector<Tensor> resultSaveInvVariance;
};
