#pragma once

#include "DeepLearning/Implementation/Layers/TrainableWeightsBiasesLayer.h"

namespace ThorImplementation {

/**
 * Parameter epsilon is used in the batch normalization formula and must be >= 0.00001.
 * Parameter exponentialRunningAverageFactor is used to determine how long to remember past results and how much weight to give the newest
 * result, via equation: runningMean = runningMean*(1-exponentialRunningAverageFactor) + newMean*exponentialRunningAverageFactor
 *
 * This layer will serialize processing of each input by synchronizing all streams with stream[0] when multiple connections are present, for
 * compatibility with cuDNN.
 *
 * Batch Normalization supports 2d or 4d input tensor
 */

class BatchNormalization : public TrainableWeightsBiasesLayer {
   public:
    virtual ~BatchNormalization() {}

    BatchNormalization(bool training,
                       Optional<double> exponentialRunningAverageFactor = Optional<double>::empty(),
                       Optional<double> epsilon = Optional<double>::empty())
        : TrainableWeightsBiasesLayer(true),
          exponentialRunningAverageFactor(exponentialRunningAverageFactor.isPresent() ? exponentialRunningAverageFactor.get() : 0.05),
          epsilon(epsilon.isPresent() ? epsilon.get() : 0.0001) {
        learningRate = 0.001;
        setTrainingMode(training);
    }

    BatchNormalization(SharedWeightsPackage sharedWeightsPackage,
                       bool training,
                       Optional<double> exponentialRunningAverageFactor = Optional<double>::empty(),
                       Optional<double> epsilon = Optional<double>::empty())
        : TrainableWeightsBiasesLayer(sharedWeightsPackage),
          exponentialRunningAverageFactor(exponentialRunningAverageFactor.isPresent() ? exponentialRunningAverageFactor.get() : 0.05),
          epsilon(epsilon.isPresent() ? epsilon.get() : 0.0001) {
        learningRate = 0.001;
        setTrainingMode(training);

        assert(sharedWeightsPackage.otherSharedMem.size() == 2);
        resultRunningMean = sharedWeightsPackage.otherSharedMem[0];
        resultRunningVariance = sharedWeightsPackage.otherSharedMem[1];
    }

    virtual SharedWeightsPackage getSharedWeightsPackage() {
        SharedWeightsPackage sharedWeightsPackage = TrainableWeightsBiasesLayer::getSharedWeightsPackage();

        sharedWeightsPackage.otherSharedMem.push_back(resultRunningMean);
        sharedWeightsPackage.otherSharedMem.push_back(resultRunningVariance);

        return sharedWeightsPackage;
    }

    virtual void createWeightsIfNecessary() {
        // Cudnn forces the use of FP32 for the weights currently
        // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnDeriveBNTensorDescriptor
        if (!usingSharedWeights && !weights.isInitialized()) {
            vector<unsigned long> derivedBnTensorDimensions = {
                featureInputs.front().get().getDescriptor().getDimensions()[1]};  // numChannels

            weights = Tensor(featureInputs[0].get().getPlacement(),
                             TensorDescriptor(TensorDescriptor::DataType::FP32, derivedBnTensorDimensions));
            biases = Tensor(featureInputs[0].get().getPlacement(),
                            TensorDescriptor(TensorDescriptor::DataType::FP32, derivedBnTensorDimensions));
            weightsGradient = Tensor(featureInputs[0].get().getPlacement(),
                                     TensorDescriptor(TensorDescriptor::DataType::FP32, derivedBnTensorDimensions));
            biasesGradient = Tensor(featureInputs[0].get().getPlacement(),
                                    TensorDescriptor(TensorDescriptor::DataType::FP32, derivedBnTensorDimensions));
            resultRunningMean = Tensor(featureInputs[0].get().getPlacement(),
                                       TensorDescriptor(TensorDescriptor::DataType::FP32, derivedBnTensorDimensions));
            resultRunningVariance = Tensor(featureInputs[0].get().getPlacement(),
                                           TensorDescriptor(TensorDescriptor::DataType::FP32, derivedBnTensorDimensions));
        }
    }

    void setTrainingMode(bool training) {
        assert(running == false);
        assert(isInferenceOnly() == false);
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

        vector<uint64_t> inputDimensions = featureInputs.front().get().getDescriptor().getDimensions();
        assert(inputDimensions.size() == 2 || inputDimensions.size() == 4);
        batchSize = inputDimensions[0];
        numChannels = inputDimensions[1];
        if (inputDimensions.size() == 2) {
            height = 1;
            width = 1;
        } else {
            height = inputDimensions[2];
            width = inputDimensions[3];
        }

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

        vector<unsigned long> derivedBnTensorDimensions = {numChannels};

        for (unsigned int i = 0; i < featureInputs.size(); ++i) {
            resultSaveMean.push_back(Tensor(featureInputs[0].get().getPlacement(),
                                            TensorDescriptor(TensorDescriptor::DataType::FP32, derivedBnTensorDimensions)));
            resultSaveInvVariance.push_back(Tensor(featureInputs[0].get().getPlacement(),
                                                   TensorDescriptor(TensorDescriptor::DataType::FP32, derivedBnTensorDimensions)));
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

        // Start with the actual average until there are enough elements observed so that the running average is a larger divisor than the
        // actual.
        assert(exponentialRunningAverageFactor > 0.0);
        assert(exponentialRunningAverageFactor <= 1.0);
        currentExponentialRunningAverageFactor = 1.0;
        itemsObserved = 0;
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

        if (currentExponentialRunningAverageFactor != exponentialRunningAverageFactor) {
            ++itemsObserved;
            currentExponentialRunningAverageFactor = 1.0 / itemsObserved;
            if (currentExponentialRunningAverageFactor < exponentialRunningAverageFactor)
                currentExponentialRunningAverageFactor = exponentialRunningAverageFactor;
        }

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
                                                       currentExponentialRunningAverageFactor,
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

    unsigned int batchSize;
    unsigned int numChannels;
    unsigned int height;
    unsigned int width;

    bool training;
    double exponentialRunningAverageFactor;
    uint32_t itemsObserved;
    double currentExponentialRunningAverageFactor;
    double epsilon;

    Optional<cudnnTensorDescriptor_t> featureInputDescriptor;
    Optional<cudnnTensorDescriptor_t> featureOutputDescriptor;
    Optional<cudnnTensorDescriptor_t> derivedBnDescriptor;

    Tensor resultRunningMean;
    Tensor resultRunningVariance;
    vector<Tensor> resultSaveMean;
    vector<Tensor> resultSaveInvVariance;
};

}  // namespace ThorImplementation
