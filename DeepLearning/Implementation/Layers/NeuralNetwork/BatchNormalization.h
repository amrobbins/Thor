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
                       int64_t stampedId = -1,
                       Optional<double> exponentialRunningAverageFactor = Optional<double>::empty(),
                       Optional<double> epsilon = Optional<double>::empty())
        : TrainableWeightsBiasesLayer(true, stampedId),
          exponentialRunningAverageFactor(exponentialRunningAverageFactor.isPresent() ? exponentialRunningAverageFactor.get() : 0.05),
          epsilon(epsilon.isPresent() ? epsilon.get() : 0.0001) {
        setTrainingMode(training);
    }

    BatchNormalization(SharedWeightsPackage sharedWeightsPackage,
                       bool training,
                       int64_t stampedId = -1,
                       Optional<double> exponentialRunningAverageFactor = Optional<double>::empty(),
                       Optional<double> epsilon = Optional<double>::empty())
        : TrainableWeightsBiasesLayer(sharedWeightsPackage, stampedId),
          exponentialRunningAverageFactor(exponentialRunningAverageFactor.isPresent() ? exponentialRunningAverageFactor.get() : 0.05),
          epsilon(epsilon.isPresent() ? epsilon.get() : 0.0001) {
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
            uint64_t numChannels = featureInputs.front().get().getDescriptor().getDimensions()[1];
            TensorPlacement placement = featureInputs[0].get().getPlacement();
            TensorDescriptor descriptor(TensorDescriptor::DataType::FP32, {numChannels});
            weights = Tensor(placement, descriptor);
            biases = weights.clone();
            resultRunningMean = weights.clone();
            resultRunningVariance = weights.clone();
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

        std::vector<uint64_t> inputDimensions = featureInputs.front().get().getDescriptor().getDimensions();
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
        cudnnDataType_t cudnnDataType;
        if (featureInputs[0].get().getDataType() == TensorDescriptor::DataType::FP16)
            cudnnDataType = CUDNN_DATA_HALF;
        else if (featureInputs[0].get().getDataType() == TensorDescriptor::DataType::FP32)
            cudnnDataType = CUDNN_DATA_FLOAT;
        else
            assert(false);
        cudnnStatus =
            cudnnSetTensor4dDescriptor(featureInputDescriptor, CUDNN_TENSOR_NCHW, cudnnDataType, batchSize, numChannels, height, width);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

        featureOutputDescriptor = cudnnTensorDescriptor_t();
        cudnnStatus = cudnnCreateTensorDescriptor(&featureOutputDescriptor.get());
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
        cudnnStatus =
            cudnnSetTensor4dDescriptor(featureOutputDescriptor, CUDNN_TENSOR_NCHW, cudnnDataType, batchSize, numChannels, height, width);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

        derivedBnDescriptor = cudnnTensorDescriptor_t();
        cudnnStatus = cudnnCreateTensorDescriptor(&derivedBnDescriptor.get());
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
        cudnnStatus = cudnnDeriveBNTensorDescriptor(derivedBnDescriptor,
                                                    featureInputDescriptor,
                                                    inputDimensions.size() == 2 ? CUDNN_BATCHNORM_PER_ACTIVATION : CUDNN_BATCHNORM_SPATIAL);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

        // From API Reference:
        // The resulting dimensions will be:
        // 1xCx1x1 for 4D and 1xCx1x1x1 for 5D for BATCHNORM_MODE_SPATIAL
        // 1xCxHxW for 4D and 1xCxDxHxW for 5D for BATCHNORM_MODE_PER_ACTIVATION mode
        // ---------------------------------------------------------------------------
        // And support by this layer is spatial for 4D tensors and per activation for 2D tensors (aka 4D tensors of 1xCx1x1),
        // so in either case the tensor dimensions is 1xCx1x1 = C
        std::vector<unsigned long> derivedBnTensorDimensions = {numChannels};

        for (unsigned int i = 0; i < featureInputs.size(); ++i) {
            // This needs to be done here, in compile() because at this point
            // the number of feature inputs are known.
            resultSaveMean.push_back(weights.clone());
            resultSaveInvVariance.push_back(weights.clone());
        }

        assert(exponentialRunningAverageFactor > 0.0);
        assert(exponentialRunningAverageFactor <= 1.0);
        itemsObserved = 0;

        if (!isInferenceOnly())
            optimizer.get()->compile();
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

    virtual void infer(Optional<Tensor> inputTensor,
                       Optional<Tensor> outputTensor,
                       Stream stream,
                       unsigned int connectionNumber,
                       Tensor weights,
                       Optional<Tensor> biases) {
        assert(inputTensor.isPresent());
        assert(outputTensor.isPresent());

        cudnnStatus_t cudnnStatus;

        if (currentExponentialRunningAverageFactor[connectionNumber] != exponentialRunningAverageFactor) {
            ++itemsObserved;
            currentExponentialRunningAverageFactor[connectionNumber] = 1.0 / itemsObserved;
            if (currentExponentialRunningAverageFactor[connectionNumber] < exponentialRunningAverageFactor)
                currentExponentialRunningAverageFactor[connectionNumber] = exponentialRunningAverageFactor;
        }

        if (training) {
            cudnnStatus =
                cudnnBatchNormalizationForwardTraining(stream.getCudnnHandle(),
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
                                                       currentExponentialRunningAverageFactor[connectionNumber],
                                                       resultRunningMean.getMemPtr(),
                                                       resultRunningVariance.getMemPtr(),
                                                       epsilon,
                                                       resultSaveMean[connectionNumber].getMemPtr(),
                                                       resultSaveInvVariance[connectionNumber].getMemPtr());

            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
        } else {
            cudnnStatus =
                cudnnBatchNormalizationForwardInference(stream.getCudnnHandle(),
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
    }

    virtual void backProp(Optional<Tensor> dataIn,
                          Optional<Tensor> errorIn,
                          Optional<Tensor> errorOut,
                          Stream dataStream,
                          unsigned int connectionNumber,
                          bool accumulateGradient) {
        if (errorOut.isEmpty())
            return;
        assert(errorIn.isPresent());

        if (optimizer.isEmpty()) {
            throw std::runtime_error("BatchNormalization: compiled but optimizer is not present, and not in inference only mode.");
        }
        Tensor weightsGradient = optimizer.get()->getWeightsGradient();
        Optional<Tensor> biasesGradient = optimizer.get()->getBiasesGradient();
        assert(biasesGradient.isPresent());

        cudnnStatus_t cudnnStatus;
        cudnnStatus = cudnnBatchNormalizationBackward(dataStream.getCudnnHandle(),
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
                                                      weightsGradient.getMemPtr(),
                                                      biasesGradient.get().getMemPtr(),
                                                      epsilon,
                                                      resultSaveMean[connectionNumber].getMemPtr(),
                                                      resultSaveInvVariance[connectionNumber].getMemPtr());
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

        if (!isInferenceOnly()) {
            assert(optimizer.isPresent());

            // Sync gradientUpdateStream with data stream, so now gradient is up to date at end of gradientUpdateStream
            assert(optimizer.isPresent());
            Stream gradientUpdateStream = optimizer.get()->getGradientUpdateStream();
            gradientUpdateStream.waitEvent(dataStream.putEvent());

            // backward() syncs gradient stream with data stream prior to calling this to ensure error in is ready at end of gradient stream
            optimizer.get()->computeWeightsUpdate(dataIn, errorIn, accumulateGradient);

            // weights update cannot be applied to weights until errorOut has been computed since weights are part of that computation
            // so to enforce this gradientStream says that gradient is not ready to be applied until both errorOut and gradient are computed
            // But.. Since both errorOut and gradient are computed in the data stream in this case, they are always both ready at the
            // beginning of gradientUpdateStream, since that was just synchronized with data stream above.
            // gradientUpdateStream.waitEvent(dataStream.putEvent());
            // Now at the end of gradientStream errorOut and gradients are ready from the updates for this connection.

            // Upon processing the last connection, schedule the update to the weights memory.
            if (stillWaitingForErrorInputTensors.empty()) {
                optimizer.get()->updateWeights(weights, biases, batchSize);
            }

            // weights will be updated at the current end of the gradientStream
            // so Forward() must wait until gradientStream is finished.
            // This is accomplished in TrainableWeightsBiasesLayer::forward().
        }
    }

    // Compute the weights gradient for the specified connection number, accumulate as necessary.
    // This computation runs on optimizer.gradientUpdateStream.
    // Note: gradient is actually computed during backProp() since it is a single cudnn function to compute all gradients.
    virtual void computeWeightsGradient(Optional<Tensor> weightsGradient,
                                        Optional<Tensor> biasesGradient,
                                        Optional<Tensor> featureIn,
                                        Optional<Tensor> errorIn,
                                        Stream gradientUpdateStream,
                                        bool accumulateGradient) {
        // Ensure all memory properly allocated
        assert(weightsGradient.isPresent());
        assert(weightsGradient.get().getDescriptor() == weights.getDescriptor());
        assert(weightsGradient.get().getPlacement() == weights.getPlacement());
        assert(weightsGradient.get().getMemPtr() != weights.getMemPtr());
        if (hasBias) {
            assert(biasesGradient.isPresent());
            assert(biases.isPresent());
            assert(biasesGradient.get().getDescriptor() == biasesGradient.get().getDescriptor());
            assert(biasesGradient.get().getMemPtr() != biases.get().getMemPtr());
            assert(biasesGradient.get().getPlacement() == biases.get().getPlacement());
        } else {
            assert(biasesGradient.isEmpty());
        }

        if (errorIn.isEmpty())
            return;
        assert(featureIn.isPresent());

        // This was already placed into optimizer's gradient buffers, so this is a no-op
    }

    Tensor getResultRunningMean() { return resultRunningMean; }
    Tensor getResultRunningVariance() { return resultRunningVariance; }

    void dumpResultRunningMeanToFile(std::string filename, Optional<Stream> stream = Optional<Stream>::empty()) {
        if (stream.isEmpty())
            stream = optimizer.get()->getGradientUpdateStream();
        if (resultRunningMean.getAttachedFilename() != filename)
            resultRunningMean.attachFile(filename, 0, Tensor::FileAccess::READ_WRITE, true);
        resultRunningMean.dumpToFile(stream);
    }

    void loadResultRunningMeanFromFile(std::string filename, Optional<Stream> stream = Optional<Stream>::empty()) {
        if (stream.isEmpty()) {
            assert(optimizer.isPresent());
            stream = optimizer.get()->getGradientUpdateStream();
        }
        if (resultRunningMean.getAttachedFilename() != filename)
            resultRunningMean.attachFile(filename, 0, Tensor::FileAccess::READ_WRITE, false);
        resultRunningMean.loadFromFile(stream);
    }

    void dumpResultRunningVarianceToFile(std::string filename, Optional<Stream> stream = Optional<Stream>::empty()) {
        if (stream.isEmpty())
            stream = optimizer.get()->getGradientUpdateStream();
        if (resultRunningVariance.getAttachedFilename() != filename)
            resultRunningVariance.attachFile(filename, 0, Tensor::FileAccess::READ_WRITE, true);
        resultRunningVariance.dumpToFile(stream);
    }

    void loadResultRunningVarianceFromFile(std::string filename, Optional<Stream> stream = Optional<Stream>::empty()) {
        if (stream.isEmpty()) {
            assert(optimizer.isPresent());
            stream = optimizer.get()->getGradientUpdateStream();
        }
        if (resultRunningVariance.getAttachedFilename() != filename)
            resultRunningVariance.attachFile(filename, 0, Tensor::FileAccess::READ_WRITE, false);
        resultRunningVariance.loadFromFile(stream);
    }

    void setCurrentExponentialRunningAverageFactor(double value) {
        currentExponentialRunningAverageFactor = std::vector<double>(featureInputs.size(), value);
    }

    virtual std::string getType() { return "BatchNormalization"; }

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
    std::vector<double> currentExponentialRunningAverageFactor;
    double epsilon;

    Optional<cudnnTensorDescriptor_t> featureInputDescriptor;
    Optional<cudnnTensorDescriptor_t> featureOutputDescriptor;
    Optional<cudnnTensorDescriptor_t> derivedBnDescriptor;

    Tensor resultRunningMean;
    Tensor resultRunningVariance;
    std::vector<Tensor> resultSaveMean;
    std::vector<Tensor> resultSaveInvVariance;
};

}  // namespace ThorImplementation
