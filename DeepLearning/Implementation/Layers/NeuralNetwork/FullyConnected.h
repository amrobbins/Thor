#pragma once

#include "DeepLearning/Implementation/Layers/TrainableWeightsBiasesLayer.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.h"

#include <thread>
#include "omp.h"

// FIXME: inference only support

class FullyConnected : public TrainableWeightsBiasesLayer {
   public:
    virtual ~FullyConnected() {}

    FullyConnected(const uint32_t numInputFeatures,
                   const uint32_t numOutputFeatures,
                   const int batchSize,
                   const bool inferenceOnly,
                   const bool hasBias,
                   Optional<float> learningRate)
        : TrainableWeightsBiasesLayer(inferenceOnly, hasBias, learningRate),
          numInputFeatures(numInputFeatures),
          numOutputFeatures(numOutputFeatures),
          batchSize(batchSize) {}

    virtual Optional<Tensor> createFeatureOutputTensor() {
        assert(!featureInputs.empty());
        assert(featureInputs.back().isPresent());

        return Tensor(featureInputs.back().get().getPlacement(),
                      TensorDescriptor(TensorDescriptor::DataType::FP16, batchSize, numOutputFeatures));
    }

    virtual void compile() {
        int gpuNum;
        assert(!featureInputs.empty());
        assert(featureInputs[0].isPresent());
        assert(featureInputs[0].get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        assert(!streams.empty());
        gpuNum = featureInputs[0].get().getPlacement().getDeviceNum();
        ScopedGpu scopedGpu(gpuNum);

        CublasMatrixMultiply::instance().chooseOptimalKernel(
            gpuNum, batchSize, numInputFeatures, numInputFeatures, numOutputFeatures, false, false, TensorDescriptor::DataType::FP16);

        // Allocate 1 weights and 1 weights gradient, if there is more than one connection, will accumulate to weights gradient using
        // BETA=1.0. Data format is NCHW so filter format is KCRS where K = num output channels, C = num input channels, R = filter rows, S
        // = filter columns
        vector<unsigned long> weightsDimensions;
        weightsDimensions.push_back(numInputFeatures);
        weightsDimensions.push_back(numOutputFeatures);
        TensorDescriptor weightsDescriptor = TensorDescriptor(TensorDescriptor::DataType::FP16, weightsDimensions);
        weights = Tensor(featureInputs.front().get().getPlacement(), weightsDescriptor);
        if (!inferenceOnly)
            weightsGradient = weights.clone();
        if (hasBias) {
            biases =
                Tensor(featureInputs.front().get().getPlacement(), TensorDescriptor(TensorDescriptor::DataType::FP16, numOutputFeatures));
            if (!inferenceOnly)
                biasesGradient = biases.get().clone(TensorDescriptor::DataType::FP16);
        }

        // Allocate 1 workspace of each type, since it is possible that all three types of kernels may be running at the same time.
        // If there is more than one connection, the kernels of a given type will run sequentially so that the workspace will be available
        bool kernelWillRunOnGpu;
        uint64_t workspaceForwardSizeInBytes = CublasMatrixMultiply::instance().getWorkspaceSizeInBytes(gpuNum,
                                                                                                        batchSize,
                                                                                                        numInputFeatures,
                                                                                                        numInputFeatures,
                                                                                                        numOutputFeatures,
                                                                                                        false,
                                                                                                        false,
                                                                                                        TensorDescriptor::DataType::FP16,
                                                                                                        kernelWillRunOnGpu);
        assert(kernelWillRunOnGpu);
        if (workspaceForwardSizeInBytes > 0) {
            vector<unsigned long> workspaceDimensions;
            workspaceDimensions.push_back(workspaceForwardSizeInBytes);
            TensorDescriptor workspaceDescriptor(TensorDescriptor::DataType::UINT8, workspaceDimensions);
            workspaceForward = Tensor(featureInputs.front().get().getPlacement(), workspaceDescriptor);
        }

        if (!isBackPropStub() && !inferenceOnly) {
            CublasMatrixMultiply::instance().chooseOptimalKernel(
                gpuNum, batchSize, numOutputFeatures, numInputFeatures, numOutputFeatures, false, true, TensorDescriptor::DataType::FP16);

            uint64_t workspaceBackwardDataSizeInBytes =
                CublasMatrixMultiply::instance().getWorkspaceSizeInBytes(gpuNum,
                                                                         batchSize,
                                                                         numOutputFeatures,
                                                                         numInputFeatures,
                                                                         numOutputFeatures,
                                                                         false,
                                                                         true,
                                                                         TensorDescriptor::DataType::FP16,
                                                                         kernelWillRunOnGpu);
            assert(kernelWillRunOnGpu);

            if (workspaceBackwardDataSizeInBytes > 0)
                workspaceBackwardData = Tensor(featureInputs.front().get().getPlacement(),
                                               TensorDescriptor(TensorDescriptor::DataType::UINT8, {workspaceBackwardDataSizeInBytes}));
        }

        if (!inferenceOnly) {
            CublasMatrixMultiply::instance().chooseOptimalKernel(
                gpuNum, batchSize, numInputFeatures, batchSize, numOutputFeatures, true, false, TensorDescriptor::DataType::FP16);

            uint64_t workspaceBackwardWeightsSizeInBytes =
                CublasMatrixMultiply::instance().getWorkspaceSizeInBytes(gpuNum,
                                                                         batchSize,
                                                                         numInputFeatures,
                                                                         batchSize,
                                                                         numOutputFeatures,
                                                                         true,
                                                                         false,
                                                                         TensorDescriptor::DataType::FP16,
                                                                         kernelWillRunOnGpu);
            assert(kernelWillRunOnGpu);

            if (workspaceBackwardWeightsSizeInBytes > 0)
                workspaceBackwardWeights =
                    Tensor(featureInputs.front().get().getPlacement(),
                           TensorDescriptor(TensorDescriptor::DataType::UINT8, {workspaceBackwardWeightsSizeInBytes}));
        }

        if (hasBias) {
            cudnnStatus_t cudnnStatus;

            biasesDescriptor = cudnnTensorDescriptor_t();
            cudnnStatus = cudnnCreateTensorDescriptor(&biasesDescriptor.get());
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
            cudnnStatus = cudnnSetTensor4dDescriptor(biasesDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, 1, numOutputFeatures, 1, 1);
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

            featureOutputDescriptor = cudnnTensorDescriptor_t();
            cudnnStatus = cudnnCreateTensorDescriptor(&featureOutputDescriptor.get());
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
            cudnnStatus =
                cudnnSetTensor4dDescriptor(featureOutputDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize, numOutputFeatures, 1, 1);
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

            if (!inferenceOnly) {
                reduceDescriptor = cudnnReduceTensorDescriptor_t();
                cudnnStatus = cudnnCreateReduceTensorDescriptor(&reduceDescriptor.get());
                assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
                cudnnStatus = cudnnSetReduceTensorDescriptor(reduceDescriptor,
                                                             CUDNN_REDUCE_TENSOR_ADD,
                                                             CUDNN_DATA_FLOAT,
                                                             CUDNN_NOT_PROPAGATE_NAN,
                                                             CUDNN_REDUCE_TENSOR_NO_INDICES,
                                                             CUDNN_32BIT_INDICES);
                assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

                size_t reduceWorkspaceSizeInBytes;
                cudnnStatus = cudnnGetReductionWorkspaceSize(gradientUpdateStream.get().getCudnnHandle(),
                                                             reduceDescriptor,
                                                             featureOutputDescriptor,
                                                             biasesDescriptor,
                                                             &reduceWorkspaceSizeInBytes);
                assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
                if (reduceWorkspaceSizeInBytes > 0)
                    workspaceBackwardBias = Tensor(featureInputs.front().get().getPlacement(),
                                                   TensorDescriptor(TensorDescriptor::DataType::UINT8, {reduceWorkspaceSizeInBytes}));
            }
        }
    }

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream, unsigned int connectionNumber) {
        assert(inputTensor.isPresent());
        assert(outputTensor.isPresent());
        assert(inputTensor.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

        if (workspaceForward.isPresent()) {
            CublasMatrixMultiply::instance().multiply(inputTensor,
                                                      weights,
                                                      outputTensor,
                                                      workspaceForward,
                                                      batchSize,
                                                      numInputFeatures,
                                                      numInputFeatures,
                                                      numOutputFeatures,
                                                      false,
                                                      false,
                                                      false,
                                                      TensorDescriptor::DataType::FP16,
                                                      stream);
        } else {
            CublasMatrixMultiply::instance().multiply(inputTensor,
                                                      weights,
                                                      outputTensor,
                                                      batchSize,
                                                      numInputFeatures,
                                                      numInputFeatures,
                                                      numOutputFeatures,
                                                      false,
                                                      false,
                                                      false,
                                                      TensorDescriptor::DataType::FP16,
                                                      stream);
        }

        if (hasBias) {
            assert(biases.isPresent());
            assert(biasesDescriptor.isPresent());
            assert(featureOutputDescriptor.isPresent());

            cudnnStatus_t cudnnStatus;

            cudnnStatus = cudnnAddTensor(stream.getCudnnHandle(),
                                         &ALPHA_NO_SCALE,
                                         biasesDescriptor,
                                         biases.get().getMemPtr(),
                                         &BETA_ACCUMULATE,
                                         featureOutputDescriptor,
                                         outputTensor.get().getMemPtr());
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
        }
    }

    virtual void backProp(Optional<Tensor> dataIn,
                          Optional<Tensor> errorIn,
                          Optional<Tensor> errorOut,
                          Stream gradientStream,
                          Optional<Stream> dataStream,
                          unsigned int connectionNumber,
                          bool accumulateGradient) {
        assert(errorIn.isPresent());
        assert(errorIn.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

        if (errorOut.isPresent()) {
            assert(dataStream.isPresent());

            if (workspaceBackwardData.isPresent()) {
                CublasMatrixMultiply::instance().multiply(errorIn,
                                                          weights,
                                                          errorOut,
                                                          workspaceBackwardData,
                                                          batchSize,
                                                          numOutputFeatures,
                                                          numInputFeatures,
                                                          numOutputFeatures,
                                                          false,
                                                          true,
                                                          false,
                                                          TensorDescriptor::DataType::FP16,
                                                          dataStream);
            } else {
                CublasMatrixMultiply::instance().multiply(errorIn,
                                                          weights,
                                                          errorOut,
                                                          batchSize,
                                                          numOutputFeatures,
                                                          numInputFeatures,
                                                          numOutputFeatures,
                                                          false,
                                                          true,
                                                          false,
                                                          TensorDescriptor::DataType::FP16,
                                                          dataStream);
            }
        }

        if (workspaceBackwardWeights.isPresent()) {
            CublasMatrixMultiply::instance().multiply(dataIn,
                                                      errorIn,
                                                      weightsGradient,
                                                      workspaceBackwardWeights,
                                                      batchSize,
                                                      numInputFeatures,
                                                      batchSize,
                                                      numOutputFeatures,
                                                      true,
                                                      false,
                                                      accumulateGradient,
                                                      TensorDescriptor::DataType::FP16,
                                                      gradientStream);
        } else {
            CublasMatrixMultiply::instance().multiply(dataIn,
                                                      errorIn,
                                                      weightsGradient,
                                                      batchSize,
                                                      numInputFeatures,
                                                      batchSize,
                                                      numOutputFeatures,
                                                      true,
                                                      false,
                                                      accumulateGradient,
                                                      TensorDescriptor::DataType::FP16,
                                                      gradientStream);
        }

        if (hasBias) {
            /* This method worked for the first 1 or 2 invocations in a program, but returned junk after that:
                        cudnnStatus_t cudnnStatus =
                        cudnnReduceTensor(gradientStream.getCudnnHandle(),
                                          reduceDescriptor,
                                          nullptr,
                                          0,
                                          workspaceBackwardBias.isPresent() ? workspaceBackwardBias.get().getMemPtr() : nullptr,
                                          workspaceBackwardBias.isPresent() ?
               workspaceBackwardBias.get().getDescriptor().getArraySizeInBytes() : 0, &ALPHA_NO_SCALE, featureOutputDescriptor,
                                          errorIn.get().getMemPtr(),
                                          accumulateGradient ? &BETA_ACCUMULATE : &BETA_CLEAR,
                                          biasesDescriptor,
                                          biasesGradient.get().getMemPtr());
                        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
            */
            launchSumBatch((half *)errorIn.get().getMemPtr(),
                           (half *)biasesGradient.get().getMemPtr(),
                           numOutputFeatures,
                           batchSize,
                           accumulateGradient,
                           gradientStream);
        }
    }

    void cleanup() {
        cudnnStatus_t cudnnStatus;

        if (reduceDescriptor.isPresent()) {
            cudnnStatus = cudnnDestroyReduceTensorDescriptor(reduceDescriptor.get());
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
            reduceDescriptor.clear();
        }

        if (biasesDescriptor.isPresent()) {
            cudnnStatus = cudnnDestroyTensorDescriptor(biasesDescriptor.get());
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
            biasesDescriptor.clear();
        }

        if (featureOutputDescriptor.isPresent()) {
            cudnnStatus = cudnnDestroyTensorDescriptor(featureOutputDescriptor.get());
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
            featureOutputDescriptor.clear();
        }
    }

    class Initializer {
       public:
        virtual ~Initializer() {}

        virtual void initializeWeights(FullyConnected *fullyConnected) const = 0;
        virtual void initializeBiases(FullyConnected *fullyConnected) const = 0;
    };

    class UniformRandomInitializer : public Initializer {
       public:
        UniformRandomInitializer(float maxValue, float minValue) : maxValue(maxValue), minValue(minValue) {
            assert(isfinite(maxValue));
            assert(isfinite(minValue));
            assert(maxValue >= minValue);
        }

        virtual void initializeWeights(FullyConnected *fullyConnected) const { initialize(fullyConnected, fullyConnected->weights); }

        virtual void initializeBiases(FullyConnected *fullyConnected) const {
            assert(fullyConnected->hasBias);
            initialize(fullyConnected, fullyConnected->biases);
        }

       private:
        virtual void initialize(FullyConnected *fullyConnected, Tensor weights) const {
            Tensor buffer = weights.clone(TensorPlacement::MemDevices::CPU);

            std::uniform_real_distribution<float> distribution(minValue, maxValue);
            std::hash<int> threadNumHash;

            uint64_t totalNumWeights = weights.getDescriptor().getTotalNumElements();
            half *bufferMem = (half *)buffer.getMemPtr();
            int numProcessors = omp_get_num_procs();
            if (numProcessors > 1)
                numProcessors -= 1;
            int maxDesiredProcessors = (totalNumWeights + 29999) / 30000;
            if (numProcessors > maxDesiredProcessors)
                numProcessors = maxDesiredProcessors;
            assert(numProcessors >= 1);
            omp_set_num_threads(numProcessors);
            uint64_t weightsPerThread = (totalNumWeights + (numProcessors - 1)) / numProcessors;
#pragma omp parallel
            {
                int threadNum = omp_get_thread_num();
                std::default_random_engine generator(time(NULL) * threadNumHash(threadNum));
                uint64_t threadEnd = (threadNum + 1) * weightsPerThread;
                if (totalNumWeights < threadEnd)
                    threadEnd = totalNumWeights;
                for (uint64_t i = threadNum * weightsPerThread; i < threadEnd; ++i) {
                    bufferMem[i] = (half)distribution(generator);
                }
            }

            weights.copyFromAsync(buffer, fullyConnected->streams[0]);

            if (fullyConnected->streams.size() > 1) {
                Event weightsUpdatedEvent = fullyConnected->streams[0].putEvent();
                for (unsigned int i = 1; i < fullyConnected->streams.size(); ++i) {
                    fullyConnected->streams[i].waitEvent(weightsUpdatedEvent);
                }
            }
        }

       private:
        float maxValue;
        float minValue;
    };

    void initializeWeights(const Initializer *initializer) {
        assert(compiled);

        const UniformRandomInitializer *uniformRandomInitializer = dynamic_cast<const UniformRandomInitializer *>(initializer);

        if (uniformRandomInitializer) {
            uniformRandomInitializer->initializeWeights(this);
            return;
        }

        assert(false);
    }

    void initializeBiases(const Initializer *initializer) {
        assert(compiled);
        assert(hasBias);

        const UniformRandomInitializer *uniformRandomInitializer = dynamic_cast<const UniformRandomInitializer *>(initializer);

        if (uniformRandomInitializer) {
            uniformRandomInitializer->initializeBiases(this);
            return;
        }

        assert(false);
    }

   private:
    static const float ALPHA_NO_SCALE;
    static const float BETA_CLEAR;
    static const float BETA_ACCUMULATE;

    const uint32_t numInputFeatures;
    const uint32_t numOutputFeatures;
    const int batchSize;

    Optional<Tensor> workspaceForward;
    Optional<Tensor> workspaceBackwardData;
    Optional<Tensor> workspaceBackwardWeights;
    Optional<Tensor> workspaceBackwardBias;

    Optional<cudnnTensorDescriptor_t> biasesDescriptor;
    Optional<cudnnTensorDescriptor_t> featureOutputDescriptor;
    Optional<cudnnReduceTensorDescriptor_t> reduceDescriptor;
};
