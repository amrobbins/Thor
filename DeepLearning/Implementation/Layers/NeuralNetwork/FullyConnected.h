#pragma once

#include "DeepLearning/Implementation/Layers/TrainableWeightsBiasesLayer.h"
#include "Utilities/TensorOperations/DeepLearning/Add1dBias.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.h"

namespace ThorImplementation {

class FullyConnected : public TrainableWeightsBiasesLayer {
   public:
    virtual ~FullyConnected() {}

    FullyConnected(const uint32_t numOutputFeatures, const bool hasBias)
        : TrainableWeightsBiasesLayer(hasBias), numOutputFeatures(numOutputFeatures) {}

    FullyConnected(SharedWeightsPackage sharedWeightsPackage)
        : TrainableWeightsBiasesLayer(sharedWeightsPackage),
          numOutputFeatures(sharedWeightsPackage.weights.getDescriptor().getDimensions()[1]) {}

    virtual Optional<Tensor> createFeatureOutputTensor() {
        assert(!featureInputs.empty());
        assert(featureInputs.back().isPresent());

        return Tensor(featureInputs.back().get().getPlacement(),
                      TensorDescriptor(
                          TensorDescriptor::DataType::FP16, featureInputs[0].get().getDescriptor().getDimensions()[0], numOutputFeatures));
    }

    virtual void createWeightsIfNecessary() {
        if (!usingSharedWeights && !weights.isInitialized()) {
            vector<unsigned long> weightsDimensions;
            weightsDimensions.push_back(featureInputs[0].get().getDescriptor().getDimensions()[1]);
            weightsDimensions.push_back(numOutputFeatures);
            TensorDescriptor weightsDescriptor = TensorDescriptor(TensorDescriptor::DataType::FP16, weightsDimensions);
            weights = Tensor(featureInputs.front().get().getPlacement(), weightsDescriptor);
            if (!isInferenceOnly())
                weightsGradient = weights.clone();
            if (hasBias) {
                biases = Tensor(featureInputs.front().get().getPlacement(),
                                TensorDescriptor(TensorDescriptor::DataType::FP16, numOutputFeatures));
                if (!isInferenceOnly())
                    biasesGradient = biases.get().clone(TensorDescriptor::DataType::FP16);
            }
        }
    }

    virtual void compile() {
        int gpuNum;
        assert(!featureInputs.empty());
        assert(featureInputs[0].isPresent());
        assert(featureInputs[0].get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        assert(!streams.empty());
        gpuNum = featureInputs[0].get().getPlacement().getDeviceNum();
        ScopedGpu scopedGpu(gpuNum);

        batchSize = featureInputs[0].get().getDescriptor().getDimensions()[0];
        numInputFeatures = featureInputs[0].get().getDescriptor().getDimensions()[1];

        CublasMatrixMultiply::instance().chooseOptimalKernel(
            gpuNum, batchSize, numInputFeatures, numInputFeatures, numOutputFeatures, false, false, TensorDescriptor::DataType::FP16);

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

        if (!isBackPropStub() && !isInferenceOnly()) {
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

        if (!isInferenceOnly()) {
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

            launchAdd1dBias<half>(
                (half *)outputTensor.get().getMemPtr(), (half *)biases.get().getMemPtr(), batchSize, numOutputFeatures, stream);
        }
    }

    virtual void backProp(Optional<Tensor> dataIn,
                          Optional<Tensor> errorIn,
                          Optional<Tensor> errorOut,
                          Stream gradientStream,
                          Stream dataStream,
                          unsigned int connectionNumber,
                          bool accumulateGradient) {
        assert(errorIn.isPresent());
        assert(errorIn.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

        if (errorOut.isPresent()) {
            assert(dataStream.isInitialized());

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
            launchSumBatch((half *)errorIn.get().getMemPtr(),
                           (half *)biasesGradient.get().getMemPtr(),
                           numOutputFeatures,
                           batchSize,
                           accumulateGradient,
                           gradientStream);
        }
    }

   private:
    static const float ALPHA_NO_SCALE;
    static const float BETA_CLEAR;
    static const float BETA_ACCUMULATE;

    /*
        void createBiasesTensorDescriptor() {
            assert(!uninitialized());

            if (*ppBiasesDescriptor != nullptr)
                return **ppBiasesDescriptor;
            *ppBiasesDescriptor = new cudnnTensorDescriptor_t;

            cudnnStatus_t cudnnStatus;

            cudnnStatus = cudnnCreateTensorDescriptor(*ppBiasesDescriptor);
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
            cudnnStatus = cudnnSetTensor4dDescriptor(**ppBiasesDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, 1, numOutputChannels, 1, 1);
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

            return **ppBiasesDescriptor;
        }

        cudnnTensorDescriptor_t getBiasesTensorDescriptor()
    */

    uint32_t numInputFeatures;
    const uint32_t numOutputFeatures;
    uint32_t batchSize;

    Optional<Tensor> workspaceForward;
    Optional<Tensor> workspaceBackwardData;
    Optional<Tensor> workspaceBackwardWeights;
};

}  // namespace ThorImplementation
