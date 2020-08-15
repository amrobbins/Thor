#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.h"

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
            gpuNum, batchSize, numInputFeatures, numOutputFeatures, TensorDescriptor::DataType::FP16);

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
            biasesGradient = biases.get().clone();
        }

        // Allocate 1 workspace of each type, since it is possible that all three types of kernels may be running at the same time.
        // If there is more than one connection, the kernels of a given type will run sequentially so that the workspace will be available
        bool kernelWillRunOnGpu;
        uint64_t workspaceForwardSizeInBytes = CublasMatrixMultiply::instance().getWorkspaceSizeInBytes(
            0, batchSize, numInputFeatures, numOutputFeatures, TensorDescriptor::DataType::FP16, kernelWillRunOnGpu);
        assert(kernelWillRunOnGpu);
        if (workspaceForwardSizeInBytes > 0) {
            vector<unsigned long> workspaceDimensions;
            workspaceDimensions.push_back(workspaceForwardSizeInBytes);
            TensorDescriptor workspaceDescriptor(TensorDescriptor::DataType::UINT8, workspaceDimensions);
            workspaceForward = Tensor(featureInputs.front().get().getPlacement(), workspaceDescriptor);
        }
        /*
                if (!isBackPropStub()) {
                    uint64_t workspaceBackwardDataSizeInBytes =
                        GpuConvolution::instance().getBackwardDataWorkspaceSizeInBytes(convolutionKernelRequirement);
                    GpuConvolution::instance().getBackwardDataWorkspaceSizeInBytes(convolutionKernelRequirement);
                    if (workspaceBackwardDataSizeInBytes > 0) {
                        vector<unsigned long> workspaceDimensions;
                        workspaceDimensions.push_back(workspaceBackwardDataSizeInBytes);
                        TensorDescriptor workspaceDescriptor(TensorDescriptor::DataType::UINT8, workspaceDimensions);
                        workspaceBackwardData = Tensor(featureInputs.front().get().getPlacement(), workspaceDescriptor);
                    }
                }
                uint64_t workspaceBackwardFilterSizeInBytes =
                    GpuConvolution::instance().getBackwardFilterWorkspaceSizeInBytes(convolutionKernelRequirement);
                if (workspaceBackwardFilterSizeInBytes > 0) {
                    vector<unsigned long> workspaceDimensions;
                    workspaceDimensions.push_back(workspaceBackwardFilterSizeInBytes);
                    TensorDescriptor workspaceDescriptor(TensorDescriptor::DataType::UINT8, workspaceDimensions);
                    workspaceBackwardFilter = Tensor(featureInputs.front().get().getPlacement(), workspaceDescriptor);
                }
        */
    }

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream, unsigned int connectionNumber) {
        assert(inputTensor.isPresent());
        assert(outputTensor.isPresent());
        assert(inputTensor.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

        CublasMatrixMultiply::instance().multiply(inputTensor,
                                                  weights,
                                                  outputTensor,
                                                  workspaceForward,
                                                  batchSize,
                                                  numInputFeatures,
                                                  numOutputFeatures,
                                                  false,
                                                  TensorDescriptor::DataType::FP16,
                                                  stream);

        // FIXME: use cudnnAddTensor
        if (hasBias)
            addFullyConnectedLayerBias(outputTensor, biases, stream);
    }

    virtual void backProp(Optional<Tensor> dataIn,
                          Optional<Tensor> errorIn,
                          Optional<Tensor> errorOut,
                          Stream gradientStream,
                          Optional<Stream> dataStream,
                          unsigned int connectionNumber,
                          bool accumulateGradient) {
        /*
                assert(convolutionKernelRequirement.isPresent());
                assert(errorIn.isPresent());
                assert(errorIn.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

                if (errorOut.isPresent()) {
                    assert(dataStream.isPresent());
                    GpuConvolution::instance().convolutionBackwardData(
                        convolutionKernelRequirement, errorIn, weights, errorOut, workspaceBackwardData, dataStream);
                }

                GpuConvolution::instance().convolutionBackwardFilter(
                    convolutionKernelRequirement, dataIn, errorIn, weightsGradient, workspaceBackwardFilter, gradientStream,
           accumulateGradient);

                if (hasBias) {
                    GpuConvolution::instance().convolutionBackwardBias(errorIn, biasesGradient, workspaceBackwardBias, gradientStream);
                }
        */
    }

    void addFullyConnectedLayerBias(Tensor outputTensor, Tensor biases, Stream stream) {
        // FIXME: implement
    }

    void cleanup() {}

   private:
    const uint32_t numInputFeatures;
    const uint32_t numOutputFeatures;
    const int batchSize;

    Optional<Tensor> workspaceForward;
    Optional<Tensor> workspaceBackwardData;
    Optional<Tensor> workspaceBackwardWeights;
};
