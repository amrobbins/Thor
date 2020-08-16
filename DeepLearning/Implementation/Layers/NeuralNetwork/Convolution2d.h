#pragma once

#include "DeepLearning/Implementation/Layers/TrainableWeightsBiasesLayer.h"
#include "Utilities/TensorOperations/GpuConvolution/GpuConvolution.h"

// FIXME: inference only support

class Convolution2d : public TrainableWeightsBiasesLayer {
   public:
    virtual ~Convolution2d() {}

    Convolution2d(const int filterWidth,
                  const int filterHeight,
                  const int filterHorizontalStride,
                  const int filterVerticalStride,
                  const int leftAndRightPadWidth,
                  const int topAndBottomPadHeight,
                  const int numInputChannels,
                  const int numOutputChannels,
                  const int batchSize,
                  const int numInputColumns,
                  const int numInputRows,
                  const bool inferenceOnly,
                  const bool hasBias,
                  Optional<float> learningRate)
        : TrainableWeightsBiasesLayer(inferenceOnly, hasBias, learningRate),
          filterWidth(filterWidth),
          filterHeight(filterHeight),
          filterHorizontalStride(filterHorizontalStride),
          filterVerticalStride(filterVerticalStride),
          leftAndRightPadWidth(leftAndRightPadWidth),
          topAndBottomPadHeight(topAndBottomPadHeight),
          numInputChannels(numInputChannels),
          numOutputChannels(numOutputChannels),
          batchSize(batchSize),
          numInputColumns(numInputColumns),
          numInputRows(numInputRows) {
        string anyGpuType = MachineEvaluator::instance().getGpuType(0);
        ConvolutionKernelRequirement tempConvolutionKernelRequirement = ConvolutionKernelRequirement(anyGpuType,
                                                                                                     filterWidth,
                                                                                                     filterHeight,
                                                                                                     filterHorizontalStride,
                                                                                                     filterVerticalStride,
                                                                                                     leftAndRightPadWidth,
                                                                                                     topAndBottomPadHeight,
                                                                                                     numInputChannels,
                                                                                                     numOutputChannels,
                                                                                                     batchSize,
                                                                                                     numInputColumns,
                                                                                                     numInputRows);
        numOutputRows = tempConvolutionKernelRequirement.getNumOutputRows();
        numOutputColumns = tempConvolutionKernelRequirement.getNumOutputColumns();
    }

    virtual Optional<Tensor> createFeatureOutputTensor() {
        assert(!featureInputs.empty());
        assert(featureInputs.back().isPresent());

        return Tensor(featureInputs.back().get().getPlacement(),
                      TensorDescriptor(TensorDescriptor::DataType::FP16, batchSize, numOutputChannels, numOutputRows, numOutputColumns));
    }

    virtual void compile() {
        int gpuNum;
        assert(!featureInputs.empty());
        assert(featureInputs[0].isPresent());
        assert(featureInputs[0].get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        assert(!streams.empty());
        gpuNum = featureInputs[0].get().getPlacement().getDeviceNum();
        ScopedGpu scopedGpu(gpuNum);

        string gpuType = MachineEvaluator::instance().getGpuType(streams.front().getGpuNum());
        convolutionKernelRequirement = ConvolutionKernelRequirement(gpuType,
                                                                    filterWidth,
                                                                    filterHeight,
                                                                    filterHorizontalStride,
                                                                    filterVerticalStride,
                                                                    leftAndRightPadWidth,
                                                                    topAndBottomPadHeight,
                                                                    numInputChannels,
                                                                    numOutputChannels,
                                                                    batchSize,
                                                                    numInputColumns,
                                                                    numInputRows);

        GpuConvolution::instance().chooseOptimalKernelForward(convolutionKernelRequirement, streams[0]);
        GpuConvolution::instance().chooseOptimalKernelBackward(convolutionKernelRequirement, streams[0]);

        // Allocate 1 weights and 1 weights gradient, if there is more than one connection, will accumulate to weights gradient using
        // BETA=1.0. Data format is NCHW so filter format is KCRS where K = num output channels, C = num input channels, R = filter rows, S
        // = filter columns
        vector<unsigned long> weightsDimensions;
        weightsDimensions.push_back(numOutputChannels);
        weightsDimensions.push_back(numInputChannels);
        weightsDimensions.push_back(filterHeight);
        weightsDimensions.push_back(filterWidth);
        TensorDescriptor weightsDescriptor = TensorDescriptor(TensorDescriptor::DataType::FP16, weightsDimensions);
        weights = Tensor(featureInputs.front().get().getPlacement(), weightsDescriptor);
        if (!inferenceOnly)
            weightsGradient = weights.clone();
        if (hasBias) {
            biases = Tensor(featureInputs.front().get().getPlacement(),
                            TensorDescriptor(TensorDescriptor::DataType::FP16, {weightsDimensions[0]}));
            biasesGradient = biases.get().clone();
        }

        // Allocate 1 workspace of each type, since it is possible that all three types of kernels may be running at the same time.
        // If there is more than one connection, the kernels of a given type will run sequentially so that the workspace will be available
        uint64_t workspaceForwardSizeInBytes = GpuConvolution::instance().getForwardWorkspaceSizeInBytes(convolutionKernelRequirement);
        if (workspaceForwardSizeInBytes > 0) {
            vector<unsigned long> workspaceDimensions;
            workspaceDimensions.push_back(workspaceForwardSizeInBytes);
            TensorDescriptor workspaceDescriptor(TensorDescriptor::DataType::UINT8, workspaceDimensions);
            workspaceForward = Tensor(featureInputs.front().get().getPlacement(), workspaceDescriptor);
        }
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
        if (hasBias) {
            uint64_t workspaceBackwardBiasSizeInBytes =
                GpuConvolution::instance().getBackwardBiasWorkspaceSizeInBytes(convolutionKernelRequirement);
            if (workspaceBackwardBiasSizeInBytes > 0) {
                workspaceBackwardBias = Tensor(featureInputs.front().get().getPlacement(),
                                               TensorDescriptor(TensorDescriptor::DataType::UINT8, {workspaceBackwardBiasSizeInBytes}));
            }
        }
    }

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream, unsigned int connectionNumber) {
        assert(convolutionKernelRequirement.isPresent());
        assert(inputTensor.isPresent());
        assert(outputTensor.isPresent());

        assert(inputTensor.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

        GpuConvolution::instance().convolutionForward(
            convolutionKernelRequirement, inputTensor, weights, biases, outputTensor, workspaceForward, stream);
    }

    virtual void backProp(Optional<Tensor> dataIn,
                          Optional<Tensor> errorIn,
                          Optional<Tensor> errorOut,
                          Stream gradientStream,
                          Optional<Stream> dataStream,
                          unsigned int connectionNumber,
                          bool accumulateGradient) {
        assert(convolutionKernelRequirement.isPresent());
        assert(dataIn.isPresent());
        assert(errorIn.isPresent());
        assert(errorIn.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

        if (errorOut.isPresent()) {
            assert(dataStream.isPresent());
            GpuConvolution::instance().convolutionBackwardData(
                convolutionKernelRequirement, errorIn, weights, errorOut, workspaceBackwardData, dataStream);
        }

        GpuConvolution::instance().convolutionBackwardFilter(
            convolutionKernelRequirement, dataIn, errorIn, weightsGradient, workspaceBackwardFilter, gradientStream, accumulateGradient);

        if (hasBias) {
            GpuConvolution::instance().convolutionBackwardBias(
                convolutionKernelRequirement, errorIn, biasesGradient, workspaceBackwardBias, gradientStream, accumulateGradient);
        }
    }

    void cleanup() {}

   private:
    const int filterWidth;
    const int filterHeight;
    const int filterHorizontalStride;
    const int filterVerticalStride;
    const int leftAndRightPadWidth;
    const int topAndBottomPadHeight;
    const int numInputChannels;
    const int numOutputChannels;
    const int batchSize;
    const int numInputColumns;
    const int numInputRows;
    int numOutputColumns;
    int numOutputRows;

    Optional<ConvolutionKernelRequirement> convolutionKernelRequirement;

    Optional<Tensor> workspaceForward;
    Optional<Tensor> workspaceBackwardData;
    Optional<Tensor> workspaceBackwardFilter;
    Optional<Tensor> workspaceBackwardBias;
};
