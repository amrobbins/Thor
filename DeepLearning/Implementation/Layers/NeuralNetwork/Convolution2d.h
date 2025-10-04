#pragma once

#include "DeepLearning/Implementation/Layers/TrainableWeightsBiasesLayer.h"
#include "Utilities/TensorOperations/GpuConvolution/GpuConvolution.h"

namespace ThorImplementation {

class Convolution2d : public TrainableWeightsBiasesLayer {
   public:
    virtual ~Convolution2d() {}

    Convolution2d(const int filterWidth,
                  const int filterHeight,
                  const int filterHorizontalStride,
                  const int filterVerticalStride,
                  const int leftAndRightPadWidth,
                  const int topAndBottomPadHeight,
                  const int numOutputChannels,
                  const bool hasBias,
                  const int64_t stampedId = -1)
        : TrainableWeightsBiasesLayer(hasBias, stampedId),
          filterWidth(filterWidth),
          filterHeight(filterHeight),
          filterHorizontalStride(filterHorizontalStride),
          filterVerticalStride(filterVerticalStride),
          leftAndRightPadWidth(leftAndRightPadWidth),
          topAndBottomPadHeight(topAndBottomPadHeight),
          numOutputChannels(numOutputChannels) {}

    Convolution2d(SharedWeightsPackage sharedWeightsPackage,
                  const int filterHorizontalStride,
                  const int filterVerticalStride,
                  const int leftAndRightPadWidth,
                  const int topAndBottomPadHeight,
                  const int64_t stampedId = -1)
        : TrainableWeightsBiasesLayer(sharedWeightsPackage, stampedId),
          filterWidth(sharedWeightsPackage.weights.getDescriptor().getDimensions()[3]),
          filterHeight(sharedWeightsPackage.weights.getDescriptor().getDimensions()[2]),
          filterHorizontalStride(filterHorizontalStride),
          filterVerticalStride(filterVerticalStride),
          leftAndRightPadWidth(leftAndRightPadWidth),
          topAndBottomPadHeight(topAndBottomPadHeight),
          numOutputChannels(sharedWeightsPackage.weights.getDescriptor().getDimensions()[0]) {}

    virtual void createWeightsIfNecessary() {
        if (!usingSharedWeights && !weights.isInitialized()) {
            // Allocate 1 weights and 1 weights gradient, if there is more than one connection, will accumulate to weights gradient using
            // BETA=1.0. Data format is NCHW so filter format is KCRS where K = num output channels, C = num input channels, R = filter
            // rows, S = filter columns
            std::vector<unsigned long> weightsDimensions;
            weightsDimensions.push_back(numOutputChannels);
            weightsDimensions.push_back(featureInputs[0].get().getDescriptor().getDimensions()[1]);
            weightsDimensions.push_back(filterHeight);
            weightsDimensions.push_back(filterWidth);
            TensorDescriptor weightsDescriptor = TensorDescriptor(TensorDescriptor::DataType::FP16, weightsDimensions);
            weights = Tensor(featureInputs.front().get().getPlacement(), weightsDescriptor);
            if (hasBias) {
                biases = Tensor(featureInputs.front().get().getPlacement(),
                                TensorDescriptor(TensorDescriptor::DataType::FP16, {weightsDimensions[0]}));
            }
        }
    }

    virtual Optional<Tensor> createFeatureOutputTensor() {
        assert(!featureInputs.empty());
        assert(featureInputs.back().isPresent());

        batchSize = featureInputs[0].get().getDescriptor().getDimensions()[0];
        numInputChannels = featureInputs[0].get().getDescriptor().getDimensions()[1];
        numInputRows = featureInputs[0].get().getDescriptor().getDimensions()[2];
        numInputColumns = featureInputs[0].get().getDescriptor().getDimensions()[3];

        std::string anyGpuType = MachineEvaluator::instance().getGpuType(0);
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

        return Tensor(featureInputs.back().get().getPlacement(),
                      TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize, numOutputChannels, numOutputRows, numOutputColumns}));
    }

    virtual void compile() {
        TrainableWeightsBiasesLayer::compile();

        int gpuNum;
        assert(!featureInputs.empty());
        assert(featureInputs[0].isPresent());
        assert(featureInputs[0].get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        assert(!streams.empty());
        gpuNum = featureInputs[0].get().getPlacement().getDeviceNum();
        ScopedGpu scopedGpu(gpuNum);

        batchSize = featureInputs[0].get().getDescriptor().getDimensions()[0];
        numInputChannels = featureInputs[0].get().getDescriptor().getDimensions()[1];
        numInputRows = featureInputs[0].get().getDescriptor().getDimensions()[2];
        numInputColumns = featureInputs[0].get().getDescriptor().getDimensions()[3];

        // ensure that the cudnnHandle is preallocated for all streams
        for (uint32_t i = 0; i < streams.size(); ++i) {
            streams[i].getCudnnHandle();
        }

        std::string gpuType = MachineEvaluator::instance().getGpuType(streams.front().getGpuNum());
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
        numOutputRows = convolutionKernelRequirement.get().getNumOutputRows();
        numOutputColumns = convolutionKernelRequirement.get().getNumOutputColumns();

        GpuConvolution::instance().chooseOptimalKernelForward(convolutionKernelRequirement, streams[0]);
        GpuConvolution::instance().chooseOptimalKernelBackward(convolutionKernelRequirement, streams[0]);

        // Allocate 1 workspace of each type, since it is possible that all three types of kernels may be running at the same time.
        // If there is more than one connection, the kernels of a given type will run sequentially so that the workspace will be available
        uint64_t workspaceForwardSizeInBytes = GpuConvolution::instance().getForwardWorkspaceSizeInBytes(convolutionKernelRequirement);
        if (workspaceForwardSizeInBytes > 0) {
            std::vector<unsigned long> workspaceDimensions;
            workspaceDimensions.push_back(workspaceForwardSizeInBytes);
            TensorDescriptor workspaceDescriptor(TensorDescriptor::DataType::UINT8, workspaceDimensions);
            workspaceForward = Tensor(featureInputs.front().get().getPlacement(), workspaceDescriptor);
        }
        if (!isBackPropStub()) {
            uint64_t workspaceBackwardDataSizeInBytes =
                GpuConvolution::instance().getBackwardDataWorkspaceSizeInBytes(convolutionKernelRequirement);
            if (workspaceBackwardDataSizeInBytes > 0) {
                std::vector<unsigned long> workspaceDimensions;
                workspaceDimensions.push_back(workspaceBackwardDataSizeInBytes);
                TensorDescriptor workspaceDescriptor(TensorDescriptor::DataType::UINT8, workspaceDimensions);
                workspaceBackwardData = Tensor(featureInputs.front().get().getPlacement(), workspaceDescriptor);
            }
        }
        if (!isInferenceOnly()) {
            uint64_t workspaceBackwardFilterSizeInBytes =
                GpuConvolution::instance().getBackwardFilterWorkspaceSizeInBytes(convolutionKernelRequirement);
            if (workspaceBackwardFilterSizeInBytes > 0) {
                std::vector<unsigned long> workspaceDimensions;
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
    }

    virtual void infer(Optional<Tensor> inputTensor,
                       Optional<Tensor> outputTensor,
                       Stream stream,
                       unsigned int connectionNumber,
                       Tensor weights,
                       Optional<Tensor> biases) {
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
                          Stream dataStream,
                          unsigned int connectionNumber,
                          bool accumulateGradient) {
        assert(convolutionKernelRequirement.isPresent());
        assert(dataIn.isPresent());
        assert(errorIn.isPresent());
        assert(errorIn.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

        if (errorOut.isPresent()) {
            assert(dataStream.isInitialized());
            GpuConvolution::instance().convolutionBackwardData(
                convolutionKernelRequirement, errorIn, weights, errorOut, workspaceBackwardData, dataStream);
        }

        if (!isInferenceOnly()) {
            assert(optimizer.isPresent());

            // backward() syncs gradient stream with data stream prior to calling this to ensure error in is ready at end of gradient stream
            optimizer.get()->computeWeightsUpdate(dataIn, errorIn, accumulateGradient);

            // weights update cannot be applied to weights until errorOut has been computed since weights are part of that computation
            // so to enforce this gradientUpdateStream says that gradient is not ready to be applied until both errorOut and gradient are
            // computed
            optimizer.get()->getGradientUpdateStream().waitEvent(dataStream.putEvent());
            // Now at the end of gradientUpdateStream errorOut and gradients are ready from the updates for this connection.

            // Upon processing the last connection, schedule the upate to the weights memory.
            if (stillWaitingForErrorInputTensors.empty()) {
                optimizer.get()->updateWeights(weights, biases, batchSize);
            }

            // weights will be updated at the current end of the gradientUpdateStream
            // so Forward() must wait until gradientUpdateStream is finished.
            // This is accomplished in TrainableWeightsBiasesLayer::forward().
        }
    }

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

        GpuConvolution::instance().convolutionBackwardFilter(convolutionKernelRequirement,
                                                             featureIn,
                                                             errorIn,
                                                             weightsGradient,
                                                             workspaceBackwardFilter,
                                                             gradientUpdateStream,
                                                             accumulateGradient);

        if (hasBias) {
            GpuConvolution::instance().convolutionBackwardBias(
                convolutionKernelRequirement, errorIn, biasesGradient, workspaceBackwardBias, gradientUpdateStream, accumulateGradient);
        }
    }

    void cleanup() {}

    void printBackwardFilterKernelInfo() { GpuConvolution::instance().printBackwardFilterKernelInfo(convolutionKernelRequirement); }

    uint64_t flopsPerConnectionPerExample() {
        Optional<Tensor> anyFeatureInput = getFirstPresentTensor(featureInputs);
        Optional<Tensor> anyFeatureOutput = getFirstPresentTensor(featureOutputs);
        assert(anyFeatureInput.isPresent());
        assert(anyFeatureOutput.isPresent());
        uint64_t flops = 2 * filterHeight * filterWidth * numInputChannels - 1;
        if (hasBias)
            flops += 1;
        flops *= numOutputRows * numOutputColumns * numOutputChannels;
        return flops;
    }

    uint64_t flopsPerWeightUpdate() {
        // dW FLOPs per example:
        // For each output position and output channel, we do R*S*C MACs to accumulate into the filter.
        // Using 2 FLOPs per MAC (mul+add):
        const uint64_t macs_per_out_elem = static_cast<uint64_t>(filterHeight) * filterWidth * numInputChannels;
        const uint64_t flops_dW = 2ULL * macs_per_out_elem * numOutputChannels * numOutputRows * numOutputColumns;

        const uint64_t flops_dB = hasBias ? (numOutputChannels * numOutputRows * numOutputColumns) : 0ULL;

        return flops_dW + flops_dB;
    }

    //    uint64_t flopsPerWeightUpdate() {
    //        uint64_t outputFilterHeight = (1 + numInputRows - filterHeight) / filterHorizontalStride;
    //        uint64_t outputFilterWidth = (1 + numInputColumns - filterWidth) / filterVerticalStride;
    //        uint64_t flops = (2 * filterHeight * filterWidth * numInputChannels - 1) * (numOutputChannels * outputFilterHeight *
    //        outputFilterWidth); return flops;
    //    }

    virtual uint64_t floatingPointOperationsPerExampleForward() {
        uint32_t connectionMultiplier = 0;
        for (uint32_t i = 0; i < featureInputs.size(); ++i) {
            if (featureInputs[i].isPresent())
                connectionMultiplier += 1;
        }

        return connectionMultiplier * flopsPerConnectionPerExample();
    }

    virtual uint64_t floatingPointOperationsPerExampleBackward() {
        if (isInferenceOnly())
            return 0;

        uint32_t connectionMultiplier = 0;
        uint32_t sums = 0;
        for (uint32_t i = 0; i < errorInputs.size(); ++i) {
            if (errorInputs[i].isPresent()) {
                if (connectionMultiplier == 0)
                    connectionMultiplier += 2;
                else
                    sums += 1;
            }
        }
        for (uint32_t i = 0; i < errorOutputs.size(); ++i) {
            if (errorOutputs[i].isPresent())
                connectionMultiplier += 1;
        }

        Optional<Tensor> anyErrorInput = getFirstPresentTensor(errorInputs);
        assert(anyErrorInput.isPresent());

        return connectionMultiplier * flopsPerConnectionPerExample() +
               (sums * anyErrorInput.get().getDescriptor().getTotalNumElements()) / batchSize + flopsPerWeightUpdate();
    }

   private:
    const uint32_t filterWidth;
    const uint32_t filterHeight;
    const uint32_t filterHorizontalStride;
    const uint32_t filterVerticalStride;
    const uint32_t leftAndRightPadWidth;
    const uint32_t topAndBottomPadHeight;
    uint64_t numInputChannels;
    const uint64_t numOutputChannels;
    uint64_t batchSize;
    uint64_t numInputColumns;
    uint64_t numInputRows;
    uint64_t numOutputColumns;
    uint64_t numOutputRows;

    Optional<ConvolutionKernelRequirement> convolutionKernelRequirement;

    Optional<Tensor> workspaceForward;
    Optional<Tensor> workspaceBackwardData;
    Optional<Tensor> workspaceBackwardFilter;
    Optional<Tensor> workspaceBackwardBias;
};

}  // namespace ThorImplementation
