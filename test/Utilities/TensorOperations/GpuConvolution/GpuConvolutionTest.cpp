#include "Utilities/TensorOperations/GpuConvolution/GpuConvolution.h"

#include "test/Utilities/TensorOperations/GpuConvolution/ConvolutionTestHelper.h"

#include "gtest/gtest.h"

#include <vector>

using namespace ThorImplementation;
using namespace std;

TEST(GpuConvolution, ConvolutionBackwardBiasProducesCorrectResult) {
    Stream stream(0);

    for (uint64_t t = 0; t < 5; ++t) {
        uint64_t numInputColumns = (rand() % 75) + 1;
        uint64_t numInputRows = (rand() % 75) + 1;
        uint64_t filterWidth = (rand() % numInputColumns) + 1;
        uint64_t filterHeight = (rand() % numInputRows) + 1;

        uint64_t filterHorizontalStride = (rand() % numInputColumns) + 1;
        uint64_t filterVerticalStride = (rand() % numInputRows) + 1;
        uint64_t leftAndRightPadWidth = (rand() % 20) + 1;
        uint64_t topAndBottomPadHeight = (rand() % 20) + 1;
        uint64_t numFeatureInputChannels = (rand() % 10) + 1;
        uint64_t numFeatureOutputChannels = (rand() % 10) + 1;
        uint64_t batchSize = (rand() % 10) + 1;

        bool accumulate = rand() % 2;

        ConvolutionKernelRequirement convolutionKernelRequirement(MachineEvaluator::instance().getGpuType(0),
                                                                  filterWidth,
                                                                  filterHeight,
                                                                  filterHorizontalStride,
                                                                  filterVerticalStride,
                                                                  leftAndRightPadWidth,
                                                                  topAndBottomPadHeight,
                                                                  numFeatureInputChannels,
                                                                  numFeatureOutputChannels,
                                                                  batchSize,
                                                                  numInputColumns,
                                                                  numInputRows);

        uint64_t numOutputRows = convolutionKernelRequirement.getNumOutputRows();
        uint64_t numOutputColumns = convolutionKernelRequirement.getNumOutputColumns();

        // Allocate tensors
        TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
        TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

        Tensor errorInputCpu(
            cpuPlacement,
            TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize, numFeatureOutputChannels, numOutputRows, numOutputColumns}));
        Tensor errorInputGpu(
            gpuPlacement,
            TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize, numFeatureOutputChannels, numOutputRows, numOutputColumns}));

        Tensor biasesGradientCpu(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {numFeatureOutputChannels}));
        Tensor biasesGradientGpu(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {numFeatureOutputChannels}));
        Tensor biasesGradientGpu_h(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {numFeatureOutputChannels}));

        Tensor workspaceGpu(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize * numFeatureOutputChannels}));

        // Fill input tensors
        uint64_t errorInputNumElements = errorInputCpu.getDescriptor().getTotalNumElements();
        half *errorInputMem = (half *)errorInputCpu.getMemPtr();
        for (uint64_t i = 0; i < errorInputNumElements; ++i) {
            errorInputMem[i] = ((rand() % 200) / 10.0f) - 10.0f;
        }
        errorInputGpu.copyFromAsync(errorInputCpu, stream);

        if (accumulate) {
            half *biasesGradientCpuMem = (half *)biasesGradientCpu.getMemPtr();
            for (uint64_t i = 0; i < numFeatureOutputChannels; ++i) {
                biasesGradientCpuMem[i] = (half)(((rand() % 200) / 10.0f) - 10.0f);
            }
            biasesGradientGpu.copyFromAsync(biasesGradientCpu, stream);
        }

        // Perform gradient computation on GPU and CPU
        GpuConvolution::instance().convolutionBackwardBias(
            convolutionKernelRequirement, errorInputGpu, biasesGradientGpu, workspaceGpu, stream, accumulate);

        ConvolutionTestHelper::cpuConvolutionBackwardBias(errorInputCpu, biasesGradientCpu, accumulate);

        biasesGradientGpu_h.copyFromAsync(biasesGradientGpu, stream);
        stream.synchronize();

        // Verify CPU and GPU results match
        for (uint64_t i = 0; i < numFeatureOutputChannels; ++i) {
            float cpuVal = biasesGradientCpu.getElement<half>({(uint64_t)i});
            float gpuVal = biasesGradientGpu_h.getElement<half>({(uint64_t)i});
            EXPECT_EQ(cpuVal, gpuVal);
        }
    }
}

TEST(GpuConvolution, ConvolutionForwardProducesCorrectResult) {
    srand(time(nullptr));

    Stream stream(0);

    for (uint64_t t = 0; t < 5; ++t) {
        uint64_t numInputColumns = (rand() % 75) + 1;
        uint64_t numInputRows = (rand() % 75) + 1;
        uint64_t filterWidth = (rand() % numInputColumns) + 1;
        uint64_t filterHeight = (rand() % numInputRows) + 1;

        uint64_t filterHorizontalStride = (rand() % numInputColumns) + 1;
        uint64_t filterVerticalStride = (rand() % numInputRows) + 1;
        uint64_t leftAndRightPadWidth = (rand() % 20) + 1;
        uint64_t topAndBottomPadHeight = (rand() % 20) + 1;
        uint64_t numFeatureInputChannels = (rand() % 10) + 1;
        uint64_t numFeatureOutputChannels = (rand() % 10) + 1;
        uint64_t batchSize = (rand() % 10) + 1;

        ConvolutionKernelRequirement convolutionKernelRequirement(MachineEvaluator::instance().getGpuType(0),
                                                                  filterWidth,
                                                                  filterHeight,
                                                                  filterHorizontalStride,
                                                                  filterVerticalStride,
                                                                  leftAndRightPadWidth,
                                                                  topAndBottomPadHeight,
                                                                  numFeatureInputChannels,
                                                                  numFeatureOutputChannels,
                                                                  batchSize,
                                                                  numInputColumns,
                                                                  numInputRows);

        // printf("%s\n", convolutionKernelRequirement.toString().c_str()); fflush(stdout);

        assert(convolutionKernelRequirement.getNumOutputColumns() ==
               ConvolutionTestHelper::computeOutputDimensionSize(convolutionKernelRequirement.getNumInputColumns(),
                                                                 convolutionKernelRequirement.getLeftAndRightPadWidth(),
                                                                 convolutionKernelRequirement.getFilterWidth(),
                                                                 convolutionKernelRequirement.getFilterHorizontalStride()));
        assert(convolutionKernelRequirement.getNumOutputRows() ==
               ConvolutionTestHelper::computeOutputDimensionSize(convolutionKernelRequirement.getNumInputRows(),
                                                                 convolutionKernelRequirement.getTopAndBottomPadHeight(),
                                                                 convolutionKernelRequirement.getFilterHeight(),
                                                                 convolutionKernelRequirement.getFilterVerticalStride()));

        TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
        TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

        TensorDescriptor featureInputDescriptor(TensorDescriptor::DataType::FP16,
                                                {batchSize, numFeatureInputChannels, numInputRows, numInputColumns});
        Tensor featureInputCpu(cpuPlacement, featureInputDescriptor);
        Tensor featureInputGpu(gpuPlacement, featureInputDescriptor);

        uint64_t numOutputRows = convolutionKernelRequirement.getNumOutputRows();
        uint64_t numOutputColumns = convolutionKernelRequirement.getNumOutputColumns();
        TensorDescriptor featureOutputDescriptor(TensorDescriptor::DataType::FP16,
                                                 {batchSize, numFeatureOutputChannels, numOutputRows, numOutputColumns});
        Tensor featureOutputCpu(cpuPlacement, featureOutputDescriptor);
        Tensor featureOutputGpu(gpuPlacement, featureOutputDescriptor);
        Tensor featureOutputGpu_h(cpuPlacement, featureOutputDescriptor);

        Tensor biasCpu(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {numFeatureOutputChannels}));
        Tensor biasGpu(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {numFeatureOutputChannels}));

        GpuConvolution::instance().chooseOptimalKernelForward(convolutionKernelRequirement, stream);

        uint64_t workspaceSizeInBytes = GpuConvolution::instance().getForwardWorkspaceSizeInBytes(convolutionKernelRequirement);
        Optional<Tensor> workspace;
        if (workspaceSizeInBytes != 0)
            workspace = Tensor(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::UINT8, {workspaceSizeInBytes}));

        uint64_t totalNumFeatureInputElements = featureInputDescriptor.getTotalNumElements();
        half *featureInputMem = (half *)featureInputCpu.getMemPtr();
        for (uint64_t i = 0; i < totalNumFeatureInputElements; ++i) {
            float val = (rand() % 100) / 10.0f;
            featureInputMem[i] = (half)(val - 5.0f);
        }
        featureInputGpu.copyFromAsync(featureInputCpu, stream);

        TensorDescriptor weightsDescriptor(TensorDescriptor::DataType::FP16,
                                           {numFeatureOutputChannels, numFeatureInputChannels, filterHeight, filterWidth});
        Tensor weightsCpu(cpuPlacement, weightsDescriptor);
        Tensor weightsGpu(gpuPlacement, weightsDescriptor);
        half *weightsMem = (half *)weightsCpu.getMemPtr();
        uint64_t totalNumWeights = weightsDescriptor.getTotalNumElements();
        for (uint64_t i = 0; i < totalNumWeights; ++i) {
            float val = (rand() % 100) / 25.0f;
            weightsMem[i] = (half)(val - 2.0f);
        }
        weightsGpu.copyFromAsync(weightsCpu, stream);

        half *biasMem = (half *)biasCpu.getMemPtr();
        for (uint64_t i = 0; i < numFeatureOutputChannels; ++i)
            biasMem[i] = ((rand() % 200) / 10.0f) - 10.0f;
        biasGpu.copyFromAsync(biasCpu, stream);

        GpuConvolution::instance().convolutionForward(
            convolutionKernelRequirement, featureInputGpu, weightsGpu, biasGpu, featureOutputGpu, workspace, stream);

        ConvolutionTestHelper::cpuConvolutionForward(featureInputCpu, weightsCpu, biasCpu, featureOutputCpu, convolutionKernelRequirement);

        featureOutputGpu_h.copyFromAsync(featureOutputGpu, stream);
        stream.synchronize();

        uint64_t totalNumFeatureOutputElements = featureOutputCpu.getDescriptor().getTotalNumElements();
        half *cpuFeatureOut = (half *)featureOutputCpu.getMemPtr();
        half *gpuFeatureOut = (half *)featureOutputGpu_h.getMemPtr();
        for (uint64_t i = 0; i < totalNumFeatureOutputElements; ++i) {
            float thresh = std::max(abs((float)cpuFeatureOut[i]) / 500, 0.02f);
            EXPECT_LT(abs((float)(cpuFeatureOut[i]) - (float)(gpuFeatureOut[i])), thresh);
            if (abs((float)(cpuFeatureOut[i]) - (float)(gpuFeatureOut[i])) >= thresh)
                printf("%f %f\n", (float)(cpuFeatureOut[i]), (float)(gpuFeatureOut[i]));
        }
    }
}

void backwardFilterTest(bool accumulate) {
    Stream stream(0);

    for (uint64_t t = 0; t < 5; ++t) {
        uint64_t numInputColumns = (rand() % 75) + 1;
        uint64_t numInputRows = (rand() % 75) + 1;
        uint64_t filterWidth = (rand() % numInputColumns) + 1;
        uint64_t filterHeight = (rand() % numInputRows) + 1;

        uint64_t filterHorizontalStride = (rand() % numInputColumns) + 1;
        uint64_t filterVerticalStride = (rand() % numInputRows) + 1;
        uint64_t leftAndRightPadWidth = (rand() % 20) + 1;
        uint64_t topAndBottomPadHeight = (rand() % 20) + 1;
        uint64_t numFeatureInputChannels = (rand() % 10) + 1;
        uint64_t numFeatureOutputChannels = (rand() % 10) + 1;
        uint64_t batchSize = (rand() % 10) + 1;

        ConvolutionKernelRequirement convolutionKernelRequirement(MachineEvaluator::instance().getGpuType(0),
                                                                  filterWidth,
                                                                  filterHeight,
                                                                  filterHorizontalStride,
                                                                  filterVerticalStride,
                                                                  leftAndRightPadWidth,
                                                                  topAndBottomPadHeight,
                                                                  numFeatureInputChannels,
                                                                  numFeatureOutputChannels,
                                                                  batchSize,
                                                                  numInputColumns,
                                                                  numInputRows);
        uint64_t numOutputRows = convolutionKernelRequirement.getNumOutputRows();
        uint64_t numOutputColumns = convolutionKernelRequirement.getNumOutputColumns();

        // printf("%s\n", convolutionKernelRequirement.toString().c_str()); fflush(stdout);

        assert(convolutionKernelRequirement.getNumOutputColumns() ==
               ConvolutionTestHelper::computeOutputDimensionSize(convolutionKernelRequirement.getNumInputColumns(),
                                                                 convolutionKernelRequirement.getLeftAndRightPadWidth(),
                                                                 convolutionKernelRequirement.getFilterWidth(),
                                                                 convolutionKernelRequirement.getFilterHorizontalStride()));
        assert(convolutionKernelRequirement.getNumOutputRows() ==
               ConvolutionTestHelper::computeOutputDimensionSize(convolutionKernelRequirement.getNumInputRows(),
                                                                 convolutionKernelRequirement.getTopAndBottomPadHeight(),
                                                                 convolutionKernelRequirement.getFilterHeight(),
                                                                 convolutionKernelRequirement.getFilterVerticalStride()));

        // Allocate tensors
        TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
        TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

        TensorDescriptor featureInputDescriptor(TensorDescriptor::DataType::FP16,
                                                {batchSize, numFeatureInputChannels, numInputRows, numInputColumns});
        Tensor featureInputCpu(cpuPlacement, featureInputDescriptor);
        Tensor featureInputGpu(gpuPlacement, featureInputDescriptor);

        TensorDescriptor featureOutputDescriptor(TensorDescriptor::DataType::FP16,
                                                 {batchSize, numFeatureOutputChannels, numOutputRows, numOutputColumns});
        Tensor errorInputCpu(cpuPlacement, featureOutputDescriptor);
        Tensor errorInputGpu(gpuPlacement, featureOutputDescriptor);

        TensorDescriptor weightsDescriptor(TensorDescriptor::DataType::FP16,
                                           {numFeatureOutputChannels, numFeatureInputChannels, filterHeight, filterWidth});
        Tensor weightsGradientCpu(cpuPlacement, weightsDescriptor);
        Tensor weightsGradientGpu(gpuPlacement, weightsDescriptor);
        Tensor weightsGradientGpu_h(cpuPlacement, weightsDescriptor);

        // Fill input tensors
        uint64_t totalNumFeatureInputElements = featureInputDescriptor.getTotalNumElements();
        half *featureInputMem = (half *)featureInputCpu.getMemPtr();
        for (uint64_t i = 0; i < totalNumFeatureInputElements; ++i) {
            float val = (rand() % 100) / 10.0f;
            featureInputMem[i] = (half)(val - 5.0f);
        }
        featureInputGpu.copyFromAsync(featureInputCpu, stream);

        uint64_t totalNumErrorInputElements = featureOutputDescriptor.getTotalNumElements();
        half *errorInputMem = (half *)errorInputCpu.getMemPtr();
        for (uint64_t i = 0; i < totalNumErrorInputElements; ++i) {
            float val = (rand() % 100) / 10.0f;
            errorInputMem[i] = (half)(val - 5.0f);
        }
        errorInputGpu.copyFromAsync(errorInputCpu, stream);

        uint64_t totalNumWeightsGradients = weightsDescriptor.getTotalNumElements();
        half *weightsGradientMem = (half *)weightsGradientCpu.getMemPtr();
        for (uint64_t i = 0; i < totalNumWeightsGradients; ++i) {
            float val = (rand() % 100) / 10.0f;
            weightsGradientMem[i] = (half)(val - 5.0f);
        }
        weightsGradientGpu.copyFromAsync(weightsGradientCpu, stream);

        // Perform convolution on GPU and CPU
        GpuConvolution::instance().chooseOptimalKernelBackward(convolutionKernelRequirement, stream);

        uint64_t workspaceSizeInBytes = GpuConvolution::instance().getBackwardFilterWorkspaceSizeInBytes(convolutionKernelRequirement);
        Optional<Tensor> workspace;
        if (workspaceSizeInBytes != 0)
            workspace = Tensor(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::UINT8, {workspaceSizeInBytes}));

        GpuConvolution::instance().convolutionBackwardFilter(
            convolutionKernelRequirement, featureInputGpu, errorInputGpu, weightsGradientGpu, workspace, stream, accumulate);

        ConvolutionTestHelper::cpuConvolutionBackwardFilter(
            featureInputCpu, errorInputCpu, weightsGradientCpu, convolutionKernelRequirement, accumulate);

        weightsGradientGpu_h.copyFromAsync(weightsGradientGpu, stream);
        stream.synchronize();

        // Verify CPU and GPU results match
        for (uint64_t o = 0; o < numFeatureOutputChannels; ++o) {
            for (uint64_t i = 0; i < numFeatureInputChannels; ++i) {
                for (uint64_t h = 0; h < filterHeight; ++h) {
                    for (uint64_t w = 0; w < filterWidth; ++w) {
                        float cpuVal = weightsGradientCpu.getElement<half>({(uint64_t)o, (uint64_t)i, (uint64_t)h, (uint64_t)w});
                        float gpuVal = weightsGradientGpu_h.getElement<half>({(uint64_t)o, (uint64_t)i, (uint64_t)h, (uint64_t)w});
                        float thresh = batchSize * 0.1 + abs(cpuVal * 0.005);
                        EXPECT_LT(abs(cpuVal - gpuVal), thresh);
                        if (abs(cpuVal - gpuVal) >= thresh)
                            printf("%f %f   at [%ld, %ld, %ld, %ld]\n", cpuVal, gpuVal, o, i, h, w);
                    }
                }
            }
        }
    }
}

TEST(GpuConvolution, ConvolutionBackwardFilterProducesCorrectResult_NoAccumulate) { backwardFilterTest(false); }

TEST(GpuConvolution, ConvolutionBackwardFilterProducesCorrectResult_WithAccumulate) { backwardFilterTest(true); }

TEST(GpuConvolution, ConvolutionBackwardDataProducesCorrectResult) {
    Stream stream(0);

    for (uint64_t t = 0; t < 5; ++t) {
        uint64_t numInputColumns = (rand() % 75) + 1;
        uint64_t numInputRows = (rand() % 75) + 1;
        uint64_t filterWidth = (rand() % numInputColumns) + 1;
        uint64_t filterHeight = (rand() % numInputRows) + 1;

        uint64_t filterHorizontalStride = (rand() % numInputColumns) + 1;
        uint64_t filterVerticalStride = (rand() % numInputRows) + 1;
        uint64_t leftAndRightPadWidth = (rand() % 20) + 1;
        uint64_t topAndBottomPadHeight = (rand() % 20) + 1;
        uint64_t numFeatureInputChannels = (rand() % 10) + 1;
        uint64_t numFeatureOutputChannels = (rand() % 10) + 1;
        uint64_t batchSize = (rand() % 10) + 1;

        ConvolutionKernelRequirement convolutionKernelRequirement(MachineEvaluator::instance().getGpuType(0),
                                                                  filterWidth,
                                                                  filterHeight,
                                                                  filterHorizontalStride,
                                                                  filterVerticalStride,
                                                                  leftAndRightPadWidth,
                                                                  topAndBottomPadHeight,
                                                                  numFeatureInputChannels,
                                                                  numFeatureOutputChannels,
                                                                  batchSize,
                                                                  numInputColumns,
                                                                  numInputRows);
        uint64_t numOutputRows = convolutionKernelRequirement.getNumOutputRows();
        uint64_t numOutputColumns = convolutionKernelRequirement.getNumOutputColumns();

        assert(convolutionKernelRequirement.getNumOutputColumns() ==
               ConvolutionTestHelper::computeOutputDimensionSize(convolutionKernelRequirement.getNumInputColumns(),
                                                                 convolutionKernelRequirement.getLeftAndRightPadWidth(),
                                                                 convolutionKernelRequirement.getFilterWidth(),
                                                                 convolutionKernelRequirement.getFilterHorizontalStride()));
        assert(convolutionKernelRequirement.getNumOutputRows() ==
               ConvolutionTestHelper::computeOutputDimensionSize(convolutionKernelRequirement.getNumInputRows(),
                                                                 convolutionKernelRequirement.getTopAndBottomPadHeight(),
                                                                 convolutionKernelRequirement.getFilterHeight(),
                                                                 convolutionKernelRequirement.getFilterVerticalStride()));

        // Allocate tensors
        TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
        TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

        TensorDescriptor errorInputDescriptor(TensorDescriptor::DataType::FP16,
                                              {batchSize, numFeatureOutputChannels, numOutputRows, numOutputColumns});
        Tensor errorInputCpu(cpuPlacement, errorInputDescriptor);
        Tensor errorInputGpu(gpuPlacement, errorInputDescriptor);

        TensorDescriptor weightsDescriptor(TensorDescriptor::DataType::FP16,
                                           {numFeatureOutputChannels, numFeatureInputChannels, filterHeight, filterWidth});
        Tensor weightsCpu(cpuPlacement, weightsDescriptor);
        Tensor weightsGpu(gpuPlacement, weightsDescriptor);

        TensorDescriptor errorOutputDescriptor(TensorDescriptor::DataType::FP16,
                                               {batchSize, numFeatureInputChannels, numInputRows, numInputColumns});
        Tensor errorOutputCpu(cpuPlacement, errorOutputDescriptor);
        Tensor errorOutputGpu(gpuPlacement, errorOutputDescriptor);
        Tensor errorOutputGpu_h(cpuPlacement, errorOutputDescriptor);

        // Fill input tensors
        uint64_t totalNumErrorInputElements = errorInputDescriptor.getTotalNumElements();
        half *errorInputMem = (half *)errorInputCpu.getMemPtr();
        for (uint64_t i = 0; i < totalNumErrorInputElements; ++i) {
            float val = (rand() % 100) / 10.0f;
            errorInputMem[i] = (half)(val - 5.0f);
        }
        errorInputGpu.copyFromAsync(errorInputCpu, stream);

        uint64_t totalNumWeightss = weightsDescriptor.getTotalNumElements();
        half *weightsMem = (half *)weightsCpu.getMemPtr();
        for (uint64_t i = 0; i < totalNumWeightss; ++i) {
            float val = (rand() % 100) / 10.0f;
            weightsMem[i] = (half)(val - 5.0f);
        }
        weightsGpu.copyFromAsync(weightsCpu, stream);

        // Perform convolution on GPU and CPU
        GpuConvolution::instance().chooseOptimalKernelBackward(convolutionKernelRequirement, stream);

        uint64_t workspaceSizeInBytes = GpuConvolution::instance().getBackwardDataWorkspaceSizeInBytes(convolutionKernelRequirement);
        Optional<Tensor> workspace;
        if (workspaceSizeInBytes != 0)
            workspace = Tensor(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::UINT8, {workspaceSizeInBytes}));

        GpuConvolution::instance().convolutionBackwardData(
            convolutionKernelRequirement, errorInputGpu, weightsGpu, errorOutputGpu, workspace, stream);

        ConvolutionTestHelper::cpuConvolutionBackwardData(errorInputCpu, weightsCpu, errorOutputCpu, convolutionKernelRequirement);

        errorOutputGpu_h.copyFromAsync(errorOutputGpu, stream);
        stream.synchronize();

        // Verify CPU and GPU results match
        for (uint64_t n = 0; n < errorOutputDescriptor.getDimensions()[0]; ++n) {
            for (uint64_t c = 0; c < errorOutputDescriptor.getDimensions()[1]; ++c) {
                for (uint64_t h = 0; h < errorOutputDescriptor.getDimensions()[2]; ++h) {
                    for (uint64_t w = 0; w < errorOutputDescriptor.getDimensions()[3]; ++w) {
                        float cpuVal = errorOutputCpu.getElement<half>({(uint64_t)n, (uint64_t)c, (uint64_t)h, (uint64_t)w});
                        float gpuVal = errorOutputGpu_h.getElement<half>({(uint64_t)n, (uint64_t)c, (uint64_t)h, (uint64_t)w});
                        float thresh = batchSize * 0.1 + abs(cpuVal * 0.005);
                        EXPECT_LT(abs(cpuVal - gpuVal), thresh);
                        if (abs(cpuVal - gpuVal) >= thresh)
                            printf("%f %f   at [%ld, %ld, %ld, %ld]\n", cpuVal, gpuVal, n, c, h, w);
                    }
                }
            }
        }
    }
}
