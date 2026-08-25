#pragma once
#include <cassert>
#include <cstdint>
#include <optional>
#include <vector>

#include <cuda_fp16.h>
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "omp.h"

namespace ThorImplementation {

/**
 * Pure-value convolution geometry used only by CPU reference tests.
 *
 * This intentionally owns no cuDNN descriptors or other accelerator execution
 * state. The retired classic-cuDNN convolution subsystem must not be kept alive by
 * test-only geometry plumbing.
 */
class ConvolutionTestRequirement {
   public:
    ConvolutionTestRequirement(int filterWidth,
                               int filterHeight,
                               int filterHorizontalStride,
                               int filterVerticalStride,
                               int leftAndRightPadWidth,
                               int topAndBottomPadHeight,
                               int numInputChannels,
                               int numOutputChannels,
                               int batchSize,
                               int numInputColumns,
                               int numInputRows)
        : filterWidth(filterWidth),
          filterHeight(filterHeight),
          filterHorizontalStride(filterHorizontalStride),
          filterVerticalStride(filterVerticalStride),
          leftAndRightPadWidth(leftAndRightPadWidth),
          topAndBottomPadHeight(topAndBottomPadHeight),
          numInputChannels(numInputChannels),
          numOutputChannels(numOutputChannels),
          batchSize(batchSize),
          numInputColumns(numInputColumns),
          numInputRows(numInputRows),
          numOutputColumns(computeOutputDimension(numInputColumns, leftAndRightPadWidth, filterWidth, filterHorizontalStride)),
          numOutputRows(computeOutputDimension(numInputRows, topAndBottomPadHeight, filterHeight, filterVerticalStride)) {
        assert(filterWidth > 0);
        assert(filterHeight > 0);
        assert(filterHorizontalStride > 0);
        assert(filterVerticalStride > 0);
        assert(leftAndRightPadWidth >= 0);
        assert(topAndBottomPadHeight >= 0);
        assert(numInputChannels > 0);
        assert(numOutputChannels > 0);
        assert(batchSize > 0);
        assert(numInputColumns > 0);
        assert(numInputRows > 0);
    }

    int getFilterWidth() const { return filterWidth; }
    int getFilterHeight() const { return filterHeight; }
    int getFilterHorizontalStride() const { return filterHorizontalStride; }
    int getFilterVerticalStride() const { return filterVerticalStride; }
    int getLeftAndRightPadWidth() const { return leftAndRightPadWidth; }
    int getTopAndBottomPadHeight() const { return topAndBottomPadHeight; }
    int getNumInputChannels() const { return numInputChannels; }
    int getNumOutputChannels() const { return numOutputChannels; }
    int getBatchSize() const { return batchSize; }
    int getNumInputColumns() const { return numInputColumns; }
    int getNumInputRows() const { return numInputRows; }
    int getNumOutputColumns() const { return numOutputColumns; }
    int getNumOutputRows() const { return numOutputRows; }

   private:
    static int computeOutputDimension(int inputSize, int perSidePadding, int filterSize, int stride) {
        assert(inputSize > 0);
        assert(perSidePadding >= 0);
        assert(filterSize > 0);
        assert(stride > 0);
        const int paddedSize = inputSize + 2 * perSidePadding;
        assert(filterSize <= paddedSize);
        return 1 + ((paddedSize - filterSize) / stride);
    }

    int filterWidth;
    int filterHeight;
    int filterHorizontalStride;
    int filterVerticalStride;
    int leftAndRightPadWidth;
    int topAndBottomPadHeight;
    int numInputChannels;
    int numOutputChannels;
    int batchSize;
    int numInputColumns;
    int numInputRows;
    int numOutputColumns;
    int numOutputRows;
};

class ConvolutionTestHelper {
   public:
    static int computeOutputDimensionSize(int inputDimensionSize, int perSidePadding, int filterSize, int filterStride) {
        int paddedSize = inputDimensionSize + 2 * perSidePadding;
        assert(filterSize <= paddedSize);
        int outputSize = 1 + ((paddedSize - filterSize) / filterStride);
        assert(outputSize > 0);
        return outputSize;
    }

    static inline half getWeightElement(
        Tensor weights, uint64_t outputChannel, uint64_t inputChannel, uint64_t filterRow, uint64_t filterCol) {
        // Match cuDNN's deep-learning convention: convolution layers use cross-correlation,
        // so the filter is addressed in natural KCRS order.
        std::vector<unsigned long> weightIndex{(uint64_t)outputChannel, (uint64_t)inputChannel, (uint64_t)filterRow, (uint64_t)filterCol};
        return weights.getElement<half>({weightIndex});
    }

    static void cpuConvolutionForward(Tensor inputFeatures,
                                      Tensor weights,
                                      std::optional<Tensor> bias,
                                      Tensor outputFeatures,
                                      ConvolutionTestRequirement convolutionTestRequirement) {
        // Validate input tensor
        assert(inputFeatures.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU);
        std::vector<unsigned long> inputTensorDimensions = inputFeatures.getDescriptor().getDimensions();
        assert(inputTensorDimensions.size() == 4);
        assert(inputTensorDimensions[0] == (unsigned long)convolutionTestRequirement.getBatchSize());
        assert(inputTensorDimensions[1] == (unsigned long)convolutionTestRequirement.getNumInputChannels());
        assert(inputTensorDimensions[2] == (unsigned long)convolutionTestRequirement.getNumInputRows());
        assert(inputTensorDimensions[3] == (unsigned long)convolutionTestRequirement.getNumInputColumns());

        // Validate output tensor
        assert(outputFeatures.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU);
        std::vector<unsigned long> outputTensorDimensions = outputFeatures.getDescriptor().getDimensions();
        assert(outputTensorDimensions.size() == 4);
        assert(outputTensorDimensions[0] == (unsigned long)convolutionTestRequirement.getBatchSize());
        assert(outputTensorDimensions[1] == (unsigned long)convolutionTestRequirement.getNumOutputChannels());
        assert(outputTensorDimensions[2] == (unsigned long)convolutionTestRequirement.getNumOutputRows());
        assert(outputTensorDimensions[3] == (unsigned long)convolutionTestRequirement.getNumOutputColumns());

        // Validate weights tensor
        assert(weights.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU);
        std::vector<unsigned long> weightsDimensions = weights.getDescriptor().getDimensions();
        assert(weightsDimensions.size() == 4);
        assert(weightsDimensions[0] == (unsigned long)convolutionTestRequirement.getNumOutputChannels());
        assert(weightsDimensions[1] == (unsigned long)convolutionTestRequirement.getNumInputChannels());
        assert(weightsDimensions[2] == (unsigned long)convolutionTestRequirement.getFilterHeight());
        assert(weightsDimensions[3] == (unsigned long)convolutionTestRequirement.getFilterWidth());

        int imageRows = convolutionTestRequirement.getNumInputRows();
        int imageCols = convolutionTestRequirement.getNumInputColumns();
        int filterHeight = convolutionTestRequirement.getFilterHeight();
        int filterWidth = convolutionTestRequirement.getFilterWidth();
        int verticalPadding = convolutionTestRequirement.getTopAndBottomPadHeight();
        int horizontalPadding = convolutionTestRequirement.getLeftAndRightPadWidth();
        int verticalStride = convolutionTestRequirement.getFilterVerticalStride();
        int horizontalStride = convolutionTestRequirement.getFilterHorizontalStride();
        int inputChannels = convolutionTestRequirement.getNumInputChannels();
        int outputChannels = convolutionTestRequirement.getNumOutputChannels();
        int batchSize = convolutionTestRequirement.getBatchSize();

        if (omp_get_num_procs() > 1)
            omp_set_num_threads(omp_get_num_procs() - 1);

// Iterate over each item in the batch
#pragma omp parallel for schedule(static, 1)
        for (int batch = 0; batch < batchSize; ++batch) {
            // Iterate over the image, applying a filter on each iteration
            for (int outputChannel = 0; outputChannel < outputChannels; ++outputChannel) {
                for (int imageRow = -verticalPadding; imageRow + filterHeight <= imageRows + verticalPadding; imageRow += verticalStride) {
                    for (int imageCol = -horizontalPadding; imageCol + filterWidth <= imageCols + horizontalPadding;
                         imageCol += horizontalStride) {
                        // Apply filter to the inputs at this location to compute the output for this channel
                        float accum = 0.0f;
                        for (int filterRow = 0; filterRow < filterHeight; ++filterRow) {
                            for (int filterCol = 0; filterCol < filterWidth; ++filterCol) {
                                for (int inputChannel = 0; inputChannel < inputChannels; ++inputChannel) {
                                    float element;
                                    float weight;
                                    if (imageRow + filterRow < 0 || imageRow + filterRow >= imageRows || imageCol + filterCol < 0 ||
                                        imageCol + filterCol >= imageCols) {
                                        element = 0.0f;
                                        weight = 0.0f;
                                    } else {
                                        std::vector<unsigned long> inputImageIndex{(uint64_t)batch,
                                                                                   (uint64_t)inputChannel,
                                                                                   (uint64_t)imageRow + filterRow,
                                                                                   (uint64_t)imageCol + filterCol};
                                        element = inputFeatures.getElement<half>({inputImageIndex});
                                        weight = getWeightElement(weights,
                                                                  (uint64_t)outputChannel,
                                                                  (uint64_t)inputChannel,
                                                                  (uint64_t)filterRow,
                                                                  (uint64_t)filterCol);
                                    }
                                    accum += element * weight;
                                }
                            }
                        }

                        if (bias.has_value())
                            accum += (float)bias.value().getElement<half>({(uint64_t)outputChannel});

                        std::vector<unsigned long> outputImageIndex{(uint64_t)batch,
                                                                    (uint64_t)outputChannel,
                                                                    (uint64_t)(imageRow + verticalPadding) / verticalStride,
                                                                    (uint64_t)(imageCol + horizontalPadding) / horizontalStride};
                        outputFeatures.setElement<half>(outputImageIndex, accum);
                    }
                }
            }
        }
    }

    static void cpuConvolutionBackwardFilter(Tensor featureInput,
                                             Tensor errorInput,
                                             Tensor weightsGradient,
                                             ConvolutionTestRequirement convolutionTestRequirement,
                                             bool accumulate) {
        Stream copyStream(0);

        // Validate feature input tensor
        assert(featureInput.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU);
        std::vector<unsigned long> featureInputTensorDimensions = featureInput.getDescriptor().getDimensions();
        assert(featureInputTensorDimensions.size() == 4);
        assert(featureInputTensorDimensions[0] == (unsigned long)convolutionTestRequirement.getBatchSize());
        assert(featureInputTensorDimensions[1] == (unsigned long)convolutionTestRequirement.getNumInputChannels());
        assert(featureInputTensorDimensions[2] == (unsigned long)convolutionTestRequirement.getNumInputRows());
        assert(featureInputTensorDimensions[3] == (unsigned long)convolutionTestRequirement.getNumInputColumns());

        // Validate error input tensor
        assert(errorInput.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU);
        std::vector<unsigned long> errorInputTensorDimensions = errorInput.getDescriptor().getDimensions();
        assert(errorInputTensorDimensions.size() == 4);
        assert(errorInputTensorDimensions[0] == (unsigned long)convolutionTestRequirement.getBatchSize());
        assert(errorInputTensorDimensions[1] == (unsigned long)convolutionTestRequirement.getNumOutputChannels());
        assert(errorInputTensorDimensions[2] == (unsigned long)convolutionTestRequirement.getNumOutputRows());
        assert(errorInputTensorDimensions[3] == (unsigned long)convolutionTestRequirement.getNumOutputColumns());

        // Validate weightsGradient gradient tensor
        assert(weightsGradient.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU);
        std::vector<unsigned long> weightsGradientDimensions = weightsGradient.getDescriptor().getDimensions();
        assert(weightsGradientDimensions.size() == 4);
        assert(weightsGradientDimensions[0] == (unsigned long)convolutionTestRequirement.getNumOutputChannels());
        assert(weightsGradientDimensions[1] == (unsigned long)convolutionTestRequirement.getNumInputChannels());
        assert(weightsGradientDimensions[2] == (unsigned long)convolutionTestRequirement.getFilterHeight());
        assert(weightsGradientDimensions[3] == (unsigned long)convolutionTestRequirement.getFilterWidth());
        TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
        Tensor weightsGradientFloat(cpuPlacement,
                                    TensorDescriptor(DataType::FP32, weightsGradient.getDescriptor().getDimensions()));
        if (accumulate) {
            weightsGradientFloat.copyFromAsync(weightsGradient, copyStream);
            copyStream.synchronize();
        } else {
            memset(weightsGradientFloat.getMemPtr(), 0, sizeof(float) * weightsGradient.getDescriptor().getTotalNumElements());
        }

        unsigned int imageRows = convolutionTestRequirement.getNumInputRows();
        unsigned int imageCols = convolutionTestRequirement.getNumInputColumns();
        unsigned int filterHeight = convolutionTestRequirement.getFilterHeight();
        unsigned int filterWidth = convolutionTestRequirement.getFilterWidth();
        unsigned int verticalPadding = convolutionTestRequirement.getTopAndBottomPadHeight();
        unsigned int horizontalPadding = convolutionTestRequirement.getLeftAndRightPadWidth();
        unsigned int verticalStride = convolutionTestRequirement.getFilterVerticalStride();
        unsigned int horizontalStride = convolutionTestRequirement.getFilterHorizontalStride();
        unsigned int inputChannels = convolutionTestRequirement.getNumInputChannels();
        unsigned int outputChannels = convolutionTestRequirement.getNumOutputChannels();
        unsigned int batchSize = convolutionTestRequirement.getBatchSize();
        unsigned int errorInputHeight = errorInput.getDescriptor().getDimensions()[2];
        unsigned int errorInputWidth = errorInput.getDescriptor().getDimensions()[3];

        // The weights gradient is a convolution over the feature input with a filter of the error input ish
        std::vector<unsigned long> featureInputDimensionsWithPadding = featureInput.getDescriptor().getDimensions();
        featureInputDimensionsWithPadding[2] += 2 * verticalPadding;
        featureInputDimensionsWithPadding[3] += 2 * horizontalPadding;
        Tensor featureInputPadded(cpuPlacement, TensorDescriptor(DataType::FP16, featureInputDimensionsWithPadding));
        for (unsigned int n = 0; n < featureInputDimensionsWithPadding[0]; ++n) {
            for (unsigned int c = 0; c < featureInputDimensionsWithPadding[1]; ++c) {
                for (unsigned int h = 0; h < featureInputDimensionsWithPadding[2]; ++h) {
                    for (unsigned int w = 0; w < featureInputDimensionsWithPadding[3]; ++w) {
                        if (h < verticalPadding || h >= imageRows + verticalPadding || w < horizontalPadding ||
                            w >= imageCols + horizontalPadding) {
                            featureInputPadded.setElement<half>({n, c, h, w}, half(0.0f));
                        } else {
                            half nonPaddedElement = featureInput.getElement<half>({n, c, h - verticalPadding, w - horizontalPadding});
                            featureInputPadded.setElement<half>({n, c, h, w}, nonPaddedElement);
                        }
                    }
                }
            }
        }

        if (omp_get_num_procs() > 1)
            omp_set_num_threads(omp_get_num_procs() - 1);

        // Iterate over each item in the batch
        for (unsigned int batch = 0; batch < batchSize; ++batch) {
// Iterate over the image, applying a filter on each iteration
#pragma omp parallel for schedule(static, 1)
            for (unsigned int outputChannel = 0; outputChannel < outputChannels; ++outputChannel) {
                for (unsigned int errorInputRow = 0; errorInputRow < errorInputHeight; ++errorInputRow) {
                    for (unsigned int errorInputCol = 0; errorInputCol < errorInputWidth; ++errorInputCol) {
                        // Apply filter to the inputs at this location to compute the output for this channel
                        for (unsigned int filterRow = 0; filterRow < filterHeight; ++filterRow) {
                            for (unsigned int filterCol = 0; filterCol < filterWidth; ++filterCol) {
                                for (unsigned int inputChannel = 0; inputChannel < inputChannels; ++inputChannel) {
                                    float featureElement =
                                        featureInputPadded.getElement<half>({batch,
                                                                             inputChannel,
                                                                             verticalStride * errorInputRow + filterRow,
                                                                             horizontalStride * errorInputCol + filterCol});
                                    float errorElement = errorInput.getElement<half>({batch, outputChannel, errorInputRow, errorInputCol});
                                    float weightsGradientElement =
                                        weightsGradientFloat.getElement<float>({outputChannel, inputChannel, filterRow, filterCol});
                                    weightsGradientElement += featureElement * errorElement;
                                    weightsGradientFloat.setElement<float>({outputChannel, inputChannel, filterRow, filterCol},
                                                                           weightsGradientElement);
                                }
                            }
                        }
                    }
                }
            }
        }
        weightsGradient.copyFromAsync(weightsGradientFloat, copyStream);
        copyStream.synchronize();
    }

    static void cpuConvolutionBackwardData(Tensor errorInput,
                                           Tensor weights,
                                           Tensor errorOutput,
                                           ConvolutionTestRequirement convolutionTestRequirement) {
        Stream copyStream(0);

        // Validate error input tensor
        assert(errorInput.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU);
        std::vector<unsigned long> errorInputTensorDimensions = errorInput.getDescriptor().getDimensions();
        assert(errorInputTensorDimensions.size() == 4);
        assert(errorInputTensorDimensions[0] == (unsigned long)convolutionTestRequirement.getBatchSize());
        assert(errorInputTensorDimensions[1] == (unsigned long)convolutionTestRequirement.getNumOutputChannels());
        assert(errorInputTensorDimensions[2] == (unsigned long)convolutionTestRequirement.getNumOutputRows());
        assert(errorInputTensorDimensions[3] == (unsigned long)convolutionTestRequirement.getNumOutputColumns());

        // Validate weights gradient tensor
        assert(weights.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU);
        std::vector<unsigned long> weightsDimensions = weights.getDescriptor().getDimensions();
        assert(weightsDimensions.size() == 4);
        assert(weightsDimensions[0] == (unsigned long)convolutionTestRequirement.getNumOutputChannels());
        assert(weightsDimensions[1] == (unsigned long)convolutionTestRequirement.getNumInputChannels());
        assert(weightsDimensions[2] == (unsigned long)convolutionTestRequirement.getFilterHeight());
        assert(weightsDimensions[3] == (unsigned long)convolutionTestRequirement.getFilterWidth());

        // Validate error output tensor
        assert(errorOutput.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU);
        std::vector<unsigned long> errorOutputTensorDimensions = errorOutput.getDescriptor().getDimensions();
        assert(errorOutputTensorDimensions.size() == 4);
        assert(errorOutputTensorDimensions[0] == (unsigned long)convolutionTestRequirement.getBatchSize());
        assert(errorOutputTensorDimensions[1] == (unsigned long)convolutionTestRequirement.getNumInputChannels());
        assert(errorOutputTensorDimensions[2] == (unsigned long)convolutionTestRequirement.getNumInputRows());
        assert(errorOutputTensorDimensions[3] == (unsigned long)convolutionTestRequirement.getNumInputColumns());
        TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
        Tensor errorOutputFloat =
            Tensor(cpuPlacement, TensorDescriptor(DataType::FP32, errorOutput.getDescriptor().getDimensions()));
        memset(errorOutputFloat.getMemPtr(), 0, sizeof(float) * errorOutputFloat.getDescriptor().getTotalNumElements());

        unsigned int filterHeight = convolutionTestRequirement.getFilterHeight();
        unsigned int filterWidth = convolutionTestRequirement.getFilterWidth();
        unsigned int verticalStride = convolutionTestRequirement.getFilterVerticalStride();
        unsigned int horizontalStride = convolutionTestRequirement.getFilterHorizontalStride();
        unsigned int verticalPadding = convolutionTestRequirement.getTopAndBottomPadHeight();
        unsigned int horizontalPadding = convolutionTestRequirement.getLeftAndRightPadWidth();

        if (omp_get_num_procs() > 1)
            omp_set_num_threads(omp_get_num_procs() - 1);

#pragma omp parallel for schedule(static, 1)
        for (unsigned int n = 0; n < errorOutputTensorDimensions[0]; ++n) {
            for (unsigned int c = 0; c < errorOutputTensorDimensions[1]; ++c) {
                for (unsigned int h = 0; h < errorOutputTensorDimensions[2]; ++h) {
                    for (unsigned int w = 0; w < errorOutputTensorDimensions[3]; ++w) {
                        for (int f = 0; f < convolutionTestRequirement.getNumOutputChannels(); ++f) {
                            for (int k = 0; k < convolutionTestRequirement.getNumOutputRows(); ++k) {
                                for (int l = 0; l < convolutionTestRequirement.getNumOutputColumns(); ++l) {
                                    for (int p = 0; p < convolutionTestRequirement.getFilterHeight(); ++p) {
                                        for (int q = 0; q < convolutionTestRequirement.getFilterWidth(); ++q) {
                                            if ((p + k * verticalStride == h + verticalPadding) &&
                                                (q + horizontalStride * l == w + horizontalPadding)) {
                                                float *errorOutputElement = errorOutputFloat.getElementPointer<float>(
                                                    {(uint64_t)n, (uint64_t)c, (uint64_t)h, (uint64_t)w});
                                                float errorInputElement =
                                                    errorInput.getElement<half>({(uint64_t)n, (uint64_t)f, (uint64_t)k, (uint64_t)l});
                                                float weightElement =
                                                    weights.getElement<half>({(uint64_t)f, (uint64_t)c, (uint64_t)p, (uint64_t)q});
                                                *errorOutputElement += errorInputElement * weightElement;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        errorOutput.copyFromAsync(errorOutputFloat, copyStream);
        copyStream.synchronize();
    }

    static void cpuConvolutionBackwardBias(Tensor errorInput, Tensor biasesGradient, bool accumulate) {
        std::vector<unsigned long> errorInputDimensions = errorInput.getDescriptor().getDimensions();
        unsigned int n = errorInputDimensions[0];
        unsigned int c = errorInputDimensions[1];
        unsigned int h = errorInputDimensions[2];
        unsigned int w = errorInputDimensions[3];

        std::vector<unsigned long> biasesGradientDimensions = biasesGradient.getDescriptor().getDimensions();
        assert(biasesGradientDimensions.size() == 1);
        assert(biasesGradientDimensions[0] == c);

        TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
        Tensor biasesGradientFloat(cpuPlacement,
                                   TensorDescriptor(DataType::FP32, biasesGradient.getDescriptor().getDimensions()));
        float *biasesGradientFloatMem = (float *)biasesGradientFloat.getMemPtr();
        if (accumulate) {
            Stream stream(0);
            biasesGradientFloat.copyFromAsync(biasesGradient, stream);
            stream.synchronize();
        } else {
            memset(biasesGradientFloatMem, 0, sizeof(float) * biasesGradientFloat.getDescriptor().getTotalNumElements());
        }

        if (omp_get_num_procs() > 1)
            omp_set_num_threads(omp_get_num_procs() - 1);

#pragma omp parallel for schedule(static, 1)
        for (unsigned int channel = 0; channel < c; ++channel) {
            for (unsigned int batchItem = 0; batchItem < n; ++batchItem) {
                for (unsigned int height = 0; height < h; ++height) {
                    for (unsigned int width = 0; width < w; ++width) {
                        half errorElement = errorInput.getElement<half>({batchItem, channel, height, width});
                        biasesGradientFloatMem[channel] += (float)errorElement;
                    }
                }
            }
        }
        Stream copyStream(0);
        biasesGradient.copyFromAsync(biasesGradientFloat, copyStream);
        copyStream.synchronize();
    }
};

}  // namespace ThorImplementation
