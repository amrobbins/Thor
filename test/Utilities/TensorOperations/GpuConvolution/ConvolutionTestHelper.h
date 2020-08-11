#pragma once

#include "omp.h"

class ConvolutionTestHelper {
   public:
    static int computeOutputDimensionSize(int inputDimensionSize, int perSidePadding, int filterSize, int filterStride) {
        int paddedSize = inputDimensionSize + 2 * perSidePadding;
        assert(filterSize <= paddedSize);
        int outputSize = 1 + (paddedSize - filterSize) / filterStride;
        assert(outputSize > 0);
        return outputSize;
    }

    static inline half getWeightElement(
        Tensor weights, uint64_t outputChannel, uint64_t inputChannel, uint64_t filterRow, uint64_t filterCol) {
        vector<uint64_t> weightDimensions = weights.getDescriptor().getDimensions();
        uint64_t filterHeight = weightDimensions[2];
        uint64_t filterWidth = weightDimensions[3];

        // Note: From below, it appears that cuDNN sets the bottom right of the filter as (0,0).
        // Seems odd, but the math matches up that way.
        vector<unsigned long> weightIndex{(uint64_t)outputChannel,
                                          (uint64_t)inputChannel,
                                          (uint64_t)((filterHeight - 1) - filterRow),
                                          (uint64_t)((filterWidth - 1) - filterCol)};
        return *((half *)weights.getElement({weightIndex}));
    }

    static void cpuConvolutionForward(Tensor inputFeatures,
                                      Tensor weights,
                                      Optional<Tensor> bias,
                                      Tensor outputFeatures,
                                      ConvolutionKernelRequirement convolutionKernelRequirement) {
        // Validate input tensor
        assert(inputFeatures.getPlacement() == TensorPlacement::MemDevices::CPU);
        vector<unsigned long> inputTensorDimensions = inputFeatures.getDescriptor().getDimensions();
        assert(inputTensorDimensions.size() == 4);
        assert(inputTensorDimensions[0] == (unsigned long)convolutionKernelRequirement.getBatchSize());
        assert(inputTensorDimensions[1] == (unsigned long)convolutionKernelRequirement.getNumInputChannels());
        assert(inputTensorDimensions[2] == (unsigned long)convolutionKernelRequirement.getNumInputRows());
        assert(inputTensorDimensions[3] == (unsigned long)convolutionKernelRequirement.getNumInputColumns());

        // Validate output tensor
        assert(outputFeatures.getPlacement() == TensorPlacement::MemDevices::CPU);
        vector<unsigned long> outputTensorDimensions = outputFeatures.getDescriptor().getDimensions();
        assert(outputTensorDimensions.size() == 4);
        assert(outputTensorDimensions[0] == (unsigned long)convolutionKernelRequirement.getBatchSize());
        assert(outputTensorDimensions[1] == (unsigned long)convolutionKernelRequirement.getNumOutputChannels());
        assert(outputTensorDimensions[2] == (unsigned long)convolutionKernelRequirement.getNumOutputRows());
        assert(outputTensorDimensions[3] == (unsigned long)convolutionKernelRequirement.getNumOutputColumns());

        // Validate weights tensor
        assert(weights.getPlacement() == TensorPlacement::MemDevices::CPU);
        vector<unsigned long> weightsDimensions = weights.getDescriptor().getDimensions();
        assert(weightsDimensions.size() == 4);
        assert(weightsDimensions[0] == (unsigned long)convolutionKernelRequirement.getNumOutputChannels());
        assert(weightsDimensions[1] == (unsigned long)convolutionKernelRequirement.getNumInputChannels());
        assert(weightsDimensions[2] == (unsigned long)convolutionKernelRequirement.getFilterHeight());
        assert(weightsDimensions[3] == (unsigned long)convolutionKernelRequirement.getFilterWidth());

        int imageRows = convolutionKernelRequirement.getNumInputRows();
        int imageCols = convolutionKernelRequirement.getNumInputColumns();
        int filterHeight = convolutionKernelRequirement.getFilterHeight();
        int filterWidth = convolutionKernelRequirement.getFilterWidth();
        int verticalPadding = convolutionKernelRequirement.getTopAndBottomPadHeight();
        int horizontalPadding = convolutionKernelRequirement.getLeftAndRightPadWidth();
        int verticalStride = convolutionKernelRequirement.getFilterVerticalStride();
        int horizontalStride = convolutionKernelRequirement.getFilterHorizontalStride();
        int inputChannels = convolutionKernelRequirement.getNumInputChannels();
        int outputChannels = convolutionKernelRequirement.getNumOutputChannels();
        int batchSize = convolutionKernelRequirement.getBatchSize();

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
                                        vector<unsigned long> inputImageIndex{(uint64_t)batch,
                                                                              (uint64_t)inputChannel,
                                                                              (uint64_t)imageRow + filterRow,
                                                                              (uint64_t)imageCol + filterCol};
                                        element = *((half *)inputFeatures.getElement({inputImageIndex}));
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

                        if (bias.isPresent())
                            accum += *((half *)bias.get().getElement({(uint64_t)outputChannel}));

                        vector<unsigned long> outputImageIndex{(uint64_t)batch,
                                                               (uint64_t)outputChannel,
                                                               (uint64_t)(imageRow + verticalPadding) / verticalStride,
                                                               (uint64_t)(imageCol + horizontalPadding) / horizontalStride};
                        half *outputElement = (half *)outputFeatures.getElement({outputImageIndex});
                        *outputElement = (half)accum;
                    }
                }
            }
        }
    }

    static void cpuConvolutionBackwardFilter(Tensor featureInput,
                                             Tensor errorInput,
                                             Tensor weightsGradient,
                                             ConvolutionKernelRequirement convolutionKernelRequirement,
                                             bool accumulate) {
        Stream copyStream(0);

        // Validate feature input tensor
        assert(featureInput.getPlacement() == TensorPlacement::MemDevices::CPU);
        vector<unsigned long> featureInputTensorDimensions = featureInput.getDescriptor().getDimensions();
        assert(featureInputTensorDimensions.size() == 4);
        assert(featureInputTensorDimensions[0] == (unsigned long)convolutionKernelRequirement.getBatchSize());
        assert(featureInputTensorDimensions[1] == (unsigned long)convolutionKernelRequirement.getNumInputChannels());
        assert(featureInputTensorDimensions[2] == (unsigned long)convolutionKernelRequirement.getNumInputRows());
        assert(featureInputTensorDimensions[3] == (unsigned long)convolutionKernelRequirement.getNumInputColumns());

        // Validate error input tensor
        assert(errorInput.getPlacement() == TensorPlacement::MemDevices::CPU);
        vector<unsigned long> errorInputTensorDimensions = errorInput.getDescriptor().getDimensions();
        assert(errorInputTensorDimensions.size() == 4);
        assert(errorInputTensorDimensions[0] == (unsigned long)convolutionKernelRequirement.getBatchSize());
        assert(errorInputTensorDimensions[1] == (unsigned long)convolutionKernelRequirement.getNumOutputChannels());
        assert(errorInputTensorDimensions[2] == (unsigned long)convolutionKernelRequirement.getNumOutputRows());
        assert(errorInputTensorDimensions[3] == (unsigned long)convolutionKernelRequirement.getNumOutputColumns());

        // Validate weightsGradient gradient tensor
        assert(weightsGradient.getPlacement() == TensorPlacement::MemDevices::CPU);
        vector<unsigned long> weightsGradientDimensions = weightsGradient.getDescriptor().getDimensions();
        assert(weightsGradientDimensions.size() == 4);
        assert(weightsGradientDimensions[0] == (unsigned long)convolutionKernelRequirement.getNumOutputChannels());
        assert(weightsGradientDimensions[1] == (unsigned long)convolutionKernelRequirement.getNumInputChannels());
        assert(weightsGradientDimensions[2] == (unsigned long)convolutionKernelRequirement.getFilterHeight());
        assert(weightsGradientDimensions[3] == (unsigned long)convolutionKernelRequirement.getFilterWidth());
        Tensor weightsGradientFloat(TensorPlacement::MemDevices::CPU,
                                    TensorDescriptor(TensorDescriptor::DataType::FP32, weightsGradient.getDescriptor().getDimensions()));
        if (accumulate) {
            weightsGradientFloat.copyFromAsync(weightsGradient, copyStream);
            copyStream.synchronize();
        } else {
            memset(weightsGradientFloat.getMemPtr(), 0, sizeof(float) * weightsGradient.getDescriptor().getTotalNumElements());
        }

        unsigned int imageRows = convolutionKernelRequirement.getNumInputRows();
        unsigned int imageCols = convolutionKernelRequirement.getNumInputColumns();
        unsigned int filterHeight = convolutionKernelRequirement.getFilterHeight();
        unsigned int filterWidth = convolutionKernelRequirement.getFilterWidth();
        unsigned int verticalPadding = convolutionKernelRequirement.getTopAndBottomPadHeight();
        unsigned int horizontalPadding = convolutionKernelRequirement.getLeftAndRightPadWidth();
        unsigned int verticalStride = convolutionKernelRequirement.getFilterVerticalStride();
        unsigned int horizontalStride = convolutionKernelRequirement.getFilterHorizontalStride();
        unsigned int inputChannels = convolutionKernelRequirement.getNumInputChannels();
        unsigned int outputChannels = convolutionKernelRequirement.getNumOutputChannels();
        unsigned int batchSize = convolutionKernelRequirement.getBatchSize();
        unsigned int errorInputHeight = errorInput.getDescriptor().getDimensions()[2];
        unsigned int errorInputWidth = errorInput.getDescriptor().getDimensions()[3];

        // The weights gradient is a convolution over the feature input with a filter of the error input ish
        vector<unsigned long> featureInputDimensionsWithPadding = featureInput.getDescriptor().getDimensions();
        featureInputDimensionsWithPadding[2] += 2 * verticalPadding;
        featureInputDimensionsWithPadding[3] += 2 * horizontalPadding;
        Tensor featureInputPadded(TensorPlacement::MemDevices::CPU,
                                  TensorDescriptor(TensorDescriptor::DataType::FP16, featureInputDimensionsWithPadding));
        for (unsigned int n = 0; n < featureInputDimensionsWithPadding[0]; ++n) {
            for (unsigned int c = 0; c < featureInputDimensionsWithPadding[1]; ++c) {
                for (unsigned int h = 0; h < featureInputDimensionsWithPadding[2]; ++h) {
                    for (unsigned int w = 0; w < featureInputDimensionsWithPadding[3]; ++w) {
                        half *paddedElement = (half *)featureInputPadded.getElement({n, c, h, w});
                        if (h < verticalPadding || h >= imageRows + verticalPadding || w < horizontalPadding ||
                            w >= imageCols + horizontalPadding) {
                            *paddedElement = (half)0.0f;
                        } else {
                            half *nonPaddedElement = (half *)featureInput.getElement({n, c, h - verticalPadding, w - horizontalPadding});
                            *paddedElement = *nonPaddedElement;
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
                                        *(half *)featureInputPadded.getElement({batch,
                                                                                inputChannel,
                                                                                verticalStride * errorInputRow + filterRow,
                                                                                horizontalStride * errorInputCol + filterCol});
                                    float errorElement =
                                        *(half *)errorInput.getElement({batch, outputChannel, errorInputRow, errorInputCol});
                                    float *weightsGradientElement = (float *)weightsGradientFloat.getElement(
                                        {outputChannel, inputChannel, (filterHeight - 1) - filterRow, (filterWidth - 1) - filterCol});
                                    *weightsGradientElement += featureElement * errorElement;
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
                                           ConvolutionKernelRequirement convolutionKernelRequirement) {
        Stream copyStream(0);

        // Validate error input tensor
        assert(errorInput.getPlacement() == TensorPlacement::MemDevices::CPU);
        vector<unsigned long> errorInputTensorDimensions = errorInput.getDescriptor().getDimensions();
        assert(errorInputTensorDimensions.size() == 4);
        assert(errorInputTensorDimensions[0] == (unsigned long)convolutionKernelRequirement.getBatchSize());
        assert(errorInputTensorDimensions[1] == (unsigned long)convolutionKernelRequirement.getNumOutputChannels());
        assert(errorInputTensorDimensions[2] == (unsigned long)convolutionKernelRequirement.getNumOutputRows());
        assert(errorInputTensorDimensions[3] == (unsigned long)convolutionKernelRequirement.getNumOutputColumns());

        // Validate weights gradient tensor
        assert(weights.getPlacement() == TensorPlacement::MemDevices::CPU);
        vector<unsigned long> weightsDimensions = weights.getDescriptor().getDimensions();
        assert(weightsDimensions.size() == 4);
        assert(weightsDimensions[0] == (unsigned long)convolutionKernelRequirement.getNumOutputChannels());
        assert(weightsDimensions[1] == (unsigned long)convolutionKernelRequirement.getNumInputChannels());
        assert(weightsDimensions[2] == (unsigned long)convolutionKernelRequirement.getFilterHeight());
        assert(weightsDimensions[3] == (unsigned long)convolutionKernelRequirement.getFilterWidth());

        // Validate error output tensor
        assert(errorOutput.getPlacement() == TensorPlacement::MemDevices::CPU);
        vector<unsigned long> errorOutputTensorDimensions = errorOutput.getDescriptor().getDimensions();
        assert(errorOutputTensorDimensions.size() == 4);
        assert(errorOutputTensorDimensions[0] == (unsigned long)convolutionKernelRequirement.getBatchSize());
        assert(errorOutputTensorDimensions[1] == (unsigned long)convolutionKernelRequirement.getNumInputChannels());
        assert(errorOutputTensorDimensions[2] == (unsigned long)convolutionKernelRequirement.getNumInputRows());
        assert(errorOutputTensorDimensions[3] == (unsigned long)convolutionKernelRequirement.getNumInputColumns());
        Tensor errorOutputFloat = Tensor(TensorPlacement::MemDevices::CPU,
                                         TensorDescriptor(TensorDescriptor::DataType::FP32, errorOutput.getDescriptor().getDimensions()));
        memset(errorOutputFloat.getMemPtr(), 0, sizeof(float) * errorOutputFloat.getDescriptor().getTotalNumElements());

        unsigned int filterHeight = convolutionKernelRequirement.getFilterHeight();
        unsigned int filterWidth = convolutionKernelRequirement.getFilterWidth();
        unsigned int verticalStride = convolutionKernelRequirement.getFilterVerticalStride();
        unsigned int horizontalStride = convolutionKernelRequirement.getFilterHorizontalStride();
        unsigned int verticalPadding = convolutionKernelRequirement.getTopAndBottomPadHeight();
        unsigned int horizontalPadding = convolutionKernelRequirement.getLeftAndRightPadWidth();

        if (omp_get_num_procs() > 1)
            omp_set_num_threads(omp_get_num_procs() - 1);

#pragma omp parallel for schedule(static, 1)
        for (unsigned int n = 0; n < errorOutputTensorDimensions[0]; ++n) {
            for (unsigned int c = 0; c < errorOutputTensorDimensions[1]; ++c) {
                for (unsigned int h = 0; h < errorOutputTensorDimensions[2]; ++h) {
                    for (unsigned int w = 0; w < errorOutputTensorDimensions[3]; ++w) {
                        for (int f = 0; f < convolutionKernelRequirement.getNumOutputChannels(); ++f) {
                            for (int k = 0; k < convolutionKernelRequirement.getNumOutputRows(); ++k) {
                                for (int l = 0; l < convolutionKernelRequirement.getNumOutputColumns(); ++l) {
                                    for (int p = 0; p < convolutionKernelRequirement.getFilterHeight(); ++p) {
                                        for (int q = 0; q < convolutionKernelRequirement.getFilterWidth(); ++q) {
                                            if ((p + k * verticalStride == h + verticalPadding) &&
                                                (q + horizontalStride * l == w + horizontalPadding)) {
                                                float *errorOutputElement = (float *)errorOutputFloat.getElement(
                                                    {(uint64_t)n, (uint64_t)c, (uint64_t)h, (uint64_t)w});
                                                float errorInputElement =
                                                    *(half *)errorInput.getElement({(uint64_t)n, (uint64_t)f, (uint64_t)k, (uint64_t)l});
                                                float weightElement = *(half *)weights.getElement({(uint64_t)f,
                                                                                                   (uint64_t)c,
                                                                                                   (uint64_t)((filterHeight - 1) - p),
                                                                                                   (uint64_t)((filterWidth - 1) - q)});
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
        vector<unsigned long> errorInputDimensions = errorInput.getDescriptor().getDimensions();
        unsigned int n = errorInputDimensions[0];
        unsigned int c = errorInputDimensions[1];
        unsigned int h = errorInputDimensions[2];
        unsigned int w = errorInputDimensions[3];

        vector<unsigned long> biasesGradientDimensions = biasesGradient.getDescriptor().getDimensions();
        assert(biasesGradientDimensions.size() == 1);
        assert(biasesGradientDimensions[0] == c);

        Tensor biasesGradientFloat(TensorPlacement::MemDevices::CPU,
                                   TensorDescriptor(TensorDescriptor::DataType::FP32, biasesGradient.getDescriptor().getDimensions()));
        float *biasesGradientFloatMem = (float *)biasesGradientFloat.getMemPtr();
        if(accumulate) {
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
                        half errorElement = *((half *)errorInput.getElement({batchItem, channel, height, width}));
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
