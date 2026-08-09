#pragma once

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/SharedOwnership.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cudnn.h>

#include <memory>
#include <string>
#include <utility>

class ConvolutionKernelRequirement;
namespace std {
template <>
struct hash<ConvolutionKernelRequirement>;
}

/**
 * Value-type handle describing one convolution kernel requirement and its
 * shared cuDNN descriptors.
 *
 * Copies share one descriptor-owning State through std::shared_ptr. Distinct
 * handles may therefore be copied, moved, assigned, and destroyed concurrently
 * according to the shared-ownership contract in Utilities/Common/SharedOwnership.h.
 */
class ConvolutionKernelRequirement {
   public:
    ConvolutionKernelRequirement() = delete;

    ConvolutionKernelRequirement(const std::string gpuType,
                                 const int filterWidth,
                                 const int filterHeight,
                                 const int filterHorizontalStride,
                                 const int filterVerticalStride,
                                 const int leftAndRightPadWidth,
                                 const int topAndBottomPadHeight,
                                 const int numInputChannels,
                                 const int numOutputChannels,
                                 const int batchSize,
                                 const int numInputColumns,
                                 const int numInputRows) {
        construct(gpuType,
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
    }

    ConvolutionKernelRequirement(const ConvolutionKernelRequirement &other) = default;
    ConvolutionKernelRequirement(ConvolutionKernelRequirement &&other) noexcept = default;

    ConvolutionKernelRequirement &operator=(const ConvolutionKernelRequirement &other) = default;
    ConvolutionKernelRequirement &operator=(ConvolutionKernelRequirement &&other) noexcept = default;

    virtual ~ConvolutionKernelRequirement() = default;

    cudnnConvolutionDescriptor_t getConvolutionDescriptor() {
        THOR_THROW_IF_FALSE(!uninitialized());

        if (state->convolutionDescriptor != nullptr)
            return state->convolutionDescriptor;

        cudnnStatus_t cudnnStatus = cudnnCreateConvolutionDescriptor(&state->convolutionDescriptor);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);

        cudnnStatus = cudnnSetConvolution2dDescriptor(state->convolutionDescriptor,
                                                      state->topAndBottomPadHeight,
                                                      state->leftAndRightPadWidth,
                                                      state->filterVerticalStride,
                                                      state->filterHorizontalStride,
                                                      1,
                                                      1,
                                                      CUDNN_CROSS_CORRELATION,
                                                      CUDNN_DATA_FLOAT);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);

        cudnnStatus = cudnnSetConvolutionMathType(state->convolutionDescriptor, CUDNN_TENSOR_OP_MATH);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);

        return state->convolutionDescriptor;
    }

    cudnnFilterDescriptor_t getWeightsFilterDescriptor() {
        THOR_THROW_IF_FALSE(!uninitialized());

        if (state->filterDescriptor != nullptr)
            return state->filterDescriptor;

        cudnnStatus_t cudnnStatus = cudnnCreateFilterDescriptor(&state->filterDescriptor);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);

        cudnnStatus = cudnnSetFilter4dDescriptor(state->filterDescriptor,
                                                 CUDNN_DATA_HALF,
                                                 CUDNN_TENSOR_NCHW,
                                                 state->numOutputChannels,
                                                 state->numInputChannels,
                                                 state->filterHeight,
                                                 state->filterWidth);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);

        return state->filterDescriptor;
    }

    cudnnFilterDescriptor_t getWeightsGradientFilterDescriptor() {
        THOR_THROW_IF_FALSE(!uninitialized());

        // This could differ in the future if we wanted to read in fp16 weights,
        // but output fp32 gradients for subsequent accumulation.
        // That is not currently implemented.
        return getWeightsFilterDescriptor();
    }

    cudnnTensorDescriptor_t getDataInputTensorDescriptor() {
        THOR_THROW_IF_FALSE(!uninitialized());

        if (state->inputTensorDescriptor != nullptr)
            return state->inputTensorDescriptor;

        cudnnStatus_t cudnnStatus = cudnnCreateTensorDescriptor(&state->inputTensorDescriptor);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);

        cudnnStatus = cudnnSetTensor4dDescriptor(state->inputTensorDescriptor,
                                                 CUDNN_TENSOR_NCHW,
                                                 CUDNN_DATA_HALF,
                                                 state->batchSize,
                                                 state->numInputChannels,
                                                 state->numInputRows,
                                                 state->numInputColumns);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);

        return state->inputTensorDescriptor;
    }

    cudnnTensorDescriptor_t getDataOutputTensorDescriptor() {
        THOR_THROW_IF_FALSE(!uninitialized());

        if (state->outputTensorDescriptor != nullptr)
            return state->outputTensorDescriptor;

        cudnnStatus_t cudnnStatus = cudnnCreateTensorDescriptor(&state->outputTensorDescriptor);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);

        int computedBatchSize;
        int computedNumOutputChannels;
        int computedNumOutputRows;
        int computedNumOutputColumns;
        cudnnStatus = cudnnGetConvolution2dForwardOutputDim(getConvolutionDescriptor(),
                                                            getDataInputTensorDescriptor(),
                                                            getWeightsFilterDescriptor(),
                                                            &computedBatchSize,
                                                            &computedNumOutputChannels,
                                                            &computedNumOutputRows,
                                                            &computedNumOutputColumns);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);

        THOR_THROW_IF_FALSE(computedBatchSize == state->batchSize);
        THOR_THROW_IF_FALSE(computedNumOutputChannels == state->numOutputChannels);
        state->numOutputRows = computedNumOutputRows;
        state->numOutputColumns = computedNumOutputColumns;

        cudnnStatus = cudnnSetTensor4dDescriptor(state->outputTensorDescriptor,
                                                 CUDNN_TENSOR_NCHW,
                                                 CUDNN_DATA_HALF,
                                                 computedBatchSize,
                                                 computedNumOutputChannels,
                                                 computedNumOutputRows,
                                                 computedNumOutputColumns);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);

        return state->outputTensorDescriptor;
    }

    cudnnTensorDescriptor_t getBiasesTensorDescriptor() {
        THOR_THROW_IF_FALSE(!uninitialized());

        if (state->biasesDescriptor != nullptr)
            return state->biasesDescriptor;

        cudnnStatus_t cudnnStatus = cudnnCreateTensorDescriptor(&state->biasesDescriptor);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
        cudnnStatus = cudnnSetTensor4dDescriptor(
            state->biasesDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, 1, state->numOutputChannels, 1, 1);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);

        return state->biasesDescriptor;
    }

    cudnnTensorDescriptor_t getErrorInputTensorDescriptor() {
        THOR_THROW_IF_FALSE(!uninitialized());
        return getDataOutputTensorDescriptor();
    }

    cudnnTensorDescriptor_t getErrorOutputTensorDescriptor() {
        THOR_THROW_IF_FALSE(!uninitialized());
        return getDataInputTensorDescriptor();
    }

    bool operator==(const ConvolutionKernelRequirement &other) const {
        THOR_THROW_IF_FALSE(!uninitialized());
        THOR_THROW_IF_FALSE(!other.uninitialized());
        return state->gpuType == other.state->gpuType && state->filterWidth == other.state->filterWidth &&
               state->filterHeight == other.state->filterHeight && state->filterHorizontalStride == other.state->filterHorizontalStride &&
               state->filterVerticalStride == other.state->filterVerticalStride &&
               state->leftAndRightPadWidth == other.state->leftAndRightPadWidth &&
               state->topAndBottomPadHeight == other.state->topAndBottomPadHeight && state->numInputChannels == other.state->numInputChannels &&
               state->numOutputChannels == other.state->numOutputChannels && state->batchSize == other.state->batchSize &&
               state->numInputColumns == other.state->numInputColumns && state->numInputRows == other.state->numInputRows &&
               state->numOutputColumns == other.state->numOutputColumns && state->numOutputRows == other.state->numOutputRows;
    }

    std::string getGpuType() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return state->gpuType;
    }
    int getFilterWidth() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return state->filterWidth;
    }
    int getFilterHeight() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return state->filterHeight;
    }
    int getFilterHorizontalStride() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return state->filterHorizontalStride;
    }
    int getFilterVerticalStride() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return state->filterVerticalStride;
    }
    int getLeftAndRightPadWidth() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return state->leftAndRightPadWidth;
    }
    int getTopAndBottomPadHeight() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return state->topAndBottomPadHeight;
    }
    int getNumInputChannels() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return state->numInputChannels;
    }
    int getNumOutputChannels() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return state->numOutputChannels;
    }
    int getBatchSize() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return state->batchSize;
    }
    int getNumInputColumns() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return state->numInputColumns;
    }
    int getNumInputRows() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return state->numInputRows;
    }
    int getNumOutputColumns() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return state->numOutputColumns;
    }
    int getNumOutputRows() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return state->numOutputRows;
    }

    std::string toString() const {
        THOR_THROW_IF_FALSE(!uninitialized());

        std::string s;
        s = "GpuType " + getGpuType() + " FilterWidth " + std::to_string(getFilterWidth()) + " FilterHeight " +
            std::to_string(getFilterHeight()) + " FilterHorizontalStride " + std::to_string(getFilterHorizontalStride()) +
            " FilterVerticalStride " + std::to_string(getFilterVerticalStride()) + " leftAndRightPadWidth " +
            std::to_string(getLeftAndRightPadWidth()) + " TopAndBottomPadHeight " + std::to_string(getTopAndBottomPadHeight()) +
            " NumInputChannels " + std::to_string(getNumInputChannels()) + " NumOutputChannels " + std::to_string(getNumOutputChannels()) +
            " BatchSize " + std::to_string(getBatchSize()) + " NumInputColumns " + std::to_string(getNumInputColumns()) + " NumInputRows " +
            std::to_string(getNumInputRows()) + " NumOutputColumns " + std::to_string(getNumOutputColumns()) + " NumOutputRows " +
            std::to_string(getNumOutputRows());
        return s;
    }

   private:
    struct State {
        State(const std::string &gpuType,
              int filterWidth,
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
            : gpuType(gpuType),
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
              numInputRows(numInputRows) {}

        ~State() noexcept {
            destroyDescriptorNoThrow("cudnnDestroyConvolutionDescriptor", convolutionDescriptor, cudnnDestroyConvolutionDescriptor);
            destroyDescriptorNoThrow("cudnnDestroyFilterDescriptor", filterDescriptor, cudnnDestroyFilterDescriptor);
            destroyDescriptorNoThrow("cudnnDestroyTensorDescriptor(input)", inputTensorDescriptor, cudnnDestroyTensorDescriptor);
            destroyDescriptorNoThrow("cudnnDestroyTensorDescriptor(output)", outputTensorDescriptor, cudnnDestroyTensorDescriptor);
            destroyDescriptorNoThrow("cudnnDestroyTensorDescriptor(biases)", biasesDescriptor, cudnnDestroyTensorDescriptor);
        }

        template <typename Descriptor, typename Destroy>
        static void destroyDescriptorNoThrow(const char *operation, Descriptor &descriptor, Destroy destroy) noexcept {
            if (descriptor == nullptr)
                return;

            ThorImplementation::SharedOwnership::cleanupNoThrow("ConvolutionKernelRequirement", operation, [&]() {
                const cudnnStatus_t status = destroy(descriptor);
                THOR_THROW_IF_FALSE(status == CUDNN_STATUS_SUCCESS);
            });
            descriptor = nullptr;
        }

        std::string gpuType;
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
        int numOutputColumns = 0;
        int numOutputRows = 0;

        cudnnConvolutionDescriptor_t convolutionDescriptor = nullptr;
        cudnnFilterDescriptor_t filterDescriptor = nullptr;
        cudnnTensorDescriptor_t inputTensorDescriptor = nullptr;
        cudnnTensorDescriptor_t outputTensorDescriptor = nullptr;
        cudnnTensorDescriptor_t biasesDescriptor = nullptr;
    };

    std::shared_ptr<State> state;

    bool uninitialized() const { return state == nullptr; }

    void construct(const std::string gpuType,
                   const int filterWidth,
                   const int filterHeight,
                   const int filterHorizontalStride,
                   const int filterVerticalStride,
                   const int leftAndRightPadWidth,
                   const int topAndBottomPadHeight,
                   const int numInputChannels,
                   const int numOutputChannels,
                   const int batchSize,
                   const int numInputColumns,
                   const int numInputRows) {
        THOR_THROW_IF_FALSE(filterWidth > 0);
        THOR_THROW_IF_FALSE(filterHeight > 0);
        THOR_THROW_IF_FALSE(filterHorizontalStride > 0);
        THOR_THROW_IF_FALSE(filterVerticalStride > 0);
        THOR_THROW_IF_FALSE(leftAndRightPadWidth >= 0);
        THOR_THROW_IF_FALSE(topAndBottomPadHeight >= 0);
        THOR_THROW_IF_FALSE(numInputChannels > 0);
        THOR_THROW_IF_FALSE(numOutputChannels > 0);
        THOR_THROW_IF_FALSE(batchSize > 0);
        THOR_THROW_IF_FALSE(numInputColumns > 0);
        THOR_THROW_IF_FALSE(numInputRows > 0);

        state = std::make_shared<State>(gpuType,
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

        // Populate every descriptor before construction completes. Once a requirement
        // is visible to other threads, its shared State is immutable except for the
        // cuDNN objects themselves, so ordinary getters require no lazy-init lock.
        getConvolutionDescriptor();
        getWeightsFilterDescriptor();
        getWeightsGradientFilterDescriptor();
        getDataInputTensorDescriptor();
        getDataOutputTensorDescriptor();
        getBiasesTensorDescriptor();
    }

    friend class std::hash<ConvolutionKernelRequirement>;
};

namespace std {

template <>
struct hash<ConvolutionKernelRequirement> {
    size_t operator()(const ConvolutionKernelRequirement &k) const {
        THOR_THROW_IF_FALSE(!k.uninitialized());

        size_t hashValue;
        hashValue = (hash<int>()(k.state->numInputRows)) << 1;
        hashValue = (hashValue ^ (hash<int>()(k.state->filterWidth))) << 1;
        hashValue = (hashValue ^ (hash<int>()(k.state->filterHeight))) << 1;
        hashValue = (hashValue ^ (hash<int>()(k.state->filterHorizontalStride))) << 1;
        hashValue = (hashValue ^ (hash<int>()(k.state->filterVerticalStride))) << 1;
        hashValue = (hashValue ^ (hash<int>()(k.state->leftAndRightPadWidth))) << 1;
        hashValue = (hashValue ^ (hash<int>()(k.state->topAndBottomPadHeight))) << 1;
        hashValue = (hashValue ^ (hash<int>()(k.state->numInputChannels))) << 1;
        hashValue = (hashValue ^ (hash<int>()(k.state->numOutputChannels))) << 1;
        hashValue = (hashValue ^ (hash<int>()(k.state->batchSize))) << 1;
        hashValue = (hashValue ^ (hash<int>()(k.state->numInputColumns))) << 1;
        hashValue = hashValue ^ hash<string>()(k.state->gpuType);
        return hashValue;
    }
};

}  // namespace std
