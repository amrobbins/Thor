#pragma once

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/ReferenceCounted.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cudnn.h>

#include <atomic>
#include <string>
#include <utility>

class ConvolutionKernelRequirement;
namespace std {
template <>
struct std::hash<ConvolutionKernelRequirement>;
}

class ConvolutionKernelRequirement : private ReferenceCounted {
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

    ConvolutionKernelRequirement(const ConvolutionKernelRequirement &other) {
        // implemented using operator=
        *this = other;
    }

    ConvolutionKernelRequirement &operator=(const ConvolutionKernelRequirement &other) {
        copyFrom(other);
        return *this;
    }

    virtual ~ConvolutionKernelRequirement() {
        bool shouldDestroy = ReferenceCounted::removeReference();
        if (shouldDestroy)
            destroy();
    }

    cudnnConvolutionDescriptor_t getConvolutionDescriptor() {
        THOR_THROW_IF_FALSE(!uninitialized());

        if (*ppConvolutionDescriptor != nullptr)
            return **ppConvolutionDescriptor;
        // FIXME: I don't think that I should new this, cudnn will allocate it. Look for any other instance and fix too.
        *ppConvolutionDescriptor = new cudnnConvolutionDescriptor_t;

        cudnnStatus_t cudnnStatus;

        cudnnStatus = cudnnCreateConvolutionDescriptor(*ppConvolutionDescriptor);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);

        cudnnStatus = cudnnSetConvolution2dDescriptor(**ppConvolutionDescriptor,
                                                      topAndBottomPadHeight,
                                                      leftAndRightPadWidth,
                                                      filterVerticalStride,
                                                      filterHorizontalStride,
                                                      1,
                                                      1,
                                                      CUDNN_CROSS_CORRELATION,
                                                      CUDNN_DATA_FLOAT);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);

        cudnnStatus = cudnnSetConvolutionMathType(**ppConvolutionDescriptor, CUDNN_TENSOR_OP_MATH);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);

        return **ppConvolutionDescriptor;
    }

    cudnnFilterDescriptor_t getWeightsFilterDescriptor() {
        THOR_THROW_IF_FALSE(!uninitialized());

        if (*ppFilterDescriptor != nullptr)
            return **ppFilterDescriptor;
        *ppFilterDescriptor = new cudnnFilterDescriptor_t;

        cudnnStatus_t cudnnStatus;

        cudnnStatus = cudnnCreateFilterDescriptor(*ppFilterDescriptor);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);

        cudnnStatus = cudnnSetFilter4dDescriptor(
            **ppFilterDescriptor, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, numOutputChannels, numInputChannels, filterHeight, filterWidth);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);

        return **ppFilterDescriptor;
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

        if (*ppInputTensorDescriptor != nullptr)
            return **ppInputTensorDescriptor;
        *ppInputTensorDescriptor = new cudnnTensorDescriptor_t;

        cudnnStatus_t cudnnStatus;

        cudnnStatus = cudnnCreateTensorDescriptor(*ppInputTensorDescriptor);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);

        cudnnStatus = cudnnSetTensor4dDescriptor(
            **ppInputTensorDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize, numInputChannels, numInputRows, numInputColumns);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);

        return **ppInputTensorDescriptor;
    }

    cudnnTensorDescriptor_t getDataOutputTensorDescriptor() {
        THOR_THROW_IF_FALSE(!uninitialized());

        if (*ppOutputTensorDescriptor != nullptr)
            return **ppOutputTensorDescriptor;
        *ppOutputTensorDescriptor = new cudnnTensorDescriptor_t;

        cudnnStatus_t cudnnStatus;

        cudnnStatus = cudnnCreateTensorDescriptor(*ppOutputTensorDescriptor);
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

        THOR_THROW_IF_FALSE(computedBatchSize == batchSize);
        THOR_THROW_IF_FALSE(computedNumOutputChannels == numOutputChannels);
        numOutputRows = computedNumOutputRows;
        numOutputColumns = computedNumOutputColumns;

        cudnnStatus = cudnnSetTensor4dDescriptor(**ppOutputTensorDescriptor,
                                                 CUDNN_TENSOR_NCHW,
                                                 CUDNN_DATA_HALF,
                                                 computedBatchSize,
                                                 computedNumOutputChannels,
                                                 computedNumOutputRows,
                                                 computedNumOutputColumns);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);

        return **ppOutputTensorDescriptor;
    }

    cudnnTensorDescriptor_t getBiasesTensorDescriptor() {
        THOR_THROW_IF_FALSE(!uninitialized());

        if (*ppBiasesDescriptor != nullptr)
            return **ppBiasesDescriptor;
        *ppBiasesDescriptor = new cudnnTensorDescriptor_t;

        cudnnStatus_t cudnnStatus;

        cudnnStatus = cudnnCreateTensorDescriptor(*ppBiasesDescriptor);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
        cudnnStatus = cudnnSetTensor4dDescriptor(**ppBiasesDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, 1, numOutputChannels, 1, 1);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);

        return **ppBiasesDescriptor;
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
        return gpuType == other.gpuType && filterWidth == other.filterWidth && filterHeight == other.filterHeight &&
               filterHorizontalStride == other.filterHorizontalStride && filterVerticalStride == other.filterVerticalStride &&
               leftAndRightPadWidth == other.leftAndRightPadWidth && topAndBottomPadHeight == other.topAndBottomPadHeight &&
               numInputChannels == other.numInputChannels && numOutputChannels == other.numOutputChannels && batchSize == other.batchSize &&
               numInputColumns == other.numInputColumns && numInputRows == other.numInputRows &&
               numOutputColumns == other.numOutputColumns && numOutputRows == other.numOutputRows;
    }

    std::string getGpuType() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return gpuType;
    }
    int getFilterWidth() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return filterWidth;
    }
    int getFilterHeight() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return filterHeight;
    }
    int getFilterHorizontalStride() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return filterHorizontalStride;
    }
    int getFilterVerticalStride() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return filterVerticalStride;
    }
    int getLeftAndRightPadWidth() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return leftAndRightPadWidth;
    }
    int getTopAndBottomPadHeight() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return topAndBottomPadHeight;
    }
    int getNumInputChannels() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return numInputChannels;
    }
    int getNumOutputChannels() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return numOutputChannels;
    }
    int getBatchSize() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return batchSize;
    }
    int getNumInputColumns() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return numInputColumns;
    }
    int getNumInputRows() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return numInputRows;
    }
    int getNumOutputColumns() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return numOutputColumns;
    }
    int getNumOutputRows() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return numOutputRows;
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
    int numOutputColumns;
    int numOutputRows;

    cudnnConvolutionDescriptor_t **ppConvolutionDescriptor;
    cudnnFilterDescriptor_t **ppFilterDescriptor;
    cudnnTensorDescriptor_t **ppInputTensorDescriptor;
    cudnnTensorDescriptor_t **ppOutputTensorDescriptor;
    cudnnTensorDescriptor_t **ppBiasesDescriptor;

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

        ReferenceCounted::initialize();

        this->gpuType = gpuType;
        this->filterWidth = filterWidth;
        this->filterHeight = filterHeight;
        this->filterHorizontalStride = filterHorizontalStride;
        this->filterVerticalStride = filterVerticalStride;
        this->leftAndRightPadWidth = leftAndRightPadWidth;
        this->topAndBottomPadHeight = topAndBottomPadHeight;
        this->numInputChannels = numInputChannels;
        this->numOutputChannels = numOutputChannels;
        this->batchSize = batchSize;
        this->numInputColumns = numInputColumns;
        this->numInputRows = numInputRows;

        ppConvolutionDescriptor = new cudnnConvolutionDescriptor_t *;
        *ppConvolutionDescriptor = nullptr;
        ppFilterDescriptor = new cudnnFilterDescriptor_t *;
        *ppFilterDescriptor = nullptr;
        ppInputTensorDescriptor = new cudnnTensorDescriptor_t *;
        *ppInputTensorDescriptor = nullptr;
        ppOutputTensorDescriptor = new cudnnTensorDescriptor_t *;
        *ppOutputTensorDescriptor = nullptr;
        ppBiasesDescriptor = new cudnnTensorDescriptor_t *;
        *ppBiasesDescriptor = nullptr;

        // Eagerly call all the get...Descriptor() functions to avoid needing to lock when populating the structures
        getConvolutionDescriptor();
        getWeightsFilterDescriptor();
        getWeightsGradientFilterDescriptor();
        getDataOutputTensorDescriptor();
    }

    void copyFrom(const ConvolutionKernelRequirement &other) {
        *((ReferenceCounted *)this) = *((ReferenceCounted *)&other);

        gpuType = other.gpuType;
        filterWidth = other.filterWidth;
        filterHeight = other.filterHeight;
        filterHorizontalStride = other.filterHorizontalStride;
        filterVerticalStride = other.filterVerticalStride;
        leftAndRightPadWidth = other.leftAndRightPadWidth;
        topAndBottomPadHeight = other.topAndBottomPadHeight;
        numInputChannels = other.numInputChannels;
        numOutputChannels = other.numOutputChannels;
        batchSize = other.batchSize;
        numInputColumns = other.numInputColumns;
        numInputRows = other.numInputRows;
        numOutputColumns = other.numOutputColumns;
        numOutputRows = other.numOutputRows;
        ppConvolutionDescriptor = other.ppConvolutionDescriptor;
        ppFilterDescriptor = other.ppFilterDescriptor;
        ppInputTensorDescriptor = other.ppInputTensorDescriptor;
        ppOutputTensorDescriptor = other.ppOutputTensorDescriptor;
        ppBiasesDescriptor = other.ppBiasesDescriptor;
    }

    void destroy() {
        cudnnStatus_t cudnnStatus;

        if (*ppConvolutionDescriptor != nullptr) {
            cudnnStatus = cudnnDestroyConvolutionDescriptor(**ppConvolutionDescriptor);
            THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
            delete *ppConvolutionDescriptor;
        }
        delete ppConvolutionDescriptor;
        ppConvolutionDescriptor = nullptr;

        if (*ppFilterDescriptor != nullptr) {
            cudnnStatus = cudnnDestroyFilterDescriptor(**ppFilterDescriptor);
            THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
            delete *ppFilterDescriptor;
        }
        delete ppFilterDescriptor;
        ppFilterDescriptor = nullptr;

        if (*ppInputTensorDescriptor != nullptr) {
            cudnnStatus = cudnnDestroyTensorDescriptor(**ppInputTensorDescriptor);
            THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
            delete *ppInputTensorDescriptor;
        }
        delete ppInputTensorDescriptor;
        ppInputTensorDescriptor = nullptr;

        if (*ppOutputTensorDescriptor != nullptr) {
            cudnnStatus = cudnnDestroyTensorDescriptor(**ppOutputTensorDescriptor);
            THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
            delete *ppOutputTensorDescriptor;
        }
        delete ppOutputTensorDescriptor;
        ppOutputTensorDescriptor = nullptr;

        if (*ppBiasesDescriptor != nullptr) {
            cudnnStatus = cudnnDestroyTensorDescriptor(**ppBiasesDescriptor);
            THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
            delete *ppBiasesDescriptor;
        }
        delete ppBiasesDescriptor;
        ppBiasesDescriptor = nullptr;
    }

    friend class std::hash<ConvolutionKernelRequirement>;
};

namespace std {

template <>
struct hash<ConvolutionKernelRequirement> {
    size_t operator()(const ConvolutionKernelRequirement &k) const {
        size_t hashValue;
        hashValue = (hash<int>()(k.numInputRows)) << 1;
        hashValue = (hashValue ^ (hash<int>()(k.filterWidth))) << 1;
        hashValue = (hashValue ^ (hash<int>()(k.filterHeight))) << 1;
        hashValue = (hashValue ^ (hash<int>()(k.filterHorizontalStride))) << 1;
        hashValue = (hashValue ^ (hash<int>()(k.filterVerticalStride))) << 1;
        hashValue = (hashValue ^ (hash<int>()(k.leftAndRightPadWidth))) << 1;
        hashValue = (hashValue ^ (hash<int>()(k.topAndBottomPadHeight))) << 1;
        hashValue = (hashValue ^ (hash<int>()(k.numInputChannels))) << 1;
        hashValue = (hashValue ^ (hash<int>()(k.numOutputChannels))) << 1;
        hashValue = (hashValue ^ (hash<int>()(k.batchSize))) << 1;
        hashValue = (hashValue ^ (hash<int>()(k.numInputColumns))) << 1;
        hashValue = hashValue ^ hash<string>()(k.gpuType);
        return hashValue;
    }
};

}  // namespace std
