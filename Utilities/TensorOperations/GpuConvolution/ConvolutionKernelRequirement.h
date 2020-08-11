#pragma once

#include "Utilities/Common/ReferenceCounted.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cudnn.h>

#include <assert.h>

#include <atomic>
#include <string>
#include <utility>

using std::atomic;
using std::hash;
using std::string;

class ConvolutionKernelRequirement;
namespace std {
template <>
struct hash<ConvolutionKernelRequirement>;
}

class ConvolutionKernelRequirement : private ReferenceCounted {
   public:
    ConvolutionKernelRequirement() = delete;

    ConvolutionKernelRequirement(const string gpuType,
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
        assert(!uninitialized());

        if (*ppConvolutionDescriptor != nullptr)
            return **ppConvolutionDescriptor;
        *ppConvolutionDescriptor = new cudnnConvolutionDescriptor_t;

        cudnnStatus_t cudnnStatus;

        cudnnStatus = cudnnCreateConvolutionDescriptor(*ppConvolutionDescriptor);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

        cudnnStatus = cudnnSetConvolution2dDescriptor(**ppConvolutionDescriptor,
                                                      topAndBottomPadHeight,
                                                      leftAndRightPadWidth,
                                                      filterVerticalStride,
                                                      filterHorizontalStride,
                                                      1,
                                                      1,
                                                      CUDNN_CONVOLUTION,
                                                      CUDNN_DATA_FLOAT);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

        cudnnStatus = cudnnSetConvolutionMathType(**ppConvolutionDescriptor, CUDNN_TENSOR_OP_MATH);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

        return **ppConvolutionDescriptor;
    }

    cudnnFilterDescriptor_t getWeightsFilterDescriptor() {
        assert(!uninitialized());

        if (*ppFilterDescriptor != nullptr)
            return **ppFilterDescriptor;
        *ppFilterDescriptor = new cudnnFilterDescriptor_t;

        cudnnStatus_t cudnnStatus;

        cudnnStatus = cudnnCreateFilterDescriptor(*ppFilterDescriptor);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

        cudnnStatus = cudnnSetFilter4dDescriptor(
            **ppFilterDescriptor, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, numOutputChannels, numInputChannels, filterHeight, filterWidth);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

        return **ppFilterDescriptor;
    }

    cudnnFilterDescriptor_t getWeightsGradientFilterDescriptor() {
        assert(!uninitialized());

        // This could differ in the future if we wanted to read in fp16 weights,
        // but output fp32 gradients for subsequent accumulation.
        // That is not currently implemented.
        return getWeightsFilterDescriptor();
    }

    cudnnTensorDescriptor_t getDataInputTensorDescriptor() {
        assert(!uninitialized());

        if (*ppInputTensorDescriptor != nullptr)
            return **ppInputTensorDescriptor;
        *ppInputTensorDescriptor = new cudnnTensorDescriptor_t;

        cudnnStatus_t cudnnStatus;

        cudnnStatus = cudnnCreateTensorDescriptor(*ppInputTensorDescriptor);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

        cudnnStatus = cudnnSetTensor4dDescriptor(
            **ppInputTensorDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize, numInputChannels, numInputRows, numInputColumns);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

        return **ppInputTensorDescriptor;
    }

    cudnnTensorDescriptor_t getDataOutputTensorDescriptor() {
        assert(!uninitialized());

        if (*ppOutputTensorDescriptor != nullptr)
            return **ppOutputTensorDescriptor;
        *ppOutputTensorDescriptor = new cudnnTensorDescriptor_t;

        cudnnStatus_t cudnnStatus;

        cudnnStatus = cudnnCreateTensorDescriptor(*ppOutputTensorDescriptor);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

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

        assert(computedBatchSize == batchSize);
        assert(computedNumOutputChannels == numOutputChannels);
        numOutputRows = computedNumOutputRows;
        numOutputColumns = computedNumOutputColumns;

        cudnnStatus = cudnnSetTensor4dDescriptor(**ppOutputTensorDescriptor,
                                                 CUDNN_TENSOR_NCHW,
                                                 CUDNN_DATA_HALF,
                                                 computedBatchSize,
                                                 computedNumOutputChannels,
                                                 computedNumOutputRows,
                                                 computedNumOutputColumns);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

        return **ppOutputTensorDescriptor;
    }

    cudnnTensorDescriptor_t getBiasesTensorDescriptor() {
        assert(!uninitialized());

        if (*ppBiasesDescriptor != nullptr)
            return **ppBiasesDescriptor;
        *ppBiasesDescriptor = new cudnnTensorDescriptor_t;

        cudnnStatus_t cudnnStatus;

        cudnnStatus = cudnnCreateTensorDescriptor(*ppBiasesDescriptor);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
        cudnnStatus = cudnnSetTensor4dDescriptor(**ppBiasesDescriptor,
                                                 CUDNN_TENSOR_NCHW,
                                                 CUDNN_DATA_HALF,
                                                 1,
                                                 numOutputChannels,
                                                 1,
                                                 1);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

        return **ppBiasesDescriptor;
    }


    cudnnTensorDescriptor_t getErrorInputTensorDescriptor() {
        assert(!uninitialized());
        return getDataOutputTensorDescriptor();
    }

    cudnnTensorDescriptor_t getErrorOutputTensorDescriptor() {
        assert(!uninitialized());
        return getDataInputTensorDescriptor();
    }

    bool operator==(const ConvolutionKernelRequirement &other) const {
        assert(!uninitialized());
        assert(!other.uninitialized());
        return gpuType == other.gpuType && filterWidth == other.filterWidth && filterHeight == other.filterHeight &&
               filterHorizontalStride == other.filterHorizontalStride && filterVerticalStride == other.filterVerticalStride &&
               leftAndRightPadWidth == other.leftAndRightPadWidth && topAndBottomPadHeight == other.topAndBottomPadHeight &&
               numInputChannels == other.numInputChannels && numOutputChannels == other.numOutputChannels && batchSize == other.batchSize &&
               numInputColumns == other.numInputColumns && numInputRows == other.numInputRows &&
               numOutputColumns == other.numOutputColumns && numOutputRows == other.numOutputRows;
    }

    string getGpuType() {
        assert(!uninitialized());
        return gpuType;
    }
    int getFilterWidth() {
        assert(!uninitialized());
        return filterWidth;
    }
    int getFilterHeight() {
        assert(!uninitialized());
        return filterHeight;
    }
    int getFilterHorizontalStride() {
        assert(!uninitialized());
        return filterHorizontalStride;
    }
    int getFilterVerticalStride() {
        assert(!uninitialized());
        return filterVerticalStride;
    }
    int getLeftAndRightPadWidth() {
        assert(!uninitialized());
        return leftAndRightPadWidth;
    }
    int getTopAndBottomPadHeight() {
        assert(!uninitialized());
        return topAndBottomPadHeight;
    }
    int getNumInputChannels() {
        assert(!uninitialized());
        return numInputChannels;
    }
    int getNumOutputChannels() {
        assert(!uninitialized());
        return numOutputChannels;
    }
    int getBatchSize() {
        assert(!uninitialized());
        return batchSize;
    }
    int getNumInputColumns() {
        assert(!uninitialized());
        return numInputColumns;
    }
    int getNumInputRows() {
        assert(!uninitialized());
        return numInputRows;
    }
    int getNumOutputColumns() {
        assert(!uninitialized());
        return numOutputColumns;
    }
    int getNumOutputRows() {
        assert(!uninitialized());
        return numOutputRows;
    }

    string toString() {
        assert(!uninitialized());

        string s;
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
    string gpuType;
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

    void construct(const string gpuType,
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
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
            delete *ppConvolutionDescriptor;
        }
        delete ppConvolutionDescriptor;
        ppConvolutionDescriptor = nullptr;

        if (*ppFilterDescriptor != nullptr) {
            cudnnStatus = cudnnDestroyFilterDescriptor(**ppFilterDescriptor);
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
            delete *ppFilterDescriptor;
        }
        delete ppFilterDescriptor;
        ppFilterDescriptor = nullptr;

        if (*ppInputTensorDescriptor != nullptr) {
            cudnnStatus = cudnnDestroyTensorDescriptor(**ppInputTensorDescriptor);
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
            delete *ppInputTensorDescriptor;
        }
        delete ppInputTensorDescriptor;
        ppInputTensorDescriptor = nullptr;

        if (*ppOutputTensorDescriptor != nullptr) {
            cudnnStatus = cudnnDestroyTensorDescriptor(**ppOutputTensorDescriptor);
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
            delete *ppOutputTensorDescriptor;
        }
        delete ppOutputTensorDescriptor;
        ppOutputTensorDescriptor = nullptr;

        if (*ppBiasesDescriptor != nullptr) {
            cudnnStatus = cudnnDestroyTensorDescriptor(**ppBiasesDescriptor);
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
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
    std::size_t operator()(const ConvolutionKernelRequirement &k) const {
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
