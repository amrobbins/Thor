#pragma once

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

class ConvolutionKernelRequirement {
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
                                 const int numInputRows)
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
          numInputRows(numInputRows) {
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

        ppConvolutionDescriptor = new cudnnConvolutionDescriptor_t *;
        *ppConvolutionDescriptor = nullptr;
        ppFilterDescriptor = new cudnnFilterDescriptor_t *;
        *ppFilterDescriptor = nullptr;
        ppInputTensorDescriptor = new cudnnTensorDescriptor_t *;
        *ppInputTensorDescriptor = nullptr;
        ppOutputTensorDescriptor = new cudnnTensorDescriptor_t *;
        *ppOutputTensorDescriptor = nullptr;

        // Eagerly call all the get...Descriptor() functions to avoid needing to lock when populating the structures
        getConvolutionDescriptor();
        getWeightsFilterDescriptor();
        getWeightsGradientFilterDescriptor();
        getDataOutputTensorDescriptor();

        referenceCount = new atomic<int>(1);
    }

    ConvolutionKernelRequirement(const ConvolutionKernelRequirement &convolutionKernelRequirement)
        : gpuType(convolutionKernelRequirement.gpuType),
          filterWidth(convolutionKernelRequirement.filterWidth),
          filterHeight(convolutionKernelRequirement.filterHeight),
          filterHorizontalStride(convolutionKernelRequirement.filterHorizontalStride),
          filterVerticalStride(convolutionKernelRequirement.filterVerticalStride),
          leftAndRightPadWidth(convolutionKernelRequirement.leftAndRightPadWidth),
          topAndBottomPadHeight(convolutionKernelRequirement.topAndBottomPadHeight),
          numInputChannels(convolutionKernelRequirement.numInputChannels),
          numOutputChannels(convolutionKernelRequirement.numOutputChannels),
          batchSize(convolutionKernelRequirement.batchSize),
          numInputColumns(convolutionKernelRequirement.numInputColumns),
          numInputRows(convolutionKernelRequirement.numInputRows),
          numOutputColumns(convolutionKernelRequirement.numOutputColumns),
          numOutputRows(convolutionKernelRequirement.numOutputRows) {
        referenceCount = convolutionKernelRequirement.referenceCount;
        referenceCount->fetch_add(1);

        ppConvolutionDescriptor = convolutionKernelRequirement.ppConvolutionDescriptor;
        ppFilterDescriptor = convolutionKernelRequirement.ppFilterDescriptor;
        ppInputTensorDescriptor = convolutionKernelRequirement.ppInputTensorDescriptor;
        ppOutputTensorDescriptor = convolutionKernelRequirement.ppOutputTensorDescriptor;
    }

    ConvolutionKernelRequirement &operator=(const ConvolutionKernelRequirement &convolutionKernelRequirement) {
        referenceCount = convolutionKernelRequirement.referenceCount;
        referenceCount->fetch_add(1);

        gpuType = convolutionKernelRequirement.gpuType;
        filterWidth = convolutionKernelRequirement.filterWidth;
        filterHeight = convolutionKernelRequirement.filterHeight;
        filterHorizontalStride = convolutionKernelRequirement.filterHorizontalStride;
        filterVerticalStride = convolutionKernelRequirement.filterVerticalStride;
        leftAndRightPadWidth = convolutionKernelRequirement.leftAndRightPadWidth;
        topAndBottomPadHeight = convolutionKernelRequirement.topAndBottomPadHeight;
        numInputChannels = convolutionKernelRequirement.numInputChannels;
        numOutputChannels = convolutionKernelRequirement.numOutputChannels;
        batchSize = convolutionKernelRequirement.batchSize;
        numInputColumns = convolutionKernelRequirement.numInputColumns;
        numInputRows = convolutionKernelRequirement.numInputRows;
        numOutputColumns = convolutionKernelRequirement.numOutputColumns;
        numOutputRows = convolutionKernelRequirement.numOutputRows;
        ppConvolutionDescriptor = convolutionKernelRequirement.ppConvolutionDescriptor;
        ppFilterDescriptor = convolutionKernelRequirement.ppFilterDescriptor;
        ppInputTensorDescriptor = convolutionKernelRequirement.ppInputTensorDescriptor;
        ppOutputTensorDescriptor = convolutionKernelRequirement.ppOutputTensorDescriptor;

        return *this;
    }

    ~ConvolutionKernelRequirement() {
        int refCountBeforeDecrement = referenceCount->fetch_sub(1);
        if (refCountBeforeDecrement == 1) {
            delete referenceCount;
            referenceCount = nullptr;

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
        }
    }

    cudnnConvolutionDescriptor_t getConvolutionDescriptor() {
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
        // This could differ in the future if we wanted to read in fp16 weights,
        // but output fp32 gradients for subsequent accumulation.
        // That is not currently implemented.
        return getWeightsFilterDescriptor();
    }

    cudnnTensorDescriptor_t getDataInputTensorDescriptor() {
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

    cudnnTensorDescriptor_t getErrorInputTensorDescriptor() { return getDataOutputTensorDescriptor(); }

    cudnnTensorDescriptor_t getErrorOutputTensorDescriptor() { return getDataInputTensorDescriptor(); }

    bool operator==(const ConvolutionKernelRequirement &other) const {
        return gpuType == other.gpuType && filterWidth == other.filterWidth && filterHeight == other.filterHeight &&
               filterHorizontalStride == other.filterHorizontalStride && filterVerticalStride == other.filterVerticalStride &&
               leftAndRightPadWidth == other.leftAndRightPadWidth && topAndBottomPadHeight == other.topAndBottomPadHeight &&
               numInputChannels == other.numInputChannels && numOutputChannels == other.numOutputChannels && batchSize == other.batchSize &&
               numInputColumns == other.numInputColumns && numInputRows == other.numInputRows &&
               numOutputColumns == other.numOutputColumns && numOutputRows == other.numOutputRows;
    }

    string getGpuType() { return gpuType; }
    int getFilterWidth() { return filterWidth; }
    int getFilterHeight() { return filterHeight; }
    int getFilterHorizontalStride() { return filterHorizontalStride; }
    int getFilterVerticalStride() { return filterVerticalStride; }
    int getLeftAndRightPadWidth() { return leftAndRightPadWidth; }
    int getTopAndBottomPadHeight() { return topAndBottomPadHeight; }
    int getNumInputChannels() { return numInputChannels; }
    int getNumOutputChannels() { return numOutputChannels; }
    int getBatchSize() { return batchSize; }
    int getNumInputColumns() { return numInputColumns; }
    int getNumInputRows() { return numInputRows; }
    int getNumOutputColumns() { return numOutputColumns; }
    int getNumOutputRows() { return numOutputRows; }

    string toString() {
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

    atomic<int> *referenceCount;

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
