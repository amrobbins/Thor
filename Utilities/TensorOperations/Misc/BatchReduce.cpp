#include "BatchReduce.h"

using namespace ThorImplementation;

/**
 * Cudnn will reduce any dimension of the output tensor that is not equal to the input dimension and is equal to 1.
 * So, to reduce across the batch, the data goes like this:
 *
 * The 2 dimensional array of all features for each batch item is laid out like this:
 *
 * 0  1  2  3
 * 4  5  6  7
 * 8  9 10 11
 *
 * for a feature size of 4 and a batch size of 3. i.e. the elements of a batch are contiguous in memory.
 * So in this case the tensor size is batch_size X feature_size = 3 X 4
 * We want to reduce across the batch so our resulting tensor size will be 1 X 4,
 * i.e. one reduced element for each feature.
 */
BatchReduce::BatchReduce(uint32_t batchletSize,
                         uint32_t batchSize,
                         uint32_t classDimSize,
                         bool reduceBatch,
                         bool reduceClass,
                         TensorDescriptor::DataType sourceDataType,
                         TensorDescriptor::DataType destDataType,
                         Stream stream,
                         bool doBatchSizeDivide) {
    // stream is kept because the cudnn handle is associated with it.
    this->stream = stream;

    doubleType = sourceDataType == TensorDescriptor::DataType::FP64;

    this->batchletSize = batchletSize;
    this->batchSize = batchSize;
    this->classDimSize = classDimSize;
    this->reduceBatch = reduceBatch;
    this->reduceClass = reduceClass;
    this->doBatchSizeDivide = doBatchSizeDivide;

    assert(batchSize > 0);
    assert(classDimSize > 0);
    if (isWire() || isScalarDivide()) {
        return;
    }

    if (doubleType) {
        assert(destDataType == TensorDescriptor::DataType::FP64);
        zero = new double;
        ((double *)zero)[0] = 0.0;
        one = new double;
        ((double *)one)[0] = 1.0;
        batchScale = new double;
        ((double *)batchScale)[0] = 1.0 / batchSize;
    } else {
        zero = new float;
        ((float *)zero)[0] = 0.0f;
        one = new float;
        ((float *)one)[0] = 1.0f;
        batchScale = new float;
        ((float *)batchScale)[0] = 1.0f / batchSize;
    }

    workspaceSizeInBytes = computeWorkspaceSizeInBytes(batchletSize, batchSize, classDimSize, sourceDataType, destDataType);
    if (workspaceSizeInBytes > 0)
        workspace = Tensor(TensorPlacement(TensorPlacement::MemDevices::GPU, 0),
                           TensorDescriptor(TensorDescriptor::DataType::UINT8, {workspaceSizeInBytes}));
}

bool BatchReduce::isWire() { return batchletSize == 1 && classDimSize == 1 && batchSize == 1; }

bool BatchReduce::isScalarDivide() { return batchletSize == 1 && classDimSize == 1 && batchSize != 1; }

uint64_t BatchReduce::computeWorkspaceSizeInBytes(uint32_t batchletSize,
                                                  uint32_t batchSize,
                                                  uint32_t classDimSize,
                                                  TensorDescriptor::DataType sourceDataType,
                                                  TensorDescriptor::DataType destDataType) {
    if (isWire() || isScalarDivide()) {
        workspaceSizeInBytes = 0;
        return workspaceSizeInBytes;
    }

    cudnnStatus_t cudnnStatus;
    cudnnStatus = cudnnCreateReduceTensorDescriptor(&reduceTensorDescriptor);
    assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

    cudnnStatus = cudnnSetReduceTensorDescriptor(reduceTensorDescriptor,
                                                 CUDNN_REDUCE_TENSOR_ADD,
                                                 CudnnHelper::getCudnnDataType(TensorDescriptor::DataType::FP32),
                                                 CUDNN_NOT_PROPAGATE_NAN,
                                                 CUDNN_REDUCE_TENSOR_NO_INDICES,
                                                 CUDNN_32BIT_INDICES);
    assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

    sourceTensorDescriptor = Layer::createCudnnTensorDescriptor({batchletSize, classDimSize}, sourceDataType);
    destTensorDescriptor =
        Layer::createCudnnTensorDescriptor({reduceBatch ? 1 : batchletSize, reduceClass ? 1 : classDimSize}, destDataType);
    cudnnStatus = cudnnGetReductionWorkspaceSize(
        stream.getCudnnHandle(), reduceTensorDescriptor, sourceTensorDescriptor, destTensorDescriptor, &workspaceSizeInBytes);
    assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

    return workspaceSizeInBytes;
}

uint64_t BatchReduce::getWorkspaceSizeInBytes() { return workspaceSizeInBytes; }

BatchReduce::~BatchReduce() {
    if (isWire() || isScalarDivide())
        return;

    cudnnStatus_t cudnnStatus;

    cudnnStatus = cudnnDestroyReduceTensorDescriptor(reduceTensorDescriptor);
    assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

    if (doubleType) {
        if (zero != nullptr) {
            delete (double *)zero;
            zero = nullptr;
        }
        if (one != nullptr) {
            delete (double *)one;
            one = nullptr;
        }
        if (batchScale != nullptr) {
            delete (double *)batchScale;
            batchScale = nullptr;
        }
    } else {
        if (zero != nullptr) {
            delete (float *)zero;
            zero = nullptr;
        }
        if (one != nullptr) {
            delete (float *)one;
            one = nullptr;
        }
        if (batchScale != nullptr) {
            delete (float *)batchScale;
            batchScale = nullptr;
        }
    }
}

Stream BatchReduce::getStream() { return stream; }

void BatchReduce::reduce(Tensor source, Tensor dest, bool accumulate) {
    if (isWire()) {
        if (source != dest)
            dest.copyFromAsync(source, stream);
        return;
    } else if (isScalarDivide()) {
        dest.divide(source, batchSize, stream);
        return;
    }

    cudnnStatus_t cudnnStatus;

    assert(source.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(dest.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

    cudnnStatus = cudnnReduceTensor(stream.getCudnnHandle(),
                                    reduceTensorDescriptor,
                                    nullptr,
                                    0,
                                    workspaceSizeInBytes > 0 ? workspace.getMemPtr() : nullptr,
                                    workspaceSizeInBytes,
                                    reduceBatch && doBatchSizeDivide ? batchScale : one,
                                    sourceTensorDescriptor,
                                    source.getMemPtr(),
                                    accumulate ? one : zero,
                                    destTensorDescriptor,
                                    dest.getMemPtr());

    if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
        printf("cudnnStatus %d : %s\n", cudnnStatus, cudnnGetErrorString(cudnnStatus));
        fflush(stdout);
    }
    assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
}
