#include "BatchReduce.h"

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
BatchReduce::BatchReduce(uint32_t batchSize,
                         uint32_t featureSize,
                         ThorImplementation::TensorDescriptor::DataType sourceDataType,
                         ThorImplementation::TensorDescriptor::DataType destDataType,
                         ThorImplementation::TensorDescriptor::DataType computationDataType,
                         Stream stream) {
    cudnnStatus_t cudnnStatus;

    // stream is kept because the cudnn handle is associated with it.
    this->stream = stream;

    doubleType = sourceDataType == ThorImplementation::TensorDescriptor::DataType::FP64;

    if (doubleType) {
        assert(destDataType == ThorImplementation::TensorDescriptor::DataType::FP64);
        zero = new double;
        ((double *)zero)[0] = 0.0;
        one = new double;
        ((double *)one)[0] = 1.0;
    } else {
        zero = new float;
        ((float *)zero)[0] = 0.0f;
        one = new float;
        ((float *)one)[0] = 1.0f;
    }

    cudnnStatus = cudnnCreateReduceTensorDescriptor(&reduceTensorDescriptor);
    assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

    cudnnStatus = cudnnSetReduceTensorDescriptor(reduceTensorDescriptor,
                                                 CUDNN_REDUCE_TENSOR_ADD,
                                                 ThorImplementation::CudnnHelper::getCudnnDataType(computationDataType),
                                                 CUDNN_NOT_PROPAGATE_NAN,
                                                 CUDNN_REDUCE_TENSOR_NO_INDICES,
                                                 CUDNN_32BIT_INDICES);
    assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

    sourceTensorDescriptor = ThorImplementation::Layer::createCudnnTensorDescriptor({batchSize, featureSize}, sourceDataType);
    destTensorDescriptor = ThorImplementation::Layer::createCudnnTensorDescriptor({1, featureSize}, destDataType);
    cudnnGetReductionWorkspaceSize(
        stream.getCudnnHandle(), reduceTensorDescriptor, sourceTensorDescriptor, destTensorDescriptor, &workspaceSizeInBytes);

    if (workspaceSizeInBytes > 0)
        workspace = ThorImplementation::Tensor(
            ThorImplementation::TensorPlacement(ThorImplementation::TensorPlacement::MemDevices::GPU, 0),
            ThorImplementation::TensorDescriptor(ThorImplementation::TensorDescriptor::DataType::UINT8, workspaceSizeInBytes));
}

BatchReduce::~BatchReduce() {
    cudnnStatus_t cudnnStatus;

    cudnnStatus = cudnnDestroyReduceTensorDescriptor(reduceTensorDescriptor);
    assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

    if (doubleType) {
        delete (double *)zero;
        delete (double *)one;
    } else {
        delete (float *)zero;
        delete (float *)one;
    }
}

Stream BatchReduce::getStream() { return stream; }

void BatchReduce::reduce(void *sourceMem_d, void *destMem_d) {
    cudnnStatus_t cudnnStatus;

    cudnnStatus = cudnnReduceTensor(stream.getCudnnHandle(),
                                    reduceTensorDescriptor,
                                    nullptr,
                                    0,
                                    workspace.getMemPtr(),
                                    workspaceSizeInBytes,
                                    one,
                                    sourceTensorDescriptor,
                                    sourceMem_d,
                                    zero,
                                    destTensorDescriptor,
                                    destMem_d);
    assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
}
