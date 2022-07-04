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
                         uint32_t lossDimSize,
                         bool reduceBatch,
                         bool reduceLoss,
                         TensorDescriptor::DataType sourceDataType,
                         TensorDescriptor::DataType destDataType,
                         Stream stream) {
    cudnnStatus_t cudnnStatus;

    // stream is kept because the cudnn handle is associated with it.
    this->stream = stream;

    doubleType = sourceDataType == TensorDescriptor::DataType::FP64;

    if (doubleType) {
        assert(destDataType == TensorDescriptor::DataType::FP64);
        zero = new double;
        ((double *)zero)[0] = 0.0;
        batchScale = new double;
        ((double *)batchScale)[0] = 1.0 / batchSize;
    } else {
        zero = new float;
        ((float *)zero)[0] = 0.0f;
        batchScale = new float;
        ((float *)batchScale)[0] = 1.0f / batchSize;
    }

    cudnnStatus = cudnnCreateReduceTensorDescriptor(&reduceTensorDescriptor);
    assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

    cudnnStatus = cudnnSetReduceTensorDescriptor(reduceTensorDescriptor,
                                                 CUDNN_REDUCE_TENSOR_ADD,
                                                 CudnnHelper::getCudnnDataType(TensorDescriptor::DataType::FP32),
                                                 CUDNN_NOT_PROPAGATE_NAN,
                                                 CUDNN_REDUCE_TENSOR_NO_INDICES,
                                                 CUDNN_32BIT_INDICES);
    assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

    printf("(%d, %d) -> (%d, %d)\n", batchletSize, lossDimSize, reduceBatch ? 1 : batchletSize, reduceLoss ? 1 : lossDimSize);
    fflush(stdout);

    sourceTensorDescriptor = Layer::createCudnnTensorDescriptor({batchletSize, lossDimSize}, sourceDataType);
    destTensorDescriptor = Layer::createCudnnTensorDescriptor({reduceBatch ? 1 : batchletSize, reduceLoss ? 1 : lossDimSize}, destDataType);
    cudnnStatus = cudnnGetReductionWorkspaceSize(
        stream.getCudnnHandle(), reduceTensorDescriptor, sourceTensorDescriptor, destTensorDescriptor, &workspaceSizeInBytes);
    assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

    if (workspaceSizeInBytes > 0)
        workspace = Tensor(TensorPlacement(TensorPlacement::MemDevices::GPU, 0),
                           TensorDescriptor(TensorDescriptor::DataType::UINT8, workspaceSizeInBytes));
    printf("workspace size %ld\n", workspaceSizeInBytes);
}

BatchReduce::~BatchReduce() {
    cudnnStatus_t cudnnStatus;

    cudnnStatus = cudnnDestroyReduceTensorDescriptor(reduceTensorDescriptor);
    assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

    if (doubleType) {
        delete (double *)zero;
        delete (double *)batchScale;
    } else {
        delete (float *)zero;
        delete (float *)batchScale;
    }
}

Stream BatchReduce::getStream() { return stream; }

void BatchReduce::reduce(void *sourceMem_d, void *destMem_d) {
    cudnnStatus_t cudnnStatus;

    cudnnStatus = cudnnReduceTensor(stream.getCudnnHandle(),
                                    reduceTensorDescriptor,
                                    nullptr,
                                    0,
                                    workspaceSizeInBytes > 0 ? workspace.getMemPtr() : nullptr,
                                    workspaceSizeInBytes,
                                    batchScale,
                                    sourceTensorDescriptor,
                                    sourceMem_d,
                                    zero,
                                    destTensorDescriptor,
                                    destMem_d);
    printf("CUDNN_STATUS = %d\n", cudnnStatus);
    fflush(stdout);
    assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
}
