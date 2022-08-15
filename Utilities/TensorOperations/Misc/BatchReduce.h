#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "Utilities/Common/CudnnHelper.h"
#include "Utilities/Common/Stream.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cudnn.h>

namespace ThorImplementation {

class BatchReduce {
   public:
    // Stream is saved because the cudnnHandle that is created belongs to the stream
    BatchReduce(uint32_t batchletSize,
                uint32_t batchSize,
                uint32_t lossDimSize,
                bool reduceBatch,
                bool reduceLoss,
                ThorImplementation::TensorDescriptor::DataType sourceDataType,
                ThorImplementation::TensorDescriptor::DataType destDataType,
                Stream stream);

    virtual ~BatchReduce();

    uint64_t getWorkspaceSizeInBytes();

    void reduce(void *sourceMem_d, void *destMem_d);

    Stream getStream();

   protected:
    uint64_t computeWorkspaceSizeInBytes(uint32_t batchletSize,
                                         uint32_t batchSize,
                                         uint32_t lossDimSize,
                                         bool reduceBatch,
                                         bool reduceLoss,
                                         TensorDescriptor::DataType sourceDataType,
                                         TensorDescriptor::DataType destDataType);

    Stream stream;
    cudnnReduceTensorDescriptor_t reduceTensorDescriptor;
    cudnnTensorDescriptor_t sourceTensorDescriptor;
    cudnnTensorDescriptor_t destTensorDescriptor;
    ThorImplementation::Tensor workspace;
    uint64_t workspaceSizeInBytes;
    bool doubleType;
    void *batchScale;
    void *zero;
};

}  // namespace ThorImplementation