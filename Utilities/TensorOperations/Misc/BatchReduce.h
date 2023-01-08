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
                uint32_t classDimSize,
                bool reduceBatch,
                bool reduceClass,
                ThorImplementation::TensorDescriptor::DataType sourceDataType,
                ThorImplementation::TensorDescriptor::DataType destDataType,
                Stream stream);

    virtual ~BatchReduce();

    uint64_t getWorkspaceSizeInBytes();

    void reduce(Tensor source, Tensor dest);

    Stream getStream();

    bool isWire();
    bool isScalarDivide();

   protected:
    uint64_t computeWorkspaceSizeInBytes(uint32_t batchletSize,
                                         uint32_t batchSize,
                                         uint32_t classDimSize,
                                         TensorDescriptor::DataType sourceDataType,
                                         TensorDescriptor::DataType destDataType);

    Stream stream;
    cudnnReduceTensorDescriptor_t reduceTensorDescriptor;
    cudnnTensorDescriptor_t sourceTensorDescriptor;
    cudnnTensorDescriptor_t destTensorDescriptor;
    ThorImplementation::Tensor workspace;
    uint64_t workspaceSizeInBytes;
    bool doubleType;
    uint32_t batchletSize;
    uint32_t batchSize;
    uint32_t classDimSize;
    bool reduceBatch;
    bool reduceClass;
    void *batchScale = nullptr;
    void *zero = nullptr;
    void *one = nullptr;
};

}  // namespace ThorImplementation