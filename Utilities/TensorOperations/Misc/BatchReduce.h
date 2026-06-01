#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "Utilities/Common/CudnnHelper.h"
#include "Utilities/Common/Stream.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cudnn.h>

#include <memory>

namespace ThorImplementation {

class FusedEquation;

class BatchReduce {
   public:
    // Stream is saved because the cudnnHandle that is created belongs to the stream
    BatchReduce(uint32_t batchletSize,
                uint32_t batchSize,
                uint32_t classDimSize,
                bool reduceBatch,
                bool reduceClass,
                DataType sourceDataType,
                DataType destDataType,
                Stream stream,
                bool doBatchSizeDivide = true);

    virtual ~BatchReduce();

    uint64_t getWorkspaceSizeInBytes();

    void reduce(Tensor source, Tensor dest, bool accumulate = false);

    Stream getStream();

    bool isWire();
    bool isScalarDivide();

   protected:
    FusedEquation &getWireAccumulateEquation();
    FusedEquation &getScalarDivideEquation(bool accumulate);

    uint64_t computeWorkspaceSizeInBytes(uint32_t batchletSize,
                                         uint32_t batchSize,
                                         uint32_t classDimSize,
                                         DataType sourceDataType,
                                         DataType destDataType);

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
    bool doBatchSizeDivide;
    DataType destDataType;
    std::unique_ptr<FusedEquation> wireAccumulateEquation;
    std::unique_ptr<FusedEquation> scalarDivideEquation;
    std::unique_ptr<FusedEquation> scalarDivideAccumulateEquation;
    void *batchScale = nullptr;
    void *zero = nullptr;
    void *one = nullptr;
};

}  // namespace ThorImplementation