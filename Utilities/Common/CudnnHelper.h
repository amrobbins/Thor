#pragma once

#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"

#include <cudnn.h>

#include <map>
#include <mutex>
#include <thread>

namespace ThorImplementation {

class CudnnHelper {
   public:
    static cudnnDataType_t getCudnnDataType(const TensorDescriptor::DataType dataType) {
        switch (dataType) {
            case ThorImplementation::TensorDescriptor::DataType::FP64:
                return CUDNN_DATA_DOUBLE;
            case ThorImplementation::TensorDescriptor::DataType::FP32:
                return CUDNN_DATA_FLOAT;
            case ThorImplementation::TensorDescriptor::DataType::FP16:
                return CUDNN_DATA_HALF;
            case ThorImplementation::TensorDescriptor::DataType::INT8:
                return CUDNN_DATA_INT8;
            case ThorImplementation::TensorDescriptor::DataType::INT32:
                return CUDNN_DATA_INT32;
            case ThorImplementation::TensorDescriptor::DataType::UINT8:
                return CUDNN_DATA_UINT8;
            case ThorImplementation::TensorDescriptor::DataType::INT64:
                return CUDNN_DATA_INT64;
            default:
                assert(false);
        }
        assert(false);
        return CUDNN_DATA_HALF;
    }

    static cudnnHandle_t getCudnnHandle(uint32_t gpuNum) {
        std::unique_lock<std::mutex> lck(mtx);

        cudnnHandle_t cudnnHandle;

        if (cudnnHandles.count(gpuNum) == 0)
            cudnnHandles[gpuNum] = std::map<std::thread::id, cudnnHandle_t>();

        if (cudnnHandles[gpuNum].count(std::this_thread::get_id()) == 0) {
            cudnnStatus_t cudnnStatus;
            cudnnStatus = cudnnCreate(&cudnnHandle);
            if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
                printf("cudnnStatus %d : %s   gpu:%d\n", cudnnStatus, cudnnGetErrorString(cudnnStatus), gpuNum);
                fflush(stdout);
            }
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
            cudnnHandles[gpuNum][std::this_thread::get_id()] = cudnnHandle;
        } else {
            cudnnHandle = cudnnHandles[gpuNum][std::this_thread::get_id()];
        }

        return cudnnHandle;
    }

   private:
    CudnnHelper();

    static std::map<uint32_t, std::map<std::thread::id, cudnnHandle_t>> cudnnHandles;

    static std::mutex mtx;
};

}  // namespace ThorImplementation
