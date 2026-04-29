#pragma once

#include <nvrtc.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/CudaDriver/CudaDrivertApi.h"

namespace ThorImplementation {

#define NVJITLINK_CHECK(handle, call)                                                                                  \
    do {                                                                                                               \
        nvJitLinkResult status = (call);                                                                               \
        if (status != NVJITLINK_SUCCESS) {                                                                             \
            size_t errSize = 0;                                                                                        \
            nvJitLinkGetErrorLogSize((handle), &errSize);                                                              \
            if (errSize > 1) {                                                                                         \
                char* errLog = (char*)malloc(errSize);                                                                 \
                if (errLog) {                                                                                          \
                    nvJitLinkGetErrorLog((handle), errLog);                                                            \
                    fprintf(stderr, "nvJitLink error log:\n%s\n", errLog);                                             \
                    free(errLog);                                                                                      \
                }                                                                                                      \
            }                                                                                                          \
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " + #call +           \
                                     " failed with nvJitLink error code " + std::to_string(static_cast<int>(status))); \
        }                                                                                                              \
    } while (0)

#define CU_CHECK(call)                                                                                                              \
    do {                                                                                                                            \
        auto& cu__ = CudaDriverApi::instance();                                                                                     \
        CUresult status__ = (cu__.call);                                                                                            \
        if (status__ != CUDA_SUCCESS) {                                                                                             \
            const char* name__ = nullptr;                                                                                           \
            const char* desc__ = nullptr;                                                                                           \
            (void)cu__.cuGetErrorName(status__, &name__);                                                                           \
            (void)cu__.cuGetErrorString(status__, &desc__);                                                                         \
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " + #call + " failed with " +      \
                                     (name__ ? name__ : "<unknown>") + std::string(": ") + (desc__ ? desc__ : "<no description>")); \
        }                                                                                                                           \
    } while (0)

#define CUDNN_CHECK(call)                                                                                                             \
    do {                                                                                                                              \
        cudnnStatus_t status = (call);                                                                                                \
        if (status != CUDNN_STATUS_SUCCESS) {                                                                                         \
            char last_err[2048] = {0};                                                                                                \
            cudnnGetLastErrorString(last_err, sizeof(last_err));                                                                      \
            const char* status_name = cudnnGetErrorString(status);                                                                    \
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " + #call + " failed with status " + \
                                     (status_name ? status_name : "CUDNN_UNKNOWN_STATUS") + " (" +                                    \
                                     std::to_string(static_cast<int>(status)) + ")" +                                                 \
                                     (last_err[0] ? std::string("\ncuDNN detail: ") + last_err : ""));                                \
        }                                                                                                                             \
    } while (0)

#define CUDA_CHECK(call)                                                                                                       \
    do {                                                                                                                       \
        cudaError_t err__ = (call);                                                                                            \
        if (err__ != cudaSuccess) {                                                                                            \
            const char* name__ = cudaGetErrorName(err__);                                                                      \
            const char* desc__ = cudaGetErrorString(err__);                                                                    \
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " + #call + " failed with " + \
                                     (name__ ? name__ : "cudaErrorUnknown") + " (" + std::to_string(static_cast<int>(err__)) + \
                                     "): " + (desc__ ? desc__ : "<no description>"));                                          \
        }                                                                                                                      \
    } while (0)

}  // namespace ThorImplementation
