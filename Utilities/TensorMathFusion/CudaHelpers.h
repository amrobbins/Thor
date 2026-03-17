#pragma once

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

namespace ThorImplementation {

#define NVRTC_CHECK(call)                                                                                                             \
    do {                                                                                                                              \
        nvrtcResult status = (call);                                                                                                  \
        if (status != NVRTC_SUCCESS) {                                                                                                \
            const char* desc = nvrtcGetErrorString(status);                                                                           \
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " + #call + " failed with status " + \
                                     std::to_string(static_cast<int>(status)) + ": " + (desc ? desc : "<no description>"));           \
        }                                                                                                                             \
    } while (0)

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

#define CU_CHECK(call)                                                                                                         \
    do {                                                                                                                       \
        CUresult status = (call);                                                                                              \
        if (status != CUDA_SUCCESS) {                                                                                          \
            const char* name = nullptr;                                                                                        \
            const char* desc = nullptr;                                                                                        \
            cuGetErrorName(status, &name);                                                                                     \
            cuGetErrorString(status, &desc);                                                                                   \
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " + #call + " failed with " + \
                                     (name ? name : "<unknown>") + std::string(": ") + (desc ? desc : "<no description>"));    \
        }                                                                                                                      \
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
