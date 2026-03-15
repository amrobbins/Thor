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
            const char* desc = cudnnGetErrorString(status);                                                                           \
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " + #call + " failed with status " + \
                                     std::to_string(static_cast<int>(status)) + ": " + (desc ? desc : "<no description>"));           \
        }                                                                                                                             \
    } while (0)

}  // namespace ThorImplementation
