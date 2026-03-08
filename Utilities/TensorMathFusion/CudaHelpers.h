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

#define NVRTC_CHECK(call)                                                                                      \
    do {                                                                                                       \
        nvrtcResult status = (call);                                                                           \
        if (status != NVRTC_SUCCESS) {                                                                         \
            fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #call, nvrtcGetErrorString(status)); \
            exit(1);                                                                                           \
        }                                                                                                      \
    } while (0)

#define NVJITLINK_CHECK(handle, call)                                      \
    do {                                                                   \
        nvJitLinkResult status = (call);                                   \
        if (status != NVJITLINK_SUCCESS) {                                 \
            size_t errSize = 0;                                            \
            nvJitLinkGetErrorLogSize((handle), &errSize);                  \
            if (errSize > 1) {                                             \
                char* errLog = (char*)malloc(errSize);                     \
                if (errLog) {                                              \
                    nvJitLinkGetErrorLog((handle), errLog);                \
                    fprintf(stderr, "nvJitLink error log:\n%s\n", errLog); \
                    free(errLog);                                          \
                }                                                          \
            }                                                              \
            fprintf(stderr, "nvJitLink error code: %d\n", (int)status);    \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

#define CU_CHECK(call)                                 \
    do {                                               \
        CUresult status = (call);                      \
        if (status != CUDA_SUCCESS) {                  \
            const char* name = nullptr;                \
            const char* desc = nullptr;                \
            cuGetErrorName(status, &name);             \
            cuGetErrorString(status, &desc);           \
            fprintf(stderr,                            \
                    "%s:%d: %s failed with %s: %s\n",  \
                    __FILE__,                          \
                    __LINE__,                          \
                    #call,                             \
                    name ? name : "<unknown>",         \
                    desc ? desc : "<no description>"); \
            exit(1);                                   \
        }                                              \
    } while (0)

}  // namespace ThorImplementation
