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
enum class ExprOp : uint16_t { INPUT = 3, SCALAR_F32, ADD, SUB, MUL, DIV, POW_SCALAR, NEG, EXP, LOG, SQRT };

struct ExprNode {
    ExprOp op;
    uint32_t lhs = UINT32_MAX;
    uint32_t rhs = UINT32_MAX;  // unused for unary/scalar ops
    uint32_t input_index = UINT32_MAX;
    float scalar_f32 = 0.0f;  // used for SCALAR_F32 or POW_SCALAR exponent
};

struct PhysicalExpression {
    std::vector<ExprNode> nodes;
    uint32_t output_node;
    uint32_t num_inputs;
};

struct EquationSignature {
    uint32_t rank;  // maybe just 1 for flattened contiguous V1
    uint32_t num_inputs;
    cudaDataType_t dtype;  // FP32 only for now
    bool contiguous;
    int sm_major;
    int sm_minor;

    bool operator==(const EquationSignature& other) const = default;
};

struct EquationCacheKey {
    std::string canonical_expr;
    EquationSignature sig;

    bool operator==(const EquationCacheKey& other) const = default;
};

void hashCombine(std::size_t& seed, std::size_t value);

struct CompiledEquation {
    EquationCacheKey key;
    CUmodule module = nullptr;
    CUfunction kernel = nullptr;
    std::string kernel_name;
    uint32_t num_inputs = 0;

    CompiledEquation() = default;
    CompiledEquation(const CompiledEquation&) = delete;
    CompiledEquation& operator=(const CompiledEquation&) = delete;
    CompiledEquation(CompiledEquation&&) = default;
    CompiledEquation& operator=(CompiledEquation&&) = default;

    ~CompiledEquation() {
        if (module != nullptr) {
            cuModuleUnload(module);
        }
    }
};

std::string formatFloatCanonical(float x);

bool isCommutative(ExprOp op);

std::string opName(ExprOp op);

std::string canonicalizeNode(const PhysicalExpression& expr, uint32_t nodeIndex, std::unordered_map<uint32_t, std::string>& memo);

std::string canonicalize(const PhysicalExpression& expr);

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

namespace std {
template <>
struct hash<ThorImplementation::EquationSignature> {
    size_t operator()(const ThorImplementation::EquationSignature& s) const {
        size_t h = 0;
        ThorImplementation::hashCombine(h, std::hash<uint32_t>{}(s.rank));
        ThorImplementation::hashCombine(h, std::hash<uint32_t>{}(s.num_inputs));
        ThorImplementation::hashCombine(h, std::hash<int>{}(static_cast<int>(s.dtype)));
        ThorImplementation::hashCombine(h, std::hash<bool>{}(s.contiguous));
        ThorImplementation::hashCombine(h, std::hash<int>{}(s.sm_major));
        ThorImplementation::hashCombine(h, std::hash<int>{}(s.sm_minor));
        return h;
    }
};

template <>
struct hash<ThorImplementation::EquationCacheKey> {
    std::size_t operator()(const ThorImplementation::EquationCacheKey& k) const {
        std::size_t h = std::hash<std::string>{}(k.canonical_expr);
        ThorImplementation::hashCombine(h, std::hash<ThorImplementation::EquationSignature>{}(k.sig));
        return h;
    }
};

}  // namespace std
