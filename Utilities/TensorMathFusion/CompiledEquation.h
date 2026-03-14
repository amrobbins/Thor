#pragma once

#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"

namespace ThorImplementation {
struct EquationSignature {
    uint32_t rank;  // maybe just 1 for flattened contiguous V1
    uint32_t num_inputs;
    TensorDescriptor::DataType dtype;
    bool contiguous;
    int sm_major;
    int sm_minor;
    int device_num;
    bool use_fast_math;

    bool operator==(const EquationSignature& other) const = default;
};

struct EquationCacheKey {
    EquationCacheKey() = default;

    EquationCacheKey(const std::string& canonical_expr, const EquationSignature& sig, const bool broadcast_support) {
        this->canonical_expr = canonical_expr;
        this->sig = sig;
        this->sig.device_num = 0;  // Device num is not part of the kernel signature in terms of compiling, instead uses sm_major/minor
        this->broadcast_support = broadcast_support;
    }
    std::string canonical_expr;
    EquationSignature sig;
    bool broadcast_support;

    bool operator==(const EquationCacheKey& other) const = default;
};

struct CompiledEquation {
    EquationCacheKey key;
    CUmodule module = nullptr;
    CUfunction kernel = nullptr;
    std::string kernel_name;

    TensorDescriptor::DataType dtype;
    int deviceNum = 0;
    uint32_t num_inputs = 0;
    std::vector<std::string> input_names;

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

}  // namespace ThorImplementation

inline void hashCombine(std::size_t& seed, std::size_t value) { seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2); }

namespace std {
template <>
struct hash<ThorImplementation::EquationSignature> {
    size_t operator()(const ThorImplementation::EquationSignature& s) const {
        size_t h = 0;
        hashCombine(h, std::hash<uint32_t>{}(s.rank));
        hashCombine(h, std::hash<uint32_t>{}(s.num_inputs));
        hashCombine(h, std::hash<int>{}(static_cast<int>(s.dtype)));
        hashCombine(h, std::hash<bool>{}(s.contiguous));
        hashCombine(h, std::hash<int>{}(s.sm_major));
        hashCombine(h, std::hash<int>{}(s.sm_minor));
        hashCombine(h, std::hash<int>{}(s.device_num));
        return h;
    }
};

template <>
struct hash<ThorImplementation::EquationCacheKey> {
    std::size_t operator()(const ThorImplementation::EquationCacheKey& k) const {
        std::size_t h = std::hash<std::string>{}(k.canonical_expr);
        hashCombine(h, std::hash<ThorImplementation::EquationSignature>{}(k.sig));
        hashCombine(h, std::hash<bool>{}(k.broadcast_support));
        return h;
    }
};

}  // namespace std
