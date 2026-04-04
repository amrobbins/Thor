#pragma once

#include <iostream>
#include <variant>

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/TensorMathFusion/Expression.h"

namespace ThorImplementation {

struct TensorScalarBinding {
    Tensor buffer;
    uint64_t byteOffset = 0;
    TensorDescriptor::DataType sourceDType = TensorDescriptor::DataType::FP32;
};

using RuntimeInputValue = std::variant<Tensor, float, TensorScalarBinding>;

struct EquationSignature {
    uint32_t num_inputs;
    int sm_major;
    int sm_minor;
    int device_num;
    bool use_fast_math;

    bool operator==(const EquationSignature& other) const = default;
};

struct EquationCacheKey {
    EquationCacheKey() = default;

    EquationCacheKey(const std::string& canonical_expr, const EquationSignature& sig, bool use_uint32_flat_index_math = false) {
        this->canonical_expr = canonical_expr;
        this->sig = sig;
        this->sig.device_num = 0;  // Device num is not part of the kernel signature in terms of compiling, instead uses sm_major/minor
        this->use_uint32_flat_index_math = use_uint32_flat_index_math;
    }
    std::string canonical_expr;
    EquationSignature sig;
    bool use_uint32_flat_index_math = false;

    bool operator==(const EquationCacheKey& other) const = default;
};

struct CompiledEquation {
    enum class LaunchKind {
        Flat,
        BroadcastSingle,
        BroadcastGrouped,
    };

    EquationCacheKey key;
    CUmodule module = nullptr;
    CUfunction kernel = nullptr;
    std::string kernel_name;

    LaunchKind launch_kind = LaunchKind::Flat;
    uint32_t num_broadcast_groups = 0;
    uint32_t elements_per_thread = 1;
    bool uses_uint32_numel_arg = false;

    int deviceNum = 0;
    std::vector<std::string> input_names;
    std::vector<NamedInput::Kind> input_kinds;
    std::vector<TensorDescriptor::DataType> input_dtypes;
    std::vector<TensorDescriptor::DataType> output_dtypes;

    uint64_t numInputs() { return input_names.size(); }
    uint64_t numInputs() const { return input_names.size(); }
    uint64_t numOutputs() const { return output_dtypes.size(); }

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

struct CompiledReduction {
    const ExprOp op;
    std::vector<uint64_t> reduction_axes;
    std::vector<uint64_t> squeeze_axes;
    const TensorDescriptor::DataType input_dtype;
    const TensorDescriptor::DataType compute_dtype;
    const TensorDescriptor::DataType output_dtype;

    bool operator==(const CompiledReduction& other) const = default;

    CompiledReduction(ExprOp op,
                      std::vector<uint64_t> reduction_axes,
                      std::vector<uint64_t> squeeze_axes,
                      TensorDescriptor::DataType input_dtype,
                      TensorDescriptor::DataType output_dtype,
                      Optional<TensorDescriptor::DataType> compute_dtype)
        : op(op),
          reduction_axes(std::move(reduction_axes)),
          squeeze_axes(std::move(squeeze_axes)),
          input_dtype(input_dtype),
          compute_dtype(compute_dtype.isPresent() ? compute_dtype.get() : output_dtype),
          output_dtype(output_dtype) {
        // Canonical representation: sorted and uniquified
        std::sort(this->reduction_axes.begin(), this->reduction_axes.end());
        // Remove adjacent duplicates:
        this->reduction_axes.erase(std::unique(this->reduction_axes.begin(), this->reduction_axes.end()), this->reduction_axes.end());

        std::sort(this->squeeze_axes.begin(), this->squeeze_axes.end());
        this->squeeze_axes.erase(std::unique(this->squeeze_axes.begin(), this->squeeze_axes.end()), this->squeeze_axes.end());
    }
};

struct CompiledArgMinMax {
    const ExprOp op;
    std::vector<uint64_t> reduction_axes;
    std::vector<uint64_t> squeeze_axes;
    const TensorDescriptor::DataType input_dtype;
    const TensorDescriptor::DataType output_dtype;
    const TensorDescriptor::DataType compute_dtype;

    bool operator==(const CompiledArgMinMax& other) const = default;

    CompiledArgMinMax(ExprOp op,
                      std::vector<uint64_t> reduction_axes,
                      std::vector<uint64_t> squeeze_axes,
                      TensorDescriptor::DataType input_dtype,
                      TensorDescriptor::DataType output_dtype,
                      Optional<TensorDescriptor::DataType> compute_dtype)
        : op(op),
          reduction_axes(std::move(reduction_axes)),
          squeeze_axes(std::move(squeeze_axes)),
          input_dtype(input_dtype),
          output_dtype(output_dtype),
          compute_dtype(compute_dtype.isPresent() ? compute_dtype.get() : TensorDescriptor::DataType::FP32) {
        std::sort(this->reduction_axes.begin(), this->reduction_axes.end());
        this->reduction_axes.erase(std::unique(this->reduction_axes.begin(), this->reduction_axes.end()), this->reduction_axes.end());

        std::sort(this->squeeze_axes.begin(), this->squeeze_axes.end());
        this->squeeze_axes.erase(std::unique(this->squeeze_axes.begin(), this->squeeze_axes.end()), this->squeeze_axes.end());
    }
};

struct CompiledReduceMinMaxBackward {
    const ExprOp op;
    std::vector<uint64_t> reduction_axes;
    std::vector<uint64_t> squeeze_axes;
    const TensorDescriptor::DataType input_dtype;
    const TensorDescriptor::DataType output_dtype;
    const TensorDescriptor::DataType compute_dtype;

    bool operator==(const CompiledReduceMinMaxBackward& other) const = default;

    CompiledReduceMinMaxBackward(ExprOp op,
                                 std::vector<uint64_t> reduction_axes,
                                 std::vector<uint64_t> squeeze_axes,
                                 TensorDescriptor::DataType input_dtype,
                                 TensorDescriptor::DataType output_dtype,
                                 Optional<TensorDescriptor::DataType> compute_dtype)
        : op(op),
          reduction_axes(std::move(reduction_axes)),
          squeeze_axes(std::move(squeeze_axes)),
          input_dtype(input_dtype),
          output_dtype(output_dtype),
          compute_dtype(compute_dtype.isPresent() ? compute_dtype.get() : TensorDescriptor::DataType::FP32) {
        std::sort(this->reduction_axes.begin(), this->reduction_axes.end());
        this->reduction_axes.erase(std::unique(this->reduction_axes.begin(), this->reduction_axes.end()), this->reduction_axes.end());

        std::sort(this->squeeze_axes.begin(), this->squeeze_axes.end());
        this->squeeze_axes.erase(std::unique(this->squeeze_axes.begin(), this->squeeze_axes.end()), this->squeeze_axes.end());
    }
};

}  // namespace ThorImplementation

inline void hashCombine(std::size_t& seed, std::size_t value) { seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2); }

namespace std {
template <>
struct hash<ThorImplementation::EquationSignature> {
    size_t operator()(const ThorImplementation::EquationSignature& s) const noexcept {
        size_t h = 0;
        hashCombine(h, std::hash<uint32_t>{}(s.num_inputs));
        hashCombine(h, std::hash<int>{}(s.sm_major));
        hashCombine(h, std::hash<int>{}(s.sm_minor));
        hashCombine(h, std::hash<int>{}(s.device_num));
        return h;
    }
};

template <>
struct hash<ThorImplementation::EquationCacheKey> {
    std::size_t operator()(const ThorImplementation::EquationCacheKey& k) const noexcept {
        std::size_t h = std::hash<std::string>{}(k.canonical_expr);
        hashCombine(h, std::hash<ThorImplementation::EquationSignature>{}(k.sig));
        hashCombine(h, std::hash<bool>{}(k.use_uint32_flat_index_math));
        return h;
    }
};

}  // namespace std
