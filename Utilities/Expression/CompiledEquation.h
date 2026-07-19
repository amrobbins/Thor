#pragma once

#include <optional>

#include <iostream>
#include <variant>
#include <utility>

#include "CudaHelpers.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Expression/Expression.h"

namespace ThorImplementation {

struct TensorScalarBinding {
    Tensor buffer;
    uint64_t byteOffset = 0;
    DataType sourceDType = DataType::FP32;
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
        FusedTiledTranspose,
    };

    EquationCacheKey key;
    CUmodule module = nullptr;
    CUfunction kernel = nullptr;
    std::string kernel_name;

    LaunchKind launch_kind = LaunchKind::Flat;
    uint32_t num_broadcast_groups = 0;
    uint32_t elements_per_thread = 1;
    uint32_t tiled_transpose_pack_scalars = 1;
    bool uses_uint32_numel_arg = false;
    bool uses_uint32_tiled_transpose_index_math = true;
    // True only for explicitly marked ragged valuewise kernels. Such kernels
    // compute logical numel from offsets[B] on device and use a grid-stride loop.
    bool uses_device_runtime_extent = false;

    // Debug/test metadata for the tiled logical-transpose consumer auto-swizzle path.
    // These are intentionally not consulted by the runtime launcher; they let tests
    // assert that the intended optimized lowering was selected instead of merely
    // asserting that the stage is some generic FusedKernel.
    bool uses_tiled_logical_transpose_consumer = false;
    uint32_t tiled_logical_transpose_slot_bytes = sizeof(unsigned int);
    uint32_t tiled_logical_transpose_dense_packed_input_load_count = 0;
    uint32_t tiled_logical_transpose_vectorized_output_count = 0;

    int deviceNum = 0;
    std::vector<std::string> input_names;
    std::vector<NamedInput::Kind> input_kinds;
    std::vector<DataType> input_dtypes;
    std::vector<DataType> output_dtypes;

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
            try {
                CU_CHECK(cuModuleUnload(module));
            } catch (...) {
            }
        }
    }
};

struct CompiledReduction {
    const ExprOp op;
    std::vector<uint64_t> reduction_axes;
    std::vector<uint64_t> squeeze_axes;
    const DataType input_dtype;
    const DataType compute_dtype;
    const DataType output_dtype;

    bool operator==(const CompiledReduction& other) const = default;

    CompiledReduction(ExprOp op,
                      std::vector<uint64_t> reduction_axes,
                      std::vector<uint64_t> squeeze_axes,
                      DataType input_dtype,
                      DataType output_dtype,
                      std::optional<DataType> compute_dtype)
        : op(op),
          reduction_axes(std::move(reduction_axes)),
          squeeze_axes(std::move(squeeze_axes)),
          input_dtype(input_dtype),
          compute_dtype(compute_dtype.has_value() ? compute_dtype.value() : output_dtype),
          output_dtype(output_dtype) {
        // Canonical representation: sorted and uniquified
        std::sort(this->reduction_axes.begin(), this->reduction_axes.end());
        // Remove adjacent duplicates:
        this->reduction_axes.erase(std::unique(this->reduction_axes.begin(), this->reduction_axes.end()), this->reduction_axes.end());

        std::sort(this->squeeze_axes.begin(), this->squeeze_axes.end());
        this->squeeze_axes.erase(std::unique(this->squeeze_axes.begin(), this->squeeze_axes.end()), this->squeeze_axes.end());
    }
};

struct CompiledSegmentedReduction {
    ExprOp op = ExprOp::SEGMENTED_REDUCE_SUM;
    DataType input_dtype = DataType::FP32;
    DataType output_dtype = DataType::FP32;
    DataType offset_dtype = DataType::UINT32;

    bool operator==(const CompiledSegmentedReduction& other) const = default;

    CompiledSegmentedReduction(ExprOp op, DataType input_dtype, DataType output_dtype, DataType offset_dtype)
        : op(op), input_dtype(input_dtype), output_dtype(output_dtype), offset_dtype(offset_dtype) {}
};

struct CompiledScan {
    const ScanOp op;
    const ScanMode mode;
    const uint64_t axis;
    const bool reverse;
    const bool segmented_by_offsets;
    const DataType input_dtype;
    const DataType output_dtype;
    const std::optional<DataType> offset_dtype;

    bool operator==(const CompiledScan& other) const = default;

    CompiledScan(ScanOp op,
                 ScanMode mode,
                 uint64_t axis,
                 bool reverse,
                 bool segmented_by_offsets,
                 DataType input_dtype,
                 DataType output_dtype,
                 std::optional<DataType> offset_dtype = std::nullopt)
        : op(op),
          mode(mode),
          axis(axis),
          reverse(reverse),
          segmented_by_offsets(segmented_by_offsets),
          input_dtype(input_dtype),
          output_dtype(output_dtype),
          offset_dtype(offset_dtype) {}
};


struct CompiledScanMinMaxBackward {
    const ScanOp value_op;
    const ScanMode mode;
    const uint64_t axis;
    const bool reverse;
    const bool segmented_by_offsets;
    const DataType input_dtype;
    const DataType grad_output_dtype;
    const DataType output_dtype;
    const std::optional<DataType> offset_dtype;

    bool operator==(const CompiledScanMinMaxBackward& other) const = default;

    CompiledScanMinMaxBackward(ScanOp value_op,
                               ScanMode mode,
                               uint64_t axis,
                               bool reverse,
                               bool segmented_by_offsets,
                               DataType input_dtype,
                               DataType grad_output_dtype,
                               DataType output_dtype,
                               std::optional<DataType> offset_dtype = std::nullopt)
        : value_op(value_op),
          mode(mode),
          axis(axis),
          reverse(reverse),
          segmented_by_offsets(segmented_by_offsets),
          input_dtype(input_dtype),
          grad_output_dtype(grad_output_dtype),
          output_dtype(output_dtype),
          offset_dtype(offset_dtype) {}
};

struct CompiledSoftmax {
    const cudnnSoftmaxAlgorithm_t algorithm;
    const cudnnSoftmaxMode_t mode;
    const DataType input_dtype;
    const DataType output_dtype;

    bool operator==(const CompiledSoftmax& other) const = default;

    CompiledSoftmax(cudnnSoftmaxAlgorithm_t algorithm,
                    cudnnSoftmaxMode_t mode,
                    DataType input_dtype,
                    DataType output_dtype)
        : algorithm(algorithm), mode(mode), input_dtype(input_dtype), output_dtype(output_dtype) {}
};

struct CompiledArgMinMax {
    const ExprOp op;
    std::vector<uint64_t> reduction_axes;
    std::vector<uint64_t> squeeze_axes;
    const DataType input_dtype;
    const DataType output_dtype;
    const DataType compute_dtype;

    bool operator==(const CompiledArgMinMax& other) const = default;

    CompiledArgMinMax(ExprOp op,
                      std::vector<uint64_t> reduction_axes,
                      std::vector<uint64_t> squeeze_axes,
                      DataType input_dtype,
                      DataType output_dtype,
                      std::optional<DataType> compute_dtype)
        : op(op),
          reduction_axes(std::move(reduction_axes)),
          squeeze_axes(std::move(squeeze_axes)),
          input_dtype(input_dtype),
          output_dtype(output_dtype),
          compute_dtype(compute_dtype.has_value() ? compute_dtype.value() : DataType::FP32) {
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
    const DataType input_dtype;
    const DataType grad_output_dtype;
    const DataType output_dtype;
    const DataType compute_dtype;

    bool operator==(const CompiledReduceMinMaxBackward& other) const = default;

    CompiledReduceMinMaxBackward(ExprOp op,
                                 std::vector<uint64_t> reduction_axes,
                                 std::vector<uint64_t> squeeze_axes,
                                 DataType input_dtype,
                                 DataType grad_output_dtype,
                                 DataType output_dtype,
                                 std::optional<DataType> compute_dtype)
        : op(op),
          reduction_axes(std::move(reduction_axes)),
          squeeze_axes(std::move(squeeze_axes)),
          input_dtype(input_dtype),
          grad_output_dtype(grad_output_dtype),
          output_dtype(output_dtype),
          compute_dtype(compute_dtype.has_value() ? compute_dtype.value() : DataType::FP32) {
        std::sort(this->reduction_axes.begin(), this->reduction_axes.end());
        this->reduction_axes.erase(std::unique(this->reduction_axes.begin(), this->reduction_axes.end()), this->reduction_axes.end());

        std::sort(this->squeeze_axes.begin(), this->squeeze_axes.end());
        this->squeeze_axes.erase(std::unique(this->squeeze_axes.begin(), this->squeeze_axes.end()), this->squeeze_axes.end());
    }
};

struct CompiledConvolution {
    const bool is_3d;
    const int32_t stride_d;
    const int32_t stride_h;
    const int32_t stride_w;
    const int32_t pad_d;
    const int32_t pad_h;
    const int32_t pad_w;
    const DataType input_dtype;
    const DataType filter_dtype;
    const DataType output_dtype;
    const DataType compute_dtype;

    bool operator==(const CompiledConvolution& other) const = default;

    CompiledConvolution(bool is_3d,
                        int32_t stride_d,
                        int32_t stride_h,
                        int32_t stride_w,
                        int32_t pad_d,
                        int32_t pad_h,
                        int32_t pad_w,
                        DataType input_dtype,
                        DataType filter_dtype,
                        DataType output_dtype,
                        std::optional<DataType> compute_dtype)
        : is_3d(is_3d),
          stride_d(stride_d),
          stride_h(stride_h),
          stride_w(stride_w),
          pad_d(pad_d),
          pad_h(pad_h),
          pad_w(pad_w),
          input_dtype(input_dtype),
          filter_dtype(filter_dtype),
          output_dtype(output_dtype),
          compute_dtype(compute_dtype.has_value() ? compute_dtype.value() : DataType::FP32) {}
};

struct CompiledConvolutionBackward {
    const ExprOp op;
    const int32_t stride_d;
    const int32_t stride_h;
    const int32_t stride_w;
    const int32_t pad_d;
    const int32_t pad_h;
    const int32_t pad_w;
    const DataType input_dtype;
    const DataType grad_output_dtype;
    const DataType output_dtype;
    const DataType compute_dtype;
    const std::vector<uint64_t> explicit_output_dims;

    bool operator==(const CompiledConvolutionBackward& other) const = default;

    CompiledConvolutionBackward(ExprOp op,
                                int32_t stride_d,
                                int32_t stride_h,
                                int32_t stride_w,
                                int32_t pad_d,
                                int32_t pad_h,
                                int32_t pad_w,
                                DataType input_dtype,
                                DataType grad_output_dtype,
                                DataType output_dtype,
                                std::optional<DataType> compute_dtype,
                                std::vector<uint64_t> explicit_output_dims = {})
        : op(op),
          stride_d(stride_d),
          stride_h(stride_h),
          stride_w(stride_w),
          pad_d(pad_d),
          pad_h(pad_h),
          pad_w(pad_w),
          input_dtype(input_dtype),
          grad_output_dtype(grad_output_dtype),
          output_dtype(output_dtype),
          compute_dtype(compute_dtype.has_value() ? compute_dtype.value() : DataType::FP32),
          explicit_output_dims(std::move(explicit_output_dims)) {}
};

struct CompiledMatmul {
    const ExprOp op;
    const bool transpose_lhs;
    const bool transpose_rhs;
    const bool transpose_aux;
    const double alpha;
    const double beta;
    const uint32_t alpha_input_slot;
    const uint32_t beta_input_slot;
    // Retained as the promoted/common input dtype for older code paths and diagnostics.
    const DataType input_dtype;
    const DataType lhs_dtype;
    const DataType rhs_dtype;
    const DataType aux_dtype;
    const DataType output_dtype;
    const DataType compute_dtype;
    const MatmulEpilogue epilogue;
    const MatmulBackwardEpilogue backward_epilogue;
    const uint32_t epilogue_aux_input_slot;
    const std::optional<DataType> epilogue_aux_dtype;
    const std::optional<DataType> bgrad_output_dtype;

    bool operator==(const CompiledMatmul& other) const = default;

    CompiledMatmul(ExprOp op,
                   bool transpose_lhs,
                   bool transpose_rhs,
                   bool transpose_aux,
                   double alpha,
                   double beta,
                   uint32_t alpha_input_slot,
                   uint32_t beta_input_slot,
                   DataType input_dtype,
                   DataType lhs_dtype,
                   DataType rhs_dtype,
                   DataType aux_dtype,
                   DataType output_dtype,
                   std::optional<DataType> compute_dtype,
                   MatmulEpilogue epilogue = MatmulEpilogue::Default,
                   MatmulBackwardEpilogue backward_epilogue = MatmulBackwardEpilogue::Default,
                   uint32_t epilogue_aux_input_slot = UINT32_MAX,
                   std::optional<DataType> epilogue_aux_dtype = std::nullopt,
                   std::optional<DataType> bgrad_output_dtype = std::nullopt)
        : op(op),
          transpose_lhs(transpose_lhs),
          transpose_rhs(transpose_rhs),
          transpose_aux(transpose_aux),
          alpha(alpha),
          beta(beta),
          alpha_input_slot(alpha_input_slot),
          beta_input_slot(beta_input_slot),
          input_dtype(input_dtype),
          lhs_dtype(lhs_dtype),
          rhs_dtype(rhs_dtype),
          aux_dtype(aux_dtype),
          output_dtype(output_dtype),
          compute_dtype(compute_dtype.has_value() ? compute_dtype.value() : output_dtype),
          epilogue(epilogue),
          backward_epilogue(backward_epilogue),
          epilogue_aux_input_slot(epilogue_aux_input_slot),
          epilogue_aux_dtype(epilogue_aux_dtype),
          bgrad_output_dtype(bgrad_output_dtype) {}
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
