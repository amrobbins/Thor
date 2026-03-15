#pragma once

#include <unordered_set>

#include "Utilities/TensorMathFusion/BroadcastStructs.h"
#include "Utilities/TensorMathFusion/EquationRunner.h"
#include "Utilities/TensorMathFusion/Expression.h"
#include "Utilities/TensorMathFusion/StampedEquation.h"

namespace ThorImplementation {

struct ReductionCacheKey {
    const ExprOp op;
    const std::vector<uint64_t> input_dims;
    std::vector<uint64_t> reduction_axes;
    const bool keepdim;
    const TensorDescriptor::DataType inout_dtype;
    const TensorDescriptor::DataType compute_dtype;
    const int device_num;

    bool operator==(const ReductionCacheKey& other) const = default;

    ReductionCacheKey(ExprOp op,
                      std::vector<uint64_t> input_dims,
                      std::vector<uint64_t> reduction_axes,
                      bool keepdim,
                      TensorDescriptor::DataType inout_dtype,
                      TensorDescriptor::DataType compute_dtype,
                      int device_num)
        : op(op),
          input_dims(std::move(input_dims)),
          reduction_axes(std::move(reduction_axes)),
          keepdim(keepdim),
          inout_dtype(inout_dtype),
          compute_dtype(compute_dtype),
          device_num(device_num) {
        std::sort(this->reduction_axes.begin(), this->reduction_axes.end());
        this->reduction_axes.erase(std::unique(this->reduction_axes.begin(), this->reduction_axes.end()), this->reduction_axes.end());
    }
};

struct BuiltReduction {
    ReductionCacheKey key;

    cudnnTensorDescriptor_t a_desc = nullptr;
    cudnnTensorDescriptor_t c_desc = nullptr;
    cudnnReduceTensorDescriptor_t reduce_desc = nullptr;

    size_t workspace_bytes = 0;

    explicit BuiltReduction(ReductionCacheKey key) : key(std::move(key)) {}

    ~BuiltReduction() {
        if (a_desc)
            cudnnDestroyTensorDescriptor(a_desc);
        if (c_desc)
            cudnnDestroyTensorDescriptor(c_desc);
        if (reduce_desc)
            cudnnDestroyReduceTensorDescriptor(reduce_desc);
    }

    BuiltReduction(const BuiltReduction&) = delete;
    BuiltReduction& operator=(const BuiltReduction&) = delete;
};

class FusedEquation {
   public:
    static FusedEquation compile(const PhysicalExpression& expr,
                                 TensorDescriptor::DataType dtype,
                                 int device_num,
                                 bool use_fast_math = false);

    [[nodiscard]] StampedEquation stampEquation(const std::unordered_map<std::string, Tensor>& inputs,
                                                const Stream& stream,
                                                const std::vector<uint64_t>& requestedOutputShape = {}) const;
    void runEquation(const std::unordered_map<std::string, Tensor>& inputs, Tensor& output, Stream& stream) const;

    StampedReduction stampReduction(Tensor& input, const Stream& stream) const;

   private:
    explicit FusedEquation(std::shared_ptr<CompiledEquation> flatEquation, std::shared_ptr<CompiledEquation> broadcastEquation)
        : compiledFlatEquation(std::move(flatEquation)),
          compiledBroadcastEquation(std::move(broadcastEquation)),
          compiledReduction(nullptr) {}

    explicit FusedEquation(std::shared_ptr<CompiledReduction> compiledReduction)
        : compiledFlatEquation(nullptr), compiledBroadcastEquation(nullptr), compiledReduction(std::move(compiledReduction)) {}

    [[nodiscard]] StampedEquation stampEquation(std::vector<Tensor>& inputs,
                                                const Stream& stream,
                                                const std::vector<uint64_t>& requestedOutputShape = {}) const;
    void runEquation(std::vector<Tensor> inputs, Tensor output, Stream stream) const;

    static bool resolveLayout(std::vector<Tensor>& inputs, std::vector<uint64_t>& outputDimensions);
    static Tensor createDeviceBroadcastInfo(const std::vector<Tensor>& inputs,
                                            const std::vector<uint64_t>& outputDimensions,
                                            Stream stream);
    [[nodiscard]] std::vector<Tensor> bindNamedInputs(const std::unordered_map<std::string, Tensor>& namedInputs) const;

    static std::shared_ptr<BuiltReduction> buildReduction(const std::shared_ptr<CompiledReduction>& compiled_reduction,
                                                          const Tensor& input,
                                                          int device_num);
    static std::vector<uint64_t> computeReductionOutputDims(const std::vector<uint64_t>& input_dims,
                                                            const std::vector<uint64_t>& reduction_axes,
                                                            bool keepdim);
    static size_t getReductionWorkspaceSize(int device_num,
                                            cudnnReduceTensorDescriptor_t reduce_desc,
                                            cudnnTensorDescriptor_t a_desc,
                                            cudnnTensorDescriptor_t c_desc);
    static cudnnReduceTensorDescriptor_t createCudnnReduceDescriptor(ExprOp op, TensorDescriptor::DataType compute_dtype);
    static cudnnTensorDescriptor_t createCudnnTensorDescriptor(const std::vector<uint64_t>& dims, TensorDescriptor::DataType dtype);

    const std::shared_ptr<CompiledEquation> compiledFlatEquation;
    const std::shared_ptr<CompiledEquation> compiledBroadcastEquation;
    const std::shared_ptr<CompiledReduction> compiledReduction;
    std::shared_ptr<BuiltReduction> builtReduction = nullptr;
};

}  // namespace ThorImplementation

namespace std {
template <>
struct hash<ThorImplementation::ReductionCacheKey> {
    size_t operator()(const ThorImplementation::ReductionCacheKey& k) const noexcept {
        size_t h = hash<ThorImplementation::ExprOp>{}(k.op);

        hashCombine(h, hash<size_t>{}(k.input_dims.size()));
        for (uint64_t d : k.input_dims)
            hashCombine(h, hash<uint64_t>{}(d));

        hashCombine(h, hash<size_t>{}(k.reduction_axes.size()));
        for (uint64_t axis : k.reduction_axes)
            hashCombine(h, hash<uint64_t>{}(axis));

        hashCombine(h, hash<bool>{}(k.keepdim));
        hashCombine(h, hash<ThorImplementation::TensorDescriptor::DataType>{}(k.inout_dtype));
        hashCombine(h, hash<ThorImplementation::TensorDescriptor::DataType>{}(k.compute_dtype));
        hashCombine(h, hash<int>{}(k.device_num));
        return h;
    }
};
}  // namespace std
