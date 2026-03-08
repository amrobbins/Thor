#pragma once

#include <cstdint>

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include "Utilities/TensorMathFusion/CudaHelpers.h"

namespace ThorImplementation {

struct EquationSignature {
    uint32_t rank;  // maybe just 1 for flattened contiguous V1
    uint32_t num_inputs;
    cudaDataType_t dtype;  // FP32 only for now
    bool contiguous;
    int sm_major;
    int sm_minor;
    int device_num;

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

    TensorDescriptor::DataType dtype;  // FIXME: dtype redundant with descriptor
    int deviceNum = 0;
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

class EquationRunner {
   public:
    static void run(const std::shared_ptr<CompiledEquation>& compiledEquation,
                    const std::vector<Tensor>& inputs,
                    Tensor& output,
                    Stream& stream);
};

class StampedEquation {
   public:
    StampedEquation(std::shared_ptr<CompiledEquation> compiledEquation, const Tensor& output, const Stream& stream)
        : compiledEquation(std::move(compiledEquation)), output(output), stream(stream) {}

    void run();
    Tensor getOutputTensor() const { return output; }

   private:
    std::shared_ptr<CompiledEquation> compiledEquation;
    std::vector<Tensor> inputs;
    Tensor output;
    Stream stream;
};

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
        ThorImplementation::hashCombine(h, std::hash<int>{}(s.device_num));
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
