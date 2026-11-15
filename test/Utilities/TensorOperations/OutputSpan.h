#pragma once

#include "DeepLearning/Implementation/Tensor/StridedSpan.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include <cstddef>
#include <optional>

namespace ThorImplementation {

// The reduction launchers take optional outputs as const StridedSpan*. Tests and benchmarks hold their outputs as
// Tensors or spans (or optional ones); outputSpan(x) wraps one for a single call. The span lives until the end of the full
// expression that names it, so the pointer is good for that call only: never keep it.
class OutputSpan {
   public:
    OutputSpan() = default;
    OutputSpan(std::nullptr_t) {}
    explicit OutputSpan(const Tensor& tensor) : span_(std::in_place, tensor) {}
    explicit OutputSpan(const std::optional<Tensor>& tensor) {
        if (tensor.has_value())
            span_.emplace(*tensor);
    }
    explicit OutputSpan(const StridedSpan& span) : span_(span) {}
    explicit OutputSpan(const std::optional<StridedSpan>& span) : span_(span) {}

    operator const StridedSpan*() const { return span_.has_value() ? &*span_ : nullptr; }

   private:
    std::optional<StridedSpan> span_;
};

inline OutputSpan outputSpan(const Tensor& tensor) { return OutputSpan(tensor); }
inline OutputSpan outputSpan(const std::optional<Tensor>& tensor) { return OutputSpan(tensor); }
inline OutputSpan outputSpan(const StridedSpan& span) { return OutputSpan(span); }
inline OutputSpan outputSpan(const std::optional<StridedSpan>& span) { return OutputSpan(span); }

}  // namespace ThorImplementation
