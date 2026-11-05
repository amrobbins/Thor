#pragma once

// 1215: the contest input of a carried-index (intermediate) FP32 ARG site (plans/Thor_FP32_Dense_Reduction_Plan,
// phase 8). The shared contest (ReductionAutotuning::autotune) rotates one input through its cold slots; a carried
// site reads two tensors, the FP32 values and the carried index, so its contest input is one UINT8 tensor holding the
// values, then (256-byte aligned) the carried index, and every slot streams both cold, as the carried sweep cells do.
// The RK and KR carried selectors share it.

#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include <cstdint>
#include <vector>

namespace ThorImplementation::CubReductionInternal {

struct Fp32CarriedArgContestInput {
    uint64_t elements = 0;
    DataType index_dtype = DataType::UINT32;
    uint64_t index_offset = 0;  // bytes, 256-byte aligned
    uint64_t bytes = 0;

    /** The FP32 values of a contest slot, viewed with `dims` (elements in total). */
    [[nodiscard]] Tensor values(const Tensor& slot, const std::vector<uint64_t>& dims) const;
    /** The carried index of a contest slot, viewed with `dims`. */
    [[nodiscard]] Tensor carriedIndex(const Tensor& slot, const std::vector<uint64_t>& dims) const;
};

/** The layout for `elements` values and a UINT32 / UINT64 carried index. */
[[nodiscard]] Fp32CarriedArgContestInput fp32CarriedArgContestInput(uint64_t elements, DataType index_dtype);

/** A new contest input on the stream's GPU holding copies of `values` (FP32) and `carried_index` (the layout's dtype),
 *  each of the layout's element count; or, with no tensors given, values 1.0 and index 0 (a representative input). */
[[nodiscard]] Tensor makeFp32CarriedArgContestInput(const Fp32CarriedArgContestInput& layout, const Stream& stream,
                                                    const Tensor* values = nullptr,
                                                    const Tensor* carried_index = nullptr);

}  // namespace ThorImplementation::CubReductionInternal
