#pragma once

// The contest input of a carried-index (intermediate) FP32 ARG site. The shared contest (ReductionAutotuning::autotune)
// rotates one input through its cold slots; a carried site reads two tensors, the FP32 values and the carried index, so
// its contest input is one UINT8 tensor holding the values, then the carried index (at nextSliceOffset: 256-byte
// aligned, at least 128 bytes past the values), and every slot streams both cold, as the carried sweep cells do.
// The RK and KR carried selectors share it.

#include "DeepLearning/Implementation/Tensor/StridedSpan.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include <cstdint>
#include <vector>

namespace ThorImplementation::ReductionInternal {

struct Fp32CarriedArgContestInput {
    uint64_t elements = 0;
    DataType index_dtype = DataType::UINT32;
    uint64_t index_offset = 0;  // bytes: 256-byte aligned, at least 128 bytes past the values
    uint64_t bytes = 0;

    /** The FP32 values of a contest slot (a span of its bytes, or the whole contest input), read as `dims`. */
    [[nodiscard]] StridedSpan values(const StridedSpan& slot, const std::vector<uint64_t>& dims) const;
    /** The carried index of a contest slot, read as `dims`. */
    [[nodiscard]] StridedSpan carriedIndex(const StridedSpan& slot, const std::vector<uint64_t>& dims) const;
};

/** The layout for `elements` values and a UINT32 / UINT64 carried index. */
[[nodiscard]] Fp32CarriedArgContestInput fp32CarriedArgContestInput(uint64_t elements, DataType index_dtype);

/** A new contest input on the stream's GPU holding copies of `values` (FP32) and `carried_index` (the layout's dtype),
 *  each of the layout's element count; or, with no tensors given, values 1.0 and index 0 (a representative input). */
[[nodiscard]] Tensor makeFp32CarriedArgContestInput(const Fp32CarriedArgContestInput& layout, const Stream& stream,
                                                    const Tensor* values = nullptr,
                                                    const Tensor* carried_index = nullptr);

}  // namespace ThorImplementation::ReductionInternal
