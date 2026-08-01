#pragma once

namespace Thor {

// A batch's authoritative runtime validity state is its valid-example count: valid rows are always a contiguous prefix.
// Expression-backed custom operations currently receive that state through this reserved FP32 prefix-mask input.
inline constexpr const char* BATCH_VALIDITY_MASK_NAME = "__thor_batch_validity_mask";

}  // namespace Thor
