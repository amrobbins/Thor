#pragma once

#include "DeepLearning/Implementation/Tensor/StridedSpan.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Copy/DenseCopyPlan.h"

namespace ThorImplementation {

// Thor's permutation / transpose / materialization operation (1388): copy a span's elements, in row-major order of its
// dimensions, into a dense tensor of the same dtype and shape. The words are copied as they are: no conversion.
// Source and destination must be on the same GPU, and destination must be dense. DenseCopyPlan.h has the families: one
// memcpy for a dense span, the matrix transpose kernel for a plain rotation of a dense matrix (1389e), a shared-tile
// transpose when rows narrower than a sector change order otherwise (1389f: rows of up to two sectors too, from
// 4 MiB), a packed gather for other permutations of single words, a row copy for the rest.

/** The plan for copying `source` into the dense `destination` (a Tensor converts), for this pair's layouts, addresses
 *  and GPU. Throws std::invalid_argument unless they have one descriptor and GPU and the destination is dense. */
[[nodiscard]] DenseCopyPlan planCopyToDense(const StridedSpan& source, const StridedSpan& destination);

/**
 * One copy by `plan`: one launch (or memcpy), stream-ordered, no allocation and no synchronization. The spans may be
 * others than the plan was made for, with the same layouts, dtype and GPU, and addresses aligned to its word (a
 * packed gather's destination: to 16 bytes). Throws std::invalid_argument otherwise.
 */
void launchCopyToDense(const DenseCopyPlan& plan, const StridedSpan& source, const StridedSpan& destination,
                       Stream stream);

/** planCopyToDense, then launchCopyToDense. */
void copyToDenseAsync(const StridedSpan& source, Tensor& destination, Stream stream);

}  // namespace ThorImplementation
