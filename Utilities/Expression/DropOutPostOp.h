#pragma once

#include "Utilities/Expression/CudaKernelExpression.h"
#include "Utilities/Common/Stream.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include <cstdint>
#include <string>

namespace ThorImplementation {

// Small tensor-backed Philox state shared by expression-backed dropout users.
// The forward pass uploads (seed, sequence), then advances sequence so backward
// can deterministically regenerate that forward mask from the device scalar values,
// which remain unchanged until the next forward on this physical layer.
class DropOutRuntimeState {
   public:
    DropOutRuntimeState(int64_t seed, int64_t initialSequence, std::string ownerName);

    void setSequenceAdvance(uint64_t advance);
    TensorScalarBinding seedBinding(TensorPlacement placement);
    TensorScalarBinding sequenceBinding(TensorPlacement placement);
    void uploadForForward(Stream& stream);

   private:
    static constexpr uint64_t kSeedByteOffset = 0;
    static constexpr uint64_t kSequenceByteOffset = sizeof(int64_t);

    void ensureBuffer(TensorPlacement placement);

    int64_t seed;
    int64_t nextSequence;
    uint64_t sequenceAdvance = 1;
    std::string ownerName;
    Tensor seedSequenceBuffer;
};

// Build a pointwise post-op with exact semantics:
//
//     output = [residual +] dropout(projected)
//
// Backward regenerates the Philox mask, producing the masked/scaled gradient
// for projected and an identity gradient for residual. Ragged mode limits all
// reads/writes to offsets[batch] * featuresPerValue, leaving packed tail storage
// untouched.
CudaKernelExpression makeDropOutPostOpKernel(DataType dataType,
                                             float probability,
                                             bool useResidual,
                                             bool ragged,
                                             DataType offsetsDataType,
                                             uint64_t raggedBatchSize,
                                             uint64_t featuresPerValue,
                                             const std::string& debugName);

}  // namespace ThorImplementation
