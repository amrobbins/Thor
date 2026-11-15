#pragma once

#include "Utilities/Expression/CudaKernelExpression.h"
#include "Utilities/Common/Stream.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include <cstdint>
#include <string>

namespace ThorImplementation {

// The one tensor-backed Philox state for every dropout user: the dropout post-op's fused kernel and cuDNN
// attention (where the sequence is cuDNN's dropout offset). The forward pass uploads (seed, sequence), then
// advances sequence so backward can deterministically regenerate that forward mask from the device scalar
// values, which remain unchanged until the next forward on this physical layer.
class DropOutRuntimeState {
   public:
    DropOutRuntimeState(int64_t seed, int64_t initialSequence, std::string ownerName);

    void setSequenceAdvance(uint64_t advance);
    TensorScalarBinding seedBinding(TensorPlacement placement);
    TensorScalarBinding sequenceBinding(TensorPlacement placement);
    void uploadForForward(Stream& stream);

   private:
    // The seed and sequence are separate one-element tensors; one launch writes both every forward.
    void ensureBuffer(TensorPlacement placement);

    int64_t seed;
    int64_t nextSequence;
    uint64_t sequenceAdvance = 1;
    std::string ownerName;
    Tensor seedTensor;
    Tensor sequenceTensor;
};

// Build a pointwise post-op with exact semantics:
//
//     output = [residual +] dropout(projected)
//
// Backward regenerates the Philox mask, producing the masked/scaled gradient
// for projected and an identity gradient for residual. Ragged mode limits all
// reads/writes to activeValueCount * featuresPerValue, leaving packed tail
// storage untouched. Partition-preserving ragged dropout uses the managed [1]
// active-count tensor as its structural carrier.
CudaKernelExpression makeDropOutPostOpKernel(
    DataType dataType,
    float probability,
    bool useResidual,
    bool ragged,
    DataType partitionDataType,
    uint64_t raggedBatchSize,
    uint64_t featuresPerValue,
    const std::string& debugName,
    RaggedRuntimeExtentSource raggedRuntimeExtentSource = RaggedRuntimeExtentSource::DEVICE_OFFSETS);

}  // namespace ThorImplementation
