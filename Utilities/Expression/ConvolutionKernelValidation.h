#pragma once

#include <cstdint>
#include <string>

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

namespace ThorImplementation {

enum class ConvolutionKernelValidationKind {
    Forward,
    BackwardData,
    BackwardFilter,
};

// Geometry needed by Thor's independent convolution correctness oracle.  The
// tensors themselves carry N/C/spatial/filter dimensions and storage dtypes;
// this structure carries only the convolution mapping between those tensors.
struct ConvolutionKernelValidationSpec {
    ConvolutionKernelValidationKind kind = ConvolutionKernelValidationKind::Forward;
    bool is_3d = false;
    uint64_t groups = 1;

    int32_t stride_d = 1;
    int32_t stride_h = 1;
    int32_t stride_w = 1;

    int32_t pre_padding_d = 0;
    int32_t pre_padding_h = 0;
    int32_t pre_padding_w = 0;

    int32_t dilation_d = 1;
    int32_t dilation_h = 1;
    int32_t dilation_w = 1;

    DataType compute_dtype = DataType::FP32;
};

struct ConvolutionKernelValidationResult {
    bool passed = false;
    uint64_t checked_elements = 0;
    uint64_t bad_elements = 0;
    uint64_t first_bad_index = UINT64_MAX;
    float first_bad_actual = 0.0f;
    float first_bad_expected = 0.0f;
    float first_bad_tolerance = 0.0f;
    float max_abs_error = 0.0f;
};

// Populate one validation input tensor with a deterministic, sign-varying,
// non-zero pattern representable by every floating dtype accepted by the
// expression convolution path.  Different seeds produce independent operands.
void fillConvolutionKernelValidationTensor(Tensor& tensor, uint64_t seed, Stream& stream);

// Verify that a candidate kernel did not modify an input tensor.  The expected
// contents are reconstructed from the same deterministic seed rather than from
// a cuDNN-produced buffer.
ConvolutionKernelValidationResult validateConvolutionKernelValidationInputUnchanged(const Tensor& tensor,
                                                                                     uint64_t seed,
                                                                                     Stream& stream);

// Recompute the mathematical convolution independently of cuDNN and compare
// every candidate output element against that reference. lhs and rhs must contain
// patterns produced by fillConvolutionKernelValidationTensor; that lets the CUDA
// oracle accumulate their dyadic products as exact scaled integers rather than
// trusting another floating convolution implementation. lhs/rhs roles are:
//   Forward:        lhs=X,  rhs=W,  candidate_output=Y
//   BackwardData:   lhs=W,  rhs=dY, candidate_output=dX
//   BackwardFilter: lhs=X,  rhs=dY, candidate_output=dW
// The implementation is intentionally simple CUDA code rather than another
// cuDNN algorithm, so a cuDNN engine cannot validate itself through a sibling
// cuDNN implementation.  Its GPU hot path is intentionally 32-bit: validation
// rejects tensors with more than UINT32_MAX elements or any single-output
// reference reduction with more than INT32_MAX terms rather than using a slow
// 64-bit accumulator/indexing fallback.
ConvolutionKernelValidationResult validateConvolutionKernelOutput(const Tensor& lhs,
                                                                  const Tensor& rhs,
                                                                  const Tensor& candidate_output,
                                                                  const ConvolutionKernelValidationSpec& spec,
                                                                  Stream& stream);

std::string describeConvolutionKernelValidationFailure(const ConvolutionKernelValidationResult& result);

}  // namespace ThorImplementation
