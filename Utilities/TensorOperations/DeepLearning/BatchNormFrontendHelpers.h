#pragma once

#include "Utilities/Common/Stream.h"

#include <cstdint>

namespace ThorImplementation {

void launchComputeBatchNormInvVarianceFp32(const float* variance_d, float* inv_variance_d, float epsilon, uint64_t num_channels, Stream stream);

void launchAccumulateBatchNormGradientFp32(float* dest_d, const float* addend_d, uint64_t num_channels, Stream stream);

}  // namespace ThorImplementation
