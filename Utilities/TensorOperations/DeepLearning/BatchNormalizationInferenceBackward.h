#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include <cstdint>

namespace ThorImplementation {

/**
 * Back-propagates through batch normalization when forward used fixed running
 * statistics. In this mode mean and variance are constants, so only dx is
 * produced and BatchNormalization parameters receive no gradients.
 */
void launchBatchNormalizationInferenceBackward(const Tensor& errorInput,
                                                Tensor& errorOutput,
                                                const Tensor& scale,
                                                const Tensor& runningVariance,
                                                double epsilon,
                                                uint32_t numChannels,
                                                Stream stream);

}  // namespace ThorImplementation
