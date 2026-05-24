#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include <cstdint>
#include <optional>

namespace ThorImplementation {

void launchEmbeddingForward(const Tensor& indices,
                            const Tensor& weights,
                            Tensor& output,
                            std::optional<uint64_t> paddingIndex,
                            Stream stream);

void launchEmbeddingSparseSgdUpdate(const Tensor& indices,
                                    const Tensor& outputGradient,
                                    Tensor& weights,
                                    float step,
                                    std::optional<uint64_t> paddingIndex,
                                    Stream stream);

}  // namespace ThorImplementation
