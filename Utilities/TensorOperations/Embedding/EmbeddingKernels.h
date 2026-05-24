#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include <cstdint>
#include <memory>
#include <optional>

namespace ThorImplementation {

struct PreparedEmbeddingForward;

std::shared_ptr<PreparedEmbeddingForward> prepareEmbeddingForward(const Tensor& indices,
                                                                 const Tensor& weights,
                                                                 const Tensor& output,
                                                                 std::optional<uint64_t> paddingIndex);

void launchPreparedEmbeddingForward(const PreparedEmbeddingForward& prepared,
                                    const Tensor& indices,
                                    const Tensor& weights,
                                    Tensor& output,
                                    Stream stream);

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
