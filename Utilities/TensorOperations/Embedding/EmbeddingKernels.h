#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ThorImplementation {

struct PreparedEmbeddingForward;

struct EmbeddingForwardEpilogue {
    // C++ scalar expression evaluated per output element.  The expression may
    // refer to `v` for the gathered embedding value and to same-shape extra
    // tensor inputs as `arg0[linear]`, `arg1[linear]`, ... .
    std::string expression;
    std::vector<TensorDescriptor::DataType> extra_input_dtypes;

    [[nodiscard]] bool enabled() const { return !expression.empty(); }
};

std::shared_ptr<PreparedEmbeddingForward> prepareEmbeddingForward(const Tensor& indices,
                                                                 const Tensor& weights,
                                                                 const Tensor& output,
                                                                 std::optional<uint64_t> paddingIndex,
                                                                 const EmbeddingForwardEpilogue& epilogue = {});

void launchPreparedEmbeddingForward(const PreparedEmbeddingForward& prepared,
                                    const Tensor& indices,
                                    const Tensor& weights,
                                    Tensor& output,
                                    Stream stream,
                                    const std::vector<Tensor>& epilogue_inputs = {});

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
