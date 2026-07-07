#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Loaders/LocalNamedExampleLayout.h"
#include "Utilities/Loaders/Shard.h"

#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/**
 * CPU snapshot of a named local dataset split after direct tensors and any
 * assembled windowed tensors/masks have been materialized into one contiguous
 * tensor per logical name.
 *
 * Tensors have shape [num_examples, *example_shape]. Empty splits intentionally
 * hold no tensors because Thor TensorDescriptor does not permit zero-sized
 * dimensions.
 */
struct MaterializedNamedSplitSnapshot {
    ExampleType exampleType = ExampleType::TRAIN;
    std::string splitName;
    std::vector<uint64_t> sourceIndices;
    std::map<std::string, ThorImplementation::Tensor> tensors;
    bool randomized = false;
    std::optional<uint64_t> seed{};
    uint64_t batchesPerEpoch = 0;

    [[nodiscard]] uint64_t numExamples() const { return static_cast<uint64_t>(sourceIndices.size()); }
    [[nodiscard]] uint64_t totalBytes() const;
    [[nodiscard]] const ThorImplementation::Tensor &tensor(const std::string &name) const;
};

/**
 * CPU materialization result used as the staging representation before split
 * tensors are uploaded to a target device.
 */
struct MaterializedNamedDatasetSnapshot {
    LocalNamedExampleLayout layout;
    uint64_t numDatasetExamples = 0;
    uint64_t batchSize = 0;
    std::vector<MaterializedNamedSplitSnapshot> splits;
    double materializationSeconds = 0.0;

    [[nodiscard]] uint64_t totalExamples() const;
    [[nodiscard]] uint64_t totalBytes() const;
    [[nodiscard]] const MaterializedNamedSplitSnapshot *findSplit(ExampleType exampleType) const;
    [[nodiscard]] const MaterializedNamedSplitSnapshot &split(ExampleType exampleType) const;
};
