#pragma once

#include "Utilities/Loaders/LocalNamedExampleLayout.h"
#include "Utilities/Loaders/Shard.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * Read-only description of a loader split that can be staged into a
 * device-resident dataset without consuming batches from the live loader.
 *
 * The indices are expressed in the source loader's canonical indexed row space.
 * Later materialization code can use this view to build an independent staging
 * reader instead of mutating the training loader's getBatch()/returnBuffers()
 * state.
 */
struct DeviceDatasetMaterializationSplitView {
    ExampleType exampleType = ExampleType::TRAIN;
    std::string splitName;
    std::vector<uint64_t> indices;
    uint64_t batchesPerEpoch = 0;
    bool randomized = false;
    std::optional<uint64_t> seed{};

    [[nodiscard]] uint64_t numExamples() const { return static_cast<uint64_t>(indices.size()); }
};

/**
 * Read-only description of a named local dataset that can be materialized onto
 * a device.  This is a contract for staging, not a batch API.
 */
struct DeviceDatasetMaterializationView {
    std::filesystem::path datasetPath;
    LocalNamedExampleLayout layout;
    uint64_t numDatasetExamples = 0;
    uint64_t batchSize = 0;
    std::vector<DeviceDatasetMaterializationSplitView> splits;

    [[nodiscard]] const DeviceDatasetMaterializationSplitView *findSplit(ExampleType exampleType) const;
    [[nodiscard]] const DeviceDatasetMaterializationSplitView &split(ExampleType exampleType) const;
};

[[nodiscard]] const char *deviceDatasetMaterializationSplitName(ExampleType exampleType);
