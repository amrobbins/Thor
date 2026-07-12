#pragma once

#include "DeepLearning/Implementation/Data/Materialization/DeviceDatasetMaterialization.h"
#include "DeepLearning/Implementation/Data/Materialization/MaterializedNamedDatasetSnapshot.h"

#include <cstdint>
#include <string>

struct NamedDatasetMaterializationSupport {
    bool supported = false;
    std::string reason;
};

/** Returns whether the canonical dataset can be staged into a CPU snapshot. */
[[nodiscard]] NamedDatasetMaterializationSupport checkNamedDatasetSnapshotMaterializationSupport(
    const Thor::DatasetMaterializationDescription &description);

/**
 * Materialize every row of a FILE_DATASET exactly once, in source row order,
 * without touching or advancing any live BatchSession. In-memory backends
 * materialize through their owning NamedDataset implementation instead.
 */
[[nodiscard]] MaterializedNamedDatasetSnapshot materializeNamedDatasetSnapshot(
    const Thor::DatasetMaterializationDescription &description,
    uint64_t readerQueueDepth = 32);
