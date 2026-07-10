#pragma once

#include "DeepLearning/Api/Loaders/DeviceDatasetMaterialization.h"
#include "Utilities/Loaders/MaterializedNamedDatasetSnapshot.h"

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
 * Materialize every canonical dataset row exactly once, in source row order,
 * without touching or advancing any live BatchSession.
 */
[[nodiscard]] MaterializedNamedDatasetSnapshot materializeNamedDatasetSnapshot(
    const Thor::DatasetMaterializationDescription &description,
    uint64_t readerQueueDepth = 32);
