#pragma once

#include "DeepLearning/Api/Loaders/DeviceDatasetMaterialization.h"
#include "Utilities/Loaders/MaterializedNamedDatasetSnapshot.h"

#include <cstdint>
#include <string>

struct NamedDatasetMaterializationSupport {
    bool supported = false;
    std::string reason;
};

/**
 * Returns whether the CPU snapshot materializer can stage the supplied loader
 * materialization view. It supports direct dense tensors plus assembled
 * windowed tensors/masks without touching the source loader's live batch stream.
 */
[[nodiscard]] NamedDatasetMaterializationSupport checkNamedDatasetSnapshotMaterializationSupport(
    const DeviceDatasetMaterializationView &view);

/**
 * Materialize all non-empty splits described by the view into contiguous CPU
 * tensors without touching or advancing the source loader's live batch stream.
 */
[[nodiscard]] MaterializedNamedDatasetSnapshot materializeNamedDatasetSnapshot(const DeviceDatasetMaterializationView &view,
                                                                              uint64_t readerQueueDepth = 32);
