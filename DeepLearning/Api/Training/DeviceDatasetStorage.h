#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace Thor {

enum class DeviceDatasetStorage {
    OFF,
    BEST_EFFORT,
    STRICT,
    /**
     * Require compact residency for the windowed fields of a file-backed
     * windowed dataset. Direct fields remain source-backed even when they
     * could also fit on the device.
     */
    STRICT_WINDOWED_ONLY
};

[[nodiscard]] const char* deviceDatasetStorageName(DeviceDatasetStorage storage);
[[nodiscard]] DeviceDatasetStorage deviceDatasetStorageFromName(std::string_view name);

struct DeviceDatasetStorageReport {
    DeviceDatasetStorage requested = DeviceDatasetStorage::OFF;
    bool attempted = false;
    bool used = false;
    std::string reason{};
    uint64_t examples = 0;
    uint64_t requiredBytes = 0;
    uint64_t availableBytesAfterPlacement = 0;
    uint64_t residentBytes = 0;
    bool residentCacheHit = false;
    bool residentConstructionJoined = false;
    bool residentConstructionStarted = false;
    double materializationSeconds = 0.0;
};

}  // namespace Thor
