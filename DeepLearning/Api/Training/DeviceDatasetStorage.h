#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace Thor {

enum class DeviceDatasetStorage { OFF, BEST_EFFORT, STRICT };

[[nodiscard]] const char* deviceDatasetStorageName(DeviceDatasetStorage storage);
[[nodiscard]] DeviceDatasetStorage deviceDatasetStorageFromName(std::string_view name);

struct DeviceDatasetStorageReport {
    DeviceDatasetStorage requested = DeviceDatasetStorage::BEST_EFFORT;
    bool attempted = false;
    bool used = false;
    std::string reason{};
    uint64_t examples = 0;
    uint64_t requiredBytes = 0;
    uint64_t availableBytesAfterPlacement = 0;
    double materializationSeconds = 0.0;
};

}  // namespace Thor
