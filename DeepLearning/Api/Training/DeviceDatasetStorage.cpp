#include "DeepLearning/Api/Training/DeviceDatasetStorage.h"

#include <stdexcept>
#include <string>

namespace Thor {

const char* deviceDatasetStorageName(DeviceDatasetStorage storage) {
    switch (storage) {
        case DeviceDatasetStorage::OFF:
            return "off";
        case DeviceDatasetStorage::BEST_EFFORT:
            return "best_effort";
        case DeviceDatasetStorage::STRICT:
            return "strict";
        case DeviceDatasetStorage::STRICT_WINDOWED_ONLY:
            return "strict_windowed_only";
        default:
            return "unknown";
    }
}

DeviceDatasetStorage deviceDatasetStorageFromName(std::string_view name) {
    if (name == "off") {
        return DeviceDatasetStorage::OFF;
    }
    if (name == "best_effort" || name == "best-effort" || name == "bestEffort") {
        return DeviceDatasetStorage::BEST_EFFORT;
    }
    if (name == "strict") {
        return DeviceDatasetStorage::STRICT;
    }
    if (name == "strict_windowed_only" ||
        name == "strict-windowed-only" ||
        name == "strictWindowedOnly") {
        return DeviceDatasetStorage::STRICT_WINDOWED_ONLY;
    }
    throw std::runtime_error(
        "device_storage must be one of: 'best_effort', 'strict', "
        "'strict_windowed_only', 'off'.");
}

}  // namespace Thor
