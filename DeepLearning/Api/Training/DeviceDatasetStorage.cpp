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
    throw std::runtime_error("device_storage must be one of: 'best_effort', 'strict', 'off'.");
}

}  // namespace Thor
