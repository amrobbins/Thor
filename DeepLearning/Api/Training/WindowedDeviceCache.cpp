#include "DeepLearning/Api/Training/WindowedDeviceCache.h"

#include <stdexcept>

namespace Thor {

const char *windowedDeviceCacheName(WindowedDeviceCache policy) {
    switch (policy) {
        case WindowedDeviceCache::OFF:
            return "off";
        case WindowedDeviceCache::AUTO:
            return "auto";
        case WindowedDeviceCache::REQUIRED:
            return "required";
    }
    throw std::runtime_error("Unknown WindowedDeviceCache value.");
}

WindowedDeviceCache windowedDeviceCacheFromName(std::string_view name) {
    if (name == "off") {
        return WindowedDeviceCache::OFF;
    }
    if (name == "auto") {
        return WindowedDeviceCache::AUTO;
    }
    if (name == "required") {
        return WindowedDeviceCache::REQUIRED;
    }
    throw std::runtime_error(
        "windowed_device_cache must be one of: 'off', 'auto', 'required'.");
}

}  // namespace Thor
