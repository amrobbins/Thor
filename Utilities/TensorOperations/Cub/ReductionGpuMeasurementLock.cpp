#include "Utilities/TensorOperations/Cub/ReductionGpuMeasurementLock.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace ThorImplementation::CubReductionInternal {
namespace {
// Device ordinal is process-local and identifies the GPU whose measurements
// must not overlap. The registry lock is held only to locate its mutex.
std::shared_ptr<std::mutex> measurementMutexForDevice(int device_ordinal) {
    static std::mutex registry_mutex;
    static std::unordered_map<int, std::shared_ptr<std::mutex>> device_mutexes;
    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    auto& device_mutex = device_mutexes[device_ordinal];
    if (!device_mutex) device_mutex = std::make_shared<std::mutex>();
    return device_mutex;
}

// Detect accidental nested contests on the same thread BEFORE entering a
// nonrecursive mutex. Selecting/tuning FP32 dependencies is legal, but must
// finish before an enclosing FP8 contest begins.
std::vector<int>& activeMeasurementDevices() {
    static thread_local std::vector<int> devices;
    return devices;
}
} // namespace

ScopedReductionGpuMeasurement::ScopedReductionGpuMeasurement(int device_ordinal) {
    if (device_ordinal < 0)
        throw std::invalid_argument("Reduction measurement requires a valid GPU ordinal");
    auto& active = activeMeasurementDevices();
    if (std::find(active.begin(), active.end(), device_ordinal) != active.end())
        throw std::logic_error("Nested GPU reduction measurement contest on the same device");
    device_mutex_ = measurementMutexForDevice(device_ordinal);
    lock_ = std::unique_lock<std::mutex>(*device_mutex_);
    active.push_back(device_ordinal);
}

ScopedReductionGpuMeasurement::~ScopedReductionGpuMeasurement() noexcept {
    // The owning thread's active device is the last acquired contest on that
    // thread. Remove it before lock_ releases the device mutex.
    activeMeasurementDevices().pop_back();
}

} // namespace ThorImplementation::CubReductionInternal
