#include "DeepLearning/Implementation/Training/DeviceStartupCoordinator.h"

#include "Utilities/Common/ScopedGpu.h"

#include <cuda_runtime_api.h>

#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ThorImplementation {
namespace {

class DeviceStartupCoordinator {
   public:
    static DeviceStartupCoordinator& instance() {
        static DeviceStartupCoordinator coordinator;
        return coordinator;
    }

    std::mutex& mutexForDevice(int deviceNum) {
        if (deviceNum < 0) {
            throw std::invalid_argument(
                "Device startup requires a non-negative CUDA device number.");
        }

        std::lock_guard<std::mutex> lock(mutex);
        auto& deviceMutex = deviceMutexes[deviceNum];
        if (deviceMutex == nullptr) {
            deviceMutex = std::make_unique<std::mutex>();
        }
        return *deviceMutex;
    }

   private:
    std::mutex mutex;
    std::map<int, std::unique_ptr<std::mutex>> deviceMutexes;
};

}  // namespace

DeviceStartupGuard::DeviceStartupGuard(int deviceNum, std::mutex& deviceMutex)
    : deviceNum(deviceNum), lock(deviceMutex) {}

void DeviceStartupGuard::complete(
    std::optional<uint64_t> availableBytesOverride) {
    if (!lock.owns_lock()) {
        throw std::logic_error(
            "Device startup transaction has already been released.");
    }
    enforceDeviceStartupSafetyReserve(deviceNum, availableBytesOverride);
    lock.unlock();
}

DeviceStartupGuard acquireDeviceStartupGuard(int deviceNum) {
    return DeviceStartupGuard(
        deviceNum,
        DeviceStartupCoordinator::instance().mutexForDevice(deviceNum));
}

uint64_t queryDeviceStartupAvailableBytes(int deviceNum) {
    ScopedGpu scopedGpu(deviceNum);
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    const cudaError_t status = cudaMemGetInfo(&freeBytes, &totalBytes);
    (void)totalBytes;
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("device_startup_memory_query_failed:") +
            cudaGetErrorString(status));
    }
    return static_cast<uint64_t>(freeBytes);
}

void enforceDeviceStartupSafetyReserve(
    int deviceNum,
    std::optional<uint64_t> availableBytesOverride) {
    const uint64_t availableBytes = availableBytesOverride.has_value()
                                        ? availableBytesOverride.value()
                                        : queryDeviceStartupAvailableBytes(deviceNum);
    if (availableBytes >= DEVICE_STARTUP_SAFETY_RESERVE_BYTES) {
        return;
    }

    std::ostringstream message;
    message << "Model startup would leave less than Thor's required GPU memory "
               "safety reserve"
            << " device=" << deviceNum
            << " available_bytes=" << availableBytes
            << " required_unused_bytes=" << DEVICE_STARTUP_SAFETY_RESERVE_BYTES
            << ". Thor requires at least 1 GiB of free GPU memory after all "
               "persistent startup allocations.";
    throw DeviceStartupSafetyReserveError(
        message.str(),
        availableBytes,
        DEVICE_STARTUP_SAFETY_RESERVE_BYTES);
}

}  // namespace ThorImplementation
