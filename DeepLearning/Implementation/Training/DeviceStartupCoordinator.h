#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace ThorImplementation {

/**
 * Thor keeps this much GPU memory unused after every serialized model startup.
 * The reserve covers small lazy/runtime allocations and memory movement by work
 * that is already executing while the next model starts.
 */
inline constexpr uint64_t DEVICE_STARTUP_SAFETY_RESERVE_BYTES =
    1ull * 1024ull * 1024ull * 1024ull;

class DeviceStartupSafetyReserveError : public std::runtime_error {
   public:
    DeviceStartupSafetyReserveError(
        std::string message,
        uint64_t availableBytes,
        uint64_t requiredUnusedBytes)
        : std::runtime_error(std::move(message)),
          availableBytes(availableBytes),
          requiredUnusedBytes(requiredUnusedBytes) {}

    [[nodiscard]] uint64_t getAvailableBytes() const { return availableBytes; }
    [[nodiscard]] uint64_t getRequiredUnusedBytes() const {
        return requiredUnusedBytes;
    }

   private:
    uint64_t availableBytes;
    uint64_t requiredUnusedBytes;
};

/**
 * Exclusive ownership of one GPU's persistent model-startup allocation phase.
 *
 * The guard is intentionally move-only. Destroying it while unwinding a failed
 * startup, or completing it after a successful startup, allows the next model
 * targeting the same GPU to proceed. Different GPUs use independent locks.
 */
class DeviceStartupGuard {
   public:
    DeviceStartupGuard(const DeviceStartupGuard&) = delete;
    DeviceStartupGuard& operator=(const DeviceStartupGuard&) = delete;

    DeviceStartupGuard(DeviceStartupGuard&&) noexcept = default;
    DeviceStartupGuard& operator=(DeviceStartupGuard&&) noexcept = default;

    ~DeviceStartupGuard() = default;

    [[nodiscard]] int getDeviceNum() const { return deviceNum; }
    [[nodiscard]] bool ownsLock() const { return lock.owns_lock(); }

    /** Verify the safety reserve and release this startup transaction. */
    void complete(
        std::optional<uint64_t> availableBytesOverride = std::nullopt);

   private:
    friend DeviceStartupGuard acquireDeviceStartupGuard(int deviceNum);

    DeviceStartupGuard(int deviceNum, std::mutex& deviceMutex);

    int deviceNum;
    std::unique_lock<std::mutex> lock;
};

/** Acquire exclusive startup ownership for one CUDA device. */
[[nodiscard]] DeviceStartupGuard acquireDeviceStartupGuard(int deviceNum);

/** Query CUDA free memory for one device. */
[[nodiscard]] uint64_t queryDeviceStartupAvailableBytes(int deviceNum);

/**
 * Require the configured safety reserve to remain after persistent startup
 * allocations have completed. availableBytesOverride exists for deterministic
 * unit testing and does not change production behavior.
 */
void enforceDeviceStartupSafetyReserve(
    int deviceNum,
    std::optional<uint64_t> availableBytesOverride = std::nullopt);

}  // namespace ThorImplementation
