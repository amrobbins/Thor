#pragma once

#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace Thor {
class PlacedNetwork;
}

namespace ThorImplementation {

/**
 * Thor keeps this much GPU memory unused after every serialized model startup.
 * The reserve covers small lazy/runtime allocations and memory movement by work
 * that is already executing while the next model starts.
 */
inline constexpr uint64_t DEVICE_STARTUP_SAFETY_RESERVE_BYTES =
    1ull * 1024ull * 1024ull * 1024ull;

/**
 * A startup failure caused specifically by insufficient GPU memory.
 *
 * The native training runner treats this as retryable while another managed
 * model remains resident on the same GPU. Other startup failures remain
 * terminal and are never retried merely because a sibling model is loaded.
 */
class DeviceStartupInsufficientMemoryError : public std::runtime_error {
   public:
    explicit DeviceStartupInsufficientMemoryError(std::string message)
        : std::runtime_error(std::move(message)) {}
};

class DeviceStartupSafetyReserveError
    : public DeviceStartupInsufficientMemoryError {
   public:
    DeviceStartupSafetyReserveError(
        std::string message,
        uint64_t availableBytes,
        uint64_t requiredUnusedBytes)
        : DeviceStartupInsufficientMemoryError(std::move(message)),
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

class DeviceStartupState;

/**
 * Lifetime token for one successfully started, GPU-resident PlacedNetwork.
 *
 * The token is normally attached to PlacedNetwork itself. Its destructor runs
 * only after that placement has released its GPU tensors, then wakes the FIFO
 * startup waiter for the device.
 */
class DeviceModelResidencyLease {
   public:
    DeviceModelResidencyLease(const DeviceModelResidencyLease&) = delete;
    DeviceModelResidencyLease& operator=(
        const DeviceModelResidencyLease&) = delete;
    ~DeviceModelResidencyLease();

   private:
    friend class DeviceStartupGuard;

    explicit DeviceModelResidencyLease(
        std::shared_ptr<DeviceStartupState> state);

    std::shared_ptr<DeviceStartupState> state;
    bool active = false;
};

/**
 * Exclusive ownership of one GPU's persistent model-startup allocation phase.
 *
 * Startup turns are FIFO. A model that cannot fit while sibling models are
 * loaded retains its place at the head of the queue, releases the device mutex
 * while waiting, and retries after each resident-model release. Later eligible
 * models cannot bypass that waiting model.
 */
class DeviceStartupGuard {
   public:
    DeviceStartupGuard(const DeviceStartupGuard&) = delete;
    DeviceStartupGuard& operator=(const DeviceStartupGuard&) = delete;

    DeviceStartupGuard(DeviceStartupGuard&& other) noexcept;
    DeviceStartupGuard& operator=(DeviceStartupGuard&& other) noexcept;

    ~DeviceStartupGuard();

    [[nodiscard]] int getDeviceNum() const { return deviceNum; }
    [[nodiscard]] bool ownsLock() const {
        return ownsTurn && lock.owns_lock();
    }

    /**
     * Verify the safety reserve and release this startup turn without tracking
     * a resident model. This is useful for deterministic coordinator tests and
     * startup work that does not leave a GPU placement alive.
     */
    void complete(
        std::optional<uint64_t> availableBytesOverride = std::nullopt);

    /**
     * Verify the safety reserve, register one loaded model, attach its lifetime
     * lease to placedNetwork, and release this startup turn.
     */
    void complete(
        Thor::PlacedNetwork& placedNetwork,
        std::optional<uint64_t> availableBytesOverride = std::nullopt);

    /**
     * Register a synthetic loaded-model lifetime and release the startup turn.
     * Production model startup should use complete(PlacedNetwork&); this method
     * permits deterministic coordinator tests without constructing a GPU graph.
     */
    [[nodiscard]] std::shared_ptr<DeviceModelResidencyLease>
    completeAndTrackModel(
        std::optional<uint64_t> availableBytesOverride = std::nullopt);

    /** Number of coordinator-managed models currently resident on this GPU. */
    [[nodiscard]] uint64_t getLoadedModelCount() const;

    /**
     * Number of loaded models that may make progress for this startup attempt.
     * A retained placement belonging to the same Trainer can be excluded because
     * it cannot be freed while that Trainer is blocked inside this startup.
     */
    [[nodiscard]] uint64_t getRetryableLoadedModelCount(
        const Thor::PlacedNetwork* retainedPlacement = nullptr) const;

    /**
     * Wait until one loaded model on this device is actually released.
     *
     * The guard keeps its FIFO startup turn while the underlying mutex is
     * released for the wait. interruptionCheck is invoked periodically so
     * cancellation can interrupt a wait even if an unrelated retained model is
     * long-lived.
     */
    void waitForModelRelease(
        const std::function<void()>& interruptionCheck = {},
        const Thor::PlacedNetwork* retainedPlacement = nullptr);

   private:
    friend DeviceStartupGuard acquireDeviceStartupGuard(int deviceNum);

    DeviceStartupGuard(
        int deviceNum,
        std::shared_ptr<DeviceStartupState> state,
        uint64_t ticket,
        std::unique_lock<std::mutex>&& lock);

    void releaseTurn() noexcept;
    [[nodiscard]] bool tracksPlacement(
        const Thor::PlacedNetwork& placement) const;

    int deviceNum = -1;
    std::shared_ptr<DeviceStartupState> state;
    uint64_t ticket = 0;
    bool ownsTurn = false;
    std::unique_lock<std::mutex> lock;
};

/** Acquire the next FIFO startup turn for one CUDA device. */
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

/** Return true only for exceptions that represent GPU startup memory pressure. */
[[nodiscard]] bool isDeviceStartupMemoryFailure(
    std::exception_ptr failure);

}  // namespace ThorImplementation
