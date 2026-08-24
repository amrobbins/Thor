#pragma once

#include "Utilities/Common/PersistingL2Cache.h"

#include <cstdint>
#include <memory>
#include <string>

namespace Thor {

/**
 * One compact, device-resident window source that is eligible to compete for
 * Thor's process-wide persisting-L2 budget on a GPU.
 */
struct DeviceWindowL2CacheSource {
    int deviceNum = -1;
    uint64_t tensorId = 0;
    const void *base = nullptr;
    uint64_t bytes = 0;
};

enum class DeviceWindowL2CacheLeaseStatus : uint8_t {
    ACTIVE,
    UNSUPPORTED,
    SOURCE_TOO_LARGE,
    INVALID_ARGUMENT,
    CUDA_ERROR,
};

/**
 * Snapshot of a lease's current device-wide budget assignment.
 *
 * hitRatio is intentionally dynamic: when independent hot sources become
 * active or inactive, the manager rebalances all live sources so the sum of
 * expected persisting bytes never exceeds budgetBytes. generation changes
 * whenever that assignment changes. DeviceBatchReference propagates the
 * current snapshot to NetworkInput, and Stream includes generation in its
 * access-policy cache key so a rebalance forces refresh.
 */
struct DeviceWindowL2CacheLeaseSnapshot {
    DeviceWindowL2CacheLeaseStatus status = DeviceWindowL2CacheLeaseStatus::UNSUPPORTED;
    int deviceNum = -1;
    uint64_t tensorId = 0;
    const void *base = nullptr;
    uint64_t bytes = 0;
    uint64_t budgetBytes = 0;
    uint64_t activeUniqueBytes = 0;
    uint64_t generation = 0;
    float hitRatio = 0.0f;
    std::string detail;

    [[nodiscard]] bool active() const { return status == DeviceWindowL2CacheLeaseStatus::ACTIVE; }
};

class DeviceWindowL2CacheManager;

/**
 * Move-only RAII lease for one source registration.
 *
 * Multiple leases for the same (GPU, tensorId) are reference-counted and count
 * the source bytes only once toward the device's persisting-L2 budget.
 */
class DeviceWindowL2CacheLease {
   public:
    DeviceWindowL2CacheLease() = default;
    ~DeviceWindowL2CacheLease();

    DeviceWindowL2CacheLease(const DeviceWindowL2CacheLease &) = delete;
    DeviceWindowL2CacheLease &operator=(const DeviceWindowL2CacheLease &) = delete;

    DeviceWindowL2CacheLease(DeviceWindowL2CacheLease &&other) noexcept;
    DeviceWindowL2CacheLease &operator=(DeviceWindowL2CacheLease &&other) noexcept;

    [[nodiscard]] DeviceWindowL2CacheLeaseSnapshot snapshot() const;
    [[nodiscard]] bool active() const { return snapshot().active(); }
    explicit operator bool() const { return active(); }

   private:
    friend class DeviceWindowL2CacheManager;

    DeviceWindowL2CacheLease(DeviceWindowL2CacheManager *manager,
                             int deviceNum,
                             uint64_t tensorId,
                             DeviceWindowL2CacheLeaseSnapshot fixedSnapshot);

    void release() noexcept;

    DeviceWindowL2CacheManager *manager = nullptr;
    int registeredDeviceNum = -1;
    uint64_t registeredTensorId = 0;
    DeviceWindowL2CacheLeaseSnapshot fixedSnapshot;
};

/**
 * Process-wide per-GPU manager for persisting-L2 window-source budget.
 *
 * The manager owns admission, deduplication, set-aside configuration, and hit
 * ratio accounting. It deliberately does not attach an access-policy window to
 * any CUDA stream; DeviceBatchReference/NetworkInput consume lease snapshots at
 * materialization time.
 */
class DeviceWindowL2CacheManager {
   public:
    static DeviceWindowL2CacheManager &instance();

    [[nodiscard]] DeviceWindowL2CacheLease acquire(const DeviceWindowL2CacheSource &source) noexcept;

    struct Telemetry {
        bool initialized = false;
        bool available = false;
        uint64_t budgetBytes = 0;
        uint64_t maxAccessPolicyWindowBytes = 0;
        uint64_t activeUniqueBytes = 0;
        uint64_t activeUniqueSources = 0;
        uint64_t activeLeases = 0;
        uint64_t generation = 0;
        std::string detail;
    };

    [[nodiscard]] Telemetry telemetry(int deviceNum) const;

    // Test-only lifecycle hook. Production code should not reset a live
    // manager because leases may still be applying the previous generation.
    void resetForTesting() noexcept;

   private:
    friend class DeviceWindowL2CacheLease;
    struct Impl;

    DeviceWindowL2CacheManager();
    ~DeviceWindowL2CacheManager();
    DeviceWindowL2CacheManager(const DeviceWindowL2CacheManager &) = delete;
    DeviceWindowL2CacheManager &operator=(const DeviceWindowL2CacheManager &) = delete;

    [[nodiscard]] DeviceWindowL2CacheLeaseSnapshot snapshot(
        int deviceNum,
        uint64_t tensorId,
        const void *base,
        uint64_t bytes) const noexcept;
    void release(int deviceNum, uint64_t tensorId) noexcept;

    std::unique_ptr<Impl> impl;
};

}  // namespace Thor
