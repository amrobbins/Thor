#include "DeepLearning/Implementation/Data/Residency/DeviceWindowL2Cache.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <mutex>
#include <utility>

namespace Thor {
namespace {

using ThorImplementation::PersistingL2Capabilities;
using ThorImplementation::PersistingL2OperationResult;
using ThorImplementation::PersistingL2OperationStatus;
using ThorImplementation::queryPersistingL2Capabilities;
using ThorImplementation::trySetPersistingL2SetAsideBytes;

DeviceWindowL2CacheLeaseSnapshot fixedFailure(
    const DeviceWindowL2CacheSource &source,
    DeviceWindowL2CacheLeaseStatus status,
    std::string detail) {
    return DeviceWindowL2CacheLeaseSnapshot{
        .status = status,
        .deviceNum = source.deviceNum,
        .tensorId = source.tensorId,
        .base = source.base,
        .bytes = source.bytes,
        .detail = std::move(detail)};
}

}  // namespace

struct DeviceWindowL2CacheManager::Impl {
    struct SourceState {
        const void *base = nullptr;
        uint64_t bytes = 0;
        uint64_t refCount = 0;
        float hitRatio = 0.0f;
    };

    struct DeviceState {
        bool initialized = false;
        bool available = false;
        uint64_t budgetBytes = 0;
        uint64_t maxAccessPolicyWindowBytes = 0;
        uint64_t activeUniqueBytes = 0;
        uint64_t activeLeases = 0;
        uint64_t generation = 0;
        DeviceWindowL2CacheLeaseStatus unavailableStatus =
            DeviceWindowL2CacheLeaseStatus::UNSUPPORTED;
        std::string detail;
        std::map<uint64_t, SourceState> sources;
    };

    mutable std::mutex mutex;
    std::map<int, DeviceState> devices;

    static void rebalance(DeviceState &device) {
        uint64_t activeUniqueBytes = 0;
        uint64_t activeLeases = 0;
        for (const auto &[tensorId, source] : device.sources) {
            (void)tensorId;
            if (source.refCount == 0) {
                continue;
            }
            if (activeUniqueBytes > std::numeric_limits<uint64_t>::max() - source.bytes) {
                activeUniqueBytes = std::numeric_limits<uint64_t>::max();
            } else {
                activeUniqueBytes += source.bytes;
            }
            if (activeLeases > std::numeric_limits<uint64_t>::max() - source.refCount) {
                activeLeases = std::numeric_limits<uint64_t>::max();
            } else {
                activeLeases += source.refCount;
            }
        }

        float ratio =
            activeUniqueBytes == 0 || device.budgetBytes == 0
                ? 0.0f
                : static_cast<float>(std::min(
                      1.0,
                      static_cast<double>(device.budgetBytes) /
                          static_cast<double>(activeUniqueBytes)));
        // A float conversion can round the exact budget ratio upward. Keep
        // the aggregate expected persisting occupancy conservative even in
        // that case.
        if (ratio > 0.0f &&
            static_cast<long double>(ratio) *
                    static_cast<long double>(activeUniqueBytes) >
                static_cast<long double>(device.budgetBytes)) {
            ratio = std::nextafter(ratio, 0.0f);
        }

        bool assignmentChanged = device.activeUniqueBytes != activeUniqueBytes;
        for (auto &[tensorId, source] : device.sources) {
            (void)tensorId;
            const float desired = source.refCount == 0 ? 0.0f : ratio;
            if (source.hitRatio != desired) {
                source.hitRatio = desired;
                assignmentChanged = true;
            }
        }
        device.activeUniqueBytes = activeUniqueBytes;
        device.activeLeases = activeLeases;
        if (assignmentChanged) {
            ++device.generation;
        }
    }

    static void initializeDevice(int deviceNum, DeviceState &device) {
        if (device.initialized) {
            return;
        }
        device.initialized = true;

        const PersistingL2Capabilities capabilities =
            queryPersistingL2Capabilities(deviceNum);
        device.maxAccessPolicyWindowBytes =
            capabilities.max_access_policy_window_bytes;
        if (!capabilities.supported) {
            device.available = false;
            device.unavailableStatus =
                !capabilities.query_succeeded
                    ? (capabilities.cuda_status == cudaErrorInvalidValue
                           ? DeviceWindowL2CacheLeaseStatus::INVALID_ARGUMENT
                           : DeviceWindowL2CacheLeaseStatus::CUDA_ERROR)
                    : DeviceWindowL2CacheLeaseStatus::UNSUPPORTED;
            device.detail = capabilities.detail;
            return;
        }

        // Respect a context/MPS configuration that already established a
        // non-zero set-aside. Otherwise Thor requests the device maximum once;
        // all source competition is handled with access-window hit ratios.
        if (capabilities.current_persisting_bytes > 0) {
            device.budgetBytes = std::min(capabilities.current_persisting_bytes,
                                          capabilities.max_persisting_bytes);
        } else {
            const PersistingL2OperationResult configured =
                trySetPersistingL2SetAsideBytes(deviceNum,
                                                capabilities.max_persisting_bytes);
            if (!configured.succeeded()) {
                device.available = false;
                switch (configured.status) {
                    case PersistingL2OperationStatus::UNSUPPORTED:
                        device.unavailableStatus =
                            DeviceWindowL2CacheLeaseStatus::UNSUPPORTED;
                        break;
                    case PersistingL2OperationStatus::INVALID_ARGUMENT:
                        device.unavailableStatus =
                            DeviceWindowL2CacheLeaseStatus::INVALID_ARGUMENT;
                        break;
                    case PersistingL2OperationStatus::CUDA_ERROR:
                        device.unavailableStatus =
                            DeviceWindowL2CacheLeaseStatus::CUDA_ERROR;
                        break;
                    case PersistingL2OperationStatus::SUCCESS:
                        device.unavailableStatus =
                            DeviceWindowL2CacheLeaseStatus::CUDA_ERROR;
                        break;
                }
                device.detail = configured.detail;
                return;
            }
            device.budgetBytes = capabilities.max_persisting_bytes;
        }

        if (device.budgetBytes == 0) {
            device.available = false;
            device.detail = "CUDA persisting-L2 budget is zero";
            return;
        }
        device.available = true;
        device.unavailableStatus = DeviceWindowL2CacheLeaseStatus::ACTIVE;
        device.detail.clear();
    }
};

DeviceWindowL2CacheLease::DeviceWindowL2CacheLease(
    DeviceWindowL2CacheManager *manager,
    int deviceNum,
    uint64_t tensorId,
    DeviceWindowL2CacheLeaseSnapshot fixedSnapshot)
    : manager(manager),
      registeredDeviceNum(deviceNum),
      registeredTensorId(tensorId),
      fixedSnapshot(std::move(fixedSnapshot)) {}

DeviceWindowL2CacheLease::~DeviceWindowL2CacheLease() { release(); }

DeviceWindowL2CacheLease::DeviceWindowL2CacheLease(
    DeviceWindowL2CacheLease &&other) noexcept
    : manager(other.manager),
      registeredDeviceNum(other.registeredDeviceNum),
      registeredTensorId(other.registeredTensorId),
      fixedSnapshot(std::move(other.fixedSnapshot)) {
    other.manager = nullptr;
    other.registeredDeviceNum = -1;
    other.registeredTensorId = 0;
}

DeviceWindowL2CacheLease &DeviceWindowL2CacheLease::operator=(
    DeviceWindowL2CacheLease &&other) noexcept {
    if (this == &other) {
        return *this;
    }
    release();
    manager = other.manager;
    registeredDeviceNum = other.registeredDeviceNum;
    registeredTensorId = other.registeredTensorId;
    fixedSnapshot = std::move(other.fixedSnapshot);
    other.manager = nullptr;
    other.registeredDeviceNum = -1;
    other.registeredTensorId = 0;
    return *this;
}

DeviceWindowL2CacheLeaseSnapshot DeviceWindowL2CacheLease::snapshot() const {
    if (manager == nullptr) {
        return fixedSnapshot;
    }
    return manager->snapshot(registeredDeviceNum,
                             registeredTensorId,
                             fixedSnapshot.base,
                             fixedSnapshot.bytes);
}

void DeviceWindowL2CacheLease::release() noexcept {
    if (manager == nullptr) {
        return;
    }
    manager->release(registeredDeviceNum, registeredTensorId);
    manager = nullptr;
    registeredDeviceNum = -1;
    registeredTensorId = 0;
}

DeviceWindowL2CacheManager &DeviceWindowL2CacheManager::instance() {
    static DeviceWindowL2CacheManager manager;
    return manager;
}

DeviceWindowL2CacheManager::DeviceWindowL2CacheManager()
    : impl(std::make_unique<Impl>()) {}

DeviceWindowL2CacheManager::~DeviceWindowL2CacheManager() = default;

DeviceWindowL2CacheLease DeviceWindowL2CacheManager::acquire(
    const DeviceWindowL2CacheSource &source) noexcept {
    if (source.deviceNum < 0) {
        return DeviceWindowL2CacheLease(
            nullptr, -1, 0,
            fixedFailure(source,
                         DeviceWindowL2CacheLeaseStatus::INVALID_ARGUMENT,
                         "window L2 cache source deviceNum must be non-negative"));
    }
    if (source.tensorId == 0) {
        return DeviceWindowL2CacheLease(
            nullptr, -1, 0,
            fixedFailure(source,
                         DeviceWindowL2CacheLeaseStatus::INVALID_ARGUMENT,
                         "window L2 cache source tensorId must be non-zero"));
    }
    if (source.base == nullptr) {
        return DeviceWindowL2CacheLease(
            nullptr, -1, 0,
            fixedFailure(source,
                         DeviceWindowL2CacheLeaseStatus::INVALID_ARGUMENT,
                         "window L2 cache source base must not be null"));
    }
    if (source.bytes == 0) {
        return DeviceWindowL2CacheLease(
            nullptr, -1, 0,
            fixedFailure(source,
                         DeviceWindowL2CacheLeaseStatus::INVALID_ARGUMENT,
                         "window L2 cache source bytes must be greater than zero"));
    }

    std::lock_guard<std::mutex> lock(impl->mutex);
    Impl::DeviceState &device = impl->devices[source.deviceNum];
    Impl::initializeDevice(source.deviceNum, device);
    if (!device.available) {
        return DeviceWindowL2CacheLease(
            nullptr, -1, 0,
            fixedFailure(source, device.unavailableStatus, device.detail));
    }
    if (source.bytes > device.maxAccessPolicyWindowBytes) {
        return DeviceWindowL2CacheLease(
            nullptr, -1, 0,
            DeviceWindowL2CacheLeaseSnapshot{
                .status = DeviceWindowL2CacheLeaseStatus::SOURCE_TOO_LARGE,
                .deviceNum = source.deviceNum,
                .tensorId = source.tensorId,
                .base = source.base,
                .bytes = source.bytes,
                .budgetBytes = device.budgetBytes,
                .activeUniqueBytes = device.activeUniqueBytes,
                .generation = device.generation,
                .hitRatio = 0.0f,
                .detail = "window L2 cache source exceeds CUDA access-policy window limit"});
    }

    auto found = device.sources.find(source.tensorId);
    if (found == device.sources.end()) {
        found = device.sources.emplace(
            source.tensorId,
            Impl::SourceState{
                .base = source.base,
                .bytes = source.bytes,
                .refCount = 0,
                .hitRatio = 0.0f})
                    .first;
    } else if (found->second.base != source.base ||
               found->second.bytes != source.bytes) {
        return DeviceWindowL2CacheLease(
            nullptr, -1, 0,
            DeviceWindowL2CacheLeaseSnapshot{
                .status = DeviceWindowL2CacheLeaseStatus::INVALID_ARGUMENT,
                .deviceNum = source.deviceNum,
                .tensorId = source.tensorId,
                .base = source.base,
                .bytes = source.bytes,
                .budgetBytes = device.budgetBytes,
                .activeUniqueBytes = device.activeUniqueBytes,
                .generation = device.generation,
                .hitRatio = 0.0f,
                .detail = "window L2 cache tensorId was already registered with different allocation metadata"});
    }
    if (found->second.refCount == std::numeric_limits<uint64_t>::max()) {
        return DeviceWindowL2CacheLease(
            nullptr, -1, 0,
            DeviceWindowL2CacheLeaseSnapshot{
                .status = DeviceWindowL2CacheLeaseStatus::INVALID_ARGUMENT,
                .deviceNum = source.deviceNum,
                .tensorId = source.tensorId,
                .base = source.base,
                .bytes = source.bytes,
                .budgetBytes = device.budgetBytes,
                .activeUniqueBytes = device.activeUniqueBytes,
                .generation = device.generation,
                .hitRatio = 0.0f,
                .detail = "window L2 cache source lease refcount overflow"});
    }

    ++found->second.refCount;
    Impl::rebalance(device);

    return DeviceWindowL2CacheLease(
        this,
        source.deviceNum,
        source.tensorId,
        DeviceWindowL2CacheLeaseSnapshot{
            .status = DeviceWindowL2CacheLeaseStatus::ACTIVE,
            .deviceNum = source.deviceNum,
            .tensorId = source.tensorId,
            .base = source.base,
            .bytes = source.bytes,
            .detail = {}});
}

DeviceWindowL2CacheLeaseSnapshot DeviceWindowL2CacheManager::snapshot(
    int deviceNum,
    uint64_t tensorId,
    const void *base,
    uint64_t bytes) const noexcept {
    std::lock_guard<std::mutex> lock(impl->mutex);
    const auto deviceIt = impl->devices.find(deviceNum);
    if (deviceIt == impl->devices.end()) {
        return DeviceWindowL2CacheLeaseSnapshot{
            .status = DeviceWindowL2CacheLeaseStatus::UNSUPPORTED,
            .deviceNum = deviceNum,
            .tensorId = tensorId,
            .base = base,
            .bytes = bytes,
            .detail = "window L2 cache device state is no longer registered"};
    }
    const Impl::DeviceState &device = deviceIt->second;
    const auto sourceIt = device.sources.find(tensorId);
    if (sourceIt == device.sources.end() || sourceIt->second.refCount == 0) {
        return DeviceWindowL2CacheLeaseSnapshot{
            .status = DeviceWindowL2CacheLeaseStatus::UNSUPPORTED,
            .deviceNum = deviceNum,
            .tensorId = tensorId,
            .base = base,
            .bytes = bytes,
            .budgetBytes = device.budgetBytes,
            .activeUniqueBytes = device.activeUniqueBytes,
            .generation = device.generation,
            .detail = "window L2 cache source is no longer registered"};
    }

    const Impl::SourceState &source = sourceIt->second;
    return DeviceWindowL2CacheLeaseSnapshot{
        .status = DeviceWindowL2CacheLeaseStatus::ACTIVE,
        .deviceNum = deviceNum,
        .tensorId = tensorId,
        .base = source.base,
        .bytes = source.bytes,
        .budgetBytes = device.budgetBytes,
        .activeUniqueBytes = device.activeUniqueBytes,
        .generation = device.generation,
        .hitRatio = source.hitRatio,
        .detail = {}};
}

void DeviceWindowL2CacheManager::release(int deviceNum, uint64_t tensorId) noexcept {
    std::lock_guard<std::mutex> lock(impl->mutex);
    auto deviceIt = impl->devices.find(deviceNum);
    if (deviceIt == impl->devices.end()) {
        return;
    }
    Impl::DeviceState &device = deviceIt->second;
    auto sourceIt = device.sources.find(tensorId);
    if (sourceIt == device.sources.end() || sourceIt->second.refCount == 0) {
        return;
    }
    --sourceIt->second.refCount;
    if (sourceIt->second.refCount == 0) {
        device.sources.erase(sourceIt);
    }
    Impl::rebalance(device);
}

DeviceWindowL2CacheManager::Telemetry DeviceWindowL2CacheManager::telemetry(
    int deviceNum) const {
    std::lock_guard<std::mutex> lock(impl->mutex);
    const auto found = impl->devices.find(deviceNum);
    if (found == impl->devices.end()) {
        return {};
    }
    const Impl::DeviceState &device = found->second;
    return Telemetry{
        .initialized = device.initialized,
        .available = device.available,
        .budgetBytes = device.budgetBytes,
        .maxAccessPolicyWindowBytes = device.maxAccessPolicyWindowBytes,
        .activeUniqueBytes = device.activeUniqueBytes,
        .activeUniqueSources = static_cast<uint64_t>(device.sources.size()),
        .activeLeases = device.activeLeases,
        .generation = device.generation,
        .detail = device.detail};
}

void DeviceWindowL2CacheManager::resetForTesting() noexcept {
    std::lock_guard<std::mutex> lock(impl->mutex);
    impl->devices.clear();
}

}  // namespace Thor
