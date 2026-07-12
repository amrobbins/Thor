#pragma once

#include "DeepLearning/Api/Data/DatasetId.h"
#include "DeepLearning/Api/Data/DatasetSchema.h"
#include "DeepLearning/Api/Training/DeviceDatasetStorage.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"

#include <cstdint>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

class DeviceResidentNamedDataset;

namespace Thor {

/** Reference-counted ownership of one immutable device dataset replica. */
class DeviceDatasetLease {
   public:
    DeviceDatasetLease() = default;
    explicit DeviceDatasetLease(std::shared_ptr<const DeviceResidentNamedDataset> resident)
        : resident(std::move(resident)) {}

    [[nodiscard]] const std::shared_ptr<const DeviceResidentNamedDataset> &getShared() const {
        return resident;
    }
    [[nodiscard]] const DeviceResidentNamedDataset &get() const;
    [[nodiscard]] const DeviceResidentNamedDataset *operator->() const;
    [[nodiscard]] explicit operator bool() const { return resident != nullptr; }

   private:
    std::shared_ptr<const DeviceResidentNamedDataset> resident;
};

struct DeviceDatasetResidencyRequest {
    DeviceDatasetResidencyRequest(
        DatasetId datasetId,
        uint64_t numExamples,
        ThorImplementation::TensorPlacement placement,
        std::set<DatasetFieldId> requiredFields,
        uint64_t residentBytes,
        uint64_t requiredBytes,
        DeviceDatasetStorage storagePolicy,
        std::optional<uint64_t> availableBytesOverride,
        std::function<std::shared_ptr<const DeviceResidentNamedDataset>()> construct)
        : datasetId(std::move(datasetId)),
          numExamples(numExamples),
          placement(placement),
          requiredFields(std::move(requiredFields)),
          residentBytes(residentBytes),
          requiredBytes(requiredBytes),
          storagePolicy(storagePolicy),
          availableBytesOverride(availableBytesOverride),
          construct(std::move(construct)) {}

    DatasetId datasetId;
    uint64_t numExamples = 0;
    ThorImplementation::TensorPlacement placement;
    std::set<DatasetFieldId> requiredFields;
    uint64_t residentBytes = 0;
    uint64_t requiredBytes = 0;
    DeviceDatasetStorage storagePolicy = DeviceDatasetStorage::BEST_EFFORT;
    std::optional<uint64_t> availableBytesOverride;
    std::function<std::shared_ptr<const DeviceResidentNamedDataset>()> construct;
};

struct DeviceDatasetResidencyAcquisition {
    DeviceDatasetLease lease;
    bool cacheHit = false;
    bool joinedConstruction = false;
    bool startedConstruction = false;
    uint64_t availableBytesAtAdmission = 0;
};

struct DeviceDatasetResidencyTelemetry {
    uint64_t cacheHits = 0;
    uint64_t constructionStarts = 0;
    uint64_t constructionJoins = 0;
    uint64_t successfulConstructions = 0;
    uint64_t failedConstructions = 0;
};

struct DeviceDatasetMemoryReservationTelemetry {
    uint64_t reservationAttempts = 0;
    uint64_t reservationFailures = 0;
    uint64_t currentReservedBytes = 0;
    uint64_t peakReservedBytes = 0;
    uint64_t activeCommittedBytes = 0;
};

class DeviceDatasetResidencyAdmissionError : public std::runtime_error {
   public:
    DeviceDatasetResidencyAdmissionError(std::string message,
                                         uint64_t requiredBytes,
                                         uint64_t availableBytes)
        : std::runtime_error(std::move(message)),
          requiredBytes(requiredBytes),
          availableBytes(availableBytes) {}

    [[nodiscard]] uint64_t getRequiredBytes() const { return requiredBytes; }
    [[nodiscard]] uint64_t getAvailableBytes() const { return availableBytes; }

   private:
    uint64_t requiredBytes;
    uint64_t availableBytes;
};

/**
 * Per-logical-dataset cache of canonical device replicas.
 *
 * Entries hold weak references, so the cache does not pin GPU memory. Concurrent
 * callers for the same device/field set join one shared construction future.
 */
class DeviceDatasetResidencyCache {
   public:
    [[nodiscard]] DeviceDatasetResidencyAcquisition acquire(
        const DeviceDatasetResidencyRequest &request);

    [[nodiscard]] DeviceDatasetResidencyTelemetry getTelemetry() const;
    void clearExpired();

   private:
    struct ReplicaKey {
        int deviceNum = 0;
        std::set<DatasetFieldId> requiredFields;

        bool operator<(const ReplicaKey &rhs) const {
            if (deviceNum != rhs.deviceNum) {
                return deviceNum < rhs.deviceNum;
            }
            return requiredFields < rhs.requiredFields;
        }
    };

    struct CacheEntry {
        std::weak_ptr<const DeviceResidentNamedDataset> resident;
        std::shared_future<std::shared_ptr<const DeviceResidentNamedDataset>> construction;
    };

    mutable std::mutex mutex;
    std::map<ReplicaKey, CacheEntry> entries;
    DeviceDatasetResidencyTelemetry telemetry;
};

[[nodiscard]] DeviceDatasetMemoryReservationTelemetry
getDeviceDatasetMemoryReservationTelemetryForTesting(int deviceNum);
void resetDeviceDatasetMemoryReservationsForTesting();

}  // namespace Thor
