#include "DeepLearning/Implementation/Data/Residency/DeviceDatasetResidency.h"

#include "DeepLearning/Implementation/Training/DeviceStartupCoordinator.h"
#include "Utilities/Common/ScopedGpu.h"
#include "DeepLearning/Implementation/Data/Residency/DeviceResidentNamedDataset.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <exception>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace Thor {
namespace {

struct CommittedAllocation {
    std::weak_ptr<const DeviceResidentNamedDataset> resident;
    uint64_t bytes = 0;
};

struct DeviceReservationState {
    uint64_t reservedBytes = 0;
    uint64_t peakReservedBytes = 0;
    uint64_t reservationAttempts = 0;
    uint64_t reservationFailures = 0;
    std::vector<CommittedAllocation> committed;
};

class DeviceDatasetMemoryReservationManager {
   public:
    struct Reservation {
        int deviceNum = 0;
        uint64_t bytes = 0;
        uint64_t availableBytesAtAdmission = 0;
        bool active = false;
    };

    static DeviceDatasetMemoryReservationManager &instance() {
        static DeviceDatasetMemoryReservationManager manager;
        return manager;
    }

    Reservation reserve(const DeviceDatasetResidencyRequest &request) {
        if (request.placement.getMemDevice() !=
            ThorImplementation::TensorPlacement::MemDevices::GPU) {
            throw DeviceDatasetResidencyAdmissionError(
                "Device dataset residency requires a GPU placement.",
                request.requiredBytes,
                0);
        }
        if (request.residentBytes == 0 || request.requiredBytes < request.residentBytes) {
            throw std::runtime_error(
                "Device dataset residency request has an invalid byte estimate.");
        }

        const int deviceNum = request.placement.getDeviceNum();
        std::lock_guard<std::mutex> lock(mutex);
        DeviceReservationState &state = states[deviceNum];
        pruneCommitted(state);
        state.reservationAttempts += 1;

        const uint64_t rawAvailable = request.availableBytesOverride.has_value()
                                          ? request.availableBytesOverride.value()
                                          : queryAvailableBytes(deviceNum);
        uint64_t unavailable = state.reservedBytes;
        if (request.availableBytesOverride.has_value()) {
            unavailable = checkedAdd(
                unavailable,
                committedBytes(state),
                "Device dataset unavailable bytes");
        }
        const uint64_t effectiveAvailable =
            unavailable >= rawAvailable ? 0 : rawAvailable - unavailable;

        if (!fitsWithSafetyReserve(
                request.requiredBytes,
                effectiveAvailable)) {
            state.reservationFailures += 1;
            const uint64_t usableBytes =
                effectiveAvailable <=
                        ThorImplementation::DEVICE_STARTUP_SAFETY_RESERVE_BYTES
                    ? 0
                    : effectiveAvailable -
                          ThorImplementation::DEVICE_STARTUP_SAFETY_RESERVE_BYTES;
            std::ostringstream message;
            message << "Insufficient device memory for device dataset residency"
                    << " required_bytes=" << request.requiredBytes
                    << " available_bytes=" << effectiveAvailable
                    << " safety_reserve_bytes="
                    << ThorImplementation::DEVICE_STARTUP_SAFETY_RESERVE_BYTES
                    << " usable_bytes_after_reserve=" << usableBytes;
            throw DeviceDatasetResidencyAdmissionError(
                message.str(), request.requiredBytes, effectiveAvailable);
        }

        state.reservedBytes = checkedAdd(
            state.reservedBytes,
            request.residentBytes,
            "Device dataset reserved bytes");
        state.peakReservedBytes = std::max(state.peakReservedBytes, state.reservedBytes);
        return Reservation{
            .deviceNum = deviceNum,
            .bytes = request.residentBytes,
            .availableBytesAtAdmission = effectiveAvailable,
            .active = true};
    }

    void commit(Reservation &reservation,
                const std::shared_ptr<const DeviceResidentNamedDataset> &resident) {
        if (!reservation.active) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex);
        DeviceReservationState &state = states[reservation.deviceNum];
        releaseReservedBytes(state, reservation.bytes);
        state.committed.push_back(CommittedAllocation{resident, reservation.bytes});
        reservation.active = false;
    }

    void release(Reservation &reservation) noexcept {
        if (!reservation.active) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex);
        auto found = states.find(reservation.deviceNum);
        if (found != states.end()) {
            releaseReservedBytes(found->second, reservation.bytes);
        }
        reservation.active = false;
    }

    DeviceDatasetMemoryReservationTelemetry telemetry(int deviceNum) {
        std::lock_guard<std::mutex> lock(mutex);
        DeviceReservationState &state = states[deviceNum];
        pruneCommitted(state);
        return DeviceDatasetMemoryReservationTelemetry{
            .reservationAttempts = state.reservationAttempts,
            .reservationFailures = state.reservationFailures,
            .currentReservedBytes = state.reservedBytes,
            .peakReservedBytes = state.peakReservedBytes,
            .activeCommittedBytes = committedBytes(state)};
    }

    void resetForTesting() {
        std::lock_guard<std::mutex> lock(mutex);
        states.clear();
    }

   private:
    static uint64_t checkedAdd(uint64_t left, uint64_t right, const char *context) {
        if (left > std::numeric_limits<uint64_t>::max() - right) {
            throw std::runtime_error(std::string(context) + " overflow while adding.");
        }
        return left + right;
    }

    static bool fitsWithSafetyReserve(uint64_t requiredBytes,
                                      uint64_t availableBytes) {
        // Strict storage policies control whether selection may fall back
        // to the source session; they do not permit consuming Thor's
        // process-wide startup safety reserve.
        if (availableBytes <=
            ThorImplementation::DEVICE_STARTUP_SAFETY_RESERVE_BYTES) {
            return false;
        }
        return requiredBytes <=
               availableBytes -
                   ThorImplementation::DEVICE_STARTUP_SAFETY_RESERVE_BYTES;
    }

    static uint64_t queryAvailableBytes(int deviceNum) {
        ScopedGpu scopedGpu(deviceNum);
        size_t freeBytes = 0;
        size_t totalBytes = 0;
        const cudaError_t status = cudaMemGetInfo(&freeBytes, &totalBytes);
        (void)totalBytes;
        if (status != cudaSuccess) {
            throw std::runtime_error(
                std::string("device_memory_query_failed:") +
                cudaGetErrorString(status));
        }
        return static_cast<uint64_t>(freeBytes);
    }

    static void pruneCommitted(DeviceReservationState &state) {
        state.committed.erase(
            std::remove_if(
                state.committed.begin(),
                state.committed.end(),
                [](const CommittedAllocation &allocation) {
                    return allocation.resident.expired();
                }),
            state.committed.end());
    }

    static uint64_t committedBytes(const DeviceReservationState &state) {
        uint64_t bytes = 0;
        for (const CommittedAllocation &allocation : state.committed) {
            bytes = checkedAdd(bytes, allocation.bytes, "Device dataset committed bytes");
        }
        return bytes;
    }

    static void releaseReservedBytes(DeviceReservationState &state, uint64_t bytes) noexcept {
        if (bytes >= state.reservedBytes) {
            state.reservedBytes = 0;
        } else {
            state.reservedBytes -= bytes;
        }
    }

    std::mutex mutex;
    std::map<int, DeviceReservationState> states;
};

void validateRequest(const DeviceDatasetResidencyRequest &request) {
    if (request.numExamples == 0) {
        throw std::runtime_error("Device dataset residency requires at least one example.");
    }
    if (request.requiredFields.empty()) {
        throw std::runtime_error("Device dataset residency requires at least one field.");
    }
    if (!request.construct) {
        throw std::runtime_error("Device dataset residency requires a construction callback.");
    }
}

void validateConstructedResident(
    const DeviceDatasetResidencyRequest &request,
    const std::shared_ptr<const DeviceResidentNamedDataset> &resident) {
    if (resident == nullptr) {
        throw std::runtime_error("Device dataset residency construction returned null.");
    }
    if (resident->getDatasetId() != request.datasetId) {
        throw std::runtime_error(
            "Device dataset residency construction returned the wrong dataset identity.");
    }
    if (resident->getNumExamples() != request.numExamples) {
        throw std::runtime_error(
            "Device dataset residency construction returned the wrong dataset generation.");
    }
    if (resident->getPlacement() != request.placement) {
        throw std::runtime_error(
            "Device dataset residency construction returned the wrong placement.");
    }
    if (resident->totalBytes() != request.residentBytes) {
        throw std::runtime_error(
            "Device dataset residency construction returned an unexpected byte size.");
    }
    for (DatasetFieldId fieldId : request.requiredFields) {
        if (!resident->hasField(fieldId)) {
            throw std::runtime_error(
                "Device dataset residency construction omitted a required field.");
        }
    }
}

}  // namespace

const DeviceResidentNamedDataset &DeviceDatasetLease::get() const {
    if (resident == nullptr) {
        throw std::runtime_error("DeviceDatasetLease is empty.");
    }
    return *resident;
}

const DeviceResidentNamedDataset *DeviceDatasetLease::operator->() const {
    return &get();
}

DeviceDatasetResidencyAcquisition DeviceDatasetResidencyCache::acquire(
    const DeviceDatasetResidencyRequest &request) {
    validateRequest(request);
    const ReplicaKey key{
        .deviceNum = request.placement.getDeviceNum(),
        .requiredFields = request.requiredFields};

    std::shared_future<std::shared_ptr<const DeviceResidentNamedDataset>> joinedFuture;
    std::shared_ptr<std::promise<std::shared_ptr<const DeviceResidentNamedDataset>>> promise;
    {
        std::lock_guard<std::mutex> lock(mutex);
        CacheEntry &entry = entries[key];
        if (std::shared_ptr<const DeviceResidentNamedDataset> resident =
                entry.resident.lock()) {
            telemetry.cacheHits += 1;
            return DeviceDatasetResidencyAcquisition{
                .lease = DeviceDatasetLease(std::move(resident)),
                .cacheHit = true};
        }
        if (entry.construction.valid()) {
            telemetry.constructionJoins += 1;
            joinedFuture = entry.construction;
        } else {
            promise = std::make_shared<
                std::promise<std::shared_ptr<const DeviceResidentNamedDataset>>>();
            entry.construction = promise->get_future().share();
            telemetry.constructionStarts += 1;
        }
    }

    if (joinedFuture.valid()) {
        std::shared_ptr<const DeviceResidentNamedDataset> resident = joinedFuture.get();
        return DeviceDatasetResidencyAcquisition{
            .lease = DeviceDatasetLease(std::move(resident)),
            .joinedConstruction = true};
    }

    DeviceDatasetMemoryReservationManager::Reservation reservation;
    try {
        reservation =
            DeviceDatasetMemoryReservationManager::instance().reserve(request);
        std::shared_ptr<const DeviceResidentNamedDataset> resident = request.construct();
        validateConstructedResident(request, resident);
        DeviceDatasetMemoryReservationManager::instance().commit(reservation, resident);

        {
            std::lock_guard<std::mutex> lock(mutex);
            CacheEntry &entry = entries[key];
            entry.resident = resident;
            entry.construction = {};
            telemetry.successfulConstructions += 1;
        }
        promise->set_value(resident);
        return DeviceDatasetResidencyAcquisition{
            .lease = DeviceDatasetLease(std::move(resident)),
            .startedConstruction = true,
            .availableBytesAtAdmission = reservation.availableBytesAtAdmission};
    } catch (...) {
        const std::exception_ptr failure = std::current_exception();
        DeviceDatasetMemoryReservationManager::instance().release(reservation);
        promise->set_exception(failure);
        {
            std::lock_guard<std::mutex> lock(mutex);
            entries.erase(key);
            telemetry.failedConstructions += 1;
        }
        std::rethrow_exception(failure);
    }
}

DeviceDatasetResidencyTelemetry DeviceDatasetResidencyCache::getTelemetry() const {
    std::lock_guard<std::mutex> lock(mutex);
    return telemetry;
}

void DeviceDatasetResidencyCache::clearExpired() {
    std::lock_guard<std::mutex> lock(mutex);
    for (auto iterator = entries.begin(); iterator != entries.end();) {
        if (!iterator->second.construction.valid() &&
            iterator->second.resident.expired()) {
            iterator = entries.erase(iterator);
        } else {
            ++iterator;
        }
    }
}

DeviceDatasetMemoryReservationTelemetry
getDeviceDatasetMemoryReservationTelemetryForTesting(int deviceNum) {
    return DeviceDatasetMemoryReservationManager::instance().telemetry(deviceNum);
}

void resetDeviceDatasetMemoryReservationsForTesting() {
    DeviceDatasetMemoryReservationManager::instance().resetForTesting();
}

}  // namespace Thor
