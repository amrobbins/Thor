#include "DeepLearning/Implementation/Training/DeviceStartupCoordinator.h"

#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cctype>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ThorImplementation {

class DeviceStartupState {
   public:
    std::mutex mutex;
    std::condition_variable changed;
    uint64_t nextTicket = 0;
    uint64_t servingTicket = 0;
    uint64_t loadedModels = 0;
    uint64_t modelReleaseGeneration = 0;
};

namespace {

class DeviceStartupCoordinator {
   public:
    static DeviceStartupCoordinator& instance() {
        static DeviceStartupCoordinator coordinator;
        return coordinator;
    }

    std::shared_ptr<DeviceStartupState> stateForDevice(int deviceNum) {
        if (deviceNum < 0) {
            throw std::invalid_argument(
                "Device startup requires a non-negative CUDA device number.");
        }

        std::lock_guard<std::mutex> lock(mutex);
        auto& state = deviceStates[deviceNum];
        if (state == nullptr) {
            state = std::make_shared<DeviceStartupState>();
        }
        return state;
    }

   private:
    std::mutex mutex;
    std::map<int, std::shared_ptr<DeviceStartupState>> deviceStates;
};

std::string lowercaseAscii(std::string value) {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool messageLooksLikeStartupMemoryFailure(const std::string& message) {
    const std::string lower = lowercaseAscii(message);
    return lower.find("gpu out of memory") != std::string::npos ||
           lower.find("launch failed: out of memory") != std::string::npos ||
           lower.find("cudaerrormemoryallocation") != std::string::npos ||
           lower.find("cuda_error_out_of_memory") != std::string::npos ||
           lower.find("cublas_status_alloc_failed") != std::string::npos ||
           lower.find("cusparse_status_alloc_failed") != std::string::npos ||
           lower.find("cudnn_status_alloc_failed") != std::string::npos ||
           lower.find("insufficient_device_memory") != std::string::npos ||
           lower.find("insufficient device memory") != std::string::npos;
}

}  // namespace

DeviceModelResidencyLease::DeviceModelResidencyLease(
    std::shared_ptr<DeviceStartupState> state)
    : state(std::move(state)) {}

DeviceModelResidencyLease::~DeviceModelResidencyLease() {
    if (!active || state == nullptr) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->loadedModels == 0) {
            std::terminate();
        }
        state->loadedModels -= 1;
        state->modelReleaseGeneration += 1;
        active = false;
    }
    state->changed.notify_all();
}

DeviceStartupGuard::DeviceStartupGuard(
    int deviceNum,
    std::shared_ptr<DeviceStartupState> state,
    uint64_t ticket,
    std::unique_lock<std::mutex>&& lock)
    : deviceNum(deviceNum),
      state(std::move(state)),
      ticket(ticket),
      ownsTurn(true),
      lock(std::move(lock)) {}

DeviceStartupGuard::DeviceStartupGuard(DeviceStartupGuard&& other) noexcept
    : deviceNum(other.deviceNum),
      state(std::move(other.state)),
      ticket(other.ticket),
      ownsTurn(other.ownsTurn),
      lock(std::move(other.lock)) {
    other.deviceNum = -1;
    other.ticket = 0;
    other.ownsTurn = false;
}

DeviceStartupGuard& DeviceStartupGuard::operator=(
    DeviceStartupGuard&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    releaseTurn();
    deviceNum = other.deviceNum;
    state = std::move(other.state);
    ticket = other.ticket;
    ownsTurn = other.ownsTurn;
    lock = std::move(other.lock);
    other.deviceNum = -1;
    other.ticket = 0;
    other.ownsTurn = false;
    return *this;
}

DeviceStartupGuard::~DeviceStartupGuard() { releaseTurn(); }

void DeviceStartupGuard::releaseTurn() noexcept {
    if (!ownsTurn) {
        return;
    }
    if (state == nullptr || !lock.owns_lock()) {
        std::terminate();
    }
    if (state->servingTicket != ticket) {
        std::terminate();
    }

    state->servingTicket += 1;
    ownsTurn = false;
    lock.unlock();
    state->changed.notify_all();
}

void DeviceStartupGuard::complete(
    std::optional<uint64_t> availableBytesOverride) {
    if (!ownsLock()) {
        throw std::logic_error(
            "Device startup transaction has already been released.");
    }
    enforceDeviceStartupSafetyReserve(deviceNum, availableBytesOverride);
    releaseTurn();
}

std::shared_ptr<DeviceModelResidencyLease>
DeviceStartupGuard::completeAndTrackModel(
    std::optional<uint64_t> availableBytesOverride) {
    if (!ownsLock()) {
        throw std::logic_error(
            "Device startup transaction has already been released.");
    }
    enforceDeviceStartupSafetyReserve(deviceNum, availableBytesOverride);

    auto lease = std::shared_ptr<DeviceModelResidencyLease>(
        new DeviceModelResidencyLease(state));
    state->loadedModels += 1;
    lease->active = true;
    releaseTurn();
    return lease;
}

void DeviceStartupGuard::complete(
    Thor::PlacedNetwork& placedNetwork,
    std::optional<uint64_t> availableBytesOverride) {
    if (placedNetwork.deviceModelResidencyLease != nullptr) {
        throw std::logic_error(
            "PlacedNetwork already owns a device model residency lease.");
    }
    std::shared_ptr<DeviceModelResidencyLease> lease =
        completeAndTrackModel(availableBytesOverride);
    placedNetwork.deviceModelResidencyLease = std::move(lease);
}

uint64_t DeviceStartupGuard::getLoadedModelCount() const {
    if (!ownsLock()) {
        throw std::logic_error(
            "Loaded-model count requires an active device startup transaction.");
    }
    return state->loadedModels;
}

bool DeviceStartupGuard::tracksPlacement(
    const Thor::PlacedNetwork& placement) const {
    if (placement.deviceModelResidencyLease == nullptr) {
        return false;
    }
    const std::shared_ptr<DeviceModelResidencyLease> lease =
        std::static_pointer_cast<DeviceModelResidencyLease>(
            placement.deviceModelResidencyLease);
    return lease != nullptr && lease->active && lease->state == state;
}

uint64_t DeviceStartupGuard::getRetryableLoadedModelCount(
    const Thor::PlacedNetwork* retainedPlacement) const {
    uint64_t count = getLoadedModelCount();
    if (retainedPlacement != nullptr &&
        tracksPlacement(*retainedPlacement)) {
        if (count == 0) {
            throw std::logic_error(
                "Device startup loaded-model accounting is inconsistent.");
        }
        count -= 1;
    }
    return count;
}

void DeviceStartupGuard::waitForModelRelease(
    const std::function<void()>& interruptionCheck,
    const Thor::PlacedNetwork* retainedPlacement) {
    if (!ownsLock()) {
        throw std::logic_error(
            "Device startup wait requires an active startup transaction.");
    }
    if (getRetryableLoadedModelCount(retainedPlacement) == 0) {
        throw std::logic_error(
            "Device startup cannot wait because no retryable loaded model remains.");
    }

    const uint64_t observedGeneration = state->modelReleaseGeneration;
    while (state->modelReleaseGeneration == observedGeneration) {
        if (interruptionCheck) {
            interruptionCheck();
        }
        state->changed.wait_for(lock, std::chrono::milliseconds(100));
    }
    if (interruptionCheck) {
        interruptionCheck();
    }
}

DeviceStartupGuard acquireDeviceStartupGuard(int deviceNum) {
    std::shared_ptr<DeviceStartupState> state =
        DeviceStartupCoordinator::instance().stateForDevice(deviceNum);
    std::unique_lock<std::mutex> lock(state->mutex);
    const uint64_t ticket = state->nextTicket++;
    state->changed.wait(lock, [&]() {
        return state->servingTicket == ticket;
    });
    return DeviceStartupGuard(
        deviceNum, std::move(state), ticket, std::move(lock));
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
    message << "Device startup out of memory: model startup would leave less "
               "than Thor's required GPU memory safety reserve"
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

DeviceStartupMemoryFailureDisposition
decideDeviceStartupMemoryFailureDisposition(
    uint64_t loadedModels,
    uint64_t retryableLoadedModels,
    bool emptyDeviceRetryAlreadyUsed) {
    if (retryableLoadedModels > 0) {
        return DeviceStartupMemoryFailureDisposition::WAIT_FOR_MODEL_RELEASE;
    }
    if (loadedModels == 0 && !emptyDeviceRetryAlreadyUsed) {
        return DeviceStartupMemoryFailureDisposition::RETRY_EMPTY_DEVICE_ONCE;
    }
    return DeviceStartupMemoryFailureDisposition::FAIL;
}

void clearDeviceStartupCudaErrorState(int deviceNum) noexcept {
    try {
        ScopedGpu scopedGpu(deviceNum);
        (void)cudaGetLastError();
    } catch (...) {
        // This is cleanup for a failure path. Preserve the original exception.
    }
}

void requireCleanDeviceStartupCudaErrorState(int deviceNum) {
    ScopedGpu scopedGpu(deviceNum);
    CUDA_CHECK(cudaGetLastError());
}

void prepareDeviceForEmptyStartupRetry(int deviceNum) {
    ScopedGpu scopedGpu(deviceNum);
    const cudaError_t synchronizeStatus = cudaDeviceSynchronize();
    if (synchronizeStatus != cudaSuccess) {
        const char* name = cudaGetErrorName(synchronizeStatus);
        const char* description = cudaGetErrorString(synchronizeStatus);
        std::ostringstream message;
        message << "device_startup_empty_retry_synchronize_failed with "
                << (name != nullptr ? name : "cudaErrorUnknown")
                << " (" << static_cast<int>(synchronizeStatus) << "): "
                << (description != nullptr ? description : "<no description>");
        if (!messageLooksLikeStartupMemoryFailure(message.str())) {
            throw std::runtime_error(message.str());
        }
    }

    // cudaDeviceSynchronize() and the failed startup may both leave the runtime's
    // per-thread last-error slot populated.  The next attempt must not mistake
    // that stale error for a failure of its first kernel launch.
    (void)cudaGetLastError();
}

bool isDeviceStartupMemoryFailure(std::exception_ptr failure) {
    if (failure == nullptr) {
        return false;
    }

    try {
        std::rethrow_exception(failure);
    } catch (const DeviceStartupInsufficientMemoryError&) {
        return true;
    } catch (const std::exception& e) {
        return messageLooksLikeStartupMemoryFailure(e.what());
    } catch (...) {
        return false;
    }
}

}  // namespace ThorImplementation
