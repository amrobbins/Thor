#include "DeepLearning/Api/Data/BatchSourceResource.h"

#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Thor {

class BatchSourceResourceState {
   public:
    explicit BatchSourceResourceState(
        BatchSourceOwner::ReleaseCallback releaseCallback,
        Event producerReadyEvent = Event())
        : releaseCallback(std::move(releaseCallback)),
          producerReadyEvent(std::move(producerReadyEvent)) {
        THOR_THROW_IF_FALSE(static_cast<bool>(this->releaseCallback));
    }

    void waitUntilReady(const Stream& consumingStream) const {
        THOR_THROW_IF_FALSE(consumingStream.isInitialized());
        if (producerReadyEvent.isInitialized()) consumingStream.waitEvent(producerReadyEvent);
    }

    void recordConsumption(const Stream& consumingStream) {
        THOR_THROW_IF_FALSE(consumingStream.isInitialized());
        std::lock_guard<std::mutex> guard(mutex);
        if (producerReleased) {
            throw std::runtime_error(
                "Cannot register a BatchSourceReference consumer after its producer was released.");
        }
        // Serialize sealing with event creation so releaseProducer() can never
        // miss a consumer that has already begun registering its final read.
        //
        // Session recyclers wait for this event from a host thread before
        // returning the reusable source storage. Create it with
        // cudaEventBlockingSync so that wait sleeps instead of actively polling
        // the GPU.
        consumedEvents.push_back(consumingStream.putEvent(
            /*enableTiming=*/false,
            /*expectingHostToWaitOnThisOne=*/true));
    }

    void releaseProducer() {
        std::vector<Event> events;
        BatchSourceOwner::ReleaseCallback callback;
        {
            std::lock_guard<std::mutex> guard(mutex);
            if (producerReleased) {
                return;
            }
            producerReleased = true;
            events = std::move(consumedEvents);
            if (producerReadyEvent.isInitialized()) events.push_back(producerReadyEvent);
            callback = releaseCallback;
        }
        callback(std::move(events));
    }

   private:
    mutable std::mutex mutex;
    BatchSourceOwner::ReleaseCallback releaseCallback;
    Event producerReadyEvent;
    std::vector<Event> consumedEvents;
    bool producerReleased = false;
};

void BatchSourceReference::waitUntilReady(const Stream& consumingStream) const {
    THOR_THROW_IF_FALSE(isInitialized());
    state->waitUntilReady(consumingStream);
}

void BatchSourceReference::recordConsumption(const Stream& consumingStream) const {
    THOR_THROW_IF_FALSE(isInitialized());
    state->recordConsumption(consumingStream);
}

BatchSourceOwner::BatchSourceOwner(ReleaseCallback releaseCallback)
    : state(std::make_shared<BatchSourceResourceState>(std::move(releaseCallback))) {}

BatchSourceOwner::BatchSourceOwner(
    ReleaseCallback releaseCallback,
    Event producerReadyEvent)
    : state(std::make_shared<BatchSourceResourceState>(
          std::move(releaseCallback), std::move(producerReadyEvent))) {
    THOR_THROW_IF_FALSE(state != nullptr);
}

BatchSourceOwner::~BatchSourceOwner() {
    try {
        release();
    } catch (...) {
        // Destruction may occur while another exception is unwinding. Session
        // release callbacks are expected to be no-throw queue operations.
    }
}

BatchSourceOwner::BatchSourceOwner(BatchSourceOwner&& other) noexcept
    : state(std::move(other.state)) {}

BatchSourceOwner& BatchSourceOwner::operator=(BatchSourceOwner&& other) noexcept {
    if (this != &other) {
        try {
            release();
        } catch (...) {
        }
        state = std::move(other.state);
    }
    return *this;
}

void BatchSourceOwner::release() {
    if (state == nullptr) {
        return;
    }
    std::shared_ptr<BatchSourceResourceState> owned = std::move(state);
    owned->releaseProducer();
}

}  // namespace Thor
