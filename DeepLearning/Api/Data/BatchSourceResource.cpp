#include "DeepLearning/Api/Data/BatchSourceResource.h"

#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Thor {

class BatchSourceResourceState {
   public:
    explicit BatchSourceResourceState(BatchSourceOwner::ReleaseCallback releaseCallback)
        : releaseCallback(std::move(releaseCallback)) {
        THOR_THROW_IF_FALSE(static_cast<bool>(this->releaseCallback));
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
        consumedEvents.push_back(consumingStream.putEvent());
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
            callback = releaseCallback;
        }
        callback(std::move(events));
    }

   private:
    std::mutex mutex;
    BatchSourceOwner::ReleaseCallback releaseCallback;
    std::vector<Event> consumedEvents;
    bool producerReleased = false;
};

void BatchSourceReference::recordConsumption(const Stream& consumingStream) const {
    THOR_THROW_IF_FALSE(isInitialized());
    state->recordConsumption(consumingStream);
}

BatchSourceOwner::BatchSourceOwner(ReleaseCallback releaseCallback)
    : state(std::make_shared<BatchSourceResourceState>(std::move(releaseCallback))) {}

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
