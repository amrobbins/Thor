#pragma once

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/Stream.h"

#include <functional>
#include <memory>
#include <vector>

namespace Thor {

class BatchSourceResourceState;

/**
 * Copyable reference attached to a logical batch field whose backing storage
 * belongs to a reusable BatchSession slot.
 *
 * A source may also carry a producer-ready event when its storage is filled
 * asynchronously. NetworkInput waits on that event before reading the source,
 * then records one event after it has enqueued the final read. The owning Batch
 * later seals the resource after all physical submissions have been enqueued.
 * The session can recycle the slot as soon as producer and consumer events have
 * completed.
 */
class BatchSourceReference {
   public:
    BatchSourceReference() = default;

    [[nodiscard]] bool isInitialized() const { return state != nullptr; }

    /** Wait for asynchronous producer work, if any, before reading the source. */
    void waitUntilReady(const Stream& consumingStream) const;

    void recordConsumption(const Stream& consumingStream) const;

    [[nodiscard]] bool refersToSameResource(const BatchSourceReference& other) const {
        return state == other.state && state != nullptr;
    }

   private:
    explicit BatchSourceReference(std::shared_ptr<BatchSourceResourceState> state)
        : state(std::move(state)) {}

    std::shared_ptr<BatchSourceResourceState> state;

    friend class BatchSourceOwner;
};

/**
 * Move-only producer ownership for one reusable session resource.
 *
 * Destroying or explicitly releasing the owner seals the resource. New
 * NetworkInput consumers cannot be registered after sealing. The release
 * callback receives the source-consumed events plus any producer-ready event
 * and transfers ownership back to the BatchSession's pending-reuse queue.
 */
class BatchSourceOwner {
   public:
    using ReleaseCallback = std::function<void(std::vector<Event>)>;

    BatchSourceOwner() = default;
    explicit BatchSourceOwner(ReleaseCallback releaseCallback);
    BatchSourceOwner(ReleaseCallback releaseCallback, Event producerReadyEvent);
    ~BatchSourceOwner();

    BatchSourceOwner(const BatchSourceOwner&) = delete;
    BatchSourceOwner& operator=(const BatchSourceOwner&) = delete;
    BatchSourceOwner(BatchSourceOwner&& other) noexcept;
    BatchSourceOwner& operator=(BatchSourceOwner&& other) noexcept;

    [[nodiscard]] bool isInitialized() const { return state != nullptr; }
    [[nodiscard]] BatchSourceReference getReference() const {
        THOR_THROW_IF_FALSE(isInitialized());
        return BatchSourceReference(state);
    }

    void release();

   private:
    std::shared_ptr<BatchSourceResourceState> state;
};

}  // namespace Thor
