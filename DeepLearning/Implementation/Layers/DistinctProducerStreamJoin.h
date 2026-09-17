#pragma once

#include "Utilities/Common/Stream.h"

#include <cstddef>
#include <vector>

namespace ThorImplementation::detail {

// Join a consumer stream to the minimal set of distinct physical producer
// streams represented by a set of logical inputs. Callers invoke this only
// after every represented logical input has arrived, so a completion event
// recorded at one producer stream's current tail dominates all required input
// work already enqueued on that same stream. This intentionally performs only
// alias coalescing: it does not attempt transitive happens-before reduction.
// The O(N^2) scan avoids allocating a set/vector in the per-batch hot path;
// layer input counts are expected to be small.
template <typename IsActive, typename ProducerStreamAt, typename ReusableEventAt>
inline void waitForDistinctProducerStreams(const Stream& consumer,
                                           std::size_t logicalProducerCount,
                                           IsActive&& isActive,
                                           ProducerStreamAt&& producerStreamAt,
                                           ReusableEventAt&& reusableEventAt) {
    THOR_THROW_IF_FALSE(consumer.isInitialized());

    for (std::size_t i = 0; i < logicalProducerCount; ++i) {
        if (!isActive(i))
            continue;

        const Stream& producer = producerStreamAt(i);
        THOR_THROW_IF_FALSE(producer.isInitialized());

        // Stream ordering already supplies this dependency.
        if (consumer == producer)
            continue;

        bool producerAlreadyJoined = false;
        for (std::size_t j = 0; j < i; ++j) {
            if (!isActive(j))
                continue;
            const Stream& earlierProducer = producerStreamAt(j);
            THOR_THROW_IF_FALSE(earlierProducer.isInitialized());
            if (producer == earlierProducer) {
                producerAlreadyJoined = true;
                break;
            }
        }

        if (!producerAlreadyJoined)
            consumer.waitFor(producer, reusableEventAt(i));
    }
}

inline void waitForDistinctProducerStreams(const Stream& consumer,
                                           const std::vector<Stream>& producerStreams,
                                           std::vector<Event>& reusableEvents,
                                           std::size_t firstLogicalProducer = 0) {
    THOR_THROW_IF_FALSE(producerStreams.size() == reusableEvents.size());
    THOR_THROW_IF_FALSE(firstLogicalProducer <= producerStreams.size());

    waitForDistinctProducerStreams(
        consumer,
        producerStreams.size(),
        [firstLogicalProducer](std::size_t i) { return i >= firstLogicalProducer; },
        [&producerStreams](std::size_t i) -> const Stream& { return producerStreams[i]; },
        [&reusableEvents](std::size_t i) -> Event& { return reusableEvents[i]; });
}

}  // namespace ThorImplementation::detail
