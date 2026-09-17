#pragma once

#include "Utilities/Common/Stream.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ThorImplementation::detail {

// A completion event together with the stable identity of the physical stream
// on which that event was recorded. Keeping only the id avoids adding Stream
// shared-ownership traffic to the backward hot path while retaining exactly
// the provenance needed for join coalescing.
struct ProducerCompletionEvent {
    uint64_t producerStreamId = 0;
    Event completionEvent;
};

// Join a consumer to the latest known completion event from each distinct
// external physical producer stream. Callers append ProducerCompletionEvent
// entries in host submission order immediately after recording each event.
// For two events recorded on the same producer stream, CUDA stream ordering
// therefore guarantees that the later appended event dominates the earlier
// one. Scanning newest-to-oldest keeps exactly that dominating event.
//
// This deliberately performs no transitive reduction between different
// producer streams. The O(N^2) alias scan avoids hot-path allocations; the
// number of backward connections/applications for one layer is expected to be
// small.
inline void waitForLatestCompletionPerProducerStream(
    const Stream& consumer,
    const std::vector<ProducerCompletionEvent>& completions) {
    THOR_THROW_IF_FALSE(consumer.isInitialized());
    const uint64_t consumerStreamId = consumer.getId();
    THOR_THROW_IF_FALSE(consumerStreamId != 0);

    for (std::size_t reverseIndex = completions.size(); reverseIndex > 0; --reverseIndex) {
        const std::size_t i = reverseIndex - 1;
        const ProducerCompletionEvent& completion = completions[i];
        THOR_THROW_IF_FALSE(completion.producerStreamId != 0);
        THOR_THROW_IF_FALSE(completion.completionEvent.isInitialized());

        // Work recorded on the consumer stream is already ordered before all
        // subsequently enqueued consumer work.
        if (completion.producerStreamId == consumerStreamId)
            continue;

        bool newerCompletionFromSameProducer = false;
        for (std::size_t j = i + 1; j < completions.size(); ++j) {
            THOR_THROW_IF_FALSE(completions[j].producerStreamId != 0);
            if (completion.producerStreamId == completions[j].producerStreamId) {
                newerCompletionFromSameProducer = true;
                break;
            }
        }

        if (!newerCompletionFromSameProducer)
            consumer.waitEvent(completion.completionEvent);
    }
}

}  // namespace ThorImplementation::detail
