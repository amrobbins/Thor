#pragma once

#include "Utilities/Common/Stream.h"

#include <cstddef>
#include <vector>

namespace ThorImplementation::detail {

// Fan one producer completion event out to the minimal set of distinct
// physical target streams represented by a set of logical consumers. CUDA
// stream ordering already covers targets that alias the producer stream, and
// multiple logical targets that alias one external stream need only one wait.
// This intentionally performs only physical-stream alias coalescing; it does
// not attempt transitive happens-before reduction. The O(N^2) scan avoids a
// set/vector allocation in the per-batch hot path; layer fanout counts are
// expected to be small.
template <typename IsActive, typename TargetStreamAt>
inline void waitOnDistinctTargetStreams(const Stream& producer,
                                        const Event& producerCompletionEvent,
                                        std::size_t logicalTargetCount,
                                        IsActive&& isActive,
                                        TargetStreamAt&& targetStreamAt) {
    THOR_THROW_IF_FALSE(producer.isInitialized());
    THOR_THROW_IF_FALSE(producerCompletionEvent.isInitialized());

    for (std::size_t i = 0; i < logicalTargetCount; ++i) {
        if (!isActive(i))
            continue;

        const Stream& target = targetStreamAt(i);
        THOR_THROW_IF_FALSE(target.isInitialized());

        // Stream ordering already supplies this dependency.
        if (target == producer)
            continue;

        bool targetAlreadyJoined = false;
        for (std::size_t j = 0; j < i; ++j) {
            if (!isActive(j))
                continue;

            const Stream& earlierTarget = targetStreamAt(j);
            THOR_THROW_IF_FALSE(earlierTarget.isInitialized());
            if (target == earlierTarget) {
                targetAlreadyJoined = true;
                break;
            }
        }

        if (!targetAlreadyJoined)
            target.waitEvent(producerCompletionEvent);
    }
}

inline void waitOnDistinctTargetStreams(const Stream& producer,
                                        const Event& producerCompletionEvent,
                                        const std::vector<Stream>& targetStreams,
                                        std::size_t firstLogicalTarget = 0) {
    THOR_THROW_IF_FALSE(firstLogicalTarget <= targetStreams.size());

    waitOnDistinctTargetStreams(
        producer,
        producerCompletionEvent,
        targetStreams.size(),
        [firstLogicalTarget](std::size_t i) { return i >= firstLogicalTarget; },
        [&targetStreams](std::size_t i) -> const Stream& { return targetStreams[i]; });
}

// Record one producer-tail completion point and publish it to the minimal set
// of distinct external target streams. If every active logical target aliases
// the producer stream, stream ordering already supplies all dependencies and
// no CUDA event operation is emitted.
template <typename IsActive, typename TargetStreamAt>
inline void recordCompletionAndWaitOnDistinctTargetStreams(const Stream& producer,
                                                           Event& reusableCompletionEvent,
                                                           std::size_t logicalTargetCount,
                                                           IsActive&& isActive,
                                                           TargetStreamAt&& targetStreamAt) {
    THOR_THROW_IF_FALSE(producer.isInitialized());

    bool hasExternalTarget = false;
    for (std::size_t i = 0; i < logicalTargetCount; ++i) {
        if (!isActive(i))
            continue;

        const Stream& target = targetStreamAt(i);
        THOR_THROW_IF_FALSE(target.isInitialized());
        if (target != producer) {
            hasExternalTarget = true;
            break;
        }
    }

    if (!hasExternalTarget)
        return;

    producer.putEvent(reusableCompletionEvent);
    waitOnDistinctTargetStreams(
        producer,
        reusableCompletionEvent,
        logicalTargetCount,
        isActive,
        targetStreamAt);
}

inline void recordCompletionAndWaitOnDistinctTargetStreams(const Stream& producer,
                                                           Event& reusableCompletionEvent,
                                                           const std::vector<Stream>& targetStreams,
                                                           std::size_t firstLogicalTarget = 0) {
    THOR_THROW_IF_FALSE(firstLogicalTarget <= targetStreams.size());

    recordCompletionAndWaitOnDistinctTargetStreams(
        producer,
        reusableCompletionEvent,
        targetStreams.size(),
        [firstLogicalTarget](std::size_t i) { return i >= firstLogicalTarget; },
        [&targetStreams](std::size_t i) -> const Stream& { return targetStreams[i]; });
}

}  // namespace ThorImplementation::detail
