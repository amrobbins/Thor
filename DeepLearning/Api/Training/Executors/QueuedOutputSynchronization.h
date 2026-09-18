#pragma once

#include "Utilities/Common/Event.h"
#include "Utilities/Common/Stream.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace Thor::detail {

struct QueuedOutputReadyDependency {
    std::string outputName;
    Stream producerStream;
};

// processingFinishedEvent already dominates NetworkOutput ready events produced
// on the stamp's ordinary processing streams, so they are not represented here.
// Auxiliary output/offload streams remain independent and need one queued
// completion dependency unless the completion stream is that same physical stream.
inline uint64_t waitForIndependentOutputReadyEvents(
    Stream& completionStream,
    const std::map<std::string, Event>& outputReadyEvents,
    const std::vector<QueuedOutputReadyDependency>& dependencies) {
    uint64_t waitCount = 0;
    for (const QueuedOutputReadyDependency& dependency : dependencies) {
        if (completionStream == dependency.producerStream)
            continue;
        auto eventIt = outputReadyEvents.find(dependency.outputName);
        THOR_THROW_IF_FALSE(eventIt != outputReadyEvents.end());
        completionStream.waitEvent(eventIt->second);
        waitCount += 1;
    }
    return waitCount;
}

}  // namespace Thor::detail
