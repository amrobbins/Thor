#include "Utilities/Common/ReusableEventPool.h"

#include "DeepLearning/Implementation/ThorError.h"

#include <utility>

namespace ThorImplementation {

namespace {

size_t boolIndex(bool value) { return value ? 1u : 0u; }

}  // namespace

Event ReusableEventPool::acquire(int32_t gpuNum, bool enableTiming, bool expectingHostToWaitOnThisOne) {
    std::vector<Event> &events = freeEvents[gpuNum][boolIndex(enableTiming)][boolIndex(expectingHostToWaitOnThisOne)];
    if (events.empty()) {
        return Event(gpuNum, enableTiming, expectingHostToWaitOnThisOne);
    }

    Event event = std::move(events.back());
    events.pop_back();
    THOR_THROW_IF_FALSE(event.isInitialized());
    THOR_THROW_IF_FALSE(event.getGpuNum() == gpuNum);
    THOR_THROW_IF_FALSE(event.usesTiming() == enableTiming);
    THOR_THROW_IF_FALSE(event.usesBlockingSync() == expectingHostToWaitOnThisOne);
    return event;
}

void ReusableEventPool::release(Event event) {
    THOR_THROW_IF_FALSE(event.isInitialized());
    freeEvents[event.getGpuNum()][boolIndex(event.usesTiming())][boolIndex(event.usesBlockingSync())].push_back(
        std::move(event));
}

size_t ReusableEventPool::freeEventCountForTests(int32_t gpuNum,
                                                 bool enableTiming,
                                                 bool expectingHostToWaitOnThisOne) const {
    auto gpuIt = freeEvents.find(gpuNum);
    if (gpuIt == freeEvents.end()) {
        return 0;
    }
    return gpuIt->second[boolIndex(enableTiming)][boolIndex(expectingHostToWaitOnThisOne)].size();
}

ReusableEventPool &threadLocalReusableEventPool() {
    thread_local ReusableEventPool pool;
    return pool;
}

ReusableEventLeases::ReusableEventLeases(size_t expectedEventCount) : pool(threadLocalReusableEventPool()) {
    leasedEvents.reserve(expectedEventCount);
}

Event ReusableEventLeases::acquire(int32_t gpuNum, bool enableTiming, bool expectingHostToWaitOnThisOne) {
    Event event = pool.acquire(gpuNum, enableTiming, expectingHostToWaitOnThisOne);
    leasedEvents.push_back(event);
    return event;
}

ReusableEventLeases::~ReusableEventLeases() {
    for (Event &event : leasedEvents) {
        pool.release(std::move(event));
    }
}

}  // namespace ThorImplementation
