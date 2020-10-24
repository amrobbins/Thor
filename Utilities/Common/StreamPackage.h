#pragma once

#include "Utilities/Common/Stream.h"

#include <vector>

using std::vector;

class StreamPackage {
   public:
    StreamPackage() { initialized = false; }

    StreamPackage(uint32_t gpuNum, uint32_t maxStreams, Stream::Priority priority = Stream::Priority::REGULAR) {
        assert(maxStreams > 0);

        this->gpuNum = gpuNum;
        this->maxStreams = maxStreams;
        this->priority = priority;
        nextIndex = 0;

        for (uint32_t i = 0; i < maxStreams; ++i) {
            streams.emplace_back(gpuNum, priority);
            streams.back().getCudnnHandle();
        }

        initialized = true;
    }

    virtual ~StreamPackage() {}

    Stream getStream() {
        assert(initialized);

        if (nextIndex >= streams.size())
            nextIndex = 0;

        Stream stream = streams[nextIndex];
        nextIndex += 1;

        return stream;
    }

    bool isInitialized() { return initialized; }

   private:
    uint32_t gpuNum;
    uint32_t maxStreams;
    Stream::Priority priority;

    uint32_t nextIndex;

    vector<Stream> streams;

    bool initialized;
};
