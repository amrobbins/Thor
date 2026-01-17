#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/WorkQueue/AsyncQueue.h"
#include "Utilities/ThreadPool/ThreadPool.h"

#include <deque>

struct ArchiveFileWriteParams {
    ThorImplementation::Tensor deviceTensor;

    uint64_t offsetBytes;  // When copying tensors in segments, copy them as uint8_t.
    uint64_t sizeBytes;
    std::string path_in_tar;
    uint32_t fileShard;
    uint32_t archiveShard;
};

class ArchiveShardWriter {
   public:
    ArchiveShardWriter(AsyncQueue<ArchiveFileWriteParams>& queue) : queue(queue) {
        // Instantiate tensor...
        // dataBuffer = ...
    }

    ArchiveShardWriter(const ArchiveShardWriter& other) : queue(other.queue) { dataBuffer = other.dataBuffer.clone(); }

    // Returns true when there is more work to do.
    bool process() {
        bool workToDo = queue.pop(writeFileParams);
        if (!workToDo)
            return false;

        // do work
        // call addArchiveFile

        return true;
    }

    ArchiveFileWriteParams writeFileParams;
    AsyncQueue<ArchiveFileWriteParams>& queue;

   protected:
    ThorImplementation::Tensor dataBuffer;
};
